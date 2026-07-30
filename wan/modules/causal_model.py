# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified by Applied Intuition, Inc. in 2026.
# SPDX-License-Identifier: Apache-2.0

import functools
from wan.modules.attention import attention
import math
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    rope_apply_given_freqs,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
import time
import copy
from einops import rearrange

import os as _os
try:
    import torch._dynamo as _dynamo
    _dynamo.config.recompile_limit = int(
        _os.environ.get("TORCHDYNAMO_RECOMPILE_LIMIT", "64"))
    _dynamo.config.accumulated_recompile_limit = int(
        _os.environ.get("TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT", "4096"))
    print(
        "[causal_model] torch._dynamo recompile_limit="
        f"{_dynamo.config.recompile_limit} accumulated="
        f"{_dynamo.config.accumulated_recompile_limit}")
except Exception as _e:
    print(f"[causal_model] could not set torch._dynamo limits: {_e}")

if _os.environ.get("DISABLE_FLEX_COMPILE") == "1":
    # Env-gated escape hatch: keep flex_attention in eager mode. Removes
    # torch.compile + Triton autotune non-determinism. Used to isolate whether
    # compile-layer variability explains cross-run output divergence.
    print("[causal_model] DISABLE_FLEX_COMPILE=1 — flex_attention stays eager")
else:
    _flex_compile_mode = _os.environ.get("FLEX_COMPILE_MODE", "max-autotune-no-cudagraphs")
    print(f"[causal_model] flex_attention torch.compile mode={_flex_compile_mode}")
    flex_attention = torch.compile(
        flex_attention, dynamic=False, mode=_flex_compile_mode)

if _os.environ.get("REMIND_DMD_CHUNK_FLASH", "0") == "1":
    print("[causal_model] DMD chunk-causal attention uses chunked FlashAttention")


_VALID_CC_ROPE_MODES = (
    "standard",          # Wan original t/h/w RoPE; no camera/ProPE modules
    "dual_prope",        # full dual-attention PRoPE, no QK phase
    "cc_basic",          # QK phase (all D/2 slots), no PRoPE output residual
    "cc_output",         # QK phase (all D/2 slots) + P·x_std + prope_proj
    "cc_dual_channel",   # QK phase (last cc_phase_slots slots only), no PRoPE
    "cc_dual_output",    # QK phase (last cc_phase_slots slots only) + P·x_std + prope_proj
    "prope_residual",    # single-attention, P·x_std + prope_proj, NO QK phase
    "cc_value",          # QK phase (all D/2 slots) + V-side P_inv residual via value_proj
    "cc_full",           # QK phase (all D/2 slots) + V-side (value_proj) + O-side (prope_proj) residuals
)


def _camera_pose_features(viewmats: torch.Tensor,
                          Ks: torch.Tensor = None) -> torch.Tensor:
    """Build a compact per-frame pose descriptor from viewmats (+ Ks).

    Args:
        viewmats: [B, F, 4, 4] c2w, already translation-normalized upstream
                   (see CausalWanModel.forward).
        Ks:       [B, F, 3, 3] intrinsics, already focal-normalized upstream.
                   Provides focal/zoom signal (None for datasets without Ks).

    Returns:
        pose_c: [B, F, P] float32 descriptor. P depends on whether Ks is given:
          translation (3) + R flattened (9) + log-focal (2 if Ks else 0)  →  14 or 12.
    """
    assert viewmats.dim() == 4 and viewmats.shape[-2:] == (4, 4)
    B_, F_ = viewmats.shape[:2]
    t = viewmats[..., :3, 3].float()                           # [B, F, 3]
    R = viewmats[..., :3, :3].float().reshape(B_, F_, 9)       # [B, F, 9]
    feats = [t, R]
    if Ks is not None:
        # log-focal: Ks already ~O(1) after normalization, clamp for safety.
        fx = Ks[..., 0, 0].float().clamp(min=1e-3)
        fy = Ks[..., 1, 1].float().clamp(min=1e-3)
        lf = torch.stack([fx.log(), fy.log()], dim=-1)          # [B, F, 2]
        feats.append(lf)
    return torch.cat(feats, dim=-1)                             # [B, F, P]


class CameraPhaseMLP(nn.Module):
    """Zero-init linear layer: pose_c [B, F, P] → phase phi [B, F, D_half].

    Returns a complex phasor tensor `exp(j * phi)` ready to multiply against
    the per-token `freqs_x` of shape [F*H*W, 1, D_half]. When mask_range is
    set, phase is produced only for slots in [lo, hi); the others output 0
    phase → `exp(j*0) = 1` → those freq slots are untouched.

    At step 0 (weight=0) phase is identically 0, `exp(0)=1`, so attention
    output is bit-exact with a model that had no CameraPhaseMLP.
    """
    def __init__(self,
                 pose_dim: int,
                 head_dim_half: int,
                 mask_range: tuple = None):
        super().__init__()
        self.pose_dim = pose_dim
        self.head_dim_half = head_dim_half
        self.mask_range = mask_range  # None = all slots, else (lo, hi)
        # zero-init so phase_delta=0 ⇒ phasor=1+0j ⇒ freqs unchanged
        out_dim = head_dim_half if mask_range is None else (mask_range[1] - mask_range[0])
        self.proj = nn.Linear(pose_dim, out_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, pose_c: torch.Tensor) -> torch.Tensor:
        # pose_c: [B, F, P]  →  phi: [B, F, D_half] (float32 for stability of
        # cos/sin; the resulting complex phasor is multiplied into freqs which
        # are also float32/complex64 in rope_apply_given_freqs).
        B_, F_, _ = pose_c.shape
        w = self.proj.weight                                    # dtype-owner
        phi_active = torch.nn.functional.linear(
            pose_c.to(w.dtype), w).float()                      # [B, F, out_dim]
        if self.mask_range is None:
            phi = phi_active
        else:
            phi = pose_c.new_zeros(B_, F_, self.head_dim_half, dtype=torch.float32)
            lo, hi = self.mask_range
            phi[..., lo:hi] = phi_active
        return torch.polar(torch.ones_like(phi), phi)           # complex [B, F, D_half]


def _attention_with_weights(q, k, v):
    """Manual scaled_dot_product_attention that also returns the softmax weights.

    Used only when KV-cache importance probing is enabled. Slower than the
    fused kernel (no flash path) but needed because F.scaled_dot_product_attention
    does not expose attention weights.

    Shapes: q,k,v in [B, L, H, D]; returns (out [B, L_q, H, D], weights [B, H, L_q, L_kv]).
    Computes in float32 for numerical stability then casts back.
    """
    q_ = q.permute(0, 2, 1, 3)                             # [B, H, L_q, D]
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    scale = q_.shape[-1] ** -0.5
    scores = torch.einsum('bhld,bhmd->bhlm', q_.float(), k_.float()) * scale
    attn = torch.softmax(scores, dim=-1)                   # [B, H, L_q, L_kv], float32
    out = torch.einsum('bhlm,bhmd->bhld', attn, v_.float()).to(v_.dtype)
    return out.permute(0, 2, 1, 3), attn


def _attention_with_frame_importance(
        q, k, v, *, hist_len: int, tokens_per_frame: int,
        query_chunk_size: int = 256):
    """Exact attention plus per-history-frame weights with bounded memory."""
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    batch, heads, query_len, head_dim = q_.shape
    num_hist = (
        hist_len // tokens_per_frame
        if tokens_per_frame > 0 and hist_len % tokens_per_frame == 0
        else 0
    )
    importance_sum = torch.zeros(
        heads, num_hist, device=q.device, dtype=torch.float32)
    outputs = []
    scale = head_dim ** -0.5
    chunk = max(1, int(query_chunk_size))
    for start in range(0, query_len, chunk):
        q_part = q_[:, :, start:start + chunk]
        scores = torch.einsum(
            'bhld,bhmd->bhlm', q_part.float(), k_.float()) * scale
        attn = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum(
            'bhlm,bhmd->bhld', attn, v_.float()).to(v_.dtype))
        if num_hist:
            hist = attn[..., :hist_len].reshape(
                batch, heads, q_part.shape[2], num_hist, tokens_per_frame)
            importance_sum += hist.sum(dim=-1).sum(dim=(0, 2))
        del scores, attn
    output = torch.cat(outputs, dim=2).permute(0, 2, 1, 3)
    importance = importance_sum / max(1, batch * query_len)
    return output, importance


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6,
                 cc_rope_mode: str = "dual_prope",
                 cc_pose_dim: int = 14,
                 cc_phase_slots: int = 16):
        assert dim % num_heads == 0
        super().__init__()
        assert cc_rope_mode in _VALID_CC_ROPE_MODES, (
            f"cc_rope_mode={cc_rope_mode!r} must be one of {_VALID_CC_ROPE_MODES}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.cc_rope_mode = cc_rope_mode
        self.fused_projections = False

        # layers (standard self-attention)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # ProPE dual-path output projection (zero-init → initial no-op).
        # See HY-WorldPlay's img_attn_prope_proj.
        # Only instantiated where it's actually read in forward:
        #   - standard        : no ProPE / camera residual modules at all
        #   - dual_prope      : gate for PRoPE 2nd-attention output (existing path)
        #   - cc_output       : gate for P·x_std (single-attention, camera-projected)
        #   - cc_dual_output  : same as cc_output but with masked-range QK phase
        #   - prope_residual  : gate for P·x_std WITHOUT any QK phase (isolated
        #                       residual-only branch for clean ablation vs cc_output)
        #   - cc_full         : gate for P·x_std (paired with V-side value_proj)
        # `cc_basic` / `cc_dual_channel` / `cc_value` do NOT use this module, so
        # we leave it as None to save ~30 × (dim² + dim) params and the
        # corresponding AdamW optimizer state (~1.6 GB on the 1.3B model). Any
        # stale prope_proj.* keys from pretrained checkpoints become harmless
        # "unexpected_keys" on load (strict=False).
        if cc_rope_mode in ("dual_prope", "cc_output", "cc_dual_output", "prope_residual", "cc_full"):
            self.prope_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.prope_proj.weight)
            if self.prope_proj.bias is not None:
                nn.init.zeros_(self.prope_proj.bias)
        else:
            self.prope_proj = None

        # cc_value / cc_full: zero-init Linear that mixes (P_inv · V) back into
        # V before attention, providing a "value-side" camera-geometry residual
        # that mirrors cc_output's "x_std-side" residual. Step-0 bit-exact
        # because the residual contributes 0 when value_proj.weight = bias = 0.
        # cc_full uses BOTH value_proj (V-side) and prope_proj (O-side) to
        # close the dual_prope geometric loop (V → world → attn → P → query
        # frame), but with zero-init residuals instead of replacements.
        if cc_rope_mode in ("cc_value", "cc_full"):
            self.value_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.value_proj.weight)
            if self.value_proj.bias is not None:
                nn.init.zeros_(self.value_proj.bias)
        else:
            self.value_proj = None

        # Camera-conditioned RoPE (CC-RoPE): zero-init per-freq phase delta.
        head_dim_half = self.head_dim // 2
        self.camera_phase_mlp = None
        if cc_rope_mode in ("cc_basic", "cc_output", "cc_value", "cc_full"):
            # full-band: all freq slots receive camera phase
            self.camera_phase_mlp = CameraPhaseMLP(
                pose_dim=cc_pose_dim,
                head_dim_half=head_dim_half,
                mask_range=None,
            )
        elif cc_rope_mode in ("cc_dual_channel", "cc_dual_output"):
            # M-RoPE style: camera phase only on the LAST `cc_phase_slots` freqs.
            # cc_dual_channel: masked phase, no PRoPE output residual.
            # cc_dual_output : masked phase + PRoPE output residual (P·x_std).
            lo = max(0, head_dim_half - cc_phase_slots)
            self.camera_phase_mlp = CameraPhaseMLP(
                pose_dim=cc_pose_dim,
                head_dim_half=head_dim_half,
                mask_range=(lo, head_dim_half),
            )

        # ── KV cache importance probing (off by default; no training impact) ──
        # When enabled, the cache-read branch of forward() swaps the fused
        # attention kernel for a manual path that exposes softmax weights, then
        # accumulates per-history-frame mean weight across denoising steps.
        self._log_attn: bool = False
        self._log_tpf: int = 0          # tokens per frame (set by pipeline)
        self._log_chunk_size: int = 0   # chunks-of-frames granularity (chunk_mask path)
        self._attn_sum = None           # [H, num_hist_frames]      — kv_cache path
        self._attn_count: int = 0
        self._attn_chunk_matrix = None  # [H, T_frames_Q, num_chunks_K] — chunk_mask path

    @staticmethod
    def _chunk_flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tokens_per_chunk: int,
    ) -> torch.Tensor:
        """Exact chunk-causal attention using dense FlashAttention calls."""
        total_tokens = q.shape[1]
        if (
            k.shape[1] != total_tokens
            or v.shape[1] != total_tokens
            or tokens_per_chunk <= 0
            or total_tokens % tokens_per_chunk != 0
        ):
            raise ValueError(
                "chunk flash expects equal, unpadded q/k/v sequences divisible "
                f"by tokens_per_chunk: q={tuple(q.shape)} k={tuple(k.shape)} "
                f"v={tuple(v.shape)} tokens_per_chunk={tokens_per_chunk}"
            )

        outputs = []
        for start in range(0, total_tokens, tokens_per_chunk):
            end = start + tokens_per_chunk
            outputs.append(attention(q[:, start:end], k[:, :end], v[:, :end]))
        return torch.cat(outputs, dim=1)

    def _clean_history_chunk_flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        half_tokens: int,
        tokens_per_chunk: int,
    ) -> torch.Tensor:
        """Structured clean-history attention via FlashAttention.

        The dual-stream clean-history mask is regular at chunk granularity:
        clean chunk i attends to clean chunks <= i, and noisy chunk i attends
        to clean chunks < i plus its own noisy chunk. Splitting those blocks
        keeps the exact attention support while avoiding generic FlexAttention
        backward, which is very slow for this 65k-token mask on A100.
        """
        total_tokens = half_tokens * 2
        if (
            q.shape[1] != total_tokens
            or k.shape[1] != total_tokens
            or v.shape[1] != total_tokens
            or half_tokens <= 0
            or tokens_per_chunk <= 0
            or half_tokens % tokens_per_chunk != 0
        ):
            raise ValueError(
                "clean-history chunk flash expects unpadded [clean,noisy] "
                f"tokens: q={tuple(q.shape)} k={tuple(k.shape)} "
                f"half_tokens={half_tokens} tokens_per_chunk={tokens_per_chunk}"
            )

        clean_out = []
        noisy_out = []
        for start in range(0, half_tokens, tokens_per_chunk):
            end = start + tokens_per_chunk
            clean_out.append(attention(q[:, start:end], k[:, :end], v[:, :end]))

            noisy_start = half_tokens + start
            noisy_end = half_tokens + end
            if start > 0:
                k_noisy = torch.cat(
                    [k[:, :start], k[:, noisy_start:noisy_end]], dim=1)
                v_noisy = torch.cat(
                    [v[:, :start], v[:, noisy_start:noisy_end]], dim=1)
            else:
                k_noisy = k[:, noisy_start:noisy_end]
                v_noisy = v[:, noisy_start:noisy_end]
            noisy_out.append(
                attention(q[:, noisy_start:noisy_end], k_noisy, v_noisy))

        return torch.cat(clean_out + noisy_out, dim=1)

    def enable_attn_log(self, tokens_per_frame: int, enable: bool = True,
                         chunk_size: int = 0):
        """Toggle attention logging. Resets accumulators.

        `tokens_per_frame`: size of one video frame in token units (required
            for both the kv_cache-streaming and chunk_mask logging paths).
        `chunk_size`: only used by the chunk_mask path (single-step training-
            style forward).  Set to the same chunk_size the forward was called
            with so the K-side can be aggregated to per-chunk columns.
        """
        self._log_attn = bool(enable)
        self._log_tpf = int(tokens_per_frame)
        self._log_chunk_size = int(chunk_size)
        self._attn_sum = None
        self._attn_count = 0
        self._attn_chunk_matrix = None

    def get_attn_log(self):
        """Return mean per-history-frame attention [H, num_hist] or None.
        (kv_cache-streaming inference path only.)
        """
        if self._attn_sum is None or self._attn_count == 0:
            return None
        return self._attn_sum / self._attn_count

    def get_attn_chunk_matrix(self):
        """Return per-Q-frame → per-K-chunk attention mass for the chunk_mask
        (single-step full-sequence) forward path.  Shape [H, T_frames_Q,
        num_chunks_K] on CPU fp32, or None if never captured.
        """
        return self._attn_chunk_matrix

    def _manual_chunk_mask_attention(self, q, k, v, tpf: int, cs: int):
        """Chunk-causal self-attention with softmax-weight bookkeeping.

        Used ONLY by the `_log_attn` probe path in the chunk_mask branch of
        `forward()`.  Replaces `flex_attention` with an explicit per-Q-frame
        loop so that we can aggregate softmax mass into a
        `[H, T_frames_Q, num_chunks_K]` matrix (cached on
        `self._attn_chunk_matrix`) and visualise it as a KV-importance heatmap.

        Memory is kept manageable by looping over Q frames: peak intermediate
        tensor is `[B, H, tpf, L_kv]`, not the full `[B, H, L, L]`.

        Args:
          q, k, v : [B, L, H, D] (RoPE already applied to q, k by caller).
          tpf     : tokens per latent frame.
          cs      : chunk_size (frames per chunk).  K is aggregated at this
                     granularity; Q stays per-frame for finer rows.
        Returns:
          x_std  : [B, L, H, D] attention output (same shape/semantics as
                    flex_attention with the chunk-causal block mask).
        """
        import math
        B, L, H, D = q.shape
        assert L % (tpf * cs) == 0, \
            f"L={L} not divisible by tpf*cs={tpf*cs}"
        T_frames = L // tpf
        num_chunks = T_frames // cs
        scale = 1.0 / math.sqrt(D)

        q_h = q.permute(0, 2, 1, 3).contiguous()      # [B, H, L, D]
        k_h = k.permute(0, 2, 1, 3).contiguous()
        v_h = v.permute(0, 2, 1, 3).contiguous()

        attn_mat = torch.zeros(H, T_frames, num_chunks,
                                device=q_h.device, dtype=torch.float32)
        out_per_frame = []
        for qf in range(T_frames):
            qs, qe = qf * tpf, (qf + 1) * tpf
            qc = qf // cs
            ke = (qc + 1) * cs * tpf                  # causal K extent
            q_i = q_h[:, :, qs:qe, :]                 # [B, H, tpf, D]
            k_i = k_h[:, :, :ke, :]                   # [B, H, ke, D]
            v_i = v_h[:, :, :ke, :]

            # bf16/fp16 matmul is plenty for softmax; accumulate in fp32.
            logits = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale
            weights = logits.softmax(dim=-1)          # [B, H, tpf, ke]

            out_i = torch.matmul(weights.to(v_i.dtype), v_i)  # [B, H, tpf, D]
            out_per_frame.append(out_i)

            # K-chunk aggregation: group ke=visible*cs*tpf into (visible, cs*tpf) and sum.
            visible = qc + 1
            w_ck = weights.float().view(
                B, H, tpf, visible, cs * tpf,
            ).sum(dim=-1).mean(dim=(0, 2))             # [H, visible]
            attn_mat[:, qf, :visible] = w_ck

        x_std_h = torch.cat(out_per_frame, dim=2)     # [B, H, L, D]
        x_std = x_std_h.permute(0, 2, 1, 3)           # [B, L, H, D]
        self._attn_chunk_matrix = attn_mat.detach().cpu()
        return x_std

    def _accumulate_attn(self, attn_weights: torch.Tensor, hist_len: int):
        """attn_weights: [B, H, L_q, L_kv]. hist_len tokens at the front of L_kv
        belong to the history cache; the tail is current-chunk self-attention."""
        if hist_len <= 0 or self._log_tpf <= 0:
            return
        num_hist = hist_len // self._log_tpf
        if num_hist == 0 or hist_len % self._log_tpf != 0:
            return
        B, H, L_q, _ = attn_weights.shape
        hist = attn_weights[:, :, :, :hist_len].reshape(
            B, H, L_q, num_hist, self._log_tpf).sum(dim=-1)   # [B, H, L_q, num_hist]
        pf = hist.mean(dim=(0, 2)).detach()                    # [H, num_hist]
        self._accumulate_attn_summary(pf)

    def _accumulate_attn_summary(self, pf: torch.Tensor):
        if pf is None or pf.numel() == 0:
            return
        pf = pf.detach()
        if self._attn_sum is None or self._attn_sum.shape != pf.shape:
            self._attn_sum = pf.clone()
            self._attn_count = 1
        else:
            self._attn_sum = self._attn_sum + pf
            self._attn_count += 1

    def forward(
        self,
        x,
        seq_lens,
        freqs,
        kv_cache=None,
        kv_size=(0,0),
        viewmats=None,
        Ks=None,
        pose_c=None,
        tokens_per_frame: int = None,
        chunk_mask=None,
        prope_temporal_dim: int = 0,
        prope_freqs_offset: int = 0,
        prope_freqs_positions=None,
        sequence_parallel: bool = False,
        sequence_parallel_clean_history: bool = False,
        chunk_flash: bool = False,
        chunk_flash_tokens_per_chunk: int = 0,
        clean_history_chunk_flash: bool = False,
        clean_history_half_tokens: int = 0,
        clean_history_tokens_per_chunk: int = 0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): RoPE freqs [L, 1, C/num_heads] (already per-token)
            kv_cache: dict with keys k, v (and optionally k_prope, v_prope).
                     Used for inference (chunk-by-chunk streaming).
            kv_size: (offset, length). length<0 means context pass (write),
                     length=0 means no cache, length>0 means read [offset:offset+length]
            viewmats: [B, num_frames_in_x, 4, 4] — camera-to-world per frame of x.
                      If None and cc_rope_mode='dual_prope', ProPE path is skipped.
            Ks: [B, num_frames_in_x, 3, 3] intrinsics.
            pose_c: [B, F, P] per-frame pose descriptor (CC-RoPE modes only).
                    Produced by _camera_pose_features in CausalWanModel.forward.
            tokens_per_frame: int, needed by CC-RoPE to broadcast per-frame phase
                               to per-token freqs.
            chunk_mask: flex_attention BlockMask for full-sequence training mode.
                        When provided, skips KV cache entirely; uses flex_attention
                        with sparse block-causal mask (chunk i sees chunks 0..i).
                        Built by CausalWanModel._get_chunk_block_mask (cached).
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # ─── Build effective RoPE freqs (shared or camera-modulated) ───
        # cc_basic / cc_dual_channel / cc_output: multiply per-frame phasor
        # into the base freqs so a SINGLE attention captures camera-relative
        # position. Zero-init MLP ⇒ phasor=1 ⇒ freqs_eff ≡ freqs, i.e. step-0
        # output is bit-exact with "pure standard RoPE".
        freqs_eff = freqs
        if self.camera_phase_mlp is not None and pose_c is not None:
            assert tokens_per_frame is not None, \
                "tokens_per_frame is required when cc_rope_mode uses camera phase"
            phasor = self.camera_phase_mlp(pose_c)     # [B, F, D_half] complex
            B_, F_, Dh = phasor.shape
            # Expand per-frame phasor to per-token:
            # [B, F, 1, D_half] → [B, F, tpf, D_half] → [B, F*tpf, 1, D_half]
            phasor = phasor.unsqueeze(2).expand(B_, F_, tokens_per_frame, Dh)
            phasor = phasor.reshape(B_, F_ * tokens_per_frame, 1, Dh)
            # freqs: [L, 1, D_half] complex; broadcast to [1, L, 1, D_half]
            base = freqs.unsqueeze(0) if freqs.dim() == 3 else freqs
            freqs_eff = base * phasor                   # [B, L, 1, D_half]

        # ─── Standard RoPE path (always runs; with CC-RoPE freqs if enabled) ──
        roped_query = rope_apply_given_freqs(q, freqs_eff).type_as(v)
        roped_key   = rope_apply_given_freqs(k, freqs_eff).type_as(v)

        # ─── ProPE path (optional) ────────────────────────────────────
        # dual_prope: full dual-attention PRoPE (legacy).
        # cc_output / cc_dual_output / prope_residual: single-attention +
        #             post-hoc P-projection of x_std (apply_fn_o; skip Q/K/V
        #             transforms). They differ only in whether the QK phase
        #             MLP is active (full / masked / disabled).
        # cc_value  : single-attention + pre-hoc P_inv-projection of V mixed
        #             via zero-init Linear residual (V → V + value_proj(P_inv·V)).
        # cc_full   : single-attention + BOTH cc_value (V-side P_inv residual)
        #             AND cc_output (O-side P residual). Closes the dual_prope
        #             geometric loop (V→world→attn→P→query frame) via two
        #             zero-init Linear residuals.
        # cc_basic / cc_dual_channel: no PRoPE ops at all.
        prope_apply_fn_o = None
        q_prope = k_prope = v_prope = None
        if self.cc_rope_mode == "dual_prope":
            use_prope = (viewmats is not None)
            if use_prope:
                from prope.camera_rope import prope_qkv
                q_prope, k_prope, v_prope, prope_apply_fn_o = prope_qkv(
                    q.permute(0, 2, 1, 3),
                    k.permute(0, 2, 1, 3),
                    v.permute(0, 2, 1, 3),
                    viewmats=viewmats,
                    Ks=Ks,
                    temporal_dim=prope_temporal_dim,
                    freqs_offset=prope_freqs_offset,
                    freqs_positions=prope_freqs_positions,
                )
                q_prope = q_prope.permute(0, 2, 1, 3).type_as(v)
                k_prope = k_prope.permute(0, 2, 1, 3).type_as(v)
                v_prope = v_prope.permute(0, 2, 1, 3).type_as(v)
            use_prope_dual = use_prope
        elif self.cc_rope_mode in ("cc_output", "cc_dual_output", "prope_residual"):
            # Build only the output-side P transform (no 2nd attention).
            # head_dim must be divisible by 4 for the 4×4 block-diagonal matmul.
            # These three modes share the identical output-side PRoPE residual
            # (x_std → P → prope_proj). They differ only in the QK phase MLP:
            #   cc_output      : full-band phase (all D/2 freq slots)
            #   cc_dual_output : masked phase (last cc_phase_slots only)
            #   prope_residual : NO phase at all (camera_phase_mlp is None)
            # `prope_residual` serves as the clean "residual-only" ablation
            # cell vs cc_basic (phase-only) and cc_output (phase + residual).
            use_prope_dual = False
            if viewmats is not None and self.head_dim % 4 == 0:
                from prope.camera_rope import _prepare_apply_fns_all_dim
                _, _, prope_apply_fn_o = _prepare_apply_fns_all_dim(
                    head_dim=self.head_dim,
                    viewmats=viewmats,
                    Ks=Ks,
                    patches_x=None, patches_y=None,
                    image_width=None, image_height=None,
                )
        elif self.cc_rope_mode in ("cc_value", "cc_full"):
            # Pre-attention V-side residual:
            #   V ← V + value_proj(P_inv · V)
            # `value_proj` is zero-init ⇒ residual=0 at step 0 ⇒ bit-exact.
            # cc_value: V-side residual only (no output-side prope_proj).
            # cc_full : V-side residual AND output-side P-residual (handled
            #           by also setting prope_apply_fn_o below).
            use_prope_dual = False
            if viewmats is not None and self.head_dim % 4 == 0 and self.value_proj is not None:
                from prope.camera_rope import _prepare_apply_fns_all_dim
                _, prope_apply_fn_kv, _apply_fn_o_local = _prepare_apply_fns_all_dim(
                    head_dim=self.head_dim,
                    viewmats=viewmats,
                    Ks=Ks,
                    patches_x=None, patches_y=None,
                    image_width=None, image_height=None,
                )
                # v: [B, L, H, d] → [B, H, L, d] for tiled block-diag matmul.
                v_hnd = v.permute(0, 2, 1, 3).contiguous()        # [B, H, L, d]
                v_p_hnd = prope_apply_fn_kv(v_hnd)                # [B, H, L, d]
                v_p_flat = v_p_hnd.permute(0, 2, 1, 3).flatten(2) # [B, L, dim]
                v_residual = self.value_proj(v_p_flat)            # [B, L, dim]
                v_aug_flat = v.flatten(2) + v_residual            # [B, L, dim]
                v = v_aug_flat.view(b, s, n, d).type_as(v)        # [B, L, H, d]
                # cc_full: also feed apply_fn_o for the output-side residual.
                if self.cc_rope_mode == "cc_full":
                    prope_apply_fn_o = _apply_fn_o_local
        else:
            use_prope_dual = False

        # Back-compat alias for the branches below (training + inference).
        use_prope = use_prope_dual

        # ─── Training mode: full-sequence with chunk-causal mask ────
        # chunk_mask is a flex_attention BlockMask (sparse chunk-causal),
        # built once per sequence shape by CausalWanModel._get_chunk_block_mask.
        if chunk_mask is not None:
            def _flex(q_, k_, v_, block_mask):
                if sequence_parallel:
                    from wan.distributed.ulysses import distributed_flex_attention
                    return distributed_flex_attention(
                        q_, k_, v_, block_mask,
                        is_clean_history=sequence_parallel_clean_history,
                        clean_history_half_tokens=int(clean_history_half_tokens),
                        clean_history_tokens_per_chunk=int(
                            clean_history_tokens_per_chunk),
                        clean_history_chunk_flash=(
                            bool(sequence_parallel_clean_history)
                            and _os.environ.get(
                                "REMIND_TF_CHUNK_FLASH_SP", "1") != "0"
                        ))
                if clean_history_chunk_flash:
                    return self._clean_history_chunk_flash_attention(
                        q_, k_, v_,
                        int(clean_history_half_tokens),
                        int(clean_history_tokens_per_chunk),
                    )
                if chunk_flash:
                    return self._chunk_flash_attention(
                        q_, k_, v_, int(chunk_flash_tokens_per_chunk)
                    )
                # q,k,v: [B, L, H, d] → [B, H, L, d]
                q_h = q_.permute(0, 2, 1, 3).contiguous()
                k_h = k_.permute(0, 2, 1, 3).contiguous()
                v_h = v_.permute(0, 2, 1, 3).contiguous()
                q_len = q_h.shape[2]
                kv_len = k_h.shape[2]
                target_q_len = int(block_mask.shape[-2])
                target_kv_len = int(block_mask.shape[-1])
                q_pad = target_q_len - q_len
                kv_pad = target_kv_len - kv_len
                if q_pad < 0 or kv_pad < 0:
                    raise ValueError(
                        f"BlockMask shape {block_mask.shape} is smaller than "
                        f"q/k lengths {(q_len, kv_len)}")
                if q_pad > 0:
                    q_h = torch.cat([
                        q_h,
                        q_h.new_zeros(q_h.shape[0], q_h.shape[1], q_pad, q_h.shape[3]),
                    ], dim=2)
                if kv_pad > 0:
                    k_h = torch.cat([
                        k_h,
                        k_h.new_zeros(k_h.shape[0], k_h.shape[1], kv_pad, k_h.shape[3]),
                    ], dim=2)
                    v_h = torch.cat([
                        v_h,
                        v_h.new_zeros(v_h.shape[0], v_h.shape[1], kv_pad, v_h.shape[3]),
                    ], dim=2)
                o_h = flex_attention(q_h, k_h, v_h, block_mask=block_mask)
                if q_pad > 0:
                    o_h = o_h[:, :, :q_len]
                return o_h.permute(0, 2, 1, 3)  # back to [B, L, H, d]

            # Attention-importance probe (validation only — skips the fused
            # flex kernel and runs manual SDPA per Q frame so we can aggregate
            # softmax mass by K-chunk).  Falls back to flex if the required
            # metadata isn't set.
            if (self._log_attn and self._log_tpf > 0
                    and self._log_chunk_size > 0):
                x_std = self._manual_chunk_mask_attention(
                    roped_query, roped_key, v,
                    tpf=self._log_tpf, cs=self._log_chunk_size,
                )
                # Skip ProPE during probe (not needed for the heatmap; avoids
                # another expensive manual attention).  Falls back to zero for
                # the residual output so outputs in _combine_out aren't NaN.
                x_p = None if not use_prope else _flex(
                    q_prope, k_prope, v_prope, chunk_mask)
                return self._combine_out(x_std, x_p, use_prope, prope_apply_fn_o)

            x_std = _flex(roped_query, roped_key, v, chunk_mask)
            if use_prope:
                x_p = _flex(q_prope, k_prope, v_prope, chunk_mask)
            else:
                x_p = None
            return self._combine_out(x_std, x_p, use_prope, prope_apply_fn_o)

        # ─── Inference mode: KV cache streaming ─────────────────────
        if kv_cache is None:
            raise RuntimeError(
                "CausalWanSelfAttention: neither chunk_mask (training) nor "
                "kv_cache (inference) provided. If this is a training call, "
                "make sure chunk_size is reaching CausalWanModel.forward "
                "(beware of DDP wrapping hiding the forward signature)."
            )

        if kv_size[1] < 0:
            # Context pass: write to cache
            len_x = roped_query.shape[1]
            kv_cache["k"][:, kv_size[0]:kv_size[0]+len_x] = roped_key
            kv_cache["v"][:, kv_size[0]:kv_size[0]+len_x] = v
            if use_prope:
                # Store already-ProPE-transformed K/V (in world space via P_inv)
                kv_cache["k_prope"][:, kv_size[0]:kv_size[0]+len_x] = k_prope
                kv_cache["v_prope"][:, kv_size[0]:kv_size[0]+len_x] = v_prope
            # Attention on just current tokens (for residual)
            x_std = attention(roped_query, roped_key, v)
            if use_prope:
                x_p = attention(q_prope, k_prope, v_prope)
        else:
            if kv_size[1] == 0:
                # No cache, attend to current only
                x_std = attention(roped_query, roped_key, v)
                if use_prope:
                    x_p = attention(q_prope, k_prope, v_prope)
            else:
                # Read cache + current, run attention
                cache_k = kv_cache["k"][:, kv_size[0]:kv_size[0]+kv_size[1]]
                cache_v = kv_cache["v"][:, kv_size[0]:kv_size[0]+kv_size[1]]
                K_full = torch.cat([cache_k, roped_key], dim=1)
                V_full = torch.cat([cache_v, v], dim=1)
                # std-RoPE attention stays fused (positional-only; not the diagnostic
                # signal). When ProPE is inactive we fall back to logging std
                # attention so the heatmap isn't empty, but the preferred signal
                # lives on the ProPE branch below.
                if self._log_attn and not use_prope:
                    x_std, _attn_summary = _attention_with_frame_importance(
                        roped_query, K_full, V_full,
                        hist_len=kv_size[1], tokens_per_frame=self._log_tpf)
                    self._accumulate_attn_summary(_attn_summary)
                    del _attn_summary
                else:
                    x_std = attention(roped_query, K_full, V_full)
                if use_prope:
                    cache_kp = kv_cache["k_prope"][:, kv_size[0]:kv_size[0]+kv_size[1]]
                    cache_vp = kv_cache["v_prope"][:, kv_size[0]:kv_size[0]+kv_size[1]]
                    Kp_full = torch.cat([cache_kp, k_prope], dim=1)
                    Vp_full = torch.cat([cache_vp, v_prope], dim=1)
                    # ProPE Q/K are CAMERA-GEOMETRY-TRANSFORMED (Q' = P_iᵀ·Q,
                    # K' = P_j⁻¹·K). Their softmax weights encode how much the
                    # current-chunk camera geometry aligns with each history
                    # frame's camera — exactly the "which frame does the model
                    # rely on for memory" signal we want to log.
                    if self._log_attn:
                        x_p, _attn_summary = _attention_with_frame_importance(
                            q_prope, Kp_full, Vp_full,
                            hist_len=kv_size[1], tokens_per_frame=self._log_tpf)
                        self._accumulate_attn_summary(_attn_summary)
                        del _attn_summary
                    else:
                        x_p = attention(q_prope, Kp_full, Vp_full)

        if not use_prope:
            x_p = None
        return self._combine_out(x_std, x_p, use_prope, prope_apply_fn_o)

    def _combine_out(self, x_std, x_p, use_prope_dual, prope_apply_fn_o):
        """Fold std-RoPE (+ optional PRoPE) outputs into final [B, L, C].

        Three code paths:
          * dual_prope (use_prope_dual=True):
              `out = o(x_std) + prope_proj(P @ x_p_from_2nd_attn)`
          * cc_output / cc_dual_output / prope_residual / cc_full
            (prope_apply_fn_o is not None, use_prope_dual=False):
              `out = o(x_std) + prope_proj(P @ x_std)`  — camera geometry on x_std
          * cc_basic / cc_dual_channel / cc_value / no-camera: `out = o(x_std)`
            (cc_value's V-side residual is folded into x_std upstream;
             cc_full also adds the O-side P·x_std residual via prope_apply_fn_o.)

        All paths are bit-exact at step 0 when prope_proj / value_proj are
        zero-init (which WanDiffusionWrapper enforces for cc_* modes after
        checkpoint load).
        """
        x_std_flat = x_std.flatten(2)                         # [B, L, dim]
        out = self.o(x_std_flat)

        if use_prope_dual:
            # Existing dual-PRoPE: 2nd-attention output → P → prope_proj
            x_p_hnd = x_p.permute(0, 2, 1, 3)                 # [B, heads, L, d]
            x_p_hnd = prope_apply_fn_o(x_p_hnd)
            x_p_flat = x_p_hnd.permute(0, 2, 1, 3).flatten(2) # [B, L, dim]
            out = out + self.prope_proj(x_p_flat)
        elif prope_apply_fn_o is not None:
            # cc_output / cc_dual_output / prope_residual / cc_full: feed
            # P(x_std) through prope_proj (no 2nd attention). Identical
            # output-side math across these; they differ only in how (or
            # whether) the QK phase MLP writes into freqs_eff upstream and
            # whether V was pre-augmented (cc_full only).
            x_std_hnd = x_std.permute(0, 2, 1, 3)             # [B, heads, L, d]
            x_std_hnd = prope_apply_fn_o(x_std_hnd)
            x_p_flat = x_std_hnd.permute(0, 2, 1, 3).flatten(2)
            out = out + self.prope_proj(x_p_flat)

        return out


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 cc_rope_mode: str = "dual_prope",
                 cc_pose_dim: int = 14,
                 cc_phase_slots: int = 16):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, qk_norm, eps,
            cc_rope_mode=cc_rope_mode,
            cc_pose_dim=cc_pose_dim,
            cc_phase_slots=cc_phase_slots,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        freqs_x,
        context,
        context_lens,
        crossattn_cache=None,
        kv_cache=None,
        kv_size=(0,0),
        viewmats=None,
        Ks=None,
        pose_c=None,
        tokens_per_frame: int = None,
        chunk_mask=None,
        prope_temporal_dim: int = 0,
        prope_freqs_offset: int = 0,
        prope_freqs_positions=None,
        sequence_parallel: bool = False,
        sequence_parallel_clean_history: bool = False,
        chunk_flash: bool = False,
        chunk_flash_tokens_per_chunk: int = 0,
        clean_history_chunk_flash: bool = False,
        clean_history_half_tokens: int = 0,
        clean_history_tokens_per_chunk: int = 0,
        context_streams: int = 1,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C] — per-frame modulation.
                Tokens in x are assumed to be contiguous per-frame:
                L = F * tokens_per_frame, with tokens grouped by frame.
            viewmats/Ks: optional camera info for ProPE. If None, ProPE is skipped.
            chunk_mask: optional [L, L] mask → training mode (full-seq + chunk-causal).
            prope_temporal_dim: dims of head_dim allocated to temporal RoPE
                (0 = pure spatial ProPE). Passed through to self_attn.
            prope_freqs_offset: frame-index offset for temporal RoPE (matches
                the standard RoPE's freqs_offset in compact-RoPE inference).
        """
        B_, L_, D_ = x.shape
        F_ = e.shape[1]
        assert L_ % F_ == 0, f"L ({L_}) not divisible by F ({F_})"
        tpf = L_ // F_

        # modulation [1, 6, C] → broadcast with [B, F, 6, C] via unsqueeze(1)
        e_mod = self.modulation.unsqueeze(1) + e           # [B, F, 6, C]
        e_list = e_mod.chunk(6, dim=2)                      # 6 × [B, F, 1, C]

        def _mod(x_, scale, shift):
            # x_: [B, L, C]; scale/shift: [B, F, 1, C] → broadcast across tpf
            return (x_.view(B_, F_, tpf, D_) * (1 + scale) + shift).view(B_, L_, D_)

        def _res(x_, y_, gate):
            return (
                x_.view(B_, F_, tpf, D_) + y_.view(B_, F_, tpf, D_) * gate
            ).view(B_, L_, D_)

        y = self.self_attn(_mod(self.norm1(x), e_list[1], e_list[0]),
                           seq_lens, freqs_x,
                           kv_cache=kv_cache, kv_size=kv_size,
                           viewmats=viewmats, Ks=Ks,
                           pose_c=pose_c,
                           tokens_per_frame=tpf,
                           chunk_mask=chunk_mask,
                           prope_temporal_dim=prope_temporal_dim,
                           prope_freqs_offset=prope_freqs_offset,
                           prope_freqs_positions=prope_freqs_positions,
                           sequence_parallel=sequence_parallel,
                           sequence_parallel_clean_history=sequence_parallel_clean_history,
                           chunk_flash=chunk_flash,
                           chunk_flash_tokens_per_chunk=chunk_flash_tokens_per_chunk,
                           clean_history_chunk_flash=clean_history_chunk_flash,
                           clean_history_half_tokens=clean_history_half_tokens,
                           clean_history_tokens_per_chunk=clean_history_tokens_per_chunk)

        x = _res(x, y, e_list[2])

        # A 4D context carries one independent text prompt per latent chunk:
        # [B, N_chunk, L_text, C]. Vectorize cross-attention over
        # B * stream * chunk while leaving self-attention and its KV cache
        # untouched.
        if context.ndim == 4:
            B_ctx, num_contexts, text_len, context_dim = context.shape
            streams = int(context_streams)
            if B_ctx != B_ or streams <= 0 or F_ % streams != 0:
                raise ValueError(
                    "invalid chunk-text context layout: "
                    f"x_frames={F_}, context={tuple(context.shape)}, "
                    f"streams={streams}")
            frames_per_stream = F_ // streams
            if frames_per_stream % num_contexts != 0:
                raise ValueError(
                    "chunk-text contexts must evenly partition each stream: "
                    f"frames={frames_per_stream}, contexts={num_contexts}")
            frames_per_context = frames_per_stream // num_contexts
            tokens_per_context = frames_per_context * tpf
            x_grouped = self.norm3(x).view(
                B_, streams, num_contexts, tokens_per_context, D_
            ).reshape(B_ * streams * num_contexts, tokens_per_context, D_)
            context_grouped = context.unsqueeze(1).expand(
                B_, streams, num_contexts, text_len, context_dim,
            ).reshape(B_ * streams * num_contexts, text_len, context_dim)
            grouped_cache = (
                crossattn_cache if streams * num_contexts == 1 else None)
            cross_out = self.cross_attn(
                x_grouped, context_grouped, None,
                crossattn_cache=grouped_cache,
            ).view(B_, streams, num_contexts, tokens_per_context, D_)
            x = x + cross_out.reshape(B_, L_, D_)
        else:
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens,
                crossattn_cache=crossattn_cache)

        y = self.ffn(_mod(self.norm2(x), e_list[4], e_list[3]))
        x = _res(x, y, e_list[5])

        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] where L = F * tokens_per_frame
            e(Tensor): Shape [B, F, C] — per-frame time embedding.
        """
        B_, L_, D_ = x.shape
        F_ = e.shape[1]
        assert L_ % F_ == 0, f"L ({L_}) not divisible by F ({F_})"
        tpf = L_ // F_
        # modulation [1, 2, C] + e.unsqueeze(2) [B, F, 1, C] → [B, F, 2, C]
        ss = self.modulation.unsqueeze(1) + e.unsqueeze(2)
        shift, scale = ss.chunk(2, dim=2)  # each [B, F, 1, C]
        x_fr = self.norm(x).view(B_, F_, tpf, D_)
        x_fr = x_fr * (1 + scale) + shift
        return self.head(x_fr.view(B_, L_, D_))


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 # Spatiotemporal ProPE: split head_dim into spatial (ProPE) + temporal (RoPE).
                 # 0 = pure spatial (original ProPE). Must be even and leave head_dim - k
                 # divisible by 4. For head_dim=128, valid choices: 0, 8, 16, 24, 32...
                 prope_temporal_dim=0,
                 # Camera/pose-aware RoPE variants. 'standard' is the plain
                 # Wan original t/h/w RoPE control with no camera/ProPE modules.
                 # 'dual_prope' keeps legacy 2-attention path. 'cc_basic',
                 # 'cc_output', 'cc_dual_channel' fuse camera info into RoPE,
                 # running a SINGLE attention per block. All three are bit-exact
                 # identity at step 0 (zero-init MLP + prope_proj).
                 cc_rope_mode: str = "dual_prope",
                 cc_phase_slots: int = 16,
                 degradation_control_dim: int = 0,
                 degradation_control_hidden_dim: int = 256):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.prope_temporal_dim = prope_temporal_dim
        assert cc_rope_mode in _VALID_CC_ROPE_MODES, (
            f"cc_rope_mode={cc_rope_mode!r} must be one of {_VALID_CC_ROPE_MODES}")
        self.cc_rope_mode = cc_rope_mode
        self.cc_phase_slots = cc_phase_slots
        self.degradation_control_dim = int(degradation_control_dim)
        self.degradation_control_hidden_dim = int(
            degradation_control_hidden_dim)
        # Pose descriptor size: 3 (t) + 9 (R flat) + 2 (log fx, log fy) = 14.
        # Ks is optional; when absent, _camera_pose_features drops the 2 log-focal
        # entries (→ 12). We size the MLP for the MAX (14) and pad with zeros at
        # runtime if Ks is missing.
        self.cc_pose_dim = 14

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.degradation_control_embedding = None
        if self.degradation_control_dim > 0:
            self.degradation_control_embedding = nn.Sequential(
                nn.Linear(
                    self.degradation_control_dim,
                    self.degradation_control_hidden_dim,
                    bias=False,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.degradation_control_hidden_dim, dim, bias=False),
            )
            self.reset_degradation_control_parameters()

        # blocks
        cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads, qk_norm,
                cross_attn_norm, eps,
                cc_rope_mode=cc_rope_mode,
                cc_pose_dim=self.cc_pose_dim,
                cc_phase_slots=cc_phase_slots,
            )
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0

        self.gradient_checkpointing = False

    def reset_degradation_control_parameters(self):
        """Initialize a missing control adapter as an exact identity branch."""
        if self.degradation_control_embedding is None:
            return
        first = self.degradation_control_embedding[0]
        last = self.degradation_control_embedding[2]
        first.reset_parameters()
        last.reset_parameters()
        with torch.no_grad():
            last.weight.zero_()



    def get_transformer_module(self):
        return {type(self.blocks[0])}

    def init_freqs(self,device):
        d = self.dim // self.num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        self.freqs = self.freqs.to(device)

    def _get_chunk_block_mask(self, total_tokens, tokens_per_chunk, device):
        """
        Build (or fetch from cache) a flex_attention BlockMask for chunk-causal
        attention. Keyed on (total_tokens, tokens_per_chunk, device) so we only
        compile once per sequence shape.
        """
        if not hasattr(self, "_block_mask_cache"):
            self._block_mask_cache = {}
        padded_total_tokens = math.ceil(total_tokens / 128) * 128
        key = (total_tokens, tokens_per_chunk, str(device))
        cached = self._block_mask_cache.get(key)
        if cached is not None:
            return cached

        def chunk_causal(b, h, q_idx, kv_idx):
            # chunk index of query vs key; q's chunk must be >= kv's chunk
            valid = (q_idx < total_tokens) & (kv_idx < total_tokens)
            causal = (q_idx // tokens_per_chunk) >= (kv_idx // tokens_per_chunk)
            return (valid & causal) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            chunk_causal,
            B=None, H=None,
            Q_LEN=padded_total_tokens, KV_LEN=padded_total_tokens,
            device=device,
            _compile=True,
        )
        self._block_mask_cache[key] = block_mask
        return block_mask

    def _get_clean_history_chunk_block_mask(self, half_tokens, tokens_per_chunk, device):
        """
        Build a BlockMask for a doubled [near_clean, noisy] training sequence.

        Layout:
          - tokens [0, half_tokens) are near-clean GT history.
          - tokens [half_tokens, 2 * half_tokens) are the noisy denoising stream.

        A noisy chunk can see only its own noisy chunk plus previous
        near-clean chunks.  A near-clean chunk can see near-clean chunks up to
        itself.  This mirrors Self-Forcing's clean-history mask.
        """
        if not hasattr(self, "_block_mask_cache"):
            self._block_mask_cache = {}
        total_tokens = half_tokens * 2
        padded_total_tokens = math.ceil(total_tokens / 128) * 128
        key = ("clean_history_clean_noisy", half_tokens, tokens_per_chunk, str(device))
        cached = self._block_mask_cache.get(key)
        if cached is not None:
            return cached

        def clean_history_mask(b, h, q_idx, kv_idx):
            # Keep this as pure index math so create_block_mask can compile it.
            # Semantics match the previous lookup-table mask:
            # clean chunk i -> clean chunks [0..i]
            # noisy chunk i -> noisy chunk i + clean chunks [0..i-1]
            is_clean_q = q_idx < half_tokens
            clean_chunk_end = ((q_idx // tokens_per_chunk) + 1) * tokens_per_chunk
            clean_mask = (
                is_clean_q
                & (kv_idx < clean_chunk_end)
                & (kv_idx < half_tokens)
            )

            is_noisy_q = (q_idx >= half_tokens) & (q_idx < total_tokens)
            noisy_rel_idx = q_idx - half_tokens
            noisy_chunk_idx = noisy_rel_idx // tokens_per_chunk
            noisy_start = half_tokens + noisy_chunk_idx * tokens_per_chunk
            noisy_end = noisy_start + tokens_per_chunk
            noisy_self_chunk = (
                is_noisy_q
                & (kv_idx >= noisy_start)
                & (kv_idx < noisy_end)
            )
            clean_context_end = noisy_chunk_idx * tokens_per_chunk
            noisy_prev_clean = (
                is_noisy_q
                & (kv_idx < clean_context_end)
            )
            return clean_mask | noisy_self_chunk | noisy_prev_clean | (q_idx == kv_idx)

        compile_mask = _os.environ.get("REMIND_TF_MASK_COMPILE", "0") != "0"
        block_mask = create_block_mask(
            clean_history_mask,
            B=None, H=None,
            Q_LEN=padded_total_tokens, KV_LEN=padded_total_tokens,
            device=device,
            _compile=compile_mask,
        )
        self._block_mask_cache[key] = block_mask
        return block_mask

    def _set_gradient_checkpointing(self, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        kv_size=(0,0),
        image_latent_input: torch.Tensor = None,
        render_latent_input: torch.Tensor = None,
        freqs_offset: int = 0,
        freqs_positions: torch.Tensor = None,
        viewmats: torch.Tensor = None,     # [B, F, 4, 4] per-frame c2w, optional
        Ks: torch.Tensor = None,           # [B, F, 3, 3] per-frame intrinsics
        chunk_size: int = None,            # latent frames per chunk (training mode)
        clean_history_clean_x: torch.Tensor = None,  # [B, C, F, H, W]
        clean_history_clean_t: torch.Tensor = None,  # [B, F] or [B, 1]
        degradation_control: torch.Tensor = None,  # [B, F, control_dim]
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # params
        device = self.patch_embedding.weight.device
        if hasattr(self, 'freqs'):
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)
        else:
            self.init_freqs(device)

        clean_history = clean_history_clean_x is not None
        f, h, w = x.shape[2:]
        orig_f = f
        if clean_history:
            assert kv_cache is None, "clean_history_clean_x is training-only"
            assert clean_history_clean_x.shape[2:] == x.shape[2:], (
                f"clean_history_clean_x shape {tuple(clean_history_clean_x.shape)} "
                f"must match x shape {tuple(x.shape)}")

        sp_pad_frames = 0
        if (chunk_size is not None and self.training and torch.is_grad_enabled()):
            try:
                from wan.distributed.util import get_sequence_parallel_world_size
                sp_size_for_padding = get_sequence_parallel_world_size()
            except Exception:
                sp_size_for_padding = 1
            if sp_size_for_padding > 1:
                align = sp_size_for_padding
                if chunk_size is not None and chunk_size > 0:
                    align = align * chunk_size // math.gcd(align, chunk_size)
                padded_f = math.ceil(f / align) * align
                sp_pad_frames = padded_f - f

        def _pad_latent_frames(tensor):
            if tensor is None or sp_pad_frames <= 0:
                return tensor
            pad = tensor.new_zeros(
                tensor.shape[0], tensor.shape[1], sp_pad_frames,
                *tensor.shape[3:])
            return torch.cat([tensor, pad], dim=2)

        def _pad_frame_tensor(tensor):
            if tensor is None or sp_pad_frames <= 0:
                return tensor
            pad = tensor[:, -1:].expand(
                tensor.shape[0], sp_pad_frames, *tensor.shape[2:])
            return torch.cat([tensor, pad], dim=1)

        if sp_pad_frames > 0:
            x = _pad_latent_frames(x)
            clean_history_clean_x = _pad_latent_frames(clean_history_clean_x)
            image_latent_input = _pad_latent_frames(image_latent_input)
            render_latent_input = _pad_latent_frames(render_latent_input)
            viewmats = _pad_frame_tensor(viewmats)
            Ks = _pad_frame_tensor(Ks)
            degradation_control = _pad_frame_tensor(degradation_control)
            if freqs_positions is not None:
                freqs_positions = freqs_positions.to(device=device, dtype=torch.long)
                extra = freqs_positions[-1] + torch.arange(
                    1, sp_pad_frames + 1, device=device, dtype=torch.long)
                freqs_positions = torch.cat([freqs_positions, extra], dim=0)
            f = padded_f
        h = h//2
        w = w//2

        c = self.dim // self.num_heads // 2
        freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # Compute freqs_x once (same for all branches).  Most callers use the
        # compact contiguous convention `freqs_offset + arange(F)`.  Ref-cache
        # training/validation can pass explicit non-contiguous positive frame
        # positions, e.g. reference at 0 and current video at +40 latent frames
        # (equivalent to reference at t=-40 because RoPE attention only uses
        # relative phase differences).
        prope_freqs_positions = None
        if freqs_positions is None:
            base_freqs_positions = torch.arange(
                freqs_offset, freqs_offset + f, device=device, dtype=torch.long)
        else:
            freqs_positions = freqs_positions.to(device=device, dtype=torch.long)
            assert freqs_positions.dim() == 1 and freqs_positions.numel() == f, (
                f"freqs_positions must be [F={f}], got {tuple(freqs_positions.shape)}")
            assert int(freqs_positions.min().item()) >= 0, \
                "freqs_positions must be non-negative; shift positions instead"
            assert int(freqs_positions.max().item()) < freqs[0].shape[0], (
                f"freqs_positions max={int(freqs_positions.max().item())} exceeds "
                f"RoPE table length {freqs[0].shape[0]}")
            base_freqs_positions = freqs_positions
        if clean_history:
            effective_freqs_positions = torch.cat(
                [base_freqs_positions, base_freqs_positions], dim=0)
        else:
            effective_freqs_positions = base_freqs_positions
        assert int(effective_freqs_positions.max().item()) < freqs[0].shape[0], (
            f"freqs_positions max={int(effective_freqs_positions.max().item())} exceeds "
            f"RoPE table length {freqs[0].shape[0]}")
        temporal_freqs = freqs[0].index_select(0, effective_freqs_positions)
        if freqs_positions is not None or clean_history:
            prope_freqs_positions = effective_freqs_positions
        f_attn = f * 2 if clean_history else f
        freqs_x = torch.cat([
            temporal_freqs.view(f_attn, 1, 1, -1).expand(f_attn, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f_attn, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f_attn, h, w, -1)
        ], dim=-1).reshape(f_attn*h*w, 1, -1)

        # Input-channel handling.
        # If the model was configured with in_dim != x.shape[1], expand x with
        # render_latent_input (legacy v2v/t2v path) to reach self.in_dim.
        # If x already matches self.in_dim, pass through directly — this is the
        # "pure text + pose + RoPE/ProPE" regime where no render/mask is used.
        if x.shape[1] != self.in_dim:
            if render_latent_input is None:
                # Legacy t2v: pad to self.in_dim with zeros
                pad_ch = self.in_dim - x.shape[1]
                assert pad_ch > 0, \
                    f"x has {x.shape[1]} channels, model in_dim is {self.in_dim}"
                x = torch.cat([x, x.new_zeros(x.shape[0], pad_ch, *x.shape[2:])], dim=1)
            elif kv_size[1] >= 0:
                # Legacy v2v: concat render_latent_input
                x = torch.cat([x, render_latent_input], dim=1)
        assert x.shape[1] == self.in_dim, \
            f"x channels ({x.shape[1]}) != model in_dim ({self.in_dim})"
        if clean_history:
            if clean_history_clean_x.shape[1] != self.in_dim:
                pad_ch = self.in_dim - clean_history_clean_x.shape[1]
                assert pad_ch > 0 and render_latent_input is None, (
                    "clean_history_clean_x currently supports pure-I2V/in_dim "
                    f"padding only, got clean channels={clean_history_clean_x.shape[1]} "
                    f"model in_dim={self.in_dim}")
                clean_history_clean_x = torch.cat([
                    clean_history_clean_x,
                    clean_history_clean_x.new_zeros(
                        clean_history_clean_x.shape[0], pad_ch,
                        *clean_history_clean_x.shape[2:])
                ], dim=1)
            assert clean_history_clean_x.shape[1] == self.in_dim, (
                f"teacher clean channels ({clean_history_clean_x.shape[1]}) "
                f"!= model in_dim ({self.in_dim})")


        # embeddings
        x_noisy = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x_clean = (
            [self.patch_embedding(u.unsqueeze(0)) for u in clean_history_clean_x]
            if clean_history else None
        )
        grid_sizes = torch.stack([
            torch.as_tensor(u.shape[2:], dtype=torch.long, device=u.device)
            for u in x_noisy
        ])
        x_noisy = [u.flatten(2).transpose(1, 2) for u in x_noisy]
        seq_lens = torch.as_tensor(
            [u.size(1) for u in x_noisy], dtype=torch.long, device=x_noisy[0].device)
        assert seq_lens.max() <= seq_len
        if clean_history:
            x_clean = [u.flatten(2).transpose(1, 2) for u in x_clean]
            seq_lens_clean = torch.as_tensor(
                [u.size(1) for u in x_clean], dtype=torch.long, device=x_noisy[0].device)
            assert torch.equal(seq_lens_clean, seq_lens), (
                f"teacher clean token lengths {seq_lens_clean.tolist()} "
                f"!= noisy token lengths {seq_lens.tolist()}")
            x = torch.cat([
                torch.cat([u_clean, u_noisy], dim=1)
                for u_noisy, u_clean in zip(x_noisy, x_clean)
            ])
        else:
            x = torch.cat(x_noisy)

        # Per-frame timesteps. Accept t of shape [B], [B, 1] (uniform), or [B, F]
        # (per-frame, used for WorldPlay memory mode where past chunks have high noise).
        if t.dim() == 1:
            t_bf_noisy = t.unsqueeze(1).expand(-1, f).contiguous()
        elif t.shape[1] == 1:
            t_bf_noisy = t.expand(-1, f).contiguous()
        else:
            if t.shape[1] == orig_f and sp_pad_frames > 0:
                t = torch.cat(
                    [t, t[:, -1:].expand(-1, sp_pad_frames)],
                    dim=1)
            assert t.shape[1] == f, (
                f"timestep frames ({t.shape[1]}) must match input frames ({f})")
            t_bf_noisy = t
        if clean_history:
            if clean_history_clean_t is None:
                t_bf_clean = torch.zeros_like(t_bf_noisy)
            elif clean_history_clean_t.dim() == 1:
                t_bf_clean = clean_history_clean_t.unsqueeze(1).expand(-1, f).contiguous()
            elif clean_history_clean_t.shape[1] == 1:
                t_bf_clean = clean_history_clean_t.expand(-1, f).contiguous()
            else:
                if (clean_history_clean_t.shape[1] == orig_f
                        and sp_pad_frames > 0):
                    clean_history_clean_t = torch.cat(
                        [clean_history_clean_t,
                         clean_history_clean_t[:, -1:].expand(
                             -1, sp_pad_frames)],
                        dim=1)
                assert clean_history_clean_t.shape[1] == f, (
                    f"teacher clean timestep frames ({clean_history_clean_t.shape[1]}) "
                    f"must match input frames ({f})")
                t_bf_clean = clean_history_clean_t
            t_bf = torch.cat([t_bf_clean.to(t_bf_noisy.device), t_bf_noisy], dim=1)
        else:
            t_bf = t_bf_noisy

        B_sz = t_bf.shape[0]
        t_flat = t_bf.reshape(-1)  # [B*F]
        e_flat = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_flat).type_as(x)
        )  # [B*F, dim]
        if self.degradation_control_embedding is not None:
            if degradation_control is None:
                degradation_control = e_flat.new_zeros(
                    B_sz, f, self.degradation_control_dim)
            else:
                assert degradation_control.ndim == 3, (
                    "degradation_control must have shape [B, F, D], got "
                    f"{tuple(degradation_control.shape)}")
                assert degradation_control.shape[:2] == (B_sz, f), (
                    "degradation_control batch/frames must match input: "
                    f"got {tuple(degradation_control.shape[:2])}, "
                    f"expected {(B_sz, f)}")
                assert degradation_control.shape[2] == self.degradation_control_dim, (
                    "degradation_control feature dim mismatch: "
                    f"got {degradation_control.shape[2]}, "
                    f"expected {self.degradation_control_dim}")
                degradation_control = degradation_control.to(
                    device=e_flat.device, dtype=e_flat.dtype)
            if clean_history:
                degradation_control = torch.cat(
                    [degradation_control, degradation_control], dim=1)
            e_flat = e_flat + self.degradation_control_embedding(
                degradation_control.reshape(-1, self.degradation_control_dim))
        elif degradation_control is not None:
            raise ValueError(
                "degradation_control was provided but the model was created "
                "with degradation_control_dim=0")
        e0_flat = self.time_projection(e_flat).unflatten(1, (6, self.dim))  # [B*F, 6, dim]

        # Reshape to per-frame representation. Blocks expand to per-token internally.
        e = e_flat.view(B_sz, f_attn, self.dim)           # [B, F_attn, dim]
        e0 = e0_flat.view(B_sz, f_attn, 6, self.dim)      # [B, F_attn, 6, dim]
        e_head = e[:, f:] if clean_history else e

        if isinstance(context, torch.Tensor) and context.ndim == 4:
            B_context, N_context, L_context, D_context = context.shape
            if L_context > self.text_len:
                context = context[:, :, :self.text_len]
                L_context = self.text_len
            elif L_context < self.text_len:
                pad = context.new_zeros(
                    B_context, N_context,
                    self.text_len - L_context, D_context)
                context = torch.cat([context, pad], dim=2)
                L_context = self.text_len
            context = self.text_embedding(
                context.reshape(
                    B_context * N_context, L_context, D_context)
            ).reshape(B_context, N_context, L_context, self.dim)
        else:
            context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

        # Keep viewmats/Ks as [B, F, 4, 4] / [B, F, 3, 3] (one per frame).
        # prope_qkv handles the frame→token mapping natively: it treats
        # `cameras = viewmats.shape[1]` as the group count and applies each
        # camera matrix to `seqlen // cameras = h*w` consecutive tokens via
        # a batched einsum. Expanding to per-token was ~1560× redundant in
        # memory and forced the einsum into 32760 tiny per-camera matmuls
        # instead of 21 batched ones.
        viewmats_expanded = None
        Ks_expanded = None
        if viewmats is not None:
            assert viewmats.shape[1] == f, \
                f"viewmats frames ({viewmats.shape[1]}) != input frames ({f})"
            viewmats_expanded = viewmats.to(device=x.device, dtype=torch.float32)
            if Ks is not None:
                Ks_expanded = Ks.to(device=x.device, dtype=torch.float32)
                # Normalize intrinsics: pixel-unit focal lengths (400-1200) cause
                # ProPE projection matrices with O(1000) entries → bf16 overflow and
                # gradient explosion. Dividing by image width brings entries to O(1).
                # image_w = latent_w * patch * vae_spatial.
                # Wan2.1 VAE uses stride 8; Wan2.2-TI2V-5B native VAE uses
                # stride 16 and 48-channel latents.
                vae_spatial = 16.0 if self.head.out_dim == 48 else 8.0
                img_w = w * 2.0 * vae_spatial
                Ks_expanded = Ks_expanded.clone()
                Ks_expanded[..., 0, :] = Ks_expanded[..., 0, :] / img_w
                Ks_expanded[..., 1, :] = Ks_expanded[..., 1, :] / img_w
            # Normalize c2w translations per-scene: center cameras at mean position
            # and scale to unit sphere. DL3DV scene scales vary wildly (1m indoor to
            # 100m+ outdoor); without this, P = K @ c2w can have huge entries.
            viewmats_expanded = viewmats_expanded.clone()
            t_all = viewmats_expanded[:, :, :3, 3]          # [B, F, 3]
            t_center = t_all.mean(dim=1, keepdim=True)       # [B, 1, 3]
            t_scale = (t_all - t_center).norm(dim=-1).max(dim=1, keepdim=True).values.unsqueeze(-1)  # [B, 1, 1]
            t_scale = t_scale.clamp(min=0.01)                 # avoid div-by-zero
            viewmats_expanded[:, :, :3, 3] = (t_all - t_center) / t_scale
            if clean_history:
                viewmats_expanded = torch.cat(
                    [viewmats_expanded, viewmats_expanded], dim=1)
                if Ks_expanded is not None:
                    Ks_expanded = torch.cat([Ks_expanded, Ks_expanded], dim=1)

        # ── CC-RoPE: build per-frame pose descriptor once; broadcast to all blocks ──
        # Always emit shape [B, F, cc_pose_dim=14]. If Ks is None, the last 2 entries
        # (log-focal) are zero-padded so the Linear(14 → D_half) always lines up.
        pose_c = None
        if self.cc_rope_mode in {
            "cc_basic", "cc_output", "cc_value", "cc_full",
            "cc_dual_channel", "cc_dual_output",
        } and viewmats_expanded is not None:
            B_pose = viewmats_expanded.shape[0]
            F_pose = viewmats_expanded.shape[1]
            p = _camera_pose_features(viewmats_expanded, Ks_expanded)
            if p.shape[-1] < self.cc_pose_dim:
                pad = viewmats_expanded.new_zeros(
                    B_pose, F_pose, self.cc_pose_dim - p.shape[-1], dtype=torch.float32)
                p = torch.cat([p, pad], dim=-1)
            pose_c = p                                           # [B, F, 14] float32

        # Build chunk-wise causal BlockMask (flex_attention) if training.
        # Mask semantics: tokens in chunk i can attend to tokens in chunks [0..i];
        # within a chunk, full attention.
        #
        # BlockMask is sparse in chunk-block granularity — flex_attention skips
        # upper-triangular blocks entirely rather than multiplying by zero. This
        # is 2-3× faster than dense SDPA with a bool attn_mask (which falls
        # back to the math backend because flash requires is_causal/None).
        chunk_block_mask = None
        clean_history_half_tokens = 0
        clean_history_tokens_per_chunk = 0
        if chunk_size is not None:
            tokens_per_frame = h * w
            tokens_per_chunk = chunk_size * tokens_per_frame
            num_chunks = f // chunk_size
            half_tokens = num_chunks * tokens_per_chunk
            assert half_tokens == f * tokens_per_frame, \
                f"f ({f}) must be divisible by chunk_size ({chunk_size})"

            if clean_history:
                clean_history_half_tokens = half_tokens
                clean_history_tokens_per_chunk = tokens_per_chunk
                chunk_block_mask = self._get_clean_history_chunk_block_mask(
                    half_tokens, tokens_per_chunk, x.device)
            else:
                chunk_block_mask = self._get_chunk_block_mask(
                    half_tokens, tokens_per_chunk, x.device)

        sequence_parallel_enabled = False
        sequence_parallel_clean_history = False
        grid_sizes_for_unpatchify = grid_sizes
        if (chunk_block_mask is not None and self.training
                and torch.is_grad_enabled()):
            try:
                from wan.distributed.util import (
                    get_sequence_parallel_rank,
                    get_sequence_parallel_world_size,
                )
                sp_size = get_sequence_parallel_world_size()
            except Exception:
                sp_size = 1
            if sp_size > 1:
                sp_rank = get_sequence_parallel_rank()
                if f % sp_size != 0:
                    raise ValueError(
                        f"sequence_parallel_size={sp_size} requires latent "
                        f"frame count f={f} to be divisible by sp_size")
                tokens_per_frame = h * w
                local_f = f // sp_size
                fs = sp_rank * local_f
                fe = fs + local_f
                half_tokens = f * tokens_per_frame
                local_half_tokens = local_f * tokens_per_frame

                def _slice_frames(tensor):
                    if tensor is None:
                        return None
                    if clean_history:
                        return torch.cat(
                            [tensor[:, fs:fe], tensor[:, f + fs:f + fe]],
                            dim=1)
                    return tensor[:, fs:fe]

                def _slice_token_freqs(tensor):
                    if clean_history:
                        return torch.cat(
                            [tensor[fs * tokens_per_frame:fe * tokens_per_frame],
                             tensor[half_tokens + fs * tokens_per_frame:
                                    half_tokens + fe * tokens_per_frame]],
                            dim=0)
                    return tensor[fs * tokens_per_frame:fe * tokens_per_frame]

                if clean_history:
                    clean_slice = slice(fs * tokens_per_frame, fe * tokens_per_frame)
                    noisy_slice = slice(
                        half_tokens + fs * tokens_per_frame,
                        half_tokens + fe * tokens_per_frame)
                    x = torch.cat([x[:, clean_slice], x[:, noisy_slice]], dim=1)
                    seq_lens = torch.full_like(seq_lens, local_half_tokens)
                    sequence_parallel_clean_history = True
                else:
                    x = x[:, fs * tokens_per_frame:fe * tokens_per_frame]
                    seq_lens = torch.full_like(seq_lens, local_half_tokens)

                freqs_x = _slice_token_freqs(freqs_x)
                e = _slice_frames(e)
                e0 = _slice_frames(e0)
                e_head = e[:, local_f:] if clean_history else e
                viewmats_expanded = _slice_frames(viewmats_expanded)
                Ks_expanded = _slice_frames(Ks_expanded)
                pose_c = _slice_frames(pose_c)
                if context.ndim == 4:
                    if fs % chunk_size != 0 or fe % chunk_size != 0:
                        raise ValueError(
                            "sequence-parallel frame shards must align to "
                            "chunk-text prompt boundaries")
                    context = context[:, fs // chunk_size:fe // chunk_size]
                if prope_freqs_positions is not None:
                    if clean_history:
                        prope_freqs_positions = torch.cat(
                            [prope_freqs_positions[fs:fe],
                             prope_freqs_positions[f + fs:f + fe]],
                            dim=0)
                    else:
                        prope_freqs_positions = prope_freqs_positions[fs:fe]
                grid_sizes_for_unpatchify = grid_sizes.clone()
                grid_sizes_for_unpatchify[:, 0] = local_f
                sequence_parallel_enabled = True

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            freqs_x=freqs_x,
            context=context,
            context_lens=None,
            kv_size=kv_size,
            viewmats=viewmats_expanded,
            Ks=Ks_expanded,
            pose_c=pose_c,                   # [B, F, 14] or None
            tokens_per_frame=h * w,          # per-token broadcast for CC-RoPE
            chunk_mask=chunk_block_mask,   # BlockMask (flex) or None
            # Spatiotemporal ProPE: per-frame temporal position = freqs_offset + frame_idx
            prope_temporal_dim=self.prope_temporal_dim,
            prope_freqs_offset=freqs_offset,
            prope_freqs_positions=prope_freqs_positions,
            sequence_parallel=sequence_parallel_enabled,
            sequence_parallel_clean_history=sequence_parallel_clean_history,
            chunk_flash=(
                not bool(clean_history)
                and bool(chunk_block_mask is not None)
                and not bool(sequence_parallel_enabled)
                and _os.environ.get("REMIND_DMD_CHUNK_FLASH", "0") == "1"
            ),
            chunk_flash_tokens_per_chunk=(
                tokens_per_chunk if chunk_block_mask is not None else 0
            ),
            clean_history_chunk_flash=(
                bool(clean_history)
                and bool(chunk_block_mask is not None)
                and not bool(sequence_parallel_enabled)
                and _os.environ.get("REMIND_TF_CHUNK_FLASH", "1") != "0"
            ),
            clean_history_half_tokens=clean_history_half_tokens,
            clean_history_tokens_per_chunk=clean_history_tokens_per_chunk,
            context_streams=(2 if clean_history else 1),
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            kwargs['kv_cache'] = kv_cache[block_index] if kv_cache is not None else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x= torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x= block(x, **kwargs)

        if clean_history:
            x = x[:, seq_lens[0]:]

        x = self.head(x, e_head)
        x = self.unpatchify(x, grid_sizes_for_unpatchify)
        x = torch.stack(x)
        if sequence_parallel_enabled:
            from wan.distributed.util import gather_forward_with_grad
            x = gather_forward_with_grad(x, dim=2)
        if sp_pad_frames > 0:
            x = x[:, :, :orig_f]

        return x

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
