# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Inference adapter for ReMind's causal Wan generator and full-history KV cache."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _raw_model(module):
    while hasattr(module, "module"):
        module = module.module
    if hasattr(module, "get_base_model"):
        module = module.get_base_model()
    return module


class InferencePipelineAdapter:
    """Expose fixed-shape rollout primitives over ``CausalInferencePipeline``.

    Cache offsets are derived from the actual latent resolution. Temporal RoPE
    positions remain absolute even though K/V are stored in compact contiguous
    slots. This is the contract needed by ReMind's time + camera 3D-RoPE.
    """

    def __init__(self, pipe, chunk_size: int = 3, context_timestep: int = 0):
        self.pipe = pipe
        self.generator = pipe.generator
        self.scheduler = pipe.scheduler
        self.cs = int(chunk_size)
        self.context_timestep = int(context_timestep)
        self.kv_cache = None
        self.tokens_per_frame = 0
        self._hist_tokens = 0
        self._capture_attention = False
        self._attention_layer_stride = 1
        self._attn_logs = {}

    def set_attention_logging(self, enabled: bool, *, layer_stride: int = 1):
        self._capture_attention = bool(enabled)
        self._attention_layer_stride = max(1, int(layer_stride))
        self._attn_logs = {}

    def _attention_blocks(self):
        raw = _raw_model(self.generator.model)
        return list(raw.blocks)[:: self._attention_layer_stride]

    def _toggle_attention_logging(self, enabled: bool):
        raw = _raw_model(self.generator.model)
        selected = {id(block) for block in self._attention_blocks()}
        for block in raw.blocks:
            attention = block.self_attn
            if hasattr(attention, "enable_attn_log"):
                attention.enable_attn_log(
                    self.tokens_per_frame,
                    bool(enabled and id(block) in selected),
                )

    def _harvest_attention(self, chunk_index: int):
        if not self._capture_attention:
            return
        layer_maps = []
        for block in self._attention_blocks():
            attention = block.self_attn
            value = (
                attention.get_attn_log() if hasattr(attention, "get_attn_log") else None
            )
            if value is not None:
                layer_maps.append(value.detach().cpu())
        if layer_maps:
            hist_frames = int(layer_maps[0].shape[-1])
            self._attn_logs[int(chunk_index)] = (
                torch.stack(layer_maps, dim=0),
                list(range(hist_frames)),
            )

    def render_attention_heatmap(self, *, title: str):
        if not self._attn_logs:
            return None
        from pipeline.dl3dv_inference import DL3DVInferencePipeline

        return DL3DVInferencePipeline._render_attn_heatmap(
            self,
            chunk_size=self.cs,
            title=title,
            num_layer_groups=min(6, len(self._attention_blocks())),
        )

    def _allocate_cache(self, noise: torch.Tensor):
        batch, frames, _, height, width = noise.shape
        raw = _raw_model(self.generator.model)
        patch = getattr(raw, "patch_size", (1, 2, 2))
        patch_h, patch_w = int(patch[-2]), int(patch[-1])
        self.tokens_per_frame = (height // patch_h) * (width // patch_w)
        max_tokens = frames * self.tokens_per_frame
        num_heads = getattr(raw, "num_heads", None)
        dim = getattr(raw, "dim", None)
        if num_heads is None:
            num_heads = raw.config.num_heads
        if dim is None:
            dim = raw.config.dim
        num_heads, dim = int(num_heads), int(dim)
        head_dim = dim // num_heads
        use_prope_cache = any(
            getattr(getattr(block, "self_attn", None), "cc_rope_mode", "")
            == "dual_prope"
            for block in raw.blocks
        )
        self.kv_cache = []
        for _ in raw.blocks:
            cache = {
                "k": torch.zeros(
                    batch,
                    max_tokens,
                    num_heads,
                    head_dim,
                    device=noise.device,
                    dtype=noise.dtype,
                ),
                "v": torch.zeros(
                    batch,
                    max_tokens,
                    num_heads,
                    head_dim,
                    device=noise.device,
                    dtype=noise.dtype,
                ),
            }
            if use_prope_cache:
                cache["k_prope"] = torch.zeros_like(cache["k"])
                cache["v_prope"] = torch.zeros_like(cache["v"])
            self.kv_cache.append(cache)

    def reset_cache(self, noise: torch.Tensor):
        self._allocate_cache(noise)
        self._hist_tokens = 0
        self._attn_logs = {}
        self._toggle_attention_logging(self._capture_attention)

    def clear_cache(self):
        if self.tokens_per_frame:
            self._toggle_attention_logging(False)
        self.kv_cache = None
        self._hist_tokens = 0

    @torch.no_grad()
    def encode_chunk_to_cache(
        self,
        chunk_x0: torch.Tensor,
        *,
        chunk_index: int,
        conditional_dict: Dict,
        context_timestep: Optional[int] = None,
        viewmats=None,
        Ks=None,
    ):
        self._harvest_attention(chunk_index)
        if self._capture_attention:
            self._toggle_attention_logging(True)
        t_ctx = (
            self.context_timestep if context_timestep is None else int(context_timestep)
        )
        batch, frames = chunk_x0.shape[:2]
        timestep = torch.full(
            (batch, frames), t_ctx, device=chunk_x0.device, dtype=torch.long
        )
        context = chunk_x0
        if t_ctx > 0:
            context = self.scheduler.add_noise(
                chunk_x0.flatten(0, 1),
                torch.randn_like(chunk_x0.flatten(0, 1)),
                timestep.flatten(0, 1),
            ).unflatten(0, chunk_x0.shape[:2])
        self.generator(
            noisy_image_or_video=context,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache,
            kv_size=(self._hist_tokens, -1),
            render_latent_input=None,
            freqs_offset=int(chunk_index) * self.cs,
            viewmats=viewmats,
            Ks=Ks,
        )
        self._hist_tokens += frames * self.tokens_per_frame

    @torch.no_grad()
    def replace_chunk_in_cache(
        self,
        chunk_x0: torch.Tensor,
        *,
        chunk_index: int,
        conditional_dict: Dict,
        context_timestep: Optional[int] = None,
        viewmats=None,
        Ks=None,
    ):
        """Rewrite one historical cache slot without changing cache length."""
        if self.kv_cache is None:
            raise RuntimeError("cache is not initialized")
        t_ctx = (
            self.context_timestep if context_timestep is None else int(context_timestep)
        )
        batch, frames = chunk_x0.shape[:2]
        expected_frames = self.cs
        if frames != expected_frames:
            raise ValueError(
                f"cache replacement expects {expected_frames} frames, got {frames}"
            )
        token_offset = int(chunk_index) * self.cs * self.tokens_per_frame
        token_end = token_offset + frames * self.tokens_per_frame
        if token_end > self._hist_tokens:
            raise ValueError(
                f"cannot replace uncached chunk {chunk_index}: "
                f"token_end={token_end} hist_tokens={self._hist_tokens}"
            )
        timestep = torch.full(
            (batch, frames), t_ctx, device=chunk_x0.device, dtype=torch.long
        )
        context = chunk_x0
        if t_ctx > 0:
            context = self.scheduler.add_noise(
                chunk_x0.flatten(0, 1),
                torch.randn_like(chunk_x0.flatten(0, 1)),
                timestep.flatten(0, 1),
            ).unflatten(0, chunk_x0.shape[:2])
        self.generator(
            noisy_image_or_video=context,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache,
            kv_size=(token_offset, -1),
            render_latent_input=None,
            freqs_offset=int(chunk_index) * self.cs,
            viewmats=viewmats,
            Ks=Ks,
        )

    def denoise_step_outputs(
        self,
        noisy_chunk: torch.Tensor,
        *,
        chunk_index: int,
        timestep,
        conditional_dict: Dict,
        viewmats=None,
        Ks=None,
    ):
        """Return the native flow prediction and reconstructed clean latent."""
        batch, frames = noisy_chunk.shape[:2]
        if not torch.is_tensor(timestep):
            timestep = torch.full(
                (batch, frames),
                int(timestep),
                device=noisy_chunk.device,
                dtype=torch.long,
            )
        else:
            timestep = timestep.to(device=noisy_chunk.device, dtype=torch.long)
            if timestep.ndim == 0:
                timestep = timestep.expand(batch, frames)
        return self.generator(
            noisy_image_or_video=noisy_chunk,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache,
            kv_size=(0, self._hist_tokens),
            render_latent_input=None,
            freqs_offset=int(chunk_index) * self.cs,
            viewmats=viewmats,
            Ks=Ks,
        )

    def denoise_step(
        self,
        noisy_chunk: torch.Tensor,
        *,
        chunk_index: int,
        timestep,
        conditional_dict: Dict,
        viewmats=None,
        Ks=None,
    ) -> torch.Tensor:
        _, pred_x0 = self.denoise_step_outputs(
            noisy_chunk,
            chunk_index=chunk_index,
            timestep=timestep,
            conditional_dict=conditional_dict,
            viewmats=viewmats,
            Ks=Ks,
        )
        return pred_x0
