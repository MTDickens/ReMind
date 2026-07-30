# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Chunk-by-chunk autoregressive inference pipeline for DL3DV.

Flow:
  1. Init KV cache (all None)
  2. Cache text KV once (cache_txt=True-ish)
  3. For each chunk after the first:
     a. Select all history or a bounded recent-history window
     b. Re-encode selected history to KV cache (clean/low-noise context pass)
     c. Denoise current chunk reading from the fresh cache
  4. Decode full video

Compact cache RoPE convention:
  - Selected history frames at positions [0..N-1] (compact)
  - Current chunk at positions [N..N+chunk_size-1]
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


TOKENS_PER_FRAME = 1560  # 30 × 52 for 480×832 at Wan2.1 VAE


class DL3DVInferencePipeline:
    """
    Chunk-by-chunk autoregressive inference with bounded KV cache selection.

    Usage:
        pipe = DL3DVInferencePipeline(generator, text_encoder, vae, scheduler,
                                       denoising_steps, chunk_size=3)
        video = pipe.inference(
            ref_latent=first_chunk_latent,   # [B, chunk_size, 16, H_lat, W_lat]
            viewmats=cams,                    # [B, T_lat, 4, 4]
            Ks=intr,                          # [B, T_lat, 3, 3]
            text_prompts=[caption],
            num_latent_frames=21,
            memory_frames=5,
            temporal_context_size=3,
        )  # → [B, T_lat, 16, H_lat, W_lat]
    """

    def __init__(
        self,
        generator,  # WanDiffusionWrapper
        text_encoder,
        vae,
        scheduler,
        denoising_steps: torch.Tensor,
        chunk_size: int = 3,
        kv_cache_max_tokens: int = 12480,  # enough for 8 history frames
        tokens_per_frame: int = TOKENS_PER_FRAME,
    ):
        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.denoising_steps = denoising_steps
        self.chunk_size = chunk_size
        self.kv_cache_max_tokens = kv_cache_max_tokens
        self.tokens_per_frame = tokens_per_frame
        # Unwrap DDP/FSDP to access model attributes (blocks, num_heads, dim).
        raw_model = generator.model
        while hasattr(raw_model, "module") and not hasattr(raw_model, "blocks"):
            raw_model = raw_model.module
        self._raw_model = raw_model
        self.num_transformer_blocks = len(raw_model.blocks)
        self.num_heads = raw_model.num_heads
        self.head_dim = raw_model.dim // self.num_heads
        self.kv_cache = None
        try:
            self.clean_context_timestep = int(self.scheduler.timesteps[-1].item())
        except Exception:
            self.clean_context_timestep = 0
        # Attention logging state (only populated when inference(log_attention=True))
        self._attn_logs: dict = {}
        self._attn_summary: str = ""

    def _encode_text_prompts(self, text_prompts) -> dict:
        cond = self.text_encoder(text_prompts=text_prompts)
        if cond["prompt_embeds"].ndim == 4:
            cond["prompt_chunk_size"] = int(self.chunk_size)
        return cond

    # --- KV cache importance probing ---
    def _enable_attn_log(self, enable: bool):
        """Broadcast the log-attn flag to every self-attention module and reset
        accumulators. Called at chunk boundaries so each chunk's summary is
        averaged only over its own denoising steps."""
        for blk in self._raw_model.blocks:
            blk.self_attn.enable_attn_log(self.tokens_per_frame, enable)

    def _collect_attn_log(self):
        """Pull per-layer per-head per-history-frame weights. Returns a stacked
        tensor [L_layer, H_heads, num_hist_frames] on device, or None."""
        layer_maps = []
        for blk in self._raw_model.blocks:
            m = blk.self_attn.get_attn_log()
            if m is not None:
                layer_maps.append(m)
        if not layer_maps:
            return None
        # All layers should see the same history size in one chunk — align defensively.
        max_hist = max(m.shape[-1] for m in layer_maps)
        padded = []
        for m in layer_maps:
            if m.shape[-1] < max_hist:
                pad = torch.zeros(
                    *m.shape[:-1],
                    max_hist - m.shape[-1],
                    device=m.device,
                    dtype=m.dtype,
                )
                m = torch.cat([m, pad], dim=-1)
            padded.append(m)
        return torch.stack(padded, dim=0)

    def _format_attn_summary(self, chunk_size: int, topk: int = 3) -> str:
        """Turn per-chunk attn logs into a human-readable multiline report.

        For each chunk: prints mean-over-layers-and-heads weight per history
        chunk (if hist frames align to chunk_size) with top-k flagged, plus an
        ASCII bar chart of the full per-chunk distribution.
        """
        lines = []
        if not self._attn_logs:
            return "(no attention logs captured)"
        bar_chars = " ▁▂▃▄▅▆▇█"
        for chunk_i in sorted(self._attn_logs.keys()):
            w, sel_idx = self._attn_logs[chunk_i]  # [L, H, num_hist], list
            # Normalize: mean over layers & heads → [num_hist]
            w_flat = w.float().mean(dim=(0, 1))
            num_hist = w_flat.shape[0]
            # Roll frames into chunks for readability if they tile cleanly.
            if num_hist % chunk_size == 0 and num_hist // chunk_size >= 1:
                w_by_chunk = w_flat.view(-1, chunk_size).sum(dim=-1)
                # Map back to absolute history-chunk index via sel_idx[0] // chunk_size.
                first_hist_frame = sel_idx[0] if sel_idx else 0
                first_chunk_idx = first_hist_frame // chunk_size
                labels = [f"c{first_chunk_idx + i}" for i in range(w_by_chunk.shape[0])]
                unit = "chunk"
            else:
                w_by_chunk = w_flat
                labels = [f"f{fi}" for fi in sel_idx[:num_hist]]
                unit = "frame"
            vals = w_by_chunk.tolist()
            vmax = max(vals) if vals else 1.0
            k = min(topk, len(vals))
            top_pairs = sorted(enumerate(vals), key=lambda kv: -kv[1])[:k]
            top_str = ", ".join(f"{labels[i]}:{v:.3f}" for i, v in top_pairs)
            bar = "".join(
                bar_chars[int(v / (vmax + 1e-9) * (len(bar_chars) - 1))] for v in vals
            )
            cur_fs = chunk_i * chunk_size
            lines.append(
                f"  chunk {chunk_i} (f{cur_fs}..{cur_fs + chunk_size - 1}) "
                f"| top-{k} {unit}s: {top_str}"
            )
            lines.append(f"    hist-{unit}s [{bar}] (max={vmax:.3f})")
        return "\n".join(lines)

    def _kv_memory_retrieval_metrics(self, chunk_size: int, peak_chunk: int):
        """Quantify the reverse attention diagonal from self._attn_logs.

        The out-of-sight training goal: generated chunks AFTER the event peak
        (recovery) should retrieve scene state from PRE-EVENT history chunks
        rather than only copying their local neighbor.  On the KV-importance
        heatmap this shows as a reverse diagonal: recovery chunk peak+k
        attending to mirror chunk peak-k.

        Returns {} when no logs / no post-peak chunks, else:
          kv_mem_pre_event_frac : mean over post-peak generated chunks of the
              attention-mass fraction on history chunks < peak_chunk.
          kv_mem_mirror_frac    : mass fraction on the mirrored chunk
              (peak - (g - peak)), +-1 chunk tolerance, clipped to pre-event.
          kv_mem_local_frac     : mass fraction on chunk g-1 (local copy).
          kv_mem_num_chunks     : number of post-peak chunks measured.
        """
        if not self._attn_logs or peak_chunk is None or peak_chunk < 1:
            return {}
        pre_fracs, mirror_fracs, local_fracs = [], [], []
        for gi in sorted(self._attn_logs.keys()):
            if gi <= peak_chunk:
                continue
            w, sel_idx = self._attn_logs[gi]
            w_flat = w.float().mean(dim=(0, 1))  # [num_hist]
            num_hist = int(w_flat.shape[0])
            if num_hist == 0 or num_hist % chunk_size != 0:
                continue
            w_by_chunk = w_flat.view(-1, chunk_size).sum(dim=-1)
            first_chunk = (sel_idx[0] if sel_idx else 0) // chunk_size
            total = float(w_by_chunk.sum().item())
            if total <= 0:
                continue

            def _chunk_mass(lo, hi):  # inclusive absolute chunk range
                lo_i = max(0, lo - first_chunk)
                hi_i = min(w_by_chunk.shape[0] - 1, hi - first_chunk)
                if hi_i < lo_i:
                    return 0.0
                return float(w_by_chunk[lo_i : hi_i + 1].sum().item())

            pre_fracs.append(_chunk_mass(0, peak_chunk - 1) / total)
            local_fracs.append(_chunk_mass(gi - 1, gi - 1) / total)
            mirror = peak_chunk - (gi - peak_chunk)
            m_lo = max(0, min(mirror - 1, peak_chunk - 1))
            m_hi = max(0, min(mirror + 1, peak_chunk - 1))
            mirror_fracs.append(_chunk_mass(m_lo, m_hi) / total)
        if not pre_fracs:
            return {}
        n = float(len(pre_fracs))
        return {
            "kv_mem_pre_event_frac": sum(pre_fracs) / n,
            "kv_mem_mirror_frac": sum(mirror_fracs) / n,
            "kv_mem_local_frac": sum(local_fracs) / n,
            "kv_mem_num_chunks": n,
        }

    def _render_attn_heatmap(
        self, chunk_size: int, title: str = None, num_layer_groups: int = 6
    ):
        """Turn self._attn_logs into a multi-panel heatmap.

        The logged weights come from the **ProPE attention branch** whenever
        ProPE is active (DL3DV real cameras, OpenVid synthesized zoom cameras).
        ProPE's Q/K are camera-geometry-transformed (Q' = P_iᵀ·Q, K' = P_j⁻¹·K),
        so softmax(Q'·K'ᵀ/√d) measures how much each history frame is
        geometrically relevant to the current chunk's camera — the semantic
        "which frame does the model remember" signal we want.

        (Falls back to logging std-RoPE attention only if use_prope is False;
        that signal is positional-only and less informative.)

        Layers split into `num_layer_groups` equal groups (default 6 of 5 layers
        each for a 30-block model). Within a group, aggregate across layers AND
        heads by **max** — if ANY (layer, head) attends strongly to a history
        chunk, the cell lights up. This surfaces long-range signals that a mean
        aggregate would wash out against the majority of locally-attending
        layers.

        Layout: 2×3 subplot grid, each subplot
            rows = generated chunks (chunk_i ≥ 1, vertically ordered)
            cols = history chunks (absolute index; NaN = not yet generated)
            color = max-over-(layer,head) attention mass on that history chunk,
                    summed across frames within the chunk.

        All 6 subplots share the same vmin=0 / vmax=global_max colorbar so
        cross-group intensity comparisons are meaningful.

        Returns np.uint8 [H, W, 4] RGBA, or None on failure.
        """
        if not self._attn_logs:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        gen_chunks = sorted(self._attn_logs.keys())
        # Probe shape from first entry
        first_w, _ = self._attn_logs[gen_chunks[0]]
        L_total = first_w.shape[0]
        layers_per_group = max(1, L_total // num_layer_groups)
        actual_groups = [
            (
                g * layers_per_group,
                L_total if g == num_layer_groups - 1 else (g + 1) * layers_per_group,
            )
            for g in range(num_layer_groups)
        ]

        # Build per-group matrix [num_gen_chunks, x_max]
        x_max = 0
        per_group_rows = [[] for _ in range(num_layer_groups)]
        first_chunk_idxs = []
        unit = "chunk"
        for gi in gen_chunks:
            w, sel_idx = self._attn_logs[gi]
            num_hist = w.shape[-1]
            if num_hist > 0 and num_hist % chunk_size == 0:
                first_chunk = (sel_idx[0] if sel_idx else 0) // chunk_size
            else:
                first_chunk = sel_idx[0] if sel_idx else 0
                unit = "frame"
            first_chunk_idxs.append(first_chunk)

            for g, (l0, l1) in enumerate(actual_groups):
                # Max over [l0:l1] layers and all heads, per history frame.
                w_slice = w[l0:l1].float()  # [Lg, H, num_hist]
                w_agg = w_slice.amax(dim=(0, 1))  # [num_hist]
                if unit == "chunk":
                    w_by_chunk = w_agg.view(-1, chunk_size).sum(dim=-1)
                else:
                    w_by_chunk = w_agg
                per_group_rows[g].append(w_by_chunk.numpy())
            x_max = max(x_max, per_group_rows[0][-1].shape[0] + first_chunk)

        # Pack to 2D matrix per group, NaN-padded, absolute-indexed.
        group_mats = []
        for g in range(num_layer_groups):
            mat = np.full((len(gen_chunks), x_max), np.nan, dtype=np.float32)
            for i, (row, fc) in enumerate(zip(per_group_rows[g], first_chunk_idxs)):
                mat[i, fc : fc + row.shape[0]] = row
            group_mats.append(mat)

        # Shared colorbar range.
        global_vmax = 0.0
        for m in group_mats:
            finite = np.isfinite(m)
            if finite.any():
                global_vmax = max(global_vmax, float(m[finite].max()))
        if global_vmax <= 0:
            global_vmax = 1.0

        # 2×3 grid of subplots.
        n_rows_grid = 2
        n_cols_grid = (num_layer_groups + n_rows_grid - 1) // n_rows_grid  # ceil
        panel_w = max(3.5, x_max * 0.5 + 1.2)
        panel_h = max(2.2, len(gen_chunks) * 0.5 + 1.0)
        fig_w = panel_w * n_cols_grid
        fig_h = panel_h * n_rows_grid + 0.8  # extra space for suptitle
        fig, axes = plt.subplots(
            n_rows_grid, n_cols_grid, figsize=(fig_w, fig_h), squeeze=False
        )
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="#333333")

        im = None
        for g in range(num_layer_groups):
            ax = axes[g // n_cols_grid][g % n_cols_grid]
            l0, l1 = actual_groups[g]
            im = ax.imshow(
                np.ma.masked_invalid(group_mats[g]),
                aspect="auto",
                cmap=cmap,
                interpolation="nearest",
                vmin=0.0,
                vmax=global_vmax,
            )
            ax.set_title(f"layers {l0}..{l1 - 1}  (max)", fontsize=9)
            ax.set_xticks(range(x_max))
            ax.set_xticklabels([f"c{i}" for i in range(x_max)], fontsize=6)
            ax.set_yticks(range(len(gen_chunks)))
            ax.set_yticklabels([f"c{gi}" for gi in gen_chunks], fontsize=6)
            if g % n_cols_grid == 0:
                ax.set_ylabel("gen chunk")
            if g // n_cols_grid == n_rows_grid - 1:
                ax.set_xlabel(f"history {unit}")
            # Cell annotations — use 2 decimals; skip NaN; contrast color.
            M = group_mats[g]
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    if np.isnan(v):
                        continue
                    color = "white" if v < 0.4 * global_vmax else "black"
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=6,
                    )

        # Turn off any unused cells.
        for g in range(num_layer_groups, n_rows_grid * n_cols_grid):
            axes[g // n_cols_grid][g % n_cols_grid].axis("off")

        # Shared colorbar on the right.
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.12, 0.012, 0.76])
        fig.colorbar(im, cax=cbar_ax, label="max attn weight (over layers × heads)")

        if title:
            fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.93, 0.96])

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(h_px, w_px, 4).copy()
        plt.close(fig)
        return img

    # --- KV cache lifecycle ---
    def _init_kv_cache(self, batch_size, device, dtype):
        """Allocate (or zero out) KV cache with ProPE slots."""
        need_alloc = (
            self.kv_cache is None
            or self.kv_cache[0]["k"].shape[0] != batch_size
            or self.kv_cache[0]["k"].shape[1] < self.kv_cache_max_tokens
            or self.kv_cache[0]["k"].device != device
            or self.kv_cache[0]["k"].dtype != dtype
        )
        if need_alloc:
            self.kv_cache = []
            for _ in range(self.num_transformer_blocks):
                self.kv_cache.append(
                    {
                        "k": torch.zeros(
                            batch_size,
                            self.kv_cache_max_tokens,
                            self.num_heads,
                            self.head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                        "v": torch.zeros(
                            batch_size,
                            self.kv_cache_max_tokens,
                            self.num_heads,
                            self.head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                        "k_prope": torch.zeros(
                            batch_size,
                            self.kv_cache_max_tokens,
                            self.num_heads,
                            self.head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                        "v_prope": torch.zeros(
                            batch_size,
                            self.kv_cache_max_tokens,
                            self.num_heads,
                            self.head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                    }
                )
        else:
            for bc in self.kv_cache:
                bc["k"].zero_()
                bc["v"].zero_()
                bc["k_prope"].zero_()
                bc["v_prope"].zero_()

    def _encode_history_to_cache(
        self,
        history_latents: torch.Tensor,  # [B, N, 16, H_lat, W_lat] clean history frames
        history_viewmats: torch.Tensor,  # [B, N, 4, 4] or None
        history_Ks: torch.Tensor,  # [B, N, 3, 3] or None
        cond: dict,  # text conditioning
        context_noise_t: Optional[int] = None,  # low noise for context (near-clean)
        cache_offset: int = 0,  # where in the cache buffer to write
        rope_offset: int = 0,  # RoPE starting position (default 0 = compact)
        prompt_frame_indices: Optional[List[int]] = None,
    ):
        """
        Encode history frames to KV cache via a context pass (kv_size=(cache_offset, -1)).

        Cache-offset and RoPE-offset are decoupled on purpose: you can write K/V
        into compact cache slots while using whatever RoPE positions reflect the
        frames' actual temporal locations. This is critical for node-insertion
        inference, where we want to glue e.g. frames [0..2] and [15..17] into
        contiguous cache slots [0..5] but preserve their original RoPE phases
        (so the model sees the temporal gap rather than thinking they're
        adjacent).
        """
        B, N, C, H_lat, W_lat = history_latents.shape
        device = history_latents.device
        if context_noise_t is None:
            context_noise_t = self.clean_context_timestep

        # Pure 16-ch video latent — no mask/render channels.
        t = torch.full([B, N], context_noise_t, device=device, dtype=torch.int64)

        cond_local = cond
        if prompt_frame_indices is not None and cond["prompt_embeds"].ndim == 4:
            cond_local = dict(cond)
            cond_local["prompt_frame_indices"] = list(prompt_frame_indices)

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=history_latents,
                conditional_dict=cond_local,
                timestep=t,
                kv_cache=self.kv_cache,
                kv_size=(cache_offset, -1),  # write at cache_offset
                render_latent_input=None,
                freqs_offset=rope_offset,  # RoPE positions start here
                viewmats=history_viewmats,
                Ks=history_Ks,
            )

        return N * self.tokens_per_frame

    def _denoise_current_chunk(
        self,
        noise: torch.Tensor,  # [B, chunk_size, 16, H_lat, W_lat]
        current_viewmats: torch.Tensor,  # [B, chunk_size, 4, 4] or None
        current_Ks: torch.Tensor,  # [B, chunk_size, 3, 3] or None
        cond: dict,
        history_tokens: int,  # size of history in cache (token count)
        current_rope_start: Optional[
            int
        ] = None,  # explicit RoPE start for current chunk
        known_prefix_latent: Optional[torch.Tensor] = None,
        noise_generator: Optional[torch.Generator] = None,
        uncond: Optional[dict] = None,
        guidance_scale: float = 1.0,
        guidance_end_frac: float = 1.0,
        cfg_rescale: float = 0.0,
    ) -> torch.Tensor:
        """
        Multi-step denoising of current chunk, reading from KV cache.

        By default, `current_rope_start = history_tokens / TOKENS_PER_FRAME`
        (compact: current follows right after history). For node-insertion
        inference where history came from non-contiguous frames, pass the
        current chunk's ACTUAL frame index explicitly.
        """
        B, cs, C, H_lat, W_lat = noise.shape
        device, dtype = noise.device, noise.dtype

        if current_rope_start is None:
            current_rope_start = history_tokens // self.tokens_per_frame

        known_n = 0
        if known_prefix_latent is not None:
            known_n = min(int(known_prefix_latent.shape[1]), cs)
            known_prefix_latent = known_prefix_latent[:, :known_n].to(
                device=device, dtype=dtype
            )

        current = noise
        if known_n > 0:
            current[:, :known_n] = known_prefix_latent
        num_steps = len(self.denoising_steps)
        # Guidance scheduling: apply CFG only on the first
        # `guidance_end_frac` fraction of denoising steps (high noise, where
        # motion and content are decided), then fall back to the conditional
        # pass alone.  Low-noise CFG mostly shifts color/exposure, and in
        # chunk-AR generation that drift feeds back through the history and
        # compounds across chunks.
        guided_steps = int(round(num_steps * max(0.0, min(1.0, guidance_end_frac))))
        for di, denoise_t in enumerate(self.denoising_steps):
            t = torch.full(
                [B, cs], int(denoise_t.item()), device=device, dtype=torch.int64
            )
            if known_n > 0:
                t[:, :known_n] = int(self.clean_context_timestep)
            with torch.no_grad():
                _, denoised = self.generator(
                    noisy_image_or_video=current,
                    conditional_dict=cond,
                    timestep=t,
                    kv_cache=self.kv_cache,
                    kv_size=(0, history_tokens),  # read history from cache start
                    render_latent_input=None,
                    freqs_offset=current_rope_start,  # current chunk's RoPE start
                    viewmats=current_viewmats,
                    Ks=current_Ks,
                )
                if uncond is not None and guidance_scale != 1.0 and di < guided_steps:
                    # Text-only CFG: second pass with the negative/empty prompt,
                    # sharing the cond-encoded history KV cache (history stays
                    # fully conditioned; guidance only pushes the current
                    # chunk's text alignment).  x0 predictions combine linearly
                    # under the flow-matching scheduler.
                    _, denoised_u = self.generator(
                        noisy_image_or_video=current,
                        conditional_dict=uncond,
                        timestep=t,
                        kv_cache=self.kv_cache,
                        kv_size=(0, history_tokens),
                        render_latent_input=None,
                        freqs_offset=current_rope_start,
                        viewmats=current_viewmats,
                        Ks=current_Ks,
                    )
                    denoised_cfg = denoised_u + guidance_scale * (denoised - denoised_u)
                    if cfg_rescale > 0.0:
                        # CFG-rescale (Lin et al., "Common Diffusion Noise
                        # Schedules and Sample Steps are Flawed"): match the
                        # per-frame std of the guided prediction back to the
                        # conditional prediction to suppress the
                        # over-saturation / exposure shift that raw CFG
                        # extrapolation introduces.
                        dims = tuple(range(2, denoised.ndim))
                        std_c = denoised.std(dim=dims, keepdim=True) + 1e-6
                        std_g = denoised_cfg.std(dim=dims, keepdim=True) + 1e-6
                        rescaled = denoised_cfg * (std_c / std_g)
                        denoised_cfg = (
                            cfg_rescale * rescaled + (1.0 - cfg_rescale) * denoised_cfg
                        )
                    denoised = denoised_cfg
            if known_n > 0:
                denoised[:, :known_n] = known_prefix_latent
            if di < len(self.denoising_steps) - 1:
                next_t = self.denoising_steps[di + 1]
                renoise = torch.randn(
                    denoised.flatten(0, 1).shape,
                    device=device,
                    dtype=dtype,
                    generator=noise_generator,
                )
                current = self.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    renoise,
                    next_t.to(device)
                    * torch.ones([B * cs], device=device, dtype=torch.long),
                ).unflatten(0, denoised.shape[:2])
                if known_n > 0:
                    current[:, :known_n] = known_prefix_latent
        return denoised

    @torch.no_grad()
    def inference(
        self,
        ref_latent: torch.Tensor,  # [B, N*chunk_size, 16, H_lat, W_lat]
        viewmats: torch.Tensor,  # [B, T_lat, 4, 4] or None
        Ks: torch.Tensor,  # [B, T_lat, 3, 3] or None
        text_prompts: List[str],
        num_latent_frames: int,
        memory_frames: int = 5,
        temporal_context_size: int = 3,
        noise_seed: Optional[int] = None,
        decode: bool = True,
        select_all_history: bool = False,
        log_attention: bool = False,
        degradation_control: Optional[torch.Tensor] = None,
        gt_history_latent: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        negative_prompt: str = "",
        guidance_end_frac: float = 1.0,
        cfg_rescale: float = 0.0,
    ):
        """
        Autoregressive chunk-by-chunk generation starting from ref_latent.

        If `gt_history_latent` ([B, T_lat, 16, H_lat, W_lat]) is given, every
        chunk is conditioned on the CLEAN ground-truth history instead of the
        model's own rollout (clean-history window diagnostic): drift cannot
        accumulate, so per-chunk quality isolates the model's one-chunk
        predictive ability from exposure bias.

        `ref_latent` may be a partial first chunk or any number of full chunks:
          - T=1 → strict first-frame I2V; frame 0 is pinned clean while the rest
            of block 0 is denoised normally.
          - T=N*cs → V2V/video continuation; the first N chunks are pinned to
            GT and generation picks up at chunk N.

        When *select_all_history* is True every previously generated frame is
        fed as KV-cache history (mirrors the chunk-causal training mask).
        Otherwise a bounded window of the most recent frames is selected.

        If *log_attention* is True, after every chunk denoise we harvest
        per-history-frame mean attention weight from every transformer block
        (averaged over heads, query tokens, and denoising steps). Result is
        stashed on `self._attn_logs` as {chunk_i: [num_layers, num_heads, num_hist_frames]}
        and its per-chunk text summary returned in `self._attn_summary`.
        Slow (~2× per attention layer, no flash kernel) — intended for val only.

        Returns either decoded video [B, T, 3, H, W] in [0, 1] or latents
        [B, T_lat, 16, H_lat, W_lat].
        """
        B, C = ref_latent.shape[0], ref_latent.shape[2]
        H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]
        device, dtype = ref_latent.device, ref_latent.dtype
        cs = self.chunk_size

        ref_n = ref_latent.shape[1]

        num_latent_frames = num_latent_frames - num_latent_frames % cs
        num_chunks = num_latent_frames // cs
        assert 0 < ref_n < num_latent_frames, (
            f"ref_latent must seed at least one frame but not the full window; "
            f"got seed={ref_n}, total={num_latent_frames}"
        )
        full_seed_chunks, partial_seed = divmod(ref_n, cs)
        start_chunk = full_seed_chunks

        # Text embedding (reused across chunks)
        cond = self._encode_text_prompts(text_prompts)
        if degradation_control is not None:
            if degradation_control.shape[1] < num_latent_frames:
                raise ValueError(
                    "degradation_control is shorter than inference window: "
                    f"{degradation_control.shape[1]} < {num_latent_frames}"
                )
            cond["degradation_control"] = degradation_control[:, :num_latent_frames]
        uncond = None
        if guidance_scale != 1.0:
            # Global (non-chunked) negative/empty prompt; keep non-text controls
            # identical so guidance isolates the text signal.
            uncond = self.text_encoder(text_prompts=[negative_prompt] * B)
            if "degradation_control" in cond:
                uncond["degradation_control"] = cond["degradation_control"]

        # Output buffer (start with the seed chunks pinned to the GT).
        output = torch.zeros(
            B, num_latent_frames, C, H_lat, W_lat, device=device, dtype=dtype
        )
        output[:, :ref_n] = ref_latent

        # Noise generator
        gen = torch.Generator(device=device)
        if noise_seed is not None:
            gen.manual_seed(noise_seed)

        # Init cache
        self._init_kv_cache(B, device, dtype)

        # KV cache importance probing (off by default)
        if log_attention:
            self._attn_logs = {}
            self._attn_summary = ""  # clear any stale summary
            self._enable_attn_log(True)

        # Generate each subsequent chunk.  For a partial first chunk seed
        # (strict I2V), chunk 0 is generated with its first frame pinned clean.
        for chunk_i in range(start_chunk, num_chunks):
            s = chunk_i * cs
            e = s + cs
            current_frame_idx = s
            known_prefix = None
            if partial_seed > 0 and chunk_i == full_seed_chunks:
                known_prefix = ref_latent[:, full_seed_chunks * cs : ref_n]

            # 1. Select history frames.
            if select_all_history:
                selected_idx = list(range(s))
            else:
                # Bound recent history by both the requested window and cache size.
                max_hist_tokens = self.kv_cache_max_tokens
                max_hist_frames = min(s, max_hist_tokens // self.tokens_per_frame)
                n_select = min(s, max(memory_frames + temporal_context_size, cs))
                n_select = min(n_select, max_hist_frames)
                selected_idx = list(range(max(0, s - n_select), s))

            logger.debug(
                "Chunk %d/%d: selected frames %s (curr=%d)",
                chunk_i,
                num_chunks - 1,
                selected_idx,
                current_frame_idx,
            )

            # 2. Refresh cache with selected history
            self._init_kv_cache(B, device, dtype)  # clear
            if selected_idx:
                hist_src = (
                    gt_history_latent if gt_history_latent is not None else output
                )
                hist_lat = hist_src[:, selected_idx]  # [B, N_sel, 16, H, W]
                hist_vm = viewmats[:, selected_idx] if viewmats is not None else None
                hist_Ks = Ks[:, selected_idx] if Ks is not None else None
                history_tokens = self._encode_history_to_cache(
                    hist_lat, hist_vm, hist_Ks, cond, prompt_frame_indices=selected_idx
                )
            else:
                history_tokens = 0

            # 3. Denoise current chunk
            noise_curr = torch.randn(
                B, cs, C, H_lat, W_lat, device=device, dtype=dtype, generator=gen
            )
            # Reset per-chunk attention accumulators (keeps logging enabled).
            if log_attention:
                self._enable_attn_log(True)

            denoised = self._denoise_current_chunk(
                noise_curr,
                viewmats[:, s:e] if viewmats is not None else None,
                Ks[:, s:e] if Ks is not None else None,
                cond,
                history_tokens,
                known_prefix_latent=known_prefix,
                noise_generator=gen,
                uncond=uncond,
                guidance_scale=guidance_scale,
                guidance_end_frac=guidance_end_frac,
                cfg_rescale=cfg_rescale,
            )
            output[:, s:e] = denoised

            if log_attention:
                logged = self._collect_attn_log()
                if logged is not None:
                    # [L_layers, H_heads, num_hist_frames] — selected_idx maps
                    # the history-frame axis back to absolute frame indices.
                    self._attn_logs[chunk_i] = (logged.cpu(), list(selected_idx))

        if log_attention:
            self._enable_attn_log(False)
            self._attn_summary = self._format_attn_summary(cs)

        if not decode:
            return output

        # VAE decode
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
        return video

    @torch.no_grad()
    def inference_ref_cache(
        self,
        ref_latent: torch.Tensor,  # [B, R, 16, H_lat, W_lat]
        target_seed_latent: Optional[torch.Tensor],  # [B, S, 16, H_lat, W_lat] or None
        target_viewmats: torch.Tensor,  # [B, T_target, 4, 4] or None
        target_Ks: torch.Tensor,  # [B, T_target, 3, 3] or None
        text_prompts: List[str],
        target_num_latent_frames: int,
        ref_viewmats: torch.Tensor = None,  # [B, R, 4, 4] or None
        ref_Ks: torch.Tensor = None,  # [B, R, 3, 3] or None
        ref_rope_start: int = 0,
        target_rope_start: int = 48,
        noise_seed: Optional[int] = None,
        decode: bool = True,
        select_all_history: bool = True,
        log_attention: bool = False,
    ):
        """Autoregressive generation with an old reference cached separately.

        This is the reference-cache inference mode. `ref_latent` is encoded
        into compact cache slots
        but uses RoPE positions starting at `ref_rope_start`; target history
        starts at `target_rope_start`.  Setting ref=0 and target=+gap is
        RoPE-equivalent to putting the reference at t=-gap.
        """
        B, C = ref_latent.shape[0], ref_latent.shape[2]
        H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]
        device, dtype = ref_latent.device, ref_latent.dtype
        cs = self.chunk_size

        assert ref_latent.shape[1] % cs == 0, (
            f"ref_latent must be chunk-aligned, got {ref_latent.shape[1]}"
        )
        if target_seed_latent is None:
            seed_n = 0
        else:
            assert target_seed_latent.shape[1] % cs == 0, (
                f"target_seed_latent must be chunk-aligned, got {target_seed_latent.shape[1]}"
            )
            seed_n = target_seed_latent.shape[1]

        target_num_latent_frames = (
            target_num_latent_frames - target_num_latent_frames % cs
        )
        num_chunks = target_num_latent_frames // cs
        seed_chunks = seed_n // cs
        assert seed_chunks < num_chunks, (
            f"target seed covers {seed_chunks}/{num_chunks} chunks; nothing to generate"
        )

        cond = self._encode_text_prompts(text_prompts)
        output = torch.zeros(
            B, target_num_latent_frames, C, H_lat, W_lat, device=device, dtype=dtype
        )
        if seed_n > 0:
            output[:, :seed_n] = target_seed_latent

        gen = torch.Generator(device=device)
        if noise_seed is not None:
            gen.manual_seed(noise_seed)

        if log_attention:
            self._attn_logs = {}
            self._attn_summary = ""
            self._enable_attn_log(True)

        for chunk_i in range(seed_chunks, num_chunks):
            s = chunk_i * cs
            e = s + cs
            self._init_kv_cache(B, device, dtype)

            cache_off = 0
            cache_off += self._encode_history_to_cache(
                ref_latent,
                ref_viewmats,
                ref_Ks,
                cond,
                cache_offset=cache_off,
                rope_offset=ref_rope_start,
            )

            if select_all_history:
                selected_idx = list(range(s))
            else:
                selected_idx = list(range(max(0, s - cs), s))

            if selected_idx:
                # Split selected target history into contiguous groups so each
                # group can keep its true target RoPE offset after the gap.
                groups = [[selected_idx[0]]]
                for fi in selected_idx[1:]:
                    if fi == groups[-1][-1] + 1:
                        groups[-1].append(fi)
                    else:
                        groups.append([fi])
                for grp in groups:
                    gs, ge = grp[0], grp[-1] + 1
                    cache_off += self._encode_history_to_cache(
                        output[:, gs:ge],
                        target_viewmats[:, gs:ge]
                        if target_viewmats is not None
                        else None,
                        target_Ks[:, gs:ge] if target_Ks is not None else None,
                        cond,
                        cache_offset=cache_off,
                        rope_offset=target_rope_start + gs,
                    )

            history_tokens = cache_off
            noise_curr = torch.randn(
                B, cs, C, H_lat, W_lat, device=device, dtype=dtype, generator=gen
            )
            if log_attention:
                self._enable_attn_log(True)
            denoised = self._denoise_current_chunk(
                noise_curr,
                target_viewmats[:, s:e] if target_viewmats is not None else None,
                target_Ks[:, s:e] if target_Ks is not None else None,
                cond,
                history_tokens,
                current_rope_start=target_rope_start + s,
                noise_generator=gen,
            )
            output[:, s:e] = denoised

            if log_attention:
                logged = self._collect_attn_log()
                if logged is not None:
                    hist_labels = [
                        ref_rope_start + i for i in range(ref_latent.shape[1])
                    ] + [target_rope_start + i for i in selected_idx]
                    self._attn_logs[chunk_i] = (logged.cpu(), hist_labels)

        if log_attention:
            self._enable_attn_log(False)
            self._attn_summary = self._format_attn_summary(cs)

        if not decode:
            return output

        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video

    @torch.no_grad()
    def inference_node_insertion(
        self,
        ref_latent: torch.Tensor,  # [B, seed_frames, 16, H_lat, W_lat]
        text_prompts: List[str],
        num_latent_frames: int,
        zoom_info: dict,  # from openvid_zoom.zoom_info_to_latent_chunks
        viewmats: torch.Tensor = None,  # optional; None → no ProPE
        Ks: torch.Tensor = None,
        noise_seed: Optional[int] = None,
        decode: bool = True,
        degradation_control: Optional[torch.Tensor] = None,
    ):
        """
        Alternative validation inference with SPARSE HISTORY.

        For every chunk_i ≥ 1, the KV cache holds only:

            {frame 0}  ∪  {anchor_chunk frames}  ∪  {chunk_i - 1 frames}

        (anchor_chunk is only added once chunk_i > anchor_chunk; before that the
        anchor frames haven't been generated yet.)

        Each group is packed contiguously into cache slots for memory efficiency
        but KEEPS ITS ORIGINAL RoPE POSITION (rope_offset = true frame index), so
        temporal ProPE + standard RoPE see the true time gaps rather than a
        compacted sequence. The current chunk uses its actual frame index as
        freqs_offset.

        This tests whether the model can "jump back" to the pre-zoom framing by
        cross-attending over a long temporal gap instead of drifting with the
        zoomed history.
        """
        B, C = ref_latent.shape[0], ref_latent.shape[2]
        H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]
        device, dtype = ref_latent.device, ref_latent.dtype
        cs = self.chunk_size
        ref_n = int(ref_latent.shape[1])
        assert 0 < ref_n <= cs, (
            f"node insertion expects 1..chunk_size seed frames, got {ref_n}"
        )

        num_latent_frames = num_latent_frames - num_latent_frames % cs
        num_chunks = num_latent_frames // cs
        full_seed_chunks, partial_seed = divmod(ref_n, cs)
        start_chunk = full_seed_chunks

        anchor_chunk = int(zoom_info["anchor_chunk"])
        zoom_out_chunk = int(zoom_info["zoom_out_chunk"])
        anchor_chunk = max(0, min(anchor_chunk, num_chunks - 1))
        zoom_out_chunk = max(1, min(zoom_out_chunk, num_chunks))

        cond = self._encode_text_prompts(text_prompts)
        if degradation_control is not None:
            if degradation_control.shape[1] < num_latent_frames:
                raise ValueError(
                    "degradation_control is shorter than inference window: "
                    f"{degradation_control.shape[1]} < {num_latent_frames}"
                )
            cond["degradation_control"] = degradation_control[:, :num_latent_frames]

        output = torch.zeros(
            B, num_latent_frames, C, H_lat, W_lat, device=device, dtype=dtype
        )
        output[:, :ref_n] = ref_latent

        gen = torch.Generator(device=device)
        if noise_seed is not None:
            gen.manual_seed(noise_seed)

        for chunk_i in range(start_chunk, num_chunks):
            s = chunk_i * cs
            e = s + cs
            known_prefix = None
            if partial_seed > 0 and chunk_i == full_seed_chunks:
                known_prefix = ref_latent[:, full_seed_chunks * cs : ref_n]
            # Fresh cache per chunk — history composition changes each step.
            self._init_kv_cache(B, device, dtype)

            # Build sparse history: {frame 0} ∪ {anchor chunk} ∪ {prev chunk}.
            # Each group keeps its ORIGINAL RoPE position; cache slots are packed.
            prev_s = (chunk_i - 1) * cs
            anchor_s = anchor_chunk * cs
            anchor_e = anchor_s + cs

            frame_set = set()
            if chunk_i > 0:
                frame_set.add(0)  # first frame always
                if chunk_i > anchor_chunk:
                    frame_set.update(range(anchor_s, anchor_e))
                frame_set.update(range(prev_s, prev_s + cs))
                frame_set = {fi for fi in frame_set if 0 <= fi < s}

            # Split into contiguous groups for _encode_history_to_cache
            frames_sorted = sorted(frame_set)
            groups = []
            if frames_sorted:
                groups = [[frames_sorted[0]]]
                for fi in frames_sorted[1:]:
                    if fi == groups[-1][-1] + 1:
                        groups[-1].append(fi)
                    else:
                        groups.append([fi])

            cache_off = 0
            for grp in groups:
                gs, ge = grp[0], grp[-1] + 1
                self._encode_history_to_cache(
                    output[:, gs:ge],
                    viewmats[:, gs:ge] if viewmats is not None else None,
                    Ks[:, gs:ge] if Ks is not None else None,
                    cond,
                    cache_offset=cache_off,
                    rope_offset=gs,
                )
                cache_off += len(grp) * self.tokens_per_frame

            history_tokens = cache_off
            cur_rope = s  # actual frame index

            logger.debug(
                "Chunk %d/%d sparse-hist: frames %s (%d tokens), cur RoPE %d",
                chunk_i,
                num_chunks - 1,
                frames_sorted,
                history_tokens,
                cur_rope,
            )

            noise_curr = torch.randn(
                B, cs, C, H_lat, W_lat, device=device, dtype=dtype, generator=gen
            )
            denoised = self._denoise_current_chunk(
                noise_curr,
                viewmats[:, s:e] if viewmats is not None else None,
                Ks[:, s:e] if Ks is not None else None,
                cond,
                history_tokens,
                current_rope_start=cur_rope,
                known_prefix_latent=known_prefix,
                noise_generator=gen,
            )
            output[:, s:e] = denoised

        if not decode:
            return output

        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video
