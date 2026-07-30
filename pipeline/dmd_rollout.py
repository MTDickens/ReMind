# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Four-step chunk-autoregressive rollout for ReMind inference."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


class DMDInferenceRollout:
    def __init__(
        self,
        pipeline,
        scheduler,
        denoising_step_list: List[int],
        *,
        chunk_size: int = 3,
        context_timestep: Optional[int] = None,
    ):
        self.pipe = pipeline
        self.scheduler = scheduler
        self.steps = [int(value) for value in denoising_step_list if int(value) > 0]
        self.chunk_size = int(chunk_size)
        self.context_timestep = context_timestep

    @staticmethod
    def _overwrite_known(value, known, mask):
        if known is None or mask is None:
            return value
        return torch.where(mask, known.to(value.dtype), value)

    @torch.no_grad()
    def rollout(
        self,
        noise: torch.Tensor,
        cond: Dict,
        cam: Dict,
        *,
        known_latents: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
        exit_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        batch, frames = noise.shape[:2]
        chunk_size = self.chunk_size
        if frames % chunk_size:
            raise ValueError(
                f"latent frames ({frames}) must be divisible by "
                f"chunk_size ({chunk_size})"
            )
        num_chunks = frames // chunk_size
        if exit_indices is None:
            exit_indices = [len(self.steps) - 1] * num_chunks
        if len(exit_indices) != num_chunks:
            raise ValueError("exit_indices must contain one value per chunk")

        self.pipe.reset_cache(noise)
        outputs = []
        for chunk_index in range(num_chunks):
            lo = chunk_index * chunk_size
            hi = (chunk_index + 1) * chunk_size
            current = noise[:, lo:hi]
            camera_chunk = {
                key: (value[:, lo:hi] if torch.is_tensor(value) else value)
                for key, value in cam.items()
            }
            known_chunk = known_latents[:, lo:hi] if known_latents is not None else None
            mask_chunk = known_mask[:, lo:hi] if known_mask is not None else None
            current = self._overwrite_known(current, known_chunk, mask_chunk)

            predicted_clean = None
            for step_index, timestep_value in enumerate(self.steps):
                timestep = torch.full(
                    (batch, chunk_size),
                    int(timestep_value),
                    device=current.device,
                    dtype=torch.long,
                )
                if mask_chunk is not None:
                    timestep = timestep.masked_fill(mask_chunk[:, :, 0, 0, 0], 0)
                predicted_clean = self.pipe.denoise_step(
                    current,
                    chunk_index=chunk_index,
                    timestep=timestep,
                    conditional_dict=cond,
                    **camera_chunk,
                )
                predicted_clean = self._overwrite_known(
                    predicted_clean, known_chunk, mask_chunk
                )
                if step_index == int(exit_indices[chunk_index]):
                    break
                next_timestep = torch.full(
                    (batch, chunk_size),
                    self.steps[step_index + 1],
                    device=current.device,
                    dtype=torch.long,
                )
                current = self.scheduler.add_noise(
                    predicted_clean.flatten(0, 1),
                    torch.randn_like(predicted_clean.flatten(0, 1)),
                    next_timestep.flatten(0, 1),
                ).unflatten(0, (batch, chunk_size))
                current = self._overwrite_known(current, known_chunk, mask_chunk)

            if predicted_clean is None:
                raise RuntimeError("empty denoising schedule")
            outputs.append(predicted_clean)
            self.pipe.encode_chunk_to_cache(
                predicted_clean,
                chunk_index=chunk_index,
                conditional_dict=cond,
                context_timestep=self.context_timestep,
                **camera_chunk,
            )
        return torch.cat(outputs, dim=1)
