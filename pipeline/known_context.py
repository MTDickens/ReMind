# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Known-prefix conditioning for I2V and V2V inference."""

from __future__ import annotations

import torch


def build_known_context(
    clean_latents: torch.Tensor,
    prefix_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    known = clean_latents.detach()
    mask = torch.zeros(
        clean_latents.shape[0],
        clean_latents.shape[1],
        1,
        1,
        1,
        dtype=torch.bool,
        device=clean_latents.device,
    )
    mask[:, : max(0, int(prefix_frames))] = True
    return known, mask
