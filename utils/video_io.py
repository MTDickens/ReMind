# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Robust MP4 writer that works across environments.

Previous versions used imageio, which silently falls back to PyAVPlugin when
`imageio-ffmpeg` is not installed. PyAVPlugin.write() does not accept the
`output_params` kwarg (and internal translation of `quality`/`pixelformat` can
also break) with:
    "PyAVPlugin.write() got an unexpected keyword argument 'output_params'"

This module uses `torchvision.io.write_video` as the primary backend: it ships
with torchvision (always present in this repo), writes h264 in a single call,
and has no plugin-detection layer. imageio is kept as a fallback for
environments where torchvision.io can't find its video codec.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import torch

logger = logging.getLogger(__name__)


def write_mp4(
    path: str,
    frames: torch.Tensor | np.ndarray,
    fps: int = 24,
    crf: int = 18,
) -> None:
    """Write an H.264 MP4 to `path`.

    Args:
        frames: either
            • torch.Tensor [T, C, H, W] in [0, 1] float, OR
            • torch.Tensor [T, H, W, C] in [0, 1] float, OR
            • uint8 numpy array [T, H, W, C]
        fps: frames per second
        crf: H.264 quality (lower = better, 18 ≈ visually lossless)
    """
    # Normalize to uint8 [T, H, W, C]
    if isinstance(frames, torch.Tensor):
        f = frames.detach().cpu()
        if f.dtype != torch.uint8:
            if f.ndim == 4 and f.shape[1] in (1, 3):      # [T, C, H, W]
                f = f.permute(0, 2, 3, 1)
            f = (f.clamp(0, 1) * 255).to(torch.uint8)
        else:
            if f.ndim == 4 and f.shape[1] in (1, 3):
                f = f.permute(0, 2, 3, 1)
    else:
        f = torch.as_tensor(frames)
        if f.dtype != torch.uint8:
            f = (f.clamp(0, 1) * 255).to(torch.uint8) if f.dtype.is_floating_point else f.to(torch.uint8)

    assert f.ndim == 4 and f.shape[-1] in (1, 3), \
        f"frames must be [T, H, W, C] after normalization, got {tuple(f.shape)}"
    if f.shape[-1] == 1:
        f = f.expand(-1, -1, -1, 3)
    f = f.contiguous()

    # Ensure dims divisible by 2 for yuv420p (H.264 requirement).
    T, H, W, C = f.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        f = torch.nn.functional.pad(
            f.permute(0, 3, 1, 2).float(), (0, pad_w, 0, pad_h)
        ).permute(0, 2, 3, 1).to(torch.uint8).contiguous()

    # Primary: torchvision.io.write_video
    try:
        import torchvision.io as tvio
        tvio.write_video(
            filename=path,
            video_array=f,
            fps=fps,
            video_codec="h264",
            options={"crf": str(crf), "pix_fmt": "yuv420p"},
        )
        return
    except Exception as e_tv:
        logger.debug("torchvision.io.write_video failed (%s), falling back to imageio", e_tv)

    # Fallback: imageio (try FFMPEG plugin, then PyAV, then bare)
    import imageio
    attempts: Iterable[dict[str, Any]] = [
        dict(format="FFMPEG", fps=fps, codec="libx264",
             quality=10 - min(10, crf // 5), pixelformat="yuv420p"),
        dict(format="pyav", fps=fps, codec="libx264"),
        dict(fps=fps),
    ]
    last_err: Exception | None = None
    for kw in attempts:
        try:
            writer = imageio.get_writer(path, **kw)
            for t in range(f.shape[0]):
                writer.append_data(f[t].numpy())
            writer.close()
            return
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not write mp4 to {path}: {last_err}")


# Back-compat shim: old callers that want a writer-like object.
def mp4_writer(path: str, fps: int = 24):
    """Deprecated: prefer write_mp4(path, frames, fps).

    Returns a minimal writer that buffers frames then writes on close().
    """
    return _BufferedMp4Writer(path, fps)


class _BufferedMp4Writer:
    def __init__(self, path: str, fps: int):
        self.path = path
        self.fps = fps
        self._frames: list[np.ndarray] = []

    def append_data(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    def close(self) -> None:
        if not self._frames:
            return
        arr = np.stack(self._frames, axis=0)  # [T, H, W, C]
        write_mp4(self.path, arr, fps=self.fps)
        self._frames.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
