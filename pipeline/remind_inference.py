# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Public ReMind 5B DMD-EMA inference helpers.

The released DMD checkpoint is an adapter, not a standalone model. Inference
therefore loads the official Wan base, overlays the ReMind-5B generator,
merges the EMA student LoRA, and runs the same four-step
chunk-autoregressive path used for the project-page samples.
"""

from __future__ import annotations

import json
import io
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image, ImageOps

from pipeline.cache_adapter import InferencePipelineAdapter
from pipeline.causal_inference import CausalInferencePipeline
from pipeline.checkpoints import (
    STUDENT_ADAPTER,
    configure_inference_lora,
    load_adapter_state_dict,
    load_generator_checkpoint,
)
from pipeline.dmd_rollout import DMDInferenceRollout
from pipeline.known_context import build_known_context
from pipeline.prompt_schedule import chunk_text_prompts
from utils.video_io import write_mp4


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_KINDS = {"clean", "camera", "occluder", "light"}
TASKS = {"i2v", "v2v"}
CANONICAL_5B_DMD_STEPS = [1000, 937, 833, 625]


def load_preset(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load and validate one public demo preset."""
    preset_path = Path(path).expanduser().resolve()
    if not preset_path.is_file():
        raise FileNotFoundError(preset_path)
    data = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"preset must be a mapping: {preset_path}")
    missing = [key for key in ("name", "prompt") if not data.get(key)]
    if missing:
        raise ValueError(f"preset is missing required keys {missing}: {preset_path}")
    task = str(data.get("task", "i2v")).lower()
    if task not in TASKS:
        raise ValueError(f"unsupported task {task!r}; choose {sorted(TASKS)}")
    data["task"] = task
    input_key = "input_image" if task == "i2v" else "input_video"
    if not data.get(input_key):
        raise ValueError(
            f"{task} preset is missing required key {input_key!r}: {preset_path}"
        )
    control = data.get("control") or {"kind": "clean"}
    if not isinstance(control, dict):
        raise ValueError("preset control must be a mapping")
    kind = str(control.get("kind", "clean")).lower()
    if kind not in CONTROL_KINDS:
        raise ValueError(
            f"unsupported control kind {kind!r}; choose {sorted(CONTROL_KINDS)}"
        )
    control["kind"] = kind
    data["control"] = control
    input_path = (preset_path.parent / str(data[input_key])).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    data[f"_{input_key}_path"] = input_path

    chunk_size = int(data.get("chunk_size", 3))
    latent_frames = (int(data.get("num_frames", 81)) - 1) // 4 + 1
    latent_frames -= latent_frames % chunk_size
    default_seed_frames = 1 if task == "i2v" else 2 * chunk_size
    seed_latent_frames = int(data.get("seed_latent_frames", default_seed_frames))
    if not 0 < seed_latent_frames < latent_frames:
        raise ValueError(
            f"seed_latent_frames must be in [1, {latent_frames - 1}], got "
            f"{seed_latent_frames}"
        )
    if task == "i2v" and seed_latent_frames != 1:
        raise ValueError("i2v presets must use exactly one seed latent frame")
    if task == "v2v" and seed_latent_frames % chunk_size:
        raise ValueError("v2v seed_latent_frames must be a whole number of chunks")
    data["seed_latent_frames"] = seed_latent_frames
    trajectory = control.get("trajectory")
    if trajectory:
        trajectory_path = (preset_path.parent / str(trajectory)).resolve()
        if not trajectory_path.is_file():
            raise FileNotFoundError(trajectory_path)
        control["_trajectory_path"] = trajectory_path
    return preset_path, data


def resolve_checkpoint(path: str | Path, filename: str) -> Path:
    """Accept either a concrete weight file or its checkpoint directory."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / filename
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _load_model_config(config_path: Path, model_folder: Path):
    default_path = REPO_ROOT / "configs" / "default_config.yaml"
    cfg = OmegaConf.merge(OmegaConf.load(default_path), OmegaConf.load(config_path))
    folder = str(model_folder)
    cfg.wan_model_folder = folder
    cfg.text_encoder_model_folder = folder
    cfg.vae_model_folder = folder
    if cfg.get("generator", {}).get("weight_list"):
        for weight in cfg.generator.weight_list:
            weight.path = folder
    return cfg


def load_dmd_ema_pipeline(
    *,
    config_path: str | Path,
    model_folder: str | Path,
    base_checkpoint: str | Path,
    ema_checkpoint: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    lora_rank: int = 128,
    lora_alpha: int = 128,
) -> CausalInferencePipeline:
    """Load Wan + the ReMind-5B generator + the DMD EMA adapter."""
    config_path = Path(config_path).expanduser().resolve()
    model_folder = Path(model_folder).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not model_folder.is_dir():
        raise FileNotFoundError(model_folder)
    base_path = resolve_checkpoint(base_checkpoint, "generator_model.safetensors")
    ema_path = resolve_checkpoint(ema_checkpoint, "student_lora_ema.safetensors")

    model_cfg = _load_model_config(config_path, model_folder)
    pipeline = CausalInferencePipeline(model_cfg, device=device)
    load_generator_checkpoint(pipeline.generator, str(base_path), "generator")

    # Match the project-page runner exactly: cast the ReMind-5B model
    # before constructing and merging the LoRA. Merging in fp32 and casting
    # afterward produces measurably different videos from the released EMA.
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device, dtype=dtype).requires_grad_(False).eval()
    pipeline.vae.to(device=device).requires_grad_(False).eval()
    pipeline.generator.to(device=device).requires_grad_(False).eval()

    from safetensors.torch import load_file

    lora_model, targets = configure_inference_lora(
        pipeline.generator.model,
        rank=int(lora_rank),
        alpha=int(lora_alpha),
        dropout=0.0,
    )
    load_adapter_state_dict(lora_model, load_file(str(ema_path)), STUDENT_ADAPTER)
    pipeline.generator.model = lora_model.merge_and_unload()
    logger.info(
        "Merged DMD EMA adapter %s (rank=%d, alpha=%d, targets=%d)",
        ema_path,
        lora_rank,
        lora_alpha,
        len(targets),
    )

    pipeline.generator.requires_grad_(False).eval()
    pipeline.scheduler.set_timesteps(1000, training=True)
    return pipeline


def _resize_input(path: Path, width: int, height: int) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def image_to_repeated_video(
    image: Image.Image,
    *,
    frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    frame = torch.from_numpy(array).permute(2, 0, 1)
    video = frame.unsqueeze(0).repeat(frames, 1, 1, 1)
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return video.to(device=device, dtype=dtype).mul(2.0).sub(1.0)


def video_prefix_to_padded_video(
    path: Path,
    *,
    seed_latent_frames: int,
    frames: int,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    """Load only the V2V prefix and pad its last frame for causal VAE encoding."""
    import imageio

    # Wan's temporal VAE maps 1 + 4 * (T_latent - 1) pixel frames to T_latent.
    seed_pixel_frames = 1 + 4 * (int(seed_latent_frames) - 1)
    decoded: list[np.ndarray] = []
    reader = imageio.get_reader(str(path))
    try:
        for frame in reader:
            if len(decoded) >= seed_pixel_frames:
                break
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            decoded.append(np.asarray(image, dtype=np.uint8).copy())
    finally:
        reader.close()
    if len(decoded) < seed_pixel_frames:
        raise ValueError(
            f"V2V input {path} has {len(decoded)} decoded frames; "
            f"{seed_pixel_frames} are required for {seed_latent_frames} seed latents"
        )
    decoded.extend([decoded[-1]] * (frames - len(decoded)))
    array = np.stack(decoded[:frames], axis=0).astype(np.float32) / 255.0
    video = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.to(device=device, dtype=dtype).mul(2.0).sub(1.0)
    return video, seed_pixel_frames


def _neutral_cameras(
    frames: int, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    extrinsics = (
        torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(1, frames, 1, 1)
    )
    intrinsics = (
        torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, frames, 1, 1)
    )
    focal = float(max(height, width))
    intrinsics[:, :, 0, 0] = focal
    intrinsics[:, :, 1, 1] = focal
    intrinsics[:, :, 0, 2] = float(width) * 0.5
    intrinsics[:, :, 1, 2] = float(height) * 0.5
    return extrinsics, intrinsics


def _parse_resolution(value) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        return None, None
    width, height = text.split("x", 1)
    try:
        return int(width), int(height)
    except ValueError:
        return None, None


def _camera_npz_to_tensors(
    camera_bytes: bytes,
    frame_indices: list[int],
    height: int,
    width: int,
    source_resolution=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the public camera-trajectory NPZ format without dataset code."""
    with np.load(io.BytesIO(camera_bytes)) as archive:
        keys = set(archive.files)
        extrinsics_key = next(
            (key for key in ("extrinsics", "c2w", "poses") if key in keys),
            None,
        )
        intrinsics_key = next(
            (key for key in ("intrinsics", "K", "Ks") if key in keys),
            None,
        )
        if extrinsics_key is None:
            raise KeyError(f"camera npz has no extrinsics key; keys={sorted(keys)}")
        extrinsics_full = archive[extrinsics_key].astype(np.float32)
        intrinsics_full = (
            archive[intrinsics_key].astype(np.float32)
            if intrinsics_key is not None
            else None
        )

    if extrinsics_full.ndim == 2:
        extrinsics_full = extrinsics_full[None]
    homogeneous = np.zeros((extrinsics_full.shape[0], 4, 4), dtype=np.float32)
    if extrinsics_full.shape[-2:] == (4, 4):
        homogeneous[:] = extrinsics_full
    else:
        homogeneous[:, :3, :4] = extrinsics_full[:, :3, :4]
        homogeneous[:, 3, 3] = 1.0
    indices = np.asarray(
        [min(max(int(index), 0), homogeneous.shape[0] - 1) for index in frame_indices],
        dtype=np.int64,
    )
    extrinsics = homogeneous[indices]

    if intrinsics_full is None:
        _, intrinsics = _neutral_cameras(len(frame_indices), height, width)
        return torch.from_numpy(extrinsics), intrinsics[0]
    if intrinsics_full.ndim == 2:
        intrinsics_full = intrinsics_full[None]
    intrinsics_indices = np.asarray(
        [
            min(max(int(index), 0), intrinsics_full.shape[0] - 1)
            for index in frame_indices
        ],
        dtype=np.int64,
    )
    intrinsics = intrinsics_full[intrinsics_indices].copy()
    source_width, source_height = _parse_resolution(source_resolution)
    if source_width and source_height:
        scale_x = float(width) / float(source_width)
        scale_y = float(height) / float(source_height)
        intrinsics[:, 0, 0] *= scale_x
        intrinsics[:, 1, 1] *= scale_y
        intrinsics[:, 0, 2] *= scale_x
        intrinsics[:, 1, 2] *= scale_y
    return torch.from_numpy(extrinsics), torch.from_numpy(intrinsics)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def build_pixel_cameras(
    control: dict[str, Any],
    *,
    frames: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a paired camera trajectory or create a static/analytic one."""
    trajectory_path = control.get("_trajectory_path")
    if trajectory_path:
        source_resolution = control.get("source_resolution")
        extrinsics, intrinsics = _camera_npz_to_tensors(
            Path(trajectory_path).read_bytes(),
            list(range(frames)),
            height,
            width,
            source_resolution=source_resolution,
        )
        return extrinsics.unsqueeze(0), intrinsics.unsqueeze(0)

    extrinsics, intrinsics = _neutral_cameras(frames, height, width)
    if str(control.get("kind", "clean")) != "camera":
        return extrinsics, intrinsics

    translation_path = control.get("translation_path")
    if translation_path is not None:
        translations = torch.as_tensor(translation_path, dtype=torch.float32)
        if translations.ndim != 2 or translations.shape[1] != 3:
            raise ValueError(
                "camera translation_path must have shape [frames, 3], got "
                f"{tuple(translations.shape)}"
            )
        if translations.shape[0] != frames:
            raise ValueError(
                "camera translation_path length must match num_frames: "
                f"{translations.shape[0]} != {frames}"
            )
        # ReMind/PRoPE consumes camera<-world view matrices.  The adapter
        # precompiles PlayWorld movement into the translation column while
        # leaving rotation fixed, exactly matching GC002's viewing-direction
        # constraint.
        extrinsics[0, :, :3, 3] = translations
        return extrinsics, intrinsics

    axis = str(control.get("axis", "yaw"))
    if axis not in {"yaw", "pitch"}:
        raise ValueError("camera axis must be 'yaw' or 'pitch'")
    target = math.radians(float(control.get("degrees", 20.0)))
    returns = bool(control.get("return", True))
    peak_fraction = float(control.get("peak_fraction", 0.50))
    return_fraction = float(control.get("return_fraction", 0.86))
    if not 0.0 < peak_fraction < return_fraction <= 1.0:
        raise ValueError("camera fractions must satisfy 0 < peak < return <= 1")

    for index in range(frames):
        phase = index / max(1, frames - 1)
        if returns:
            if phase <= peak_fraction:
                scale = _smoothstep(phase / peak_fraction)
            elif phase <= return_fraction:
                scale = 1.0 - _smoothstep(
                    (phase - peak_fraction) / (return_fraction - peak_fraction)
                )
            else:
                scale = 0.0
        else:
            scale = (
                _smoothstep(phase / return_fraction)
                if phase <= return_fraction
                else 1.0
            )
        angle = target * scale
        cosine, sine = math.cos(angle), math.sin(angle)
        if axis == "yaw":
            rotation = torch.tensor(
                [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
                dtype=torch.float32,
            )
        else:
            rotation = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
                dtype=torch.float32,
            )
        extrinsics[0, index, :3, :3] = rotation
    return extrinsics, intrinsics


def pixel_to_latent_cameras(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    latent_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = (
        torch.linspace(
            0, extrinsics.shape[1] - 1, latent_frames, device=extrinsics.device
        )
        .round()
        .long()
    )
    return extrinsics[:, indices], intrinsics[:, indices]


def _event_span(control: dict[str, Any], num_chunks: int) -> tuple[int, int]:
    start = int(control.get("start_chunk", 2))
    end = int(control.get("end_chunk", min(4, num_chunks - 2)))
    if start < 1 or end < start or end >= num_chunks - 1:
        raise ValueError(
            f"event chunks must leave a clean seed and recovery tail; got "
            f"[{start}, {end}] for {num_chunks} chunks"
        )
    return start, end


def _loop_caption(
    base: str, loop: dict[str, Any], total_frames: int, motion: str = ""
) -> str:
    base = (base or "A video.").rstrip(". ")
    peak = int(loop.get("peak_frame", 0))
    returned = int(loop.get("return_frame", total_frames - 1))
    motion_label = str(motion or "").strip().replace("_", " ")
    motion_clause = (
        f" The camera-motion label is {motion_label}; follow this camera "
        "path/direction exactly."
        if motion_label and motion_label.lower() not in {"unknown", "loop"}
        else ""
    )
    return (
        f"{base}. The camera trajectory is a loop: starting from frame 0, "
        f"the camera moves away to its furthest point at frame {peak}, then "
        f"returns to its starting framing by frame {returned} (out of "
        f"{total_frames} frames).{motion_clause} This is camera motion only: "
        "exposure and illumination stay stable and normal throughout; do not "
        "dim the scene, fade to black, toggle the lights, or insert an "
        "occluder. During the away interval, the original subject or main "
        "action may leave the frame because of the camera motion while the "
        "visible adjacent background remains normally lit and from the same "
        "environment. Do not duplicate, teleport, refresh, or reintroduce the "
        "original subject in the off-screen view. When the camera returns, "
        "the same scene is revealed naturally and the visible state has "
        "advanced continuously rather than resetting. No scene cuts; no new "
        "people or objects enter the scene; preserve object identity, "
        "background, stable lighting, scale, and time continuity."
    )


def _pixel_event_bounds(
    control: dict[str, Any],
    *,
    pixel_frames: int,
    latent_frames: int,
    chunk_size: int,
) -> tuple[int, int, int, int]:
    """Create a three-phase event schedule for custom presets."""
    num_chunks = math.ceil(latent_frames / chunk_size)
    start_chunk, end_chunk = _event_span(control, num_chunks)
    latent_start = min(latent_frames - 1, start_chunk * chunk_size)
    latent_end = min(latent_frames - 1, (end_chunk + 1) * chunk_size - 1)
    pixel_start = int(
        round(latent_start * (pixel_frames - 1) / max(1, latent_frames - 1))
    )
    pixel_end = int(round(latent_end * (pixel_frames - 1) / max(1, latent_frames - 1)))
    span = max(6, pixel_end - pixel_start)
    ramp = max(2, span // 4)
    hold_end = max(pixel_start + ramp + 2, pixel_end - ramp)
    return pixel_start, pixel_start + ramp, hold_end, pixel_end


def _default_event_params(
    kind: str,
    control: dict[str, Any],
    *,
    pixel_frames: int,
    latent_frames: int,
    chunk_size: int,
) -> dict[str, Any]:
    start, enter_end, hold_end, end = _pixel_event_bounds(
        control,
        pixel_frames=pixel_frames,
        latent_frames=latent_frames,
        chunk_size=chunk_size,
    )
    if kind == "light":
        return {
            "t_off_start": start,
            "t_off_end": enter_end,
            "t_hold_start": enter_end,
            "t_hold_end": hold_end,
            "t_on_start": hold_end,
            "t_on_end": end,
            "low_gain": float(control.get("low_gain", 0.04)),
            "high_gain": 1.0,
        }
    label = str(control.get("label", "brown cardboard panel"))
    slug = str(control.get("slug", label.lower().replace(" ", "_")))
    return {
        "rgba_path": f"procedural/{slug}/occluder.png",
        "enter_side": str(control.get("enter_side", "top")),
        "exit_side": str(control.get("exit_side", "bottom")),
        "t_enter_start": start,
        "t_enter_end": enter_end,
        "t_hold_start": enter_end,
        "t_hold_end": hold_end,
        "t_exit_start": hold_end,
        "t_exit_end": end,
        "peak_scale": float(control.get("peak_scale", 1.9)),
        "force_rect_mask": True,
    }


def build_prompt(
    prompt: str,
    control: dict[str, Any],
    *,
    pixel_frames: int,
    latent_frames: int,
    chunk_size: int,
) -> str | list[str]:
    """Build the global or chunk-local prompt used by ReMind inference."""
    kind = str(control.get("kind", "clean"))
    num_chunks = math.ceil(latent_frames / chunk_size)
    if kind == "clean":
        return chunk_text_prompts(
            str(prompt), "none", {}, pixel_frames, latent_frames, chunk_size
        )
    if kind == "camera":
        if (
            control.get("_trajectory_path")
            or control.get("translation_path") is not None
        ):
            camera_prompt = str(prompt)
        else:
            peak = int(
                round((pixel_frames - 1) * float(control.get("peak_fraction", 0.50)))
            )
            returned = int(
                round((pixel_frames - 1) * float(control.get("return_fraction", 0.86)))
            )
            degrees = float(control.get("degrees", 20.0))
            if str(control.get("axis", "yaw")) == "yaw":
                direction = "left" if degrees < 0 else "right"
            else:
                direction = "down" if degrees < 0 else "up"
            camera_prompt = _loop_caption(
                str(prompt),
                {"peak_frame": peak, "return_frame": returned},
                pixel_frames,
                motion=f"pan_{control.get('axis', 'yaw')}_{direction}",
            )
        return [camera_prompt] * num_chunks

    explicit_params = control.get("params")
    if explicit_params is not None:
        params = dict(explicit_params)
        meta = dict(control.get("meta") or params.get("meta") or {})
        return chunk_text_prompts(
            str(prompt),
            kind,
            params,
            pixel_frames,
            latent_frames,
            chunk_size,
            meta=meta,
            recovery_chunks=int(control.get("recovery_chunks", 2)),
        )

    params = _default_event_params(
        kind,
        control,
        pixel_frames=pixel_frames,
        latent_frames=latent_frames,
        chunk_size=chunk_size,
    )
    if kind == "light":
        return chunk_text_prompts(
            str(prompt),
            "light",
            params,
            pixel_frames,
            latent_frames,
            chunk_size,
            recovery_chunks=int(control.get("recovery_chunks", 2)),
        )

    label = str(control.get("label", "brown cardboard panel"))
    slug = str(control.get("slug", label.lower().replace(" ", "_")))
    return chunk_text_prompts(
        str(prompt),
        "occluder",
        params,
        pixel_frames,
        latent_frames,
        chunk_size,
        meta={"slug": slug, "label": label, "caption": ""},
        recovery_chunks=int(control.get("recovery_chunks", 2)),
    )


def camera_metrics(extrinsics: torch.Tensor) -> dict[str, float]:
    rotation = extrinsics[0, :, :3, :3].double()
    relative = rotation @ rotation[0].transpose(0, 1)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    angles = torch.rad2deg(torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)))
    translation = extrinsics[0, :, :3, 3].double()
    translation = torch.linalg.norm(translation - translation[0], dim=-1)
    return {
        "max_angle_deg": float(angles.max()),
        "final_angle_deg": float(angles[-1]),
        "max_translation": float(translation.max()),
        "final_translation": float(translation[-1]),
    }


def dry_run_summary(preset_path: Path, preset: dict[str, Any]) -> dict[str, Any]:
    frames = int(preset.get("num_frames", 81))
    height = int(preset.get("height", 480))
    width = int(preset.get("width", 832))
    chunk_size = int(preset.get("chunk_size", 3))
    latent_frames = (frames - 1) // 4 + 1
    latent_frames -= latent_frames % chunk_size
    prompts = build_prompt(
        str(preset["prompt"]),
        preset["control"],
        pixel_frames=frames,
        latent_frames=latent_frames,
        chunk_size=chunk_size,
    )
    extrinsics, _ = build_pixel_cameras(
        preset["control"], frames=frames, height=height, width=width
    )
    task = str(preset["task"])
    input_key = "input_image" if task == "i2v" else "input_video"
    seed_latent_frames = int(preset["seed_latent_frames"])
    return {
        "name": preset["name"],
        "preset": str(preset_path),
        "task": task,
        input_key: str(preset[f"_{input_key}_path"]),
        "control_kind": preset["control"]["kind"],
        "num_frames": frames,
        "latent_frames": latent_frames,
        "seed_latent_frames": seed_latent_frames,
        "seed_pixel_frames": 1 + 4 * (seed_latent_frames - 1),
        "chunk_size": chunk_size,
        "rollout_steps": CANONICAL_5B_DMD_STEPS,
        "prompt_mode": "chunk_local" if isinstance(prompts, list) else "global",
        "prompt_schedule": prompts if isinstance(prompts, list) else [prompts],
        "camera": camera_metrics(extrinsics),
    }


@torch.no_grad()
def generate_from_preset(
    *,
    preset_path: Path,
    preset: dict[str, Any],
    config_path: str | Path,
    model_folder: str | Path,
    base_checkpoint: str | Path,
    ema_checkpoint: str | Path,
    output_path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    lora_rank: int = 128,
    lora_alpha: int = 128,
    seed_override: int | None = None,
    preloaded_pipeline: CausalInferencePipeline | None = None,
) -> dict[str, Any]:
    frames = int(preset.get("num_frames", 81))
    height = int(preset.get("height", 480))
    width = int(preset.get("width", 832))
    fps = int(preset.get("fps", 16))
    chunk_size = int(preset.get("chunk_size", 3))
    seed = int(preset.get("seed", 20260711) if seed_override is None else seed_override)

    pipeline = preloaded_pipeline
    if pipeline is None:
        pipeline = load_dmd_ema_pipeline(
            config_path=config_path,
            model_folder=model_folder,
            base_checkpoint=base_checkpoint,
            ema_checkpoint=ema_checkpoint,
            device=device,
            dtype=dtype,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    task = str(preset["task"])
    seed_latent_frames = int(preset["seed_latent_frames"])
    if task == "i2v":
        input_key = "input_image"
        image = _resize_input(preset["_input_image_path"], width, height)
        source_video = image_to_repeated_video(
            image, frames=frames, device=device, dtype=dtype
        )
        seed_pixel_frames = 1
    else:
        input_key = "input_video"
        source_video, seed_pixel_frames = video_prefix_to_padded_video(
            preset["_input_video_path"],
            seed_latent_frames=seed_latent_frames,
            frames=frames,
            width=width,
            height=height,
            device=device,
            dtype=dtype,
        )
    source_latent = pipeline.vae.encode_to_latent(source_video).to(
        device=device, dtype=dtype
    )
    latent_frames = int(source_latent.shape[1])
    latent_frames -= latent_frames % chunk_size
    if latent_frames <= 1:
        raise RuntimeError(f"invalid latent length {latent_frames}")
    source_latent = source_latent[:, :latent_frames]
    pixel_extrinsics, pixel_intrinsics = build_pixel_cameras(
        preset["control"], frames=frames, height=height, width=width
    )
    viewmats, intrinsics = pixel_to_latent_cameras(
        pixel_extrinsics.to(device), pixel_intrinsics.to(device), latent_frames
    )
    prompt = build_prompt(
        str(preset["prompt"]),
        preset["control"],
        pixel_frames=frames,
        latent_frames=latent_frames,
        chunk_size=chunk_size,
    )

    num_chunks = latent_frames // chunk_size
    denoising_steps = [
        int(value) for value in pipeline.denoising_step_list.tolist() if int(value) > 0
    ]
    conditional = pipeline.text_encoder(text_prompts=[prompt])
    if isinstance(prompt, list):
        conditional["prompt_chunk_size"] = chunk_size

    adapter = InferencePipelineAdapter(
        pipeline,
        chunk_size=chunk_size,
        context_timestep=int(preset.get("context_timestep", 0)),
    )
    rollout = DMDInferenceRollout(
        adapter,
        pipeline.scheduler,
        denoising_steps,
        chunk_size=chunk_size,
        context_timestep=int(preset.get("context_timestep", 0)),
    )
    known_latents, known_mask = build_known_context(source_latent, seed_latent_frames)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    noise = torch.randn_like(source_latent)
    generated = rollout.rollout(
        noise,
        conditional,
        {"viewmats": viewmats, "Ks": intrinsics},
        known_latents=known_latents,
        known_mask=known_mask,
        exit_indices=[len(denoising_steps) - 1] * num_chunks,
    )
    adapter.clear_cache()
    decoded = pipeline.vae.decode_to_pixel(generated, use_cache=False)
    decoded = decoded.float().mul(0.5).add(0.5).clamp(0, 1)
    if decoded.shape[1] in (1, 3, 4):
        output_frames = decoded[0].permute(1, 0, 2, 3)
    elif decoded.shape[2] in (1, 3, 4):
        output_frames = decoded[0]
    else:
        raise RuntimeError(f"cannot identify decoded layout {tuple(decoded.shape)}")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(str(output_path), output_frames, fps=fps)
    metadata = {
        "name": preset["name"],
        "preset": str(preset_path),
        "task": task,
        input_key: str(preset[f"_{input_key}_path"]),
        "output": str(output_path),
        "control_kind": preset["control"]["kind"],
        "seed": seed,
        "num_frames": frames,
        "latent_frames": latent_frames,
        "seed_latent_frames": seed_latent_frames,
        "seed_pixel_frames": seed_pixel_frames,
        "rollout_steps": denoising_steps,
        "prompt_mode": "chunk_local" if isinstance(prompt, list) else "global",
        "prompt_schedule": prompt if isinstance(prompt, list) else [prompt],
        "camera": camera_metrics(pixel_extrinsics),
        "website_case": preset.get("website_case"),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata
