#!/usr/bin/env python3
# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Minimal open-loop PlayWorld adapter for ReMind."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PLAYWORLD_ROOT = HERE.parents[1]
DEFAULT_OUTPUT_ROOT = PLAYWORLD_ROOT / "outputs" / "remind"
DATASET_SPLITS = {"gc", "if", "insight", "outsight"}
FPS = 16
WINDOW_FRAMES = 81
WINDOW_STRIDE = WINDOW_FRAMES - 1
DEFAULT_DURATION_SECONDS = 5.0
DEFAULT_TRANSLATION_PER_ACTION = 0.005
ACTION_PATTERN = re.compile(r"([A-Za-z]+)\s*\*\s*(\d+)")
# ReMind's PRoPE expects camera<-world view matrices.  These deltas are
# therefore the inverse of camera-center motion in an OpenCV-style frame.
VIEW_TRANSLATION_DELTAS = {
    "W": (0.0, 0.0, -1.0),
    "S": (0.0, 0.0, 1.0),
    "A": (1.0, 0.0, 0.0),
    "D": (-1.0, 0.0, 0.0),
}


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _default_image_root(mapping_json: Path) -> Path:
    parent = mapping_json.resolve().parent
    return parent.parent if parent.name.lower() in DATASET_SPLITS else parent


def load_tasks(
    mapping_json: str | Path,
    *,
    images_dir: str | Path | None = None,
    task_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Load PlayWorld records and resolve their image paths."""

    mapping_path = Path(mapping_json).expanduser().resolve()
    records = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"PlayWorld mapping must be a JSON array: {mapping_path}")

    image_root = (
        Path(images_dir).expanduser().resolve()
        if images_dir is not None
        else _default_image_root(mapping_path)
    )
    tasks: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError(f"PlayWorld task must be a JSON object: {raw_record!r}")
        record = dict(raw_record)
        task_id = str(record.get("task_id") or "").strip()
        image_value = str(record.get("image_path") or "").strip()
        if not task_id or not image_value:
            raise ValueError("Each PlayWorld task requires task_id and image_path")
        if task_id in by_id:
            raise ValueError(f"Duplicate PlayWorld task_id: {task_id}")

        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = image_root / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"PlayWorld input image not found: {image_path}")

        record["task_id"] = task_id
        record["_image_path"] = image_path
        tasks.append(record)
        by_id[task_id] = record

    requested = [str(task_id) for task_id in task_ids]
    if not requested:
        return tasks
    missing = [task_id for task_id in requested if task_id not in by_id]
    if missing:
        raise KeyError(f"PlayWorld task IDs not found: {', '.join(missing)}")
    return [by_id[task_id] for task_id in requested]


def world_prompt_for_task(task: dict[str, Any]) -> str:
    """Keep the benchmark objective separate from ReMind content conditioning."""

    image_caption = str(task.get("image_caption") or "").strip()
    objective = str(task.get("prompt") or "").strip()
    prompt = image_caption or objective
    if not prompt:
        raise ValueError(f"Task {task.get('task_id')} has no image_caption or prompt")
    return prompt


def rollout_layout(duration_seconds: float, fps: int = FPS) -> tuple[int, int]:
    """Snap a requested duration to overlapping 81-frame ReMind windows."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    num_segments = max(1, round(float(duration_seconds) * fps / WINDOW_STRIDE))
    return num_segments, 1 + num_segments * WINDOW_STRIDE


def _movement_segments(task: dict[str, Any]) -> list[tuple[str, int]]:
    action = str(task.get("action") or "")
    segments = [
        (name.upper(), int(count))
        for name, count in ACTION_PATTERN.findall(action)
    ]
    if not segments or any(name not in VIEW_TRANSLATION_DELTAS for name, _ in segments):
        return []
    return segments


def camera_translation_path(
    task: dict[str, Any],
    *,
    frames: int,
    translation_per_action: float = DEFAULT_TRANSLATION_PER_ACTION,
) -> list[list[float]] | None:
    """Compile W/S/A/D task actions into a fixed-orientation camera path."""

    segments = _movement_segments(task)
    if not segments:
        return None
    total_actions = sum(count for _, count in segments)
    if total_actions <= 0:
        return None

    starts: list[tuple[float, float, float]] = []
    position = [0.0, 0.0, 0.0]
    for action, count in segments:
        starts.append(tuple(position))
        delta = VIEW_TRANSLATION_DELTAS[action]
        for axis in range(3):
            position[axis] += delta[axis] * count * translation_per_action

    path: list[list[float]] = []
    for frame_index in range(frames):
        progress = frame_index * total_actions / max(1, frames - 1)
        consumed = 0
        value = [0.0, 0.0, 0.0]
        for segment_index, (action, count) in enumerate(segments):
            if progress <= consumed + count or segment_index == len(segments) - 1:
                local = max(0.0, min(float(count), progress - consumed))
                delta = VIEW_TRANSLATION_DELTAS[action]
                value = [
                    starts[segment_index][axis]
                    + delta[axis] * local * translation_per_action
                    for axis in range(3)
                ]
                break
            consumed += count
        path.append(value)
    path[-1] = position
    return path


def build_preset(
    task: dict[str, Any],
    seed: int | None = None,
    *,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    apply_actions: bool = False,
    translation_per_action: float = DEFAULT_TRANSLATION_PER_ACTION,
) -> dict[str, Any]:
    """Translate one PlayWorld image task into a windowed ReMind I2V preset."""

    image_path = Path(task["_image_path"]).resolve()
    num_segments, num_frames = rollout_layout(duration_seconds)
    translation_path = (
        camera_translation_path(
            task,
            frames=num_frames,
            translation_per_action=translation_per_action,
        )
        if apply_actions
        else None
    )
    if apply_actions and translation_path is None:
        raise ValueError(
            f"Task {task['task_id']} has no supported W/S/A/D action path"
        )
    control: dict[str, Any] = {"kind": "clean"}
    if translation_path is not None:
        control = {
            "kind": "camera",
            "translation_path": translation_path,
        }
    preset: dict[str, Any] = {
        "name": str(task["task_id"]),
        "task": "i2v",
        "input_image": str(image_path),
        "_input_image_path": image_path,
        "prompt": world_prompt_for_task(task),
        "num_frames": num_frames,
        "height": 480,
        "width": 832,
        "fps": FPS,
        "chunk_size": 3,
        "seed_latent_frames": 1,
        "control": control,
        "_num_segments": num_segments,
    }
    if seed is not None:
        preset["seed"] = int(seed)
    return preset


def _result_base(task: dict[str, Any], preset: dict[str, Any]) -> dict[str, Any]:
    actions_applied = preset["control"].get("translation_path") is not None
    return {
        "task_id": str(task["task_id"]),
        "model": "ReMind",
        "open_loop": True,
        "objective_prompt": str(task.get("prompt") or ""),
        "world_prompt": str(preset["prompt"]),
        "image_path": str(preset["_input_image_path"]),
        "action_sequence": task.get("action_sequence"),
        "action_sequence_steps": task.get("action_sequence_steps") or [],
        "actions_applied": actions_applied,
        "action_control": (
            "precompiled_fixed-orientation_camera_path" if actions_applied else None
        ),
        "control_kind": str(preset["control"]["kind"]),
        "num_segments": int(preset["_num_segments"]),
        "num_frames": int(preset["num_frames"]),
        "fps": int(preset["fps"]),
        "duration_seconds": round(
            int(preset["num_frames"]) / int(preset["fps"]), 4
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument(
        "--config",
        type=Path,
        default=_env_path("REMIND_CONFIG") or HERE / "configs" / "model_5b.yaml",
    )
    parser.add_argument(
        "--model-folder", type=Path, default=_env_path("REMIND_MODEL_FOLDER")
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=_env_path("REMIND_BASE_CHECKPOINT"),
    )
    parser.add_argument(
        "--ema-checkpoint",
        type=Path,
        default=_env_path("REMIND_EMA_CHECKPOINT"),
    )
    parser.add_argument("--device", default=os.environ.get("REMIND_DEVICE", "cuda:0"))
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default=os.environ.get("REMIND_DTYPE", "bf16"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=float(
            os.environ.get("REMIND_DURATION_SECONDS", DEFAULT_DURATION_SECONDS)
        ),
    )
    parser.add_argument(
        "--apply-actions",
        action="store_true",
        help="Compile PlayWorld W/S/A/D actions into a camera trajectory.",
    )
    parser.add_argument(
        "--translation-per-action",
        type=float,
        default=float(
            os.environ.get(
                "REMIND_TRANSLATION_PER_ACTION", DEFAULT_TRANSLATION_PER_ACTION
            )
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_inference_paths(args: argparse.Namespace) -> None:
    required = {
        "--model-folder or REMIND_MODEL_FOLDER": args.model_folder,
        "--base-checkpoint or REMIND_BASE_CHECKPOINT": args.base_checkpoint,
        "--ema-checkpoint or REMIND_EMA_CHECKPOINT": args.ema_checkpoint,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Missing ReMind inference paths: {', '.join(missing)}")


def _segment_preset(
    preset: dict[str, Any],
    *,
    segment_index: int,
    input_image: Path,
    seed: int,
) -> dict[str, Any]:
    start = segment_index * WINDOW_STRIDE
    stop = start + WINDOW_FRAMES
    control = dict(preset["control"])
    translation_path = control.get("translation_path")
    if translation_path is not None:
        control["translation_path"] = translation_path[start:stop]
    segment = dict(preset)
    segment.update(
        name=f"{preset['name']}_segment_{segment_index + 1:02d}",
        input_image=str(input_image),
        _input_image_path=input_image,
        num_frames=WINDOW_FRAMES,
        seed=seed,
        control=control,
    )
    return segment


def _save_last_frame(video_path: Path, image_path: Path) -> None:
    import imageio.v2 as imageio
    from PIL import Image

    last_frame = None
    reader = imageio.get_reader(str(video_path))
    try:
        for frame in reader:
            last_frame = frame
    finally:
        reader.close()
    if last_frame is None:
        raise RuntimeError(f"No frames decoded from {video_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(last_frame).save(image_path)


def _stitch_segments(segment_paths: list[Path], output_path: Path, fps: int) -> int:
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-crf", "18"],
    )
    frame_count = 0
    try:
        for segment_index, segment_path in enumerate(segment_paths):
            reader = imageio.get_reader(str(segment_path))
            try:
                for frame_index, frame in enumerate(reader):
                    if segment_index > 0 and frame_index == 0:
                        continue
                    writer.append_data(frame)
                    frame_count += 1
            finally:
                reader.close()
    finally:
        writer.close()
    return frame_count


def run_task(
    task: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    from pipeline.remind_inference import (
        dry_run_summary,
        generate_from_preset,
        load_dmd_ema_pipeline,
    )

    task_id = str(task["task_id"])
    task_dir = args.output_dir.expanduser().resolve() / task_id
    video_path = task_dir / f"{task_id}.mp4"
    result_path = task_dir / "result.json"
    preset = build_preset(
        task,
        seed=args.seed,
        duration_seconds=args.duration_seconds,
        apply_actions=args.apply_actions,
        translation_per_action=args.translation_per_action,
    )
    result = _result_base(task, preset)

    if video_path.is_file() and not args.overwrite and not args.dry_run:
        result.update(status="skipped", success=True, video_path=str(video_path))
        _write_json(result_path, result)
        return result

    try:
        if args.dry_run:
            metadata = dry_run_summary(args.mapping_json.resolve(), preset)
            result.update(
                status="dry_run",
                success=True,
                video_path=None,
                remind=metadata,
            )
        else:
            import torch

            device = torch.device(args.device)
            if device.type == "cuda":
                torch.cuda.set_device(device)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
            pipeline = load_dmd_ema_pipeline(
                config_path=args.config.expanduser().resolve(),
                model_folder=args.model_folder.expanduser().resolve(),
                base_checkpoint=args.base_checkpoint.expanduser().resolve(),
                ema_checkpoint=args.ema_checkpoint.expanduser().resolve(),
                device=device,
                dtype=dtype,
            )
            segment_dir = task_dir / "segments"
            segment_dir.mkdir(parents=True, exist_ok=True)
            segment_paths: list[Path] = []
            segment_records: list[dict[str, Any]] = []
            current_image = Path(preset["_input_image_path"])
            base_seed = int(args.seed if args.seed is not None else 20260711)
            for segment_index in range(int(preset["_num_segments"])):
                segment_seed = base_seed + segment_index
                segment = _segment_preset(
                    preset,
                    segment_index=segment_index,
                    input_image=current_image,
                    seed=segment_seed,
                )
                segment_path = segment_dir / f"segment_{segment_index + 1:02d}.mp4"
                print(
                    f"  [{task_id}] ReMind window "
                    f"{segment_index + 1}/{preset['_num_segments']}",
                    flush=True,
                )
                segment_metadata = generate_from_preset(
                    preset_path=args.mapping_json.resolve(),
                    preset=segment,
                    config_path=args.config.expanduser().resolve(),
                    model_folder=args.model_folder.expanduser().resolve(),
                    base_checkpoint=args.base_checkpoint.expanduser().resolve(),
                    ema_checkpoint=args.ema_checkpoint.expanduser().resolve(),
                    output_path=segment_path,
                    device=device,
                    dtype=dtype,
                    seed_override=segment_seed,
                    preloaded_pipeline=pipeline,
                )
                segment_paths.append(segment_path)
                segment_records.append(
                    {
                        "segment": segment_index + 1,
                        "seed": segment_seed,
                        "video_path": str(segment_path.resolve()),
                        "camera": segment_metadata["camera"],
                    }
                )
                if segment_index + 1 < int(preset["_num_segments"]):
                    current_image = (
                        segment_dir / f"segment_{segment_index + 1:02d}_last.png"
                    )
                    _save_last_frame(segment_path, current_image)

            stitched_frames = _stitch_segments(
                segment_paths, video_path, int(preset["fps"])
            )
            summary = dry_run_summary(args.mapping_json.resolve(), preset)
            metadata = {
                "name": preset["name"],
                "task": preset["task"],
                "output": str(video_path.resolve()),
                "control_kind": preset["control"]["kind"],
                "seed": base_seed,
                "num_frames": stitched_frames,
                "fps": int(preset["fps"]),
                "duration_seconds": round(
                    stitched_frames / int(preset["fps"]), 4
                ),
                "num_segments": len(segment_paths),
                "segment_frames": WINDOW_FRAMES,
                "segment_stride": WINDOW_STRIDE,
                "camera": summary["camera"],
                "segments": segment_records,
            }
            result.update(
                status="completed",
                success=True,
                video_path=str(video_path.resolve()),
                remind=metadata,
            )
    except Exception as error:
        result.update(
            status="failed",
            success=False,
            video_path=None,
            error=f"{type(error).__name__}: {error}",
        )
    _write_json(result_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.mapping_json = args.mapping_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.dry_run:
        _validate_inference_paths(args)

    tasks = load_tasks(
        args.mapping_json,
        images_dir=args.images_dir,
        task_ids=args.tasks or (),
    )
    results = []
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] ReMind open-loop {task['task_id']}", flush=True)
        result = run_task(task, args)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    _write_json(args.output_dir / "results.json", results)
    return 0 if all(result.get("success") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
