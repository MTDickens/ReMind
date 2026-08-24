#!/usr/bin/env python3
# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Minimal open-loop PlayWorld adapter for ReMind."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PLAYWORLD_ROOT = HERE.parents[1]
DEFAULT_OUTPUT_ROOT = PLAYWORLD_ROOT / "outputs" / "remind"
DATASET_SPLITS = {"gc", "if", "insight", "outsight"}


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


def build_preset(task: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    """Translate one PlayWorld image task into a clean ReMind I2V preset."""

    image_path = Path(task["_image_path"]).resolve()
    preset: dict[str, Any] = {
        "name": str(task["task_id"]),
        "task": "i2v",
        "input_image": str(image_path),
        "_input_image_path": image_path,
        "prompt": world_prompt_for_task(task),
        "num_frames": 81,
        "height": 480,
        "width": 832,
        "fps": 16,
        "chunk_size": 3,
        "seed_latent_frames": 1,
        "control": {"kind": "clean"},
    }
    if seed is not None:
        preset["seed"] = int(seed)
    return preset


def _result_base(task: dict[str, Any], preset: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(task["task_id"]),
        "model": "ReMind",
        "open_loop": True,
        "objective_prompt": str(task.get("prompt") or ""),
        "world_prompt": str(preset["prompt"]),
        "image_path": str(preset["_input_image_path"]),
        "action_sequence": task.get("action_sequence"),
        "action_sequence_steps": task.get("action_sequence_steps") or [],
        "actions_applied": False,
        "control_kind": "clean",
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


def run_task(
    task: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    from pipeline.remind_inference import dry_run_summary, generate_from_preset

    task_id = str(task["task_id"])
    task_dir = args.output_dir.expanduser().resolve() / task_id
    video_path = task_dir / f"{task_id}.mp4"
    result_path = task_dir / "result.json"
    preset = build_preset(task, seed=args.seed)
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
            metadata = generate_from_preset(
                preset_path=args.mapping_json.resolve(),
                preset=preset,
                config_path=args.config.expanduser().resolve(),
                model_folder=args.model_folder.expanduser().resolve(),
                base_checkpoint=args.base_checkpoint.expanduser().resolve(),
                ema_checkpoint=args.ema_checkpoint.expanduser().resolve(),
                output_path=video_path,
                device=device,
                dtype=dtype,
                seed_override=args.seed,
            )
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
