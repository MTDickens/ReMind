#!/usr/bin/env python3
# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Generate a ReMind video from one public demo preset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from pipeline.remind_inference import (
    dry_run_summary,
    generate_from_preset,
    load_preset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model_5b.yaml"))
    parser.add_argument("--model-folder", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--ema-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the preset and print its resolved control schedule without loading weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    preset_path, preset = load_preset(args.preset)
    if args.dry_run:
        print(
            json.dumps(
                dry_run_summary(preset_path, preset), indent=2, ensure_ascii=False
            )
        )
        return

    required = {
        "--model-folder": args.model_folder,
        "--base-checkpoint": args.base_checkpoint,
        "--ema-checkpoint": args.ema_checkpoint,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required inference arguments: {', '.join(missing)}")
    output = args.output or Path("outputs") / f"{preset['name']}.mp4"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    metadata = generate_from_preset(
        preset_path=preset_path,
        preset=preset,
        config_path=args.config,
        model_folder=args.model_folder,
        base_checkpoint=args.base_checkpoint,
        ema_checkpoint=args.ema_checkpoint,
        output_path=output,
        device=device,
        dtype=dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed_override=args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
