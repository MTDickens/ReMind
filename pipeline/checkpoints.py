# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Checkpoint and LoRA loading used by ReMind inference."""

from __future__ import annotations

import gc
import logging
import os
from typing import Dict

import torch
from safetensors.torch import load_file


logger = logging.getLogger(__name__)
STUDENT_ADAPTER = "student"


def _checkpoint_state(path: str, role: str = "generator") -> Dict[str, torch.Tensor]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    if path.endswith(".safetensors"):
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in (role, "model", "generator", "critic"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no state dict: {path}")
    return state


def _clean_key(key: str) -> str:
    prefixes = (
        "module.",
        "_fsdp_wrapped_module.",
        "_checkpoint_wrapped_module.",
        "_orig_mod.",
        "model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def load_generator_checkpoint(generator, path: str, role: str = "generator") -> None:
    """Load a released generator checkpoint with strict shape/coverage checks."""
    state = {
        _clean_key(key): value for key, value in _checkpoint_state(path, role).items()
    }
    live = generator.model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in live and tuple(value.shape) == tuple(live[key].shape)
    }
    matched_numel = sum(int(live[key].numel()) for key in compatible)
    total_numel = sum(int(value.numel()) for value in live.values())
    coverage = matched_numel / max(1, total_numel)
    if coverage < 0.90:
        raise RuntimeError(
            f"{role} checkpoint coverage is only {coverage:.1%}: {path}; "
            "refusing to load an incompatible base model"
        )
    missing, unexpected = generator.model.load_state_dict(compatible, strict=False)
    logger.info(
        "Loaded %s from %s: coverage=%.2f%% missing=%d unexpected=%d",
        role,
        path,
        coverage * 100.0,
        len(missing),
        len(unexpected),
    )
    del state, compatible, live
    gc.collect()


def causal_block_linear_targets(transformer) -> list[str]:
    """Return the causal Wan linear layers targeted by the released adapter."""
    targets = []
    for name, module in transformer.named_modules():
        if module.__class__.__name__ != "CausalWanAttentionBlock":
            continue
        for child_name, child in module.named_modules(prefix=name):
            if not isinstance(child, torch.nn.Linear):
                continue
            if child_name.endswith("camera_phase_mlp.proj"):
                continue
            targets.append(child_name)
    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError("could not find Linear modules in CausalWanAttentionBlock")
    return targets


def configure_inference_lora(
    transformer,
    *,
    rank: int = 128,
    alpha: int = 128,
    dropout: float = 0.0,
    init_seed: int = 20260714,
):
    """Attach the single student adapter expected by the released EMA weights."""
    import peft

    targets = causal_block_linear_targets(transformer)
    config = peft.LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        init_lora_weights=True,
        target_modules=targets,
    )
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(init_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(init_seed))
        model = peft.get_peft_model(transformer, config, adapter_name=STUDENT_ADAPTER)
    try:
        model.set_adapter(STUDENT_ADAPTER, inference_mode=False)
    except TypeError:
        model.set_adapter(STUDENT_ADAPTER)
    return model, targets


@torch.no_grad()
def load_adapter_state_dict(
    model, state: Dict[str, torch.Tensor], adapter_name: str = STUDENT_ADAPTER
):
    """Load one LoRA adapter with exact key and shape coverage."""
    marker = f".{adapter_name}."
    targets: Dict[str, tuple[str, torch.nn.Parameter]] = {}
    for target_name, parameter in model.named_parameters():
        if marker not in target_name or "lora_" not in target_name:
            continue
        head, tail = target_name.rsplit(marker, 1)
        saved_name = f"{head}.{tail}"
        if saved_name in targets:
            raise RuntimeError(
                f"ambiguous LoRA key {saved_name!r} for adapter {adapter_name!r}"
            )
        targets[saved_name] = (target_name, parameter)

    state_keys = set(state)
    target_keys = set(targets)
    missing = sorted(target_keys - state_keys)
    unexpected = sorted(state_keys - target_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"LoRA adapter {adapter_name!r} key mismatch: "
            f"missing={len(missing)} {missing[:5]} unexpected={len(unexpected)} "
            f"{unexpected[:5]}"
        )

    for saved_name, value in state.items():
        target_name, target = targets[saved_name]
        if tuple(value.shape) != tuple(target.shape):
            raise RuntimeError(
                f"LoRA adapter {adapter_name!r} shape mismatch for "
                f"{saved_name}: checkpoint={tuple(value.shape)} "
                f"model={tuple(target.shape)} ({target_name})"
            )
        target.copy_(value.to(device=target.device, dtype=target.dtype))
    return {"loaded_keys": len(state), "missing_keys": [], "unexpected_keys": []}
