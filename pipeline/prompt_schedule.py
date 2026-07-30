# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Inference-time chunk-local prompts for reversible visibility events."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

import torch


def _unwrap(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap(value[0])
    if isinstance(value, dict):
        return {key: _unwrap(item) for key, item in value.items()}
    return value


def _base_caption(caption: str) -> str:
    return str(caption or "A video.").strip().rstrip(". ") + "."


def _short_description(meta: Optional[Dict[str, Any]], params: Dict[str, Any]) -> str:
    meta = _unwrap(meta or {})
    label = str(meta.get("label") or "").strip()
    rich = str(meta.get("caption") or "").strip().rstrip(". ")
    if not label:
        slug = str(meta.get("slug") or "").strip()
        if not slug:
            rgba_path = str(params.get("rgba_path") or "")
            slug = os.path.basename(os.path.dirname(rgba_path))
        label = (slug or "opaque occluder").replace("_", " ")
    if rich and rich.lower() != label.lower():
        words = rich.split()
        rich = " ".join(words[:40])
        return f"{label}, {rich}"
    return label


def _pixel_to_chunk(
    pixel_index: int,
    num_pixel_frames: int,
    num_latent_frames: int,
    chunk_size: int,
) -> int:
    if num_latent_frames <= 1:
        return 0
    latent_index = round(
        max(0, int(pixel_index))
        * float(num_latent_frames - 1)
        / float(max(1, num_pixel_frames - 1))
    )
    latent_index = max(0, min(num_latent_frames - 1, latent_index))
    return int(latent_index) // chunk_size


def _event_bounds(kind: str, params: Dict[str, Any]) -> Optional[Sequence[int]]:
    if kind == "light":
        keys = ("t_off_start", "t_off_end", "t_hold_end", "t_on_end")
    elif kind == "occluder":
        keys = ("t_enter_start", "t_enter_end", "t_hold_end", "t_exit_end")
    else:
        return None
    if not all(key in params for key in keys):
        return None
    return tuple(int(params[key]) for key in keys)


def _event_chunk_plan(
    base: str,
    kind: str,
    params: Dict[str, Any],
    meta: Optional[Dict[str, Any]],
    num_pixel_frames: int,
    num_latent_frames: int,
    chunk_size: int,
    num_chunks: int,
) -> Optional[Dict[str, Any]]:
    """Per-event chunk prompt plan: phase prompts for the event's active
    chunks plus the recovery clause to write after it ends.  Returns None
    when the (kind, params) pair does not describe a reversible event."""
    bounds = _event_bounds(kind, params)
    if bounds is None:
        return None

    start, enter_end, hold_end, end = bounds
    start_chunk = _pixel_to_chunk(
        start, num_pixel_frames, num_latent_frames, chunk_size
    )
    end_chunk = _pixel_to_chunk(end, num_pixel_frames, num_latent_frames, chunk_size)
    event_chunks = list(range(start_chunk, end_chunk + 1))

    if kind == "light":
        clauses = {
            "enter": (
                "During this segment, the lights are turning off and the same "
                "scene becomes progressively darker while its action continues."
            ),
            "hold": (
                "During this entire segment, the lights remain fully off. The "
                "same scene is dark and invisible while its underlying action "
                "continues out of sight."
            ),
            "exit": (
                "During this segment, the lights turn back on and progressively "
                "reveal the same continuously evolving scene."
            ),
            "recover": (
                "The lights are fully on again in this segment. The original "
                "scene is visible and its action continues from before the "
                "darkness without resetting."
            ),
            "brief": (
                "During this segment, the lights briefly turn off and back on "
                "within the segment: the scene darkens, stays dark for a "
                "moment, then returns to full brightness while its action "
                "continues without resetting."
            ),
        }
    else:
        description = _short_description(meta or params.get("meta"), params)
        enter_side = str(params.get("enter_side") or "side")
        exit_side = str(params.get("exit_side") or "side")
        clauses = {
            "enter": (
                f"During this segment, the same {description} enters from the "
                f"{enter_side} and progressively covers the camera view without "
                "interacting with the scene."
            ),
            "hold": (
                f"Throughout this segment, the same {description} remains in "
                "front of the camera and completely blocks the scene. The "
                "underlying action continues out of sight."
            ),
            "exit": (
                f"During this segment, the same {description} moves out toward "
                f"the {exit_side} and progressively reveals the original, "
                "continuously evolving scene."
            ),
            "recover": (
                f"The {description} is completely gone in this segment. The "
                "original scene is fully visible and its action continues from "
                "before the occlusion without resetting."
            ),
            "brief": (
                f"During this segment, the same {description} briefly enters, "
                "covers the camera view, and leaves again within the segment, "
                "revealing the original continuously evolving scene by the "
                "segment's end."
            ),
        }

    last_event_chunk = end_chunk
    event_finishes_in_clip = end < num_pixel_frames
    enter_finishes_in_clip = enter_end < num_pixel_frames
    hold_finishes_in_clip = hold_end < num_pixel_frames

    def _chunk_center_pixel(chunk_idx: int) -> float:
        latent_center = chunk_idx * chunk_size + (chunk_size - 1) / 2.0
        latent_center = min(float(num_latent_frames - 1), latent_center)
        if num_latent_frames <= 1:
            return 0.0
        return (
            latent_center * float(num_pixel_frames - 1) / float(num_latent_frames - 1)
        )

    single_chunk_event = start_chunk == end_chunk and event_finishes_in_clip
    chunk_prompts: Dict[int, str] = {}
    for chunk_idx in event_chunks:
        if chunk_idx >= num_chunks:
            continue
        if single_chunk_event:
            phase = "brief"
        elif chunk_idx == start_chunk:
            phase = "enter"
        elif chunk_idx == end_chunk and event_finishes_in_clip:
            phase = "exit"
        elif not enter_finishes_in_clip:
            phase = "enter"
        elif not hold_finishes_in_clip:
            phase = "enter" if _chunk_center_pixel(chunk_idx) < enter_end else "hold"
        else:
            center = _chunk_center_pixel(chunk_idx)
            if center < enter_end:
                phase = "enter"
            elif center < hold_end:
                phase = "hold"
            else:
                phase = "exit"
        chunk_prompts[chunk_idx] = f"{base} {clauses[phase]}"

    return {
        "chunk_prompts": chunk_prompts,
        "recover_prompt": f"{base} {clauses['recover']}",
        "last_event_chunk": last_event_chunk,
        "finishes_in_clip": event_finishes_in_clip,
        "start_chunk": start_chunk,
    }


def chunk_text_prompts_multi(
    source_caption: str,
    events: Sequence[Dict[str, Any]],
    num_pixel_frames: int,
    num_latent_frames: int,
    chunk_size: int,
    *,
    recovery_chunks: int = 2,
) -> List[str]:
    """Build one prompt per latent chunk for zero or more reversible events.

    `events` is an ordered list of {"kind", "params", "meta"} dictionaries.
    Each event writes its phase clauses
    into its active chunks; recovery clauses fill the clean chunks after an
    event but never overwrite a later event's chunks — so a double-occluder
    sample reads e.g. base / enter / exit / recover / enter / exit / recover.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    base = _base_caption(source_caption)
    num_chunks = math.ceil(num_latent_frames / chunk_size)
    prompts = [base for _ in range(num_chunks)]

    plans: List[Dict[str, Any]] = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        kind = str(_unwrap(event.get("kind")) or "none").strip().lower()
        params = _unwrap(event.get("params") or {})
        if not isinstance(params, dict):
            params = {}
        meta = _unwrap(event.get("meta") or params.get("meta"))
        if not isinstance(meta, dict):
            meta = None
        plan = _event_chunk_plan(
            base,
            kind,
            params,
            meta,
            num_pixel_frames,
            num_latent_frames,
            chunk_size,
            num_chunks,
        )
        if plan is not None:
            plans.append(plan)
    plans.sort(key=lambda item: item["start_chunk"])

    claimed = set()
    for plan in plans:
        for chunk_idx, text in plan["chunk_prompts"].items():
            prompts[chunk_idx] = text
            claimed.add(chunk_idx)
    for plan in plans:
        if not plan["finishes_in_clip"]:
            continue
        for offset in range(1, max(0, int(recovery_chunks)) + 1):
            chunk_idx = plan["last_event_chunk"] + offset
            if chunk_idx >= num_chunks or chunk_idx in claimed:
                break
            prompts[chunk_idx] = plan["recover_prompt"]
    return prompts


def chunk_text_prompts(
    source_caption: str,
    kind: str,
    params: Optional[Dict[str, Any]],
    num_pixel_frames: int,
    num_latent_frames: int,
    chunk_size: int,
    *,
    meta: Optional[Dict[str, Any]] = None,
    recovery_chunks: int = 2,
) -> List[str]:
    """Build one prompt per latent chunk (single-event wrapper).

    The scene description is unchanged outside the event. Event prompts repeat
    the same appearance description in every active chunk because each T5
    encoding is independent. The first chunks after removal explicitly ask for
    recovery before returning to the content-only prompt.
    """
    return chunk_text_prompts_multi(
        source_caption,
        [{"kind": kind, "params": params, "meta": meta}],
        num_pixel_frames,
        num_latent_frames,
        chunk_size,
        recovery_chunks=recovery_chunks,
    )
