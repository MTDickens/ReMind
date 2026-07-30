# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

from pathlib import Path

import pytest
import torch

from pipeline.remind_inference import (
    build_pixel_cameras,
    build_prompt,
    camera_metrics,
    dry_run_summary,
    load_preset,
    resolve_checkpoint,
    video_prefix_to_padded_video,
)


ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "examples" / "presets"


@pytest.mark.parametrize(
    "name,task,kind",
    [
        ("01_latte_occluder_recovery.yaml", "i2v", "occluder"),
        ("02_pancake_occluder_recovery.yaml", "i2v", "occluder"),
        ("03_whisking_camera_pan.yaml", "i2v", "camera"),
        ("04_cereal_camera_pan.yaml", "i2v", "camera"),
        ("05_dog_clean_i2v.yaml", "i2v", "clean"),
        ("06_dry_ice_clean_i2v.yaml", "i2v", "clean"),
        ("07_flour_clean_i2v.yaml", "i2v", "clean"),
        ("08_cake_decorating_occluder_v2v.yaml", "v2v", "occluder"),
        ("09_ice_melting_drink_occluder_v2v.yaml", "v2v", "occluder"),
        ("10_pour_wine_beer_camera_pan_v2v.yaml", "v2v", "camera"),
        ("11_slime_stretch_camera_pan_v2v.yaml", "v2v", "camera"),
    ],
)
def test_public_presets_resolve(name, task, kind):
    path, preset = load_preset(PRESETS / name)
    summary = dry_run_summary(path, preset)
    assert summary["task"] == task
    assert summary["control_kind"] == kind
    assert summary["latent_frames"] == 21
    assert summary["rollout_steps"] == [1000, 937, 833, 625]
    input_key = "input_image" if task == "i2v" else "input_video"
    assert Path(summary[input_key]).is_file()
    assert summary["seed_latent_frames"] == (1 if task == "i2v" else 6)
    assert summary["seed_pixel_frames"] == (1 if task == "i2v" else 21)
    assert summary["prompt_mode"] == "chunk_local"
    assert len(summary["prompt_schedule"]) == 7


def test_exactly_eleven_public_presets():
    assert [path.name for path in sorted(PRESETS.glob("*.yaml"))] == [
        "01_latte_occluder_recovery.yaml",
        "02_pancake_occluder_recovery.yaml",
        "03_whisking_camera_pan.yaml",
        "04_cereal_camera_pan.yaml",
        "05_dog_clean_i2v.yaml",
        "06_dry_ice_clean_i2v.yaml",
        "07_flour_clean_i2v.yaml",
        "08_cake_decorating_occluder_v2v.yaml",
        "09_ice_melting_drink_occluder_v2v.yaml",
        "10_pour_wine_beer_camera_pan_v2v.yaml",
        "11_slime_stretch_camera_pan_v2v.yaml",
    ]


def test_v2v_examples_are_two_occluders_and_two_camera_pans():
    v2v = []
    for path in sorted(PRESETS.glob("*.yaml")):
        _, preset = load_preset(path)
        if preset["task"] == "v2v":
            v2v.append(preset)
    assert [preset["control"]["kind"] for preset in v2v].count("occluder") == 2
    assert [preset["control"]["kind"] for preset in v2v].count("camera") == 2
    assert all(preset["control"]["kind"] != "clean" for preset in v2v)


def test_v2v_examples_do_not_duplicate_i2v_content():
    i2v_cases = set()
    v2v_cases = set()
    for path in sorted(PRESETS.glob("*.yaml")):
        _, preset = load_preset(path)
        target = i2v_cases if preset["task"] == "i2v" else v2v_cases
        target.add(preset["website_case"])
    assert len(i2v_cases) == 7
    assert len(v2v_cases) == 4
    assert i2v_cases.isdisjoint(v2v_cases)


def test_v2v_loader_uses_21_frames_and_does_not_require_future_video():
    _, preset = load_preset(PRESETS / "08_cake_decorating_occluder_v2v.yaml")
    video, seed_pixel_frames = video_prefix_to_padded_video(
        preset["_input_video_path"],
        seed_latent_frames=6,
        frames=25,
        width=16,
        height=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert seed_pixel_frames == 21
    assert video.shape == (1, 3, 25, 8, 16)
    assert torch.equal(video[:, :, 20], video[:, :, 24])


def test_recovery_prompts_are_chunk_local():
    for name, phrase in [
        ("01_latte_occluder_recovery.yaml", "completely blocks the scene"),
        ("02_pancake_occluder_recovery.yaml", "completely blocks the scene"),
    ]:
        _, preset = load_preset(PRESETS / name)
        prompts = build_prompt(
            preset["prompt"],
            preset["control"],
            pixel_frames=81,
            latent_frames=21,
            chunk_size=3,
        )
        assert isinstance(prompts, list) and len(prompts) == 7
        assert any(phrase in prompt for prompt in prompts)
        assert any("without resetting" in prompt for prompt in prompts)


@pytest.mark.parametrize(
    "name,max_angle,final_angle",
    [
        ("03_whisking_camera_pan.yaml", 16.92149937206113, 0.5014356263381423),
        ("04_cereal_camera_pan.yaml", 17.987917810908208, 0.12433010922246619),
        ("10_pour_wine_beer_camera_pan_v2v.yaml", 17.0947532978402, 1.2000781353326495),
        ("11_slime_stretch_camera_pan_v2v.yaml", 21.20528106501148, 3.288428754997835),
    ],
)
def test_camera_presets_use_exact_pairfixed_trajectories(name, max_angle, final_angle):
    _, preset = load_preset(PRESETS / name)
    extrinsics, intrinsics = build_pixel_cameras(
        preset["control"], frames=81, height=480, width=832
    )
    assert extrinsics.shape == (1, 81, 4, 4)
    assert intrinsics.shape == (1, 81, 3, 3)
    metrics = camera_metrics(extrinsics)
    assert metrics["max_angle_deg"] == pytest.approx(max_angle)
    assert metrics["final_angle_deg"] == pytest.approx(final_angle)
    prompt = build_prompt(
        preset["prompt"],
        preset["control"],
        pixel_frames=81,
        latent_frames=21,
        chunk_size=3,
    )
    assert isinstance(prompt, list) and len(prompt) == 7
    assert all("camera trajectory is a loop" not in item for item in prompt)


def test_checkpoint_directory_resolution(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weight = checkpoint / "student_lora_ema.safetensors"
    weight.write_bytes(b"test")
    assert resolve_checkpoint(checkpoint, weight.name) == weight
