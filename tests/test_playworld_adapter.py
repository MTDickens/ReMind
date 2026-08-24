# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import tempfile
import unittest
from pathlib import Path

from player import _segment_preset, build_preset, load_tasks, rollout_layout


def _example_mapping(tmp_path):
    image_root = tmp_path / "example"
    split = image_root / "GC"
    split.mkdir(parents=True)
    image = split / "001.jpg"
    image.write_bytes(b"image")
    mapping = split / "001.json"
    mapping.write_text(
        json.dumps(
            [
                {
                    "task_id": "GC002",
                    "prompt": "Move around the sculpture and return.",
                    "image_path": "GC/001.jpg",
                    "image_caption": "A sculpture stands in a museum gallery.",
                    "action": "d*3 -> w*12 -> a*6 -> s*12 -> d*3",
                    "action_sequence_steps": ["hold(D,1350ms)"],
                }
            ]
        ),
        encoding="utf-8",
    )
    return mapping, image


class PlayWorldAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_playworld_example_layout(self):
        mapping, image = _example_mapping(self.root)
        tasks = load_tasks(mapping, task_ids=["GC002"])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["_image_path"], image.resolve())

    def test_preset_keeps_objective_out_of_world_prompt(self):
        mapping, _ = _example_mapping(self.root)
        task = load_tasks(mapping)[0]
        preset = build_preset(task, seed=7)
        self.assertEqual(preset["prompt"], task["image_caption"])
        self.assertNotEqual(preset["prompt"], task["prompt"])
        self.assertEqual(preset["control"], {"kind": "clean"})
        self.assertEqual(preset["seed"], 7)

    def test_missing_task_id_is_reported(self):
        mapping, _ = _example_mapping(self.root)
        with self.assertRaisesRegex(KeyError, "GC999"):
            load_tasks(mapping, task_ids=["GC999"])

    def test_one_minute_gc002_uses_twelve_overlapping_windows(self):
        self.assertEqual(rollout_layout(60), (12, 961))

    def test_gc002_actions_become_a_closed_rectangular_camera_path(self):
        mapping, _ = _example_mapping(self.root)
        task = load_tasks(mapping)[0]
        preset = build_preset(task, duration_seconds=60, apply_actions=True)
        path = preset["control"]["translation_path"]

        self.assertEqual(preset["control"]["kind"], "camera")
        self.assertEqual(preset["num_frames"], 961)
        self.assertEqual(len(path), 961)
        self.assertEqual(path[0], [0.0, 0.0, 0.0])
        for actual, expected in zip(path[80], [-0.015, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(path[400], [-0.015, 0.0, -0.06]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(path[560], [0.015, 0.0, -0.06]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(path[880], [0.015, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected)
        for actual in path[-1]:
            self.assertAlmostEqual(actual, 0.0)

        first = _segment_preset(
            preset,
            segment_index=0,
            input_image=task["_image_path"],
            seed=1,
        )
        second = _segment_preset(
            preset,
            segment_index=1,
            input_image=task["_image_path"],
            seed=2,
        )
        self.assertEqual(len(first["control"]["translation_path"]), 81)
        self.assertEqual(
            first["control"]["translation_path"][-1],
            second["control"]["translation_path"][0],
        )


if __name__ == "__main__":
    unittest.main()
