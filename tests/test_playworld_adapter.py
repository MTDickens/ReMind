# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import tempfile
import unittest
from pathlib import Path

from player import build_preset, load_tasks


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


if __name__ == "__main__":
    unittest.main()
