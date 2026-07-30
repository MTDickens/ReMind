# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

import unittest

import torch

from pipeline.checkpoints import causal_block_linear_targets


class CameraPhaseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(14, 8, bias=False)


class CausalWanAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q = torch.nn.Linear(8, 8)
        self.self_attn.camera_phase_mlp = CameraPhaseMLP()
        self.ffn = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU())


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([CausalWanAttentionBlock()])


class LoraTargetTest(unittest.TestCase):
    def test_excludes_direct_weight_camera_phase_projection(self):
        targets = causal_block_linear_targets(Transformer())
        self.assertIn("blocks.0.self_attn.q", targets)
        self.assertIn("blocks.0.ffn.0", targets)
        self.assertNotIn(
            "blocks.0.self_attn.camera_phase_mlp.proj",
            targets,
        )


if __name__ == "__main__":
    unittest.main()
