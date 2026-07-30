# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from torch import nn

from pipeline.prompt_schedule import chunk_text_prompts
from utils.wan_wrapper import _select_chunk_prompt_embeds
from wan.modules.causal_model import CausalWanAttentionBlock


def test_light_prompts_follow_chunk_phases_and_recovery():
    prompts = chunk_text_prompts(
        "A person pours a drink",
        "light",
        {
            "t_off_start": 8,
            "t_off_end": 20,
            "t_hold_end": 44,
            "t_on_end": 56,
        },
        81,
        21,
        3,
        recovery_chunks=2,
    )
    assert len(prompts) == 7
    assert "turning off" in prompts[0]
    assert "turning off" in prompts[1]
    assert "remain fully off" in prompts[2]
    # v4 midpoint phase rule: chunk 3 spans px≈36-44 and the on-ramp only
    # starts at its final pixel (44), so it is labeled hold, not exit.
    assert "remain fully off" in prompts[3]
    assert "turn back on" in prompts[4]
    assert "fully on again" in prompts[5]
    assert "fully on again" in prompts[6]


def test_occluder_description_is_repeated_in_active_and_recovery_chunks():
    prompts = chunk_text_prompts(
        "A person pours a drink",
        "occluder",
        {
            "t_enter_start": 20,
            "t_enter_end": 30,
            "t_hold_end": 50,
            "t_exit_end": 62,
            "enter_side": "left",
            "exit_side": "right",
        },
        81,
        21,
        3,
        meta={
            "label": "matte brown cardboard panel",
            "caption": "a large flat rectangular panel with a rough surface",
        },
        recovery_chunks=1,
    )
    event_prompts = [prompt for prompt in prompts if prompt != prompts[0]]
    assert event_prompts
    assert all("matte brown cardboard panel" in prompt for prompt in event_prompts)
    assert any("enters from the left" in prompt for prompt in prompts)
    assert any("moves out toward the right" in prompt for prompt in prompts)
    assert any("completely gone" in prompt for prompt in prompts)


def test_late_truncated_event_does_not_claim_exit_or_recovery():
    prompts = chunk_text_prompts(
        "A person pours a drink",
        "light",
        {
            "t_off_start": 72,
            "t_off_end": 78,
            "t_hold_end": 90,
            "t_on_end": 96,
        },
        81,
        21,
        3,
        recovery_chunks=2,
    )
    assert prompts[:6] == ["A person pours a drink."] * 6
    assert "turning off" in prompts[6]
    assert all("turn back on" not in prompt for prompt in prompts)
    assert all("fully on again" not in prompt for prompt in prompts)


def test_streaming_prompt_selection_uses_absolute_chunk_offset():
    prompt_embeds = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1)
    cond = {"prompt_chunk_size": 3}
    full = _select_chunk_prompt_embeds(
        prompt_embeds, cond, current_frames=21, freqs_offset=0
    )
    current = _select_chunk_prompt_embeds(
        prompt_embeds, cond, current_frames=3, freqs_offset=9
    )
    partial = _select_chunk_prompt_embeds(
        prompt_embeds, cond, current_frames=2, freqs_offset=2
    )
    explicit = _select_chunk_prompt_embeds(
        prompt_embeds,
        {"prompt_chunk_size": 3, "prompt_frame_indices": [0, 6, 15]},
        current_frames=3,
        freqs_offset=0,
    )
    assert torch.equal(full, prompt_embeds)
    assert current.flatten().tolist() == [3.0]
    assert partial.flatten().tolist() == [0.0, 1.0]
    assert explicit.flatten().tolist() == [0.0, 2.0, 5.0]


class _ZeroSelfAttention(nn.Module):
    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)


class _ContextValueCrossAttention(nn.Module):
    def forward(self, x, context, context_lens, crossattn_cache=None):
        value = context[:, :1, :]
        return value.expand(-1, x.shape[1], -1)


class _ZeroFFN(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_chunk_context_is_grouped_independently_for_teacher_streams():
    block = CausalWanAttentionBlock(
        "t2v_cross_attn",
        dim=8,
        ffn_dim=16,
        num_heads=2,
        qk_norm=False,
        cross_attn_norm=False,
    )
    block.self_attn = _ZeroSelfAttention()
    block.cross_attn = _ContextValueCrossAttention()
    block.ffn = _ZeroFFN()
    block.modulation.data.zero_()

    # Two clean-history streams, two chunk prompts, three frames/chunk,
    # and two tokens/frame.
    x = torch.zeros(1, 24, 8)
    e = torch.zeros(1, 12, 6, 8)
    context = torch.zeros(1, 2, 1, 8)
    context[:, 0] = 1
    context[:, 1] = 2
    out = block(
        x,
        e,
        seq_lens=torch.tensor([12]),
        freqs_x=None,
        context=context,
        context_lens=None,
        tokens_per_frame=2,
        context_streams=2,
    )
    expected_stream = torch.cat(
        [
            torch.ones(1, 6, 8),
            torch.full((1, 6, 8), 2.0),
        ],
        dim=1,
    )
    assert torch.equal(out, torch.cat([expected_stream, expected_stream], dim=1))
