# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified by Applied Intuition, Inc. in 2026.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch
from torch.nn.attention.flex_attention import flex_attention

from wan.distributed.util import (
    all_to_all_with_grad,
    get_sequence_parallel_world_size,
)


if os.environ.get("DISABLE_FLEX_COMPILE") == "1":
    print("[ulysses] DISABLE_FLEX_COMPILE=1 - distributed flex_attention stays eager")
else:
    _flex_compile_mode = os.environ.get("FLEX_COMPILE_MODE", "default")
    print(f"[ulysses] distributed flex_attention torch.compile mode={_flex_compile_mode}")
    flex_attention = torch.compile(
        flex_attention, dynamic=False, mode=_flex_compile_mode)


def distributed_flex_attention(
    roped_q: torch.Tensor,
    roped_k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    *,
    pad_multiple: int = 128,
    is_clean_history: bool = False,
    clean_history_half_tokens: int = 0,
    clean_history_tokens_per_chunk: int = 0,
    clean_history_chunk_flash: bool = False,
):
    """Ulysses-style sequence-parallel flex attention.

    Inputs are local sequence shards with full heads: [B, L_local, H, D].
    The first all-to-all gathers the global sequence and scatters heads.
    The second all-to-all restores the local sequence shard with full heads.
    """
    world_size = get_sequence_parallel_world_size()
    num_heads = int(roped_q.shape[2])
    if num_heads % world_size != 0:
        raise ValueError(
            "Ulysses sequence parallel requires num_heads to be divisible by "
            f"sequence_parallel_size. Got num_heads={num_heads}, "
            f"sequence_parallel_size={world_size}. For Wan2.1-T2V-1.3B "
            "num_heads=12, so valid 8-GPU SP sizes are 1, 2, or 4; sp8 is "
            "invalid and can otherwise corrupt the attention output width."
        )

    roped_q = all_to_all_with_grad(roped_q, scatter_dim=2, gather_dim=1)
    roped_k = all_to_all_with_grad(roped_k, scatter_dim=2, gather_dim=1)
    v = all_to_all_with_grad(v, scatter_dim=2, gather_dim=1)

    bsz, global_len, local_heads, head_dim = roped_q.shape

    if is_clean_history:
        if global_len % (2 * world_size) != 0:
            raise ValueError(
                f"clean-history SP length {global_len} must be divisible by "
                f"2 * sp_size ({2 * world_size})")
        chunk_len = global_len // (2 * world_size)

        def _tf_to_global_order(tensor):
            tensor = tensor.view(
                bsz, world_size, 2, chunk_len, local_heads, head_dim)
            clean = tensor[:, :, 0].reshape(
                bsz, world_size * chunk_len, local_heads, head_dim)
            noisy = tensor[:, :, 1].reshape(
                bsz, world_size * chunk_len, local_heads, head_dim)
            return torch.cat([clean, noisy], dim=1)

        roped_q = _tf_to_global_order(roped_q)
        roped_k = _tf_to_global_order(roped_k)
        v = _tf_to_global_order(v)
        global_len = roped_q.shape[1]

        if clean_history_chunk_flash:
            half_tokens = int(clean_history_half_tokens)
            tokens_per_chunk = int(clean_history_tokens_per_chunk)
            if half_tokens <= 0:
                half_tokens = global_len // 2
            if half_tokens * 2 != global_len:
                raise ValueError(
                    "clean-history SP chunk flash got inconsistent lengths: "
                    f"global_len={global_len}, half_tokens={half_tokens}")
            out = _clean_history_chunk_flash_attention(
                roped_q, roped_k, v,
                half_tokens=half_tokens,
                tokens_per_chunk=tokens_per_chunk,
            )
            out = out.view(bsz, 2, world_size, chunk_len, local_heads, head_dim)
            out = out.permute(0, 2, 1, 3, 4, 5).reshape(
                bsz, global_len, local_heads, head_dim)
            out = all_to_all_with_grad(out, scatter_dim=1, gather_dim=2)
            return out

    target_q_len = int(block_mask.shape[-2])
    target_kv_len = int(block_mask.shape[-1])
    if target_q_len != target_kv_len:
        raise ValueError(
            f"distributed flex attention expects square block mask, got "
            f"{block_mask.shape}")
    target_len = target_q_len
    if target_len < global_len:
        raise ValueError(
            f"BlockMask target length {target_len} is smaller than "
            f"global sequence length {global_len}")

    # The BlockMask defines the legal q/kv extent for flex_attention.  Do not
    # pad q/k/v beyond it: PyTorch validates q_len/kv_len against the BlockMask
    # shape before running the kernel.  Upstream mask builders are responsible
    # for rounding target_len when a padded kernel shape is desired.
    padded_len = target_len
    pad = padded_len - global_len
    if pad > 0:
        roped_q = torch.cat([
            roped_q,
            roped_q.new_zeros(bsz, pad, local_heads, head_dim),
        ], dim=1)
        roped_k = torch.cat([
            roped_k,
            roped_k.new_zeros(bsz, pad, local_heads, head_dim),
        ], dim=1)
        v = torch.cat([
            v,
            v.new_zeros(bsz, pad, local_heads, head_dim),
        ], dim=1)

    out = flex_attention(
        query=roped_q.transpose(1, 2),
        key=roped_k.transpose(1, 2),
        value=v.transpose(1, 2),
        block_mask=block_mask,
    )
    if pad > 0:
        out = out[:, :, :global_len]
    out = out.transpose(1, 2).contiguous()

    if is_clean_history:
        out = out.view(bsz, 2, world_size, chunk_len, local_heads, head_dim)
        out = out.permute(0, 2, 1, 3, 4, 5).reshape(
            bsz, global_len, local_heads, head_dim)

    out = all_to_all_with_grad(out, scatter_dim=1, gather_dim=2)
    return out


def _clean_history_chunk_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    half_tokens: int,
    tokens_per_chunk: int,
) -> torch.Tensor:
    """Exact dual-stream clean-history mask using chunked FlashAttention.

    After the Ulysses all-to-all, the tensor is in global [clean, noisy]
    order with only a shard of heads. The mask is regular:
      clean chunk i -> clean chunks <= i
      noisy chunk i -> clean chunks < i plus noisy chunk i
    Splitting by chunk preserves that support while avoiding the generic
    FlexAttention backward path for the large doubled sequence.
    """
    if (
        half_tokens <= 0
        or tokens_per_chunk <= 0
        or half_tokens % tokens_per_chunk != 0
        or q.shape[1] != half_tokens * 2
        or k.shape[1] != half_tokens * 2
        or v.shape[1] != half_tokens * 2
    ):
        raise ValueError(
            "SP clean-history chunk flash expects [clean,noisy] tokens: "
            f"q={tuple(q.shape)} k={tuple(k.shape)} "
            f"half_tokens={half_tokens} tokens_per_chunk={tokens_per_chunk}"
        )

    from wan.modules.attention import attention

    clean_out = []
    noisy_out = []
    for start in range(0, half_tokens, tokens_per_chunk):
        end = start + tokens_per_chunk
        clean_out.append(attention(q[:, start:end], k[:, :end], v[:, :end]))

        noisy_start = half_tokens + start
        noisy_end = half_tokens + end
        if start > 0:
            k_noisy = torch.cat(
                [k[:, :start], k[:, noisy_start:noisy_end]], dim=1)
            v_noisy = torch.cat(
                [v[:, :start], v[:, noisy_start:noisy_end]], dim=1)
        else:
            k_noisy = k[:, noisy_start:noisy_end]
            v_noisy = v[:, noisy_start:noisy_end]
        noisy_out.append(
            attention(q[:, noisy_start:noisy_end], k_noisy, v_noisy))

    return torch.cat(clean_out + noisy_out, dim=1)
