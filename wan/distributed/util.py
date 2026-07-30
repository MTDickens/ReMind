# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified by Applied Intuition, Inc. in 2026.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist


_sequence_parallel_group = None


def set_sequence_parallel_group(group):
    global _sequence_parallel_group
    _sequence_parallel_group = group


def get_sequence_parallel_group():
    return _sequence_parallel_group


def get_sequence_parallel_world_size():
    if _sequence_parallel_group is None or not dist.is_initialized():
        return 1
    return dist.get_world_size(_sequence_parallel_group)


def get_sequence_parallel_rank():
    if _sequence_parallel_group is None or not dist.is_initialized():
        return 0
    return dist.get_rank(_sequence_parallel_group)


def _resolve_group(group=None):
    return group if group is not None else _sequence_parallel_group


def all_to_all_with_grad(x, scatter_dim, gather_dim, group=None):
    group = _resolve_group(group)
    if group is None or not dist.is_initialized():
        return x
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return x
    scatter_size = x.size(scatter_dim)
    if scatter_size % world_size != 0:
        raise ValueError(
            "all_to_all_with_grad requires the scatter dimension to be "
            f"divisible by the sequence-parallel world size: "
            f"size={scatter_size}, world_size={world_size}, "
            f"scatter_dim={scatter_dim}, gather_dim={gather_dim}, "
            f"shape={tuple(x.shape)}"
        )
    return _AllToAllWithGrad.apply(x, scatter_dim, gather_dim, group)


class _AllToAllWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scatter_dim, gather_dim, group):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        world_size = dist.get_world_size(group)
        inputs = [u.contiguous() for u in input_tensor.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group)
        return torch.cat(outputs, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size(ctx.group)
        inputs = [u.contiguous() for u in grad_output.chunk(world_size, dim=ctx.gather_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=ctx.group)
        grad_input = torch.cat(outputs, dim=ctx.scatter_dim).contiguous()
        return grad_input, None, None, None


def gather_forward_with_grad(x, dim, group=None):
    group = _resolve_group(group)
    if group is None or not dist.is_initialized():
        return x
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return x
    return _AllGatherWithGrad.apply(x, dim, group)


class _AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, dim, group):
        ctx.dim = dim
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        outputs = [torch.empty_like(input_tensor) for _ in range(ctx.world_size)]
        dist.all_gather(outputs, input_tensor.contiguous(), group=group)
        return torch.cat(outputs, dim=dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        chunks = [u.contiguous() for u in grad_output.chunk(ctx.world_size, dim=ctx.dim)]
        grad_input = chunks[ctx.rank]
        dist.all_reduce(grad_input, group=ctx.group)
        return grad_input, None, None
