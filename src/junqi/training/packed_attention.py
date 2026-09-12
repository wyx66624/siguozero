"""Exact causal training on packed histories with no attention padding.

Uses the installed PyTorch variable-length Flash Attention implementation.
Every sequence has its own cumulative offset and local position indices;
different sliding-window starts are never joined into one causal history.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:  # Older CUDA/NPU installations retain padded causal SDPA.
    varlen_attn = None

AVAILABLE = varlen_attn is not None

from .accelerator import supports_pinned_memory
from .encoding import MAX_CASUALTY_BITS, history_prefix_groups
from .modes import mode_spec


def prepare(model, states):
    if len({state.mode for state in states}) != 1:
        raise ValueError('a policy batch must use one information mode')
    groups = history_prefix_groups(states)
    lengths = np.asarray([len(states[g[0]].records) for g in groups], dtype=np.int64)
    maximum = int(lengths.max())
    if maximum > model.config.max_sequence_tokens:
        raise ValueError('policy history exceeds the configured window')
    offsets = np.concatenate(([0], lengths.cumsum()))
    total = int(offsets[-1])
    mode = states[0].mode
    spec = mode_spec(mode)
    width = spec.point_count + (MAX_CASUALTY_BITS if model.config.dead_rules_enabled else 0) + 10
    pin = supports_pinned_memory(model.device)
    raw_cpu = torch.empty((1, total, width), dtype=torch.int16, pin_memory=pin)
    raw_numpy = raw_cpu.numpy()[0]
    queries = np.empty(len(states), dtype=np.int64)
    for group, start, length in zip(groups, offsets[:-1], lengths, strict=True):
        view = states[group[0]].records
        if view.mode is not mode or view.dead_rules != model.config.dead_rules_enabled:
            raise ValueError('history array mode/dead-rule features do not match the model')
        view.copy_rows(raw_numpy[start:start + length])
        for index in group:
            queries[index] = start + len(states[index].records) - 1
    index_cpu = torch.empty(total + len(states), dtype=torch.long, pin_memory=pin)
    index_cpu.numpy()[:] = np.concatenate((np.arange(total) - np.repeat(offsets[:-1], lengths), queries))
    cu_cpu = torch.empty(len(offsets), dtype=torch.int32, pin_memory=pin)
    cu_cpu.numpy()[:] = offsets
    positions, queries_tensor = index_cpu.to(model.device, non_blocking=pin).split((total, len(states)))
    return (raw_cpu.to(model.device, non_blocking=pin), positions[None, :],
            cu_cpu.to(model.device, non_blocking=pin), queries_tensor, maximum, mode), lengths


def forward(model, raw, positions, cu, queries, maximum, mode):
    tokens = model._embed_observation_rows(raw, mode, positions)[0]
    for layer in model.temporal_layers:
        norm = layer.attention_norm(tokens)
        qkv = F.linear(norm, layer.attention.in_proj_weight, layer.attention.in_proj_bias)
        heads = layer.attention.num_heads
        q, k, v = [part.view(-1, heads, tokens.shape[-1] // heads) for part in qkv.chunk(3, dim=-1)]
        attended = varlen_attn(q, k, v, cu, cu, maximum, maximum, window_size=(-1, 0)).flatten(1)
        tokens = tokens + F.linear(attended, layer.attention.out_proj.weight, layer.attention.out_proj.bias)
        tokens = tokens + layer.ffn(layer.ffn_norm(tokens))
    # LayerNorm acts independently per token. Only requested decisions need it.
    return model.temporal_norm(tokens.index_select(0, queries))
