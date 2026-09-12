"""Bounded PPO-only KV slots, multi-token causal decode and static CUDA graphs.

Weights are owned by the existing policy/critic. Cache identity is the game
instance, seat, context-window start, and the frozen behavior version. No
search prefix generations, page tables, or per-layer history concatenation.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .accelerator import supports_pinned_memory
from .encoding import PolicyState
from .history_arrays import HistoryArrayView
from .modes import mode_spec


@dataclass(slots=True)
class _Entry:
    slot: int
    window_start: int = -1
    length: int = 0


class FixedKVCache:
    def __init__(self, model, *, capacity: int, behavior_version: int,
                 cuda_graphs: bool = True) -> None:
        self.model, self.capacity, self.behavior_version = model, capacity, behavior_version
        self.entries: OrderedDict[tuple[int, int], _Entry] = OrderedDict()
        self.free = list(range(capacity - 1, -1, -1))
        self.storage: Tensor | None = None
        self.contexts: Tensor | None = None
        self.graphs: OrderedDict[tuple, tuple] = OrderedDict()
        self.cuda_graphs = cuda_graphs
        self.kernels = None
        if model.device.type == "cuda":
            try:
                from . import fixed_kv_kernels
                self.kernels = fixed_kv_kernels
            except ModuleNotFoundError:
                # CPU/NPU and installations without Triton retain exact SDPA.
                pass
        self.prefill_states = self.decode_states = self.decode_tokens = 0
        self.submitted_decode_tokens = 0
        self.graph_captures = self.graph_replays = 0

    def allocate(self) -> None:
        if self.storage is not None:
            return
        config, device = self.model.config, self.model.device
        dtype = (torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type)
                 else self.model.position_embedding.weight.dtype)
        self.storage = torch.empty(
            (len(self.model.temporal_layers), 2, self.capacity, config.temporal_heads,
             config.max_sequence_tokens, config.temporal_dim // config.temporal_heads),
            device=device, dtype=dtype,
        )
        self.contexts = torch.empty((self.capacity, config.temporal_dim), device=device,
                                     dtype=self.model.position_embedding.weight.dtype)

    def release_game(self, game_identity: int) -> None:
        for key in list(self.entries):
            if key[0] == game_identity:
                self.free.append(self.entries.pop(key).slot)

    def _entry(self, view: HistoryArrayView, pinned: set) -> _Entry:
        key = view.identity
        entry = self.entries.get(key)
        if entry is None:
            if not self.free:
                victim = next((key for key in self.entries if key not in pinned), None)
                if victim is None:
                    raise torch.OutOfMemoryError("PPO fixed KV capacity is smaller than the active batch")
                self.free.append(self.entries.pop(victim).slot)
            entry = _Entry(self.free.pop())
            self.entries[key] = entry
        self.entries.move_to_end(key)
        return entry

    def write(self, layer: int, keys: Tensor, values: Tensor, metadata: Tensor) -> None:
        cache = self.storage[layer]
        if self.kernels is not None:
            batch, heads, tokens, dim = keys.shape
            self.kernels.write_kv[(math.ceil(batch * heads * tokens * dim / 256),)](
                keys, values, cache[0], cache[1], metadata,
                batch, tokens, *keys.stride()[:3], *values.stride()[:3],
                HEADS=heads, DIM=dim, CAPACITY=self.model.config.max_sequence_tokens, BLOCK=256,
            )
        else:
            # This portable path is deliberately outside CUDA graph capture.
            for row, (slot, prefix, count) in enumerate(metadata.cpu().tolist()):
                cache[0, slot, :, prefix:prefix + count].copy_(keys[row, :, :count])
                cache[1, slot, :, prefix:prefix + count].copy_(values[row, :, :count])

    def attention(self, layer: int, queries: Tensor, metadata: Tensor, extent: int) -> Tensor:
        cache = self.storage[layer]
        batch, heads, count, dim = queries.shape
        if self.kernels is not None:
            output = torch.empty((batch, heads, count, dim), device=queries.device, dtype=queries.dtype)
            if count >= 2 and extent >= 128 and queries.dtype in (torch.bfloat16, torch.float16):
                from .grouped_kv_kernel import attend_grouped
                attend_grouped[(batch, heads)](
                    queries, cache[0], cache[1], metadata, output, *queries.stride()[:3],
                    HEADS=heads, QUERIES=count, DIM=dim, CAPACITY=self.model.config.max_sequence_tokens,
                    EXTENT=extent,
                )
                return output
            self.kernels.attend_kv[(batch, heads, count)](
                queries, cache[0], cache[1], metadata, output, *queries.stride()[:3],
                HEADS=heads, QUERIES=count, DIM=dim, CAPACITY=self.model.config.max_sequence_tokens,
                BLOCK_N=extent, BLOCK_D=1 << (dim - 1).bit_length(), SCALE=dim ** -0.5,
                num_warps=4 if extent <= 512 else 8,
            )
            return output
        slots, prefixes, counts = metadata.unbind(dim=1)
        keys = cache[0].index_select(0, slots)[:, :, :extent]
        values = cache[1].index_select(0, slots)[:, :, :extent]
        valid = torch.arange(keys.shape[2], device=queries.device)[None, :] < (prefixes + counts)[:, None]
        # SDPA masks scores, but 0 * an uninitialized NaN value can still
        # propagate through its value product on portable backends.
        keys = keys.masked_fill(~valid[:, None, :, None], 0.)
        values = values.masked_fill(~valid[:, None, :, None], 0.)
        positions = torch.arange(keys.shape[2], device=queries.device)[None, None, :]
        queries_at = prefixes[:, None, None] + torch.arange(count, device=queries.device)[None, :, None]
        allowed = (positions <= queries_at) & (positions < (prefixes + counts)[:, None, None])
        return F.scaled_dot_product_attention(queries, keys, values, attn_mask=allowed[:, None], dropout_p=0.)

    def _decode(self, raw: Tensor, metadata: Tensor, extent: int, mode) -> Tensor:
        batch, queries = raw.shape[:2]
        positions = metadata[:, 1, None] + torch.arange(queries, device=raw.device)[None, :]
        hidden = self.model._embed_observation_rows(raw, mode, positions.clamp_max(self.model.config.max_transitions))
        for layer_index, layer in enumerate(self.model.temporal_layers):
            q, k, v = layer.incremental_projection(hidden)
            self.write(layer_index, k, v, metadata)
            attended = self.attention(layer_index, q, metadata, extent)
            hidden = layer.incremental_output(hidden, attended.transpose(1, 2).reshape(batch, queries, -1))
        normalized = self.model.temporal_norm(hidden)
        return normalized[torch.arange(batch, device=raw.device), (metadata[:, 2] - 1).clamp_min(0)]

    def _run_decode(self, raw: Tensor, metadata: Tensor, extent: int, mode) -> Tensor:
        if not self.cuda_graphs or self.kernels is None:
            return self._decode(raw, metadata, extent, mode)
        key = (tuple(raw.shape), extent, mode)
        entry = self.graphs.get(key)
        if entry is None:
            # Warm up all libraries/JIT kernels on a side stream before capture.
            static_raw, static_meta = raw.clone(), metadata.clone()
            stream = torch.cuda.Stream(device=raw.device)
            stream.wait_stream(torch.cuda.current_stream(raw.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._decode(static_raw, static_meta, extent, mode)
            torch.cuda.current_stream(raw.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._decode(static_raw, static_meta, extent, mode)
            entry = graph, static_raw, static_meta, output
            self.graphs[key] = entry
            self.graph_captures += 1
            while len(self.graphs) > 8:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        graph, static_raw, static_meta, output = entry
        static_raw.copy_(raw)
        static_meta.copy_(metadata)
        graph.replay()
        self.graph_replays += 1
        return output

    def encode(self, states: Sequence[PolicyState]):
        from .models import PolicyFeatures
        if len({state.records.identity for state in states}) != len(states):
            # Duplicate or out-of-order diagnostic queries cannot mutate one
            # seat's slot twice in the same invocation.
            return self.model._encode_full(states)
        self.allocate()
        pinned = {state.records.identity for state in states}
        entries = [self._entry(state.records, pinned) for state in states]
        cold, warm, cached = [], [], []
        for index, (state, entry) in enumerate(zip(states, entries, strict=True)):
            missing = len(state.records) - entry.length
            if entry.window_start != state.records.window_start or not 0 <= missing <= 8:
                entry.length, entry.window_start = 0, -1
                cold.append(index)
            elif missing:
                warm.append(index)
            else:
                cached.append(index)
        output = torch.empty((len(states), self.model.config.temporal_dim), device=self.model.device,
                               dtype=self.contexts.dtype)
        if cold:
            features = self.model._encode_full([states[i] for i in cold], count_history_stats=False,
                                               fixed_slots=[entries[i].slot for i in cold])
            output.index_copy_(0, torch.tensor(cold, device=output.device), features.context)
            self.prefill_states += len(cold)
            self.model._temporal_cold_states += len(cold)
            self.model._temporal_computed_pairs += sum(len(states[i].records) * (len(states[i].records) + 1) // 2 for i in cold)
        if warm:
            missing = [len(states[i].records) - entries[i].length for i in warm]
            queries = 1 << (max(missing) - 1).bit_length()
            # Stable sizes avoid a separate capture for each rare partial batch.
            batch = ((len(warm) + 7) // 8) * 8 if self.kernels is not None else len(warm)
            view = states[warm[0]].records
            width = mode_spec(view.mode).point_count + (75 if view.dead_rules else 0) + 10
            pin = supports_pinned_memory(output.device)
            raw_cpu = torch.zeros((batch, queries, width), dtype=torch.int16, pin_memory=pin)
            meta_cpu = torch.zeros((batch, 3), dtype=torch.long, pin_memory=pin)
            raw_rows, meta_rows = raw_cpu.numpy(), meta_cpu.numpy()
            for row, (index, count) in enumerate(zip(warm, missing, strict=True)):
                states[index].records.copy_rows(raw_rows[row, :count], start=entries[index].length)
                meta_rows[row] = entries[index].slot, entries[index].length, count
            extent = 1 << (max(len(states[i].records) for i in warm) - 1).bit_length()
            raw = raw_cpu.to(output.device, non_blocking=pin)
            meta = meta_cpu.to(output.device, non_blocking=pin)
            contexts = self._run_decode(raw, meta, extent, states[0].mode)[:len(warm)]
            output.index_copy_(0, torch.tensor(warm, device=output.device), contexts)
            self.decode_states += len(warm)
            self.decode_tokens += sum(missing)
            self.submitted_decode_tokens += batch * queries
            self.model._board_input_tokens += sum(missing)
            self.model._board_unique_tokens += sum(missing)
            self.model._board_encoder_tokens += batch * queries
            self.model._temporal_incremental_tokens += sum(missing)
            self.model._temporal_incremental_batches += 1
            self.model._temporal_incremental_batch_rows += len(warm)
            self.model._temporal_incremental_batch_max = max(self.model._temporal_incremental_batch_max, len(warm))
            self.model._temporal_computed_pairs += sum(
                count * entries[i].length + count * (count + 1) // 2
                for i, count in zip(warm, missing, strict=True))
        if cached:
            slots = torch.tensor([entries[i].slot for i in cached], device=output.device)
            output.index_copy_(0, torch.tensor(cached, device=output.device), self.contexts.index_select(0, slots))
        # Only commit metadata after every layer succeeded. A failed append
        # leaves old prefixes intact; failed cold fills remain explicitly invalid.
        slots = torch.tensor([entry.slot for entry in entries], device=output.device)
        self.contexts.index_copy_(0, slots, output)
        for state, entry in zip(states, entries, strict=True):
            entry.window_start, entry.length = state.records.window_start, len(state.records)
        self.model._history_input_states += len(states)
        self.model._history_unique_states += len(states)
        self.model._raw_board_tokens += sum(len(state.records) for state in states)
        self.model._temporal_requested_pairs += sum(len(state.records) * (len(state.records) + 1) // 2 for state in states)
        self.model._temporal_cache_hits += len(warm) + len(cached)
        return PolicyFeatures(output, torch.ones((len(states), mode_spec(states[0].mode).point_count),
                                                dtype=torch.bool, device=output.device))
