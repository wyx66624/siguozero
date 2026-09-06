"""Bounded copy-on-write paged KV cache for one-token actor decoding.

The learner continues to use PyTorch's ordinary full-sequence attention.  This
module is only used while the policy is frozen for rollout inference, so its
storage is detached, mutable, and can be recycled between anchor waves.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor

from .accelerator import is_accelerator, supports_pinned_memory


@dataclass(slots=True)
class PagedKVState:
    """A logical immutable prefix backed by reference-counted physical pages."""

    length: int
    page_indices: tuple[int, ...]
    context: Tensor


class PagedKVCache:
    """All-layer paged KV arena shared by one frozen rollout actor."""

    page_size = 16

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_tokens: int,
        max_entries: int,
    ) -> None:
        if not is_accelerator(device):
            raise RuntimeError("paged KV is only available on CUDA or NPU")
        if dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("paged KV requires float16 or bfloat16 actor tensors")
        if min(num_layers, num_heads, head_dim, max_tokens, max_entries) <= 0:
            raise ValueError("paged KV dimensions must be positive")
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_tokens = max_tokens
        self.max_entries = max_entries
        pages_per_sequence = math.ceil(max_tokens / self.page_size)
        # The LRU is sized at roughly three cache tips per live branch.  Full
        # pages are shared across those tips, so one full live generation plus
        # four writable tail pages per LRU entry is sufficient.  A rare complete
        # sliding-window rebuild resets an obsolete generation before refill.
        live_sequences = max(1, max_entries // 3)
        self.max_pages = live_sequences * pages_per_sequence + 4 * max_entries
        self.storage = torch.empty(
            (
                num_layers,
                self.max_pages,
                2,
                self.page_size,
                num_heads,
                head_dim,
            ),
            dtype=dtype,
            device=device,
        )
        self._references = [0] * self.max_pages
        self._free_pages = list(range(self.max_pages - 1, -1, -1))
        self.peak_pages = 0

    @property
    def used_pages(self) -> int:
        return self.max_pages - len(self._free_pages)

    @property
    def free_pages(self) -> int:
        return len(self._free_pages)

    @property
    def allocated_bytes(self) -> int:
        return self.storage.numel() * self.storage.element_size()

    def reset(self) -> None:
        self._references[:] = [0] * self.max_pages
        self._free_pages[:] = range(self.max_pages - 1, -1, -1)

    def _allocate_page(self) -> int:
        if not self._free_pages:
            raise torch.OutOfMemoryError(
                "paged temporal KV arena exhausted; increase cache capacity or "
                "reduce actor concurrency"
            )
        page = self._free_pages.pop()
        if self._references[page] != 0:
            raise RuntimeError("paged KV allocator reference corruption")
        self._references[page] = 1
        self.peak_pages = max(self.peak_pages, self.used_pages)
        return page

    def _retain(self, pages: Sequence[int]) -> None:
        retained: list[int] = []
        try:
            for page in pages:
                if self._references[page] <= 0:
                    raise RuntimeError("cannot retain a free paged KV page")
                self._references[page] += 1
                retained.append(page)
        except BaseException:
            self._release_pages(retained)
            raise

    def release(self, state: PagedKVState) -> None:
        self._release_pages(state.page_indices)

    def _release_pages(self, pages: Sequence[int]) -> None:
        for page in pages:
            references = self._references[page] - 1
            if references < 0:
                raise RuntimeError("paged KV page released too many times")
            self._references[page] = references
            if references == 0:
                self._free_pages.append(page)

    def from_contiguous(
        self,
        layer_keys: Sequence[Tensor],
        layer_values: Sequence[Tensor],
        *,
        length: int,
        context: Tensor,
    ) -> PagedKVState:
        """Copy one full-sequence prefill result into physical pages once."""

        if length <= 0 or length > self.max_tokens:
            raise ValueError("paged KV prefill length is outside model capacity")
        if len(layer_keys) != self.num_layers or len(layer_values) != self.num_layers:
            raise ValueError("paged KV prefill layer count mismatch")
        pages_list: list[int] = []
        try:
            page_count = math.ceil(length / self.page_size)
            for _ in range(page_count):
                pages_list.append(self._allocate_page())
            pages = tuple(pages_list)
            for layer_index, (keys, values) in enumerate(
                zip(layer_keys, layer_values, strict=True)
            ):
                # Inputs use [heads, time, dim]; pages use [time, heads, dim].
                keys_by_time = keys[:, :length].transpose(0, 1)
                values_by_time = values[:, :length].transpose(0, 1)
                for page_offset, page in enumerate(pages):
                    start = page_offset * self.page_size
                    stop = min(start + self.page_size, length)
                    count = stop - start
                    self.storage[
                        layer_index, page, 0, :count
                    ].copy_(keys_by_time[start:stop])
                    self.storage[
                        layer_index, page, 1, :count
                    ].copy_(values_by_time[start:stop])
        except BaseException:
            self._release_pages(pages_list)
            raise
        return PagedKVState(length, pages, context)

    def fork_for_append(
        self, prefixes: Sequence[PagedKVState]
    ) -> tuple[list[PagedKVState], Tensor, Tensor]:
        """Fork variable-length prefixes, copying each final partial page."""

        if not prefixes:
            raise ValueError("cannot fork an empty paged KV batch")
        if any(prefix.length >= self.max_tokens for prefix in prefixes):
            raise ValueError("paged KV append exceeds model position capacity")
        children: list[PagedKVState] = []
        append_pages: list[int] = []
        append_offsets: list[int] = []
        clone_sources: list[int] = []
        clone_targets: list[int] = []
        try:
            for prefix in prefixes:
                length = prefix.length
                remainder = length % self.page_size
                prefix_pages = prefix.page_indices
                page = self._allocate_page()
                if remainder:
                    try:
                        self._retain(prefix_pages[:-1])
                    except BaseException:
                        self._release_pages((page,))
                        raise
                    clone_sources.append(prefix_pages[-1])
                    clone_targets.append(page)
                    child_pages = prefix_pages[:-1] + (page,)
                    append_offset = remainder
                else:
                    try:
                        self._retain(prefix_pages)
                    except BaseException:
                        self._release_pages((page,))
                        raise
                    child_pages = prefix_pages + (page,)
                    append_offset = 0
                children.append(
                    PagedKVState(length + 1, child_pages, prefix.context)
                )
                append_pages.append(page)
                append_offsets.append(append_offset)
            if clone_sources:
                sources = torch.tensor(
                    clone_sources, dtype=torch.long, device=self.device
                )
                targets = torch.tensor(
                    clone_targets, dtype=torch.long, device=self.device
                )
                self.storage[:, targets] = self.storage[:, sources]
            page_tensor = torch.tensor(
                append_pages, dtype=torch.long, device=self.device
            )
            offset_tensor = torch.tensor(
                append_offsets, dtype=torch.long, device=self.device
            )
            return children, page_tensor, offset_tensor
        except BaseException:
            for child in children:
                self.release(child)
            raise

    def make_page_table(
        self, states: Sequence[PagedKVState]
    ) -> tuple[Tensor, Tensor | None, int]:
        if not states:
            raise ValueError("cannot build an empty paged KV page table")
        max_pages_per_state = max(len(state.page_indices) for state in states)
        pin_memory = supports_pinned_memory(self.device)
        page_table_cpu = torch.zeros(
            (len(states), max_pages_per_state),
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        for index, state in enumerate(states):
            page_table_cpu[index, : len(state.page_indices)] = torch.tensor(
                state.page_indices, dtype=torch.int32
            )
        length_values = [state.length for state in states]
        maximum_length = max(length_values)
        valid_tokens = None
        if min(length_values) != maximum_length:
            lengths = torch.tensor(
                length_values,
                dtype=torch.long,
                device=self.device,
            )
            valid_tokens = (
                torch.arange(maximum_length, device=self.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            ).unsqueeze(1).unsqueeze(1)
        return (
            page_table_cpu.to(self.device, non_blocking=pin_memory),
            valid_tokens,
            maximum_length,
        )

    def append(
        self,
        layer_index: int,
        pages: Tensor,
        offsets: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> None:
        self.storage[layer_index, pages, 0, offsets] = keys
        self.storage[layer_index, pages, 1, offsets] = values

    def decode(
        self,
        layer_index: int,
        queries: Tensor,
        page_table: Tensor,
        valid_tokens: Tensor | None,
        max_tokens: int,
    ) -> Tensor:
        if queries.ndim != 3:
            raise ValueError("paged decode queries must be [batch, heads, dim]")
        cache = self.storage[layer_index]
        batch_size, page_count = page_table.shape
        selected = cache.index_select(0, page_table.reshape(-1)).view(
            batch_size,
            page_count,
            2,
            self.page_size,
            self.num_heads,
            self.head_dim,
        )
        keys = selected[:, :, 0].reshape(
            batch_size,
            page_count * self.page_size,
            self.num_heads,
            self.head_dim,
        )[:, :max_tokens].transpose(1, 2)
        values = selected[:, :, 1].reshape(
            batch_size,
            page_count * self.page_size,
            self.num_heads,
            self.head_dim,
        )[:, :max_tokens].transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(
            queries.unsqueeze(2),
            keys,
            values,
            attn_mask=valid_tokens,
            dropout_p=0.0,
            is_causal=False,
        ).squeeze(2)
