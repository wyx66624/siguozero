"""Small torch.distributed facade used by the self-play trainer.

The training process uses one process per accelerator.  Each process owns an
independent environment shard while DDP keeps the single logical policy in
sync.  This module intentionally keeps distributed concerns out of the rules
engine and model definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import nn

from .accelerator import (
    device_count,
    is_accelerator,
    resolve_device,
    set_device,
)


@dataclass(slots=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None = None
    owns_process_group: bool = False
    object_group: Any = None

    @classmethod
    def initialize(cls, requested_device: str) -> "DistributedContext":
        """Initialize from ``torchrun`` environment variables when present."""

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        requested = resolve_device(requested_device)

        if world_size <= 1:
            device = set_device(requested) if is_accelerator(requested) else requested
            return cls(0, 1, 0, device)

        if is_accelerator(requested):
            visible_devices = device_count(requested.type)
            if local_rank >= visible_devices:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} exceeds visible "
                    f"{requested.type.upper()} device count {visible_devices}"
                )
            device = set_device(torch.device(requested.type, local_rank))
            backend = "hccl" if requested.type == "npu" else "nccl"
        elif requested.type == "cpu":
            device = torch.device("cpu")
            backend = "gloo"
        else:
            raise ValueError(
                "distributed training supports only NPU, CUDA, or CPU devices"
            )

        owns_group = False
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
            owns_group = True
        # Python environment/checkpoint objects can become large.  A secondary
        # Gloo keeps serialization on CPU instead of allocating byte tensors in
        # accelerator memory.  It also avoids HCCL's lack of object gather.
        object_group = (
            dist.new_group(backend="gloo")
            if backend in ("nccl", "hccl")
            else None
        )
        return cls(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=local_rank,
            device=device,
            backend=backend,
            owns_process_group=owns_group,
            object_group=object_group,
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def _collective_device(self) -> torch.device:
        return (
            self.device
            if self.backend in ("nccl", "hccl")
            else torch.device("cpu")
        )

    def reduce_int(self, value: int, *, operation: str = "sum") -> int:
        if not self.enabled:
            return int(value)
        tensor = torch.tensor(
            int(value), dtype=torch.int64, device=self._collective_device()
        )
        operations = {
            "sum": dist.ReduceOp.SUM,
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
        }
        dist.all_reduce(tensor, op=operations[operation])
        return int(tensor.item())

    def reduce_float(self, value: float, *, operation: str = "mean") -> float:
        if not self.enabled:
            return float(value)
        # Ascend 910/HCCL does not support FP64 collectives.  Metrics do not
        # need double precision, so accelerator reductions use FP32.
        dtype = (
            torch.float32
            if self.backend in ("nccl", "hccl")
            else torch.float64
        )
        tensor = torch.tensor(
            float(value), dtype=dtype, device=self._collective_device()
        )
        if operation == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
        elif operation == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif operation == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        elif operation == "min":
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"unsupported reduction operation: {operation}")
        return float(tensor.item())

    def mean_metrics(self, values: Mapping[str, float]) -> dict[str, float]:
        return {
            key: self.reduce_float(value, operation="mean")
            for key, value in values.items()
        }

    def sum_metrics(self, values: Mapping[str, float]) -> dict[str, float]:
        """Sum scalar moments in one collective; all ranks supply the same keys."""
        if not self.enabled or not values:
            return dict(values)
        keys = sorted(values)
        tensor = torch.tensor(
            [values[key] for key in keys],
            dtype=torch.float32 if self.backend in ("nccl", "hccl") else torch.float64,
            device=self._collective_device(),
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return dict(zip(keys, tensor.tolist(), strict=True))

    def any(self, value: bool) -> bool:
        return bool(self.reduce_int(int(value), operation="max"))

    def gather_object(self, value: Any) -> list[Any] | None:
        if not self.enabled:
            return [value]
        gathered: list[Any] | None = (
            [None for _ in range(self.world_size)] if self.primary else None
        )
        dist.gather_object(value, gathered, dst=0, group=self.object_group)
        return gathered

    def broadcast_object(self, value: Any, *, source: int = 0) -> Any:
        if not self.enabled:
            return value
        values = [value if self.rank == source else None]
        dist.broadcast_object_list(values, src=source, group=self.object_group)
        return values[0]

    def broadcast_module(self, module: nn.Module, *, source: int = 0) -> None:
        if not self.enabled:
            return
        for parameter in module.parameters():
            dist.broadcast(parameter.data, src=source)
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=source)

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            if self.object_group is not None:
                dist.destroy_process_group(self.object_group)
                self.object_group = None
            dist.destroy_process_group()
            self.owns_process_group = False
