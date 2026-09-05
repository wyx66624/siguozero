"""Small torch.distributed facade used by the self-play trainer.

The training process uses one process per GPU.  Each process owns an
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
        requested = torch.device(
            "cuda" if requested_device == "auto" and torch.cuda.is_available()
            else "cpu" if requested_device == "auto"
            else requested_device
        )

        if world_size <= 1:
            if requested.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but torch.cuda.is_available() is false"
                )
            device = requested
            if device.type == "cuda":
                device = torch.device(
                    "cuda",
                    torch.cuda.current_device()
                    if device.index is None
                    else device.index,
                )
                torch.cuda.set_device(device)
            return cls(0, 1, 0, device)

        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("torchrun requested CUDA but CUDA is unavailable")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} exceeds visible CUDA device count "
                    f"{torch.cuda.device_count()}"
                )
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            backend = "nccl"
        elif requested.type == "cpu":
            device = torch.device("cpu")
            backend = "gloo"
        else:
            raise ValueError("distributed training supports only CUDA or CPU devices")

        owns_group = False
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
            owns_group = True
        # Python environment/checkpoint objects can become large.  A secondary
        # Gloo group keeps their serialization on CPU instead of allocating
        # byte tensors in scarce GPU memory under NCCL.
        object_group = dist.new_group(backend="gloo") if backend == "nccl" else None
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
        return self.device if self.backend == "nccl" else torch.device("cpu")

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
        tensor = torch.tensor(
            float(value), dtype=torch.float64, device=self._collective_device()
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
