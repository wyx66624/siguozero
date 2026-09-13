"""Reuse the warm-up stream across model versions and capture shapes.

PyTorch 2.11 retains a cuBLAS workspace per handle/stream. Creating a fresh
stream for every graph slowly fills the stream pool with persistent workspaces.
No graph or model tensor is retained here; each inference thread/device owns
one stream, reused for both temporal decoding and categorical sampling.
"""
from __future__ import annotations

import threading

import torch


_local = threading.local()


def warmup_stream(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA graph warm-up requires a CUDA device")
    index = torch.cuda.current_device() if device.index is None else device.index
    if not hasattr(_local, "streams"):
        _local.streams = {}
    if index not in _local.streams:
        _local.streams[index] = torch.cuda.Stream(device=index)
    return _local.streams[index]
