"""CUDA/NPU compatibility helpers for the PyTorch 2.7 runtime.

The project intentionally avoids the newer ``torch.accelerator`` facade so the
same code runs on the cluster's PyTorch 2.7 + torch_npu stack.  NPU support is
loaded lazily: importing the rules package on a CPU/CUDA host does not require
``torch_npu`` to be installed.
"""

from __future__ import annotations

import importlib
import re
import sys
from types import ModuleType
from typing import Any

import torch


ACCELERATOR_DEVICE_TYPES = frozenset(("cuda", "npu"))
_NPU_MODULE: ModuleType | None = None


def _version_triplet(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def load_torch_npu(*, required: bool) -> ModuleType | None:
    """Import torch_npu once and expose a useful error for explicit NPU use."""

    global _NPU_MODULE
    if _NPU_MODULE is not None:
        return _NPU_MODULE
    try:
        _NPU_MODULE = importlib.import_module("torch_npu")
    except (ImportError, ModuleNotFoundError, OSError) as error:
        if required:
            raise RuntimeError(
                "NPU was requested but torch_npu could not be loaded; install a "
                "torch_npu 2.7.1 wheel matching the cluster CANN release"
            ) from error
        return None
    return _NPU_MODULE


def validate_npu_runtime() -> None:
    """Enforce the deliberately narrow, cluster-supported NPU environment."""

    torch_npu = load_torch_npu(required=True)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "the npu branch supports Python 3.12 exactly; found "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    if _version_triplet(torch.__version__) != (2, 7, 1):
        raise RuntimeError(
            "the npu branch supports PyTorch 2.7.1 exactly; found "
            f"{torch.__version__}"
        )
    npu_version = getattr(torch_npu, "__version__", "unknown")
    if _version_triplet(str(npu_version)) != (2, 7, 1):
        raise RuntimeError(
            "torch_npu must be a 2.7.1 build matching PyTorch 2.7.1; found "
            f"{npu_version}"
        )
    try:
        numpy = importlib.import_module("numpy")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("NPU training requires NumPy 1.26.x (<2)") from error
    numpy_major = int(str(numpy.__version__).split(".", 1)[0])
    if numpy_major >= 2:
        raise RuntimeError(
            f"NPU training requires NumPy <2; found {numpy.__version__}"
        )


def accelerator_module(device_type: str, *, required: bool = True) -> Any:
    if device_type == "cuda":
        return torch.cuda
    if device_type == "npu":
        load_torch_npu(required=required)
        module = getattr(torch, "npu", None)
        if module is None and required:
            raise RuntimeError("torch_npu loaded but torch.npu is not registered")
        return module
    if required:
        raise ValueError(f"{device_type!r} is not an accelerator device type")
    return None


def accelerator_available(device_type: str) -> bool:
    module = accelerator_module(device_type, required=False)
    if module is None:
        return False
    try:
        return bool(module.is_available())
    except (RuntimeError, OSError):
        return False


def resolve_device(requested: str, *, validate_npu: bool = True) -> torch.device:
    """Resolve ``auto`` and validate explicit accelerator requests."""

    if requested == "auto":
        if accelerator_available("npu"):
            if validate_npu:
                validate_npu_runtime()
            return torch.device("npu")
        if accelerator_available("cuda"):
            return torch.device("cuda")
        return torch.device("cpu")

    requested_type = requested.split(":", 1)[0]
    if requested_type == "npu":
        load_torch_npu(required=True)
    device = torch.device(requested)
    if device.type in ACCELERATOR_DEVICE_TYPES:
        if not accelerator_available(device.type):
            raise RuntimeError(
                f"{device.type.upper()} was requested but is not available"
            )
        if device.type == "npu" and validate_npu:
            validate_npu_runtime()
    return device


def is_accelerator(device: torch.device | str) -> bool:
    return torch.device(device).type in ACCELERATOR_DEVICE_TYPES


def set_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if not is_accelerator(resolved):
        return resolved
    module = accelerator_module(resolved.type)
    index = module.current_device() if resolved.index is None else resolved.index
    indexed = torch.device(resolved.type, index)
    module.set_device(indexed)
    return indexed


def device_count(device_type: str) -> int:
    module = accelerator_module(device_type)
    return int(module.device_count())


def manual_seed_all(device_type: str, seed: int) -> None:
    accelerator_module(device_type).manual_seed_all(seed)


def is_bf16_supported(device: torch.device | str) -> bool:
    resolved = torch.device(device)
    if not is_accelerator(resolved):
        return False
    module = accelerator_module(resolved.type)
    checker = getattr(module, "is_bf16_supported", None)
    if checker is not None:
        return bool(checker())
    try:
        torch.empty(1, dtype=torch.bfloat16, device=resolved)
    except (RuntimeError, TypeError):
        return False
    return True


def make_grad_scaler(device_type: str, *, enabled: bool) -> Any:
    """Construct the PyTorch-2.7 AMP scaler registered by each backend."""

    if device_type == "npu":
        npu_amp = getattr(accelerator_module("npu"), "amp", None)
        scaler = getattr(npu_amp, "GradScaler", None)
        if scaler is not None:
            return scaler(enabled=enabled)
    scaler_device = device_type if device_type in ACCELERATOR_DEVICE_TYPES else "cuda"
    return torch.amp.GradScaler(scaler_device, enabled=enabled)


def empty_cache(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if is_accelerator(resolved):
        accelerator_module(resolved.type).empty_cache()


def trim_cuda_cache(device: torch.device | str) -> int:
    """Release unused allocator blocks at a high memory watermark.

    Live tensors, KV slots and captured graph addresses remain reserved.
    Call only at an inference-wave boundary, never during graph capture.
    """
    resolved = torch.device(device)
    if resolved.type != 'cuda':
        return 0
    before = torch.cuda.memory_reserved(resolved)
    total = torch.cuda.get_device_properties(resolved).total_memory
    if (before <= .85 * total
            or before - torch.cuda.memory_allocated(resolved) < 1024**3):
        return 0
    torch.cuda.empty_cache()
    return max(0, before - torch.cuda.memory_reserved(resolved))


def reset_peak_memory_stats(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if is_accelerator(resolved):
        accelerator_module(resolved.type).reset_peak_memory_stats(resolved)


def synchronize(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if is_accelerator(resolved):
        accelerator_module(resolved.type).synchronize(resolved)


def get_device_name(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if not is_accelerator(resolved):
        return resolved.type
    return str(accelerator_module(resolved.type).get_device_name(resolved))


def memory_allocated(device: torch.device | str) -> int:
    resolved = torch.device(device)
    return int(accelerator_module(resolved.type).memory_allocated(resolved))


def memory_reserved(device: torch.device | str) -> int:
    resolved = torch.device(device)
    return int(accelerator_module(resolved.type).memory_reserved(resolved))


def max_memory_allocated(device: torch.device | str) -> int:
    resolved = torch.device(device)
    return int(accelerator_module(resolved.type).max_memory_allocated(resolved))


def max_memory_reserved(device: torch.device | str) -> int:
    resolved = torch.device(device)
    return int(accelerator_module(resolved.type).max_memory_reserved(resolved))


def is_out_of_memory(error: BaseException, device_type: str) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    if device_type != "npu" or not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return "out of memory" in message or "npu out of memory" in message


def supports_pinned_memory(device: torch.device | str) -> bool:
    # PyTorch 2.7's default pin allocator is CUDA-specific.  NPU copies remain
    # correct without it, whereas requesting the wrong pinned allocator fails.
    return torch.device(device).type == "cuda"
