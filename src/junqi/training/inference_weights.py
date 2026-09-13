"""Coalesced immutable inference weights, shared by rollout and evaluation."""
from __future__ import annotations

import torch


def _matrix_packs(module, packs, dtype):
    """Precast just autocast's linear operands; retain embeddings/norms in FP32."""
    matrices = set()
    for prefix, child in module.named_modules():
        names = ("weight", "bias") if isinstance(child, torch.nn.Linear) else (
            ("in_proj_weight", "in_proj_bias") if isinstance(child, torch.nn.MultiheadAttention) else ())
        for name in names:
            if getattr(child, name, None) is not None:
                matrices.add(f"{prefix}.{name}" if prefix else name)
    groups = {}
    for pack in packs.values():
        for name, (start, end, shape) in pack["views"].items():
            value = pack["flat"][start:end].view(shape)
            target_dtype = dtype if name in matrices else value.dtype
            groups.setdefault(target_dtype, []).append((name, value))
    result = {}
    for target_dtype, items in groups.items():
        # Preserve tensor-core-friendly alignment even after odd-sized heads.
        offsets, total = {}, 0
        for name, value in items:
            total = (total + 127) // 128 * 128
            offsets[name] = (total, total + value.numel(), tuple(value.shape))
            total += value.numel()
        flat = torch.empty(total, dtype=target_dtype)
        for name, value in items:
            start, end, shape = offsets[name]
            flat[start:end].view(shape).copy_(value)
        result[str(target_dtype)] = dict(flat=flat, views=offsets)
    return result


def install_packed_weights(module, packs, device, *, slabs=None, matrix_dtype=None):
    """Upload one contiguous tensor per dtype; reuse installed storage on swaps.

    The module has no optimizer. Its Parameter objects alias these slabs after
    the first installation. Callers must invalidate model-specific caches first.
    """
    if matrix_dtype is not None:
        if matrix_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("frozen linear weights require an existing AMP dtype")
        packs = _matrix_packs(module, packs, matrix_dtype)
    first = slabs is None
    slabs = {} if first else slabs
    state, uploaded = {}, 0
    for dtype, pack in packs.items():
        source = pack["flat"]
        if first:
            slabs[dtype] = torch.empty(source.shape, dtype=source.dtype, device=device)
        target = slabs[dtype]
        if target.shape != source.shape or target.dtype != source.dtype:
            raise ValueError("historical weight slab architecture changed")
        target.copy_(source)
        uploaded += source.numel() * source.element_size()
        state.update({name: target[start:end].view(shape)
                      for name, (start, end, shape) in pack["views"].items()})
    if first:
        module.load_state_dict(state, strict=True, assign=True)
        module.to(device)  # also moves nonpersistent layout-rule buffers
    return slabs, uploaded, len(packs)
