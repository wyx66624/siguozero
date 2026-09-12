"""Optional atomic, inference-only exports; never used to resume training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

import torch

from .checkpoint import CHECKPOINT_FORMAT_VERSION


def export_inference_snapshot(directory, policy, layout, settings, update):
    directory = Path(directory) / "inference"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"policy_{update:09d}_{uuid.uuid4().hex[:8]}.pt"
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION, "update": update,
        "mode": settings.mode.value, "algorithm": settings.algorithm,
        "dead_rules_enabled": settings.dead_rules_enabled,
        "reason": "live_inference_only", "config": settings.serializable(),
        "policy": {k: v.detach().to("cpu", copy=True) for k, v in policy.state_dict().items()},
        "layout": {k: v.detach().to("cpu", copy=True) for k, v in layout.state_dict().items()},
    }
    temporary = directory / f".{name}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, directory / name)
    manifest = {k: payload[k] for k in ("format_version", "mode", "algorithm", "update", "dead_rules_enabled", "reason")}
    manifest.update(file=name, timestamp_unix=time.time())
    temporary = directory / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, directory / "manifest.json")
    # Readers copy weights into a CPU model; two prior generations cover loads
    # crossing publication. Files are immutable and the manifest commits last.
    older = sorted(directory.glob("policy_*.pt"), key=lambda p: p.stat().st_mtime)
    for path in older[:-3]:
        try:
            path.unlink()
        except OSError:
            pass  # A mapped Windows reader can still own this generation.
    return directory / name
