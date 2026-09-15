"""Bounded, disposable RAM cache for immutable historical inference weights.

Keep already converted CPU slabs, not hundreds of nn.Modules or optimizers.
One worker reads/prepares the next selected cohort; only that version is pinned.
No worker touches a CUDA stream, model parameters, or any random generator.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from pathlib import Path
import threading
import time

import torch

from .arena import sha256_file
from .checkpoint_format import CHECKPOINT_FORMAT_VERSION
from .inference_weights import prepare_matrix_packs


def available_memory_bytes():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_phys", "avail_phys", "total_page", "avail_page",
                    "total_virtual", "avail_virtual", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
        return 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0  # Unknown capacity disables retention, not inference.


def pack_bytes(packs):
    return sum(p["flat"].numel() * p["flat"].element_size()
               for model in packs.values() for p in model.values())


class HistoricalWeightCache:
    def __init__(self, directory, settings, *, matrix_names, matrix_dtype=None, pin=False):
        self.directory, self.settings = Path(directory), settings
        self.matrix_names, self.matrix_dtype = frozenset(matrix_names), matrix_dtype
        self.maximum = int(settings.historical_cache_gib * 2**30)
        self.reserve = int(settings.historical_cache_reserve_gib * 2**30)
        self.pin_limit = int(settings.historical_pinned_mib * 2**20) if pin else 0
        self.items = OrderedDict()
        self.bytes = self.effective_limit = 0
        self.hits = self.misses = self.reads = self.evictions = 0
        self.read_seconds = self.prepare_seconds = self.wait_seconds = 0.
        self.pinned_bytes = self.pin_failures = self.prefetch_count = 0
        self._lock = threading.RLock()
        self._worker = None
        self._pending = None
        self._staged = None
        self._closed = False

    def _trim_locked(self, incoming=0):
        # available excludes our live tensors; adding them back prevents a cache
        # from shrinking merely because it successfully populated itself.
        self.effective_limit = min(self.maximum, max(0, self.bytes + available_memory_bytes() - self.reserve))
        while self.items and self.bytes + incoming > self.effective_limit:
            _, packs = self.items.popitem(last=False)
            self.bytes -= pack_bytes(packs)
            self.evictions += 1

    def trim(self):
        with self._lock:
            self._trim_locked()

    def _prepare(self, payload):
        if (payload.get("mode") != self.settings.mode.value
                or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION
                or payload.get("dead_rules_enabled") != self.settings.dead_rules_enabled
                or payload.get("config", {}).get("no_capture_draw_plies") != self.settings.no_capture_draw_plies):
            raise ValueError("historical opponent checkpoint contract mismatch")
        started = time.perf_counter()
        result = {}
        for name, packs in payload["packed_weights"].items():
            if name == "policy" and self.matrix_dtype is not None:
                result[name] = prepare_matrix_packs(packs, self.matrix_names, self.matrix_dtype)
            else:
                # Own the storage; retaining mmap handles would defer disk page
                # faults to inference and make retained memory unaccountable.
                result[name] = {key: dict(flat=p["flat"].clone(), views=p["views"])
                                for key, p in packs.items()}
        if set(result) != {"policy", "layout"}:
            raise ValueError("historical cache requires policy and layout only")
        with self._lock:
            self.prepare_seconds += time.perf_counter() - started
        return result

    def _remember(self, key, packs):
        size = pack_bytes(packs)
        with self._lock:
            if key in self.items:
                previous = self.items.pop(key)
                self.bytes -= pack_bytes(previous)
            self._trim_locked(size)
            if size <= self.effective_limit:
                self.items[key] = packs
                self.bytes += size

    def remember_payload(self, entry, payload):
        """New archives are already on the CPU: no read-back or second D2H."""
        if self.maximum:
            self._remember(entry["sha256"], self._prepare(payload))

    def _get(self, entry):
        key = entry["sha256"]
        with self._lock:
            self._trim_locked()
            if key in self.items:
                self.hits += 1
                self.items.move_to_end(key)
                return self.items[key]
            self.misses += 1
        started = time.perf_counter()
        path = self.directory / entry["file"]
        if sha256_file(path) != key:
            raise ValueError("frozen historical snapshot hash changed")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        packs = self._prepare(payload)
        del payload
        with self._lock:
            self.reads += 1
            self.read_seconds += time.perf_counter() - started
        self._remember(key, packs)
        return packs

    def _stage(self, entry):
        packs = self._get(entry)
        if self.pin_limit and pack_bytes(packs) <= self.pin_limit:
            try:
                packs = {name: {key: dict(flat=p["flat"].pin_memory(), views=p["views"])
                                for key, p in model.items()} for name, model in packs.items()}
            except RuntimeError:
                # Pinning is an optimization; pageable RAM remains correct.
                with self._lock:
                    self.pin_failures += 1
        with self._lock:
            self.pinned_bytes = sum(p["flat"].numel() * p["flat"].element_size()
                for model in packs.values() for p in model.values() if p["flat"].is_pinned())
        return packs

    def prefetch(self, entry):
        if self._closed or not self.maximum or self._pending is not None:
            return
        if self._worker is None:
            self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="historical-weights")
        self._staged = None
        self.pinned_bytes = 0
        self.prefetch_count += 1
        self._pending = (entry["sha256"], self._worker.submit(self._stage, dict(entry)))

    def get(self, entry):
        started = time.perf_counter()
        ready = None
        if self._pending is not None:
            pending, self._pending = self._pending, None
            ready = pending[1].result()
            if pending[0] != entry["sha256"]:
                ready = None
        self._staged = None
        self._staged = ready if ready is not None else self._stage(entry)
        self.wait_seconds += time.perf_counter() - started
        return self._staged

    def finish_upload(self):
        # Caller waits for its CUDA copy event before releasing pinned sources.
        self._staged = None
        if self._pending is None:
            self.pinned_bytes = 0

    def close(self):
        self._closed = True
        if self._worker is not None:
            self._worker.shutdown(wait=True, cancel_futures=True)
        self._pending = self._staged = None
        self.items.clear()
        self.bytes = self.pinned_bytes = 0

    def metrics(self):
        with self._lock:
            return {"historical/ram_cache_bytes": self.bytes,
                    "historical/ram_cache_limit_bytes": self.maximum,
                    "historical/ram_cache_effective_limit_bytes": self.effective_limit,
                    "historical/ram_cache_models": len(self.items),
                    "historical/ram_cache_hits": self.hits, "historical/ram_cache_misses": self.misses,
                    "historical/ram_cache_evictions": self.evictions,
                    "historical/disk_load_count": self.reads,
                    "historical/disk_load_seconds": self.read_seconds,
                    "historical/cpu_prepare_seconds": self.prepare_seconds,
                    "historical/cache_wait_seconds": self.wait_seconds,
                    "historical/prefetch_count": self.prefetch_count,
                    "historical/pinned_bytes": self.pinned_bytes,
                    "historical/pin_failures": self.pin_failures}
