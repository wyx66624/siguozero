"""Detailed append-only JSONL, console, and optional TensorBoard metrics."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Callable, Mapping

import torch


class MetricLogger:
    def __init__(
        self,
        run_directory: str | Path,
        *,
        device: torch.device | None = None,
        rank: int = 0,
        write_training_metrics: bool = True,
    ) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.device = device
        self.rank = rank
        self.write_training_metrics = write_training_metrics
        self.run_directory.mkdir(parents=True, exist_ok=True)
        suffix = "" if rank == 0 else f".rank{rank:03d}"
        self.jsonl_path = self.run_directory / f"metrics{suffix}.jsonl"
        self.latest_path = self.run_directory / f"latest_metrics{suffix}.json"
        self.logger = logging.getLogger(
            f"siguozero.{abs(hash(str(self.run_directory)))}.rank{rank}"
        )
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            formatter = logging.Formatter(
                f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s"
            )
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            file_handler = logging.FileHandler(
                self.run_directory / f"train{suffix}.log", encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(console)
            self.logger.addHandler(file_handler)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = (
                SummaryWriter(self.run_directory / "tensorboard")
                if write_training_metrics
                else None
            )
        except (ImportError, ModuleNotFoundError):
            self.writer = None
        self._resource_monitor: _ResourceMonitor | None = None

    def start_resource_monitor(
        self,
        status_getter: Callable[[], Mapping[str, float | int | str]],
        *,
        interval_seconds: float,
    ) -> None:
        if self._resource_monitor is not None:
            raise RuntimeError("resource monitor is already running")
        self._resource_monitor = _ResourceMonitor(
            self.run_directory,
            self.logger,
            self.device,
            status_getter,
            interval_seconds=interval_seconds,
            rank=self.rank,
        )
        self._resource_monitor.start()

    def log(self, update: int, values: Mapping[str, float | int | str]) -> None:
        if not self.write_training_metrics:
            return
        record: dict[str, float | int | str] = {
            "update": update,
            "timestamp_unix": time.time(),
            **values,
        }
        if (
            self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            device = (
                torch.cuda.current_device()
                if self.device.index is None
                else self.device.index
            )
            record.update(
                {
                    "gpu/name": torch.cuda.get_device_name(device),
                    "gpu/memory_allocated_gb": torch.cuda.memory_allocated(device)
                    / 2**30,
                    "gpu/memory_reserved_gb": torch.cuda.memory_reserved(device)
                    / 2**30,
                    "gpu/max_memory_allocated_gb": torch.cuda.max_memory_allocated(
                        device
                    )
                    / 2**30,
                }
            )
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        suffix = "" if self.rank == 0 else f".rank{self.rank:03d}"
        temporary = self.run_directory / f".latest_metrics{suffix}.json.tmp"
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.latest_path)
        if self.writer is not None:
            for key, value in record.items():
                if isinstance(value, (float, int)) and key not in (
                    "update",
                    "timestamp_unix",
                ):
                    self.writer.add_scalar(key, value, update)
            self.writer.flush()
        compact = {
            key: round(value, 5) if isinstance(value, float) else value
            for key, value in record.items()
            if key
            in {
                "loss/policy_total",
                "loss/layout_total",
                "policy/kl_reference",
                "policy/entropy",
                "rollout/draws",
                "rollout/wins",
                "rollout/losses",
                "timing/update_seconds",
                "optimizer/policy_lr",
            }
        }
        self.logger.info("update=%d metrics=%s", update, compact)

    def event(self, message: str) -> None:
        self.logger.info(message)

    def close(self) -> None:
        if self._resource_monitor is not None:
            self._resource_monitor.close()
            self._resource_monitor = None
        if self.writer is not None:
            self.writer.close()
        for handler in tuple(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


class _ResourceMonitor:
    """Low-overhead resource heartbeat independent of learner updates."""

    def __init__(
        self,
        run_directory: Path,
        logger: logging.Logger,
        device: torch.device | None,
        status_getter: Callable[[], Mapping[str, float | int | str]],
        *,
        interval_seconds: float,
        rank: int,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource monitor interval must be positive")
        self.run_directory = run_directory
        self.logger = logger
        self.device = device
        self.status_getter = status_getter
        self.interval_seconds = interval_seconds
        suffix = "" if rank == 0 else f".rank{rank:03d}"
        self.jsonl_path = run_directory / f"resource_metrics{suffix}.jsonl"
        self.latest_path = run_directory / f"resource_latest{suffix}.json"
        self.temporary_path = run_directory / f".resource_latest{suffix}.json.tmp"
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="siguozero-resource-monitor",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=min(self.interval_seconds + 1.0, 5.0))

    def _gpu_index(self) -> int | None:
        if (
            self.device is None
            or self.device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return None
        return (
            torch.cuda.current_device()
            if self.device.index is None
            else self.device.index
        )

    def _nvidia_smi(self, gpu_index: int) -> dict[str, float]:
        command = [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return {}
        if completed.returncode != 0 or not completed.stdout.strip():
            return {}
        fields = completed.stdout.strip().splitlines()[0].split(",")
        if len(fields) != 6:
            return {}
        try:
            values = [float(field.strip()) for field in fields]
        except ValueError:
            return {}
        return {
            "gpu/utilization_percent": values[0],
            "gpu/device_memory_used_mib": values[1],
            "gpu/device_memory_total_mib": values[2],
            "gpu/temperature_c": values[3],
            "gpu/power_w": values[4],
            "gpu/power_limit_w": values[5],
        }

    def _sample(self) -> dict[str, float | int | str]:
        disk = shutil.disk_usage(self.run_directory)
        record: dict[str, float | int | str] = {
            "timestamp_unix": time.time(),
            "process/pid": os.getpid(),
            "disk/free_gib": disk.free / 2**30,
            "disk/used_gib": disk.used / 2**30,
            **self.status_getter(),
        }
        gpu_index = self._gpu_index()
        if gpu_index is not None:
            record["gpu/index"] = gpu_index
            record.update(self._nvidia_smi(gpu_index))
            record.update(
                {
                    "gpu/process_allocated_gib": torch.cuda.memory_allocated(
                        gpu_index
                    )
                    / 2**30,
                    "gpu/process_reserved_gib": torch.cuda.memory_reserved(
                        gpu_index
                    )
                    / 2**30,
                    "gpu/process_peak_allocated_gib": (
                        torch.cuda.max_memory_allocated(gpu_index) / 2**30
                    ),
                }
            )
        return record

    def _write(self, record: Mapping[str, float | int | str]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.temporary_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(self.temporary_path, self.latest_path)

    def _warn_if_needed(
        self, record: Mapping[str, float | int | str]
    ) -> None:
        used = float(record.get("gpu/device_memory_used_mib", 0.0))
        total = float(record.get("gpu/device_memory_total_mib", 0.0))
        if total and used / total >= 0.95:
            self.logger.warning(
                "GPU memory pressure %.1f/%.1f MiB (%.1f%%)",
                used,
                total,
                100.0 * used / total,
            )
        temperature = float(record.get("gpu/temperature_c", 0.0))
        if temperature >= 85.0:
            self.logger.warning("GPU temperature is %.1f C", temperature)
        disk_free = float(record["disk/free_gib"])
        if disk_free < 10.0:
            self.logger.warning("checkpoint disk has only %.2f GiB free", disk_free)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                record = self._sample()
                self._write(record)
                self._warn_if_needed(record)
            except Exception as error:  # monitoring must never stop training
                self.logger.warning("resource monitor sample failed: %s", error)
            self._stop.wait(self.interval_seconds)
