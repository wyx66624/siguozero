"""Read-only monitoring and explicit checkpoint catalog. Does not import torch."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time

from ..training.checkpoint_format import CHECKPOINT_FORMAT_VERSION

MODE_NAMES = {"four_dark": "四暗棋", "double_open": "双明棋", "two_player": "双人军棋"}


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def tail_records(path, limit=120):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            offset = max(0, stream.tell() - 4 * 1024 * 1024)
            stream.seek(offset)
            if offset:
                stream.readline()
            lines = stream.read().splitlines()[-limit:]
        records = []
        for line in lines:
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
            except ValueError:
                continue  # A writer may not yet have completed its last line.
        return records
    except OSError:
        return []


def process_info(pid):
    """Check Linux process identity and creation time, including PID reuse."""
    try:
        proc = Path("/proc") / str(int(pid))
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
        return {"pid": int(pid), "command": command,
                "started": boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")}
    except (OSError, ValueError, IndexError, StopIteration, TypeError):
        return None


def training_process(pid, run, heartbeat_time=None):
    info = process_info(pid)
    if not info:
        return None
    command = info["command"]
    modules = ("junqi.training.train_", "junqi.training.cli")
    paths = (str(run), str(run.parent), str(run.parent.parent))
    if not any(m in command for m in modules) or not any(p in command for p in paths):
        return None
    if heartbeat_time is not None and heartbeat_time < info["started"] - 2:
        return None
    return info


def clean_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_numbers(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_numbers(v) for v in value]
    return value


class Monitor:
    def __init__(self, config, root):
        self.root = Path(root).resolve()
        self.specs = []
        for spec in config["runs"]:
            path = (self.root / spec["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("run directories must be inside the project")
            self.specs.append({**spec, "path": path})

    def catalog(self):
        entries = []
        for spec in self.specs:
            run = spec["path"]
            selection = read_json(run / "model_selection/state.json")
            resume = read_json(run / "checkpoints/manifest.json")
            live = read_json(run / "inference/manifest.json")
            candidates = []
            if selection.get("best_snapshot"):
                path = (run / "model_selection" / selection["best_snapshot"]).resolve()
                proven = bool(selection.get("rounds"))
                candidates.append(("best", path, selection.get("best_update"),
                                   "评测最优" if proven else "评测初始基线", proven))
            if live.get("file"):
                candidates.append(("live", (run / "inference" / live["file"]).resolve(),
                                   live.get("update"), "最新训练快照", False))
            candidates.append(("latest", run / "checkpoints/latest.pt", resume.get("update"), "最近可恢复检查点", False))
            for kind, path, update, label, proven in candidates:
                if not path.is_relative_to(run) or not path.is_file():
                    continue
                stat = path.stat()
                version = (live if kind == "live" else resume).get("format_version")
                available = version == CHECKPOINT_FORMAT_VERSION
                entries.append({"id": f"{spec['id']}:{kind}", "run_id": spec["id"],
                                "run_name": spec["name"], "mode": spec["mode"],
                                "dead_rules_enabled": spec.get("dead_rules_enabled", True),
                                "update": update, "kind": kind, "label": label,
                                "evaluated": proven, "experimental": spec.get("experimental", False),
                                "available": available, "format_version": version,
                                "unavailable_reason": None if available else f"检查点格式 v{version or '?'}，当前网络需要 v{CHECKPOINT_FORMAT_VERSION}",
                                "modified": stat.st_mtime, "size_bytes": stat.st_size,
                                "path": str(path.relative_to(self.root)), "_path": path})
        return entries

    def public_catalog(self):
        return [{k: v for k, v in item.items() if not k.startswith("_")} for item in self.catalog()]

    def resolve_checkpoint(self, identifier):
        for entry in self.catalog():
            if entry["id"] == identifier:
                if not entry["available"]:
                    raise ValueError(entry["unavailable_reason"])
                return entry
        raise ValueError("检查点尚未就绪或已更新，请刷新模型列表")

    def run_status(self, spec):
        run, now = spec["path"], time.time()
        settings = read_json(run / "resolved_config.json")
        latest = read_json(run / "latest_metrics.json")
        history = tail_records(run / "metrics.jsonl")
        resources = [read_json(p) for p in sorted(run.glob("resource_latest*.json"))]
        fresh, processes = [], []
        timeout = max(90, settings.get("resource_monitor_interval_seconds", 30) * 3)
        for resource in resources:
            timestamp = resource.get("timestamp_unix", 0)
            info = training_process(resource.get("process/pid"), run, timestamp)
            if info:
                processes.append(info)
                if now - timestamp < timeout:
                    fresh.append(resource)
        launch = read_json(self.root / "output/local_console" / f"training-{spec['id']}.json")
        launching = training_process(launch.get("pid"), run)
        if launching and not processes:
            processes.append(launching)
        heartbeat = fresh[0] if fresh else (resources[0] if resources else {})
        counter = settings.get("step_budget_counter", latest.get("training/step_budget_counter"))
        counter = counter or ("environment_plies" if spec["mode"] != "two_player" else "continuation_plies")
        target_key = "target_" + counter
        target = settings.get(target_key) or latest.get("training/" + target_key) or 3_000_000_000
        completed = max(latest.get("cumulative/" + counter, 0), heartbeat.get("training/" + counter, 0))
        world = int(heartbeat.get("distributed/world_size", latest.get("distributed/world_size", 1)))
        status = "stopped"
        if processes:
            status = "running" if len(fresh) >= world else "stale"
            if not resources or (launching and now - launching["started"] < 120 and not fresh):
                status = "starting"
        elif completed >= target > 0:
            status = "completed"
        started = min((p["started"] for p in processes), default=None)
        current = [r for r in history if started is not None and r.get("timestamp_unix", 0) >= started]
        window = current[-7:]
        cycle_rate = None
        if len(window) >= 2:
            delta = window[-1].get("cumulative/" + counter, 0) - window[0].get("cumulative/" + counter, 0)
            elapsed = window[-1]["timestamp_unix"] - window[0]["timestamp_unix"]
            if delta > 0 and elapsed > 0:
                cycle_rate = delta / elapsed
        if status != "running":
            cycle_rate = None
        inner = current[-6:]
        seconds = sum(r.get("timing/update_seconds", 0) for r in inner)
        steps = sum(r.get("rollout/" + counter, 0) for r in inner)
        remaining = max(0, target - completed)
        checkpoint = read_json(run / "checkpoints/manifest.json")
        selection = read_json(run / "model_selection/state.json")
        loss_keys = ("loss/policy_total", "loss/critic_total", "loss/layout_total")
        chart = [{"update": r.get("update"), **{k: r.get(k) for k in loss_keys}} for r in history]
        try:
            with (run / "train.log").open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 20000))
                logs = stream.read().decode("utf-8", errors="replace").splitlines()[-14:]
        except OSError:
            logs = []
        return clean_numbers({"id": spec["id"], "name": spec["name"], "mode": spec["mode"],
                "experimental": spec.get("experimental", False), "status": status,
                "phase": heartbeat.get("training/phase", "waiting"),
                "algorithm": settings.get("algorithm", latest.get("algorithm", "grpo" if spec["mode"] == "two_player" else "ppo")),
                "update": max(latest.get("update", 0), heartbeat.get("training/update", 0)),
                "checkpoint_update": checkpoint.get("update"), "best_update": selection.get("best_update"),
                "evaluated_rounds": len(selection.get("rounds", [])), "counter": counter,
                "steps": completed, "target": target, "progress": min(100, 100 * completed / target),
                "processes": [{"pid": p["pid"], "started": p["started"]} for p in processes],
                "heartbeat_count": len(fresh), "world_size": world,
                "heartbeat_age": max((now - r["timestamp_unix"] for r in fresh), default=None),
                "cycle_rate": cycle_rate, "inner_rate": steps / seconds if seconds > 0 else None,
                "rate_updates": max(0, len(window) - 1), "eta_seconds": remaining / cycle_rate if cycle_rate else None,
                "required_rate_20d": remaining / (20 * 86400),
                "metrics_age": now - latest["timestamp_unix"] if latest else None,
                "metrics": latest, "resource": heartbeat if fresh else {}, "chart": chart,
                "checkpoint_policy": settings.get("checkpoint_policy"), "logs": logs,
                "path": str(run.relative_to(self.root))})

    def status(self):
        gpu = None
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                fields = result.stdout.strip().splitlines()[0].split(",")
                gpu = dict(zip(("name", "utilization", "memory_used", "memory_total", "temperature", "power"), [v.strip() for v in fields]))
        except (OSError, subprocess.TimeoutExpired, IndexError):
            pass
        return {"timestamp": time.time(), "runs": [self.run_status(s) for s in self.specs], "gpu": gpu}
