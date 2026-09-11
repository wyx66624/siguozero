"""Independent, restartable historical arena. Never starts/stops training.

Use --once in a scheduler, or --watch to poll completed checkpoints. All nodes
of a torchrun job must share the checkpoint AND output directories.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

import torch

from .arena import (
    ARENA_VERSION, MatchSettings, atomic_json, run_match, sha256_file, synchronized_error,
)
from .checkpoint import CHECKPOINT_FORMAT_VERSION
from .distributed import DistributedContext
from .models import ModelConfig
from .modes import TrainingMode, normalize_mode


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def protocol_fingerprint() -> str:
    """Do not silently combine scores across edited rules/encoders/evaluators."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "board.py", "pieces.py", "game.py", "training/encoding.py", "training/models.py",
        "training/modes.py", "training/accelerator.py", "training/paged_kv.py",
        "training/rollout.py", "training/inference.py", "training/arena.py",
        "training/evaluate_history.py", "training/distributed.py",
        "training/arena_two_player.py", "training/arena_four_player.py",
        "training/evaluate_two_player.py", "training/evaluate_four_player.py",
    ):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


class PublicationInProgress(RuntimeError):
    """A latest/manifest atomic-publication boundary; retry without accepting it."""


class OutputLock:
    """OS-released advisory lock: no stale PID lock after crashes.

    Lock file intentionally remains on disk. Removing a live lock file would
    allow two processes to lock different inodes for the same output directory.
    """

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / ".arena.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.stream.tell() == 0:
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError("another evaluator owns this output directory") from exc

    def close(self) -> None:
        if not self.stream.closed:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()


def read_manifest(
    checkpoint_dir: Path, *, mode: TrainingMode | str = TrainingMode.TWO_PLAYER,
) -> dict[str, Any]:
    mode = normalize_mode(mode).value
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION
            or manifest.get("latest") != "latest.pt"
            or manifest.get("mode") != mode
            or type(manifest.get("dead_rules_enabled")) is not bool
            or type(manifest.get("update")) is not int or manifest["update"] < 0):
        raise ValueError(f"invalid {mode} checkpoint manifest (cannot mix modes)")
    return manifest


def pin_checkpoint(
    source: Path, output_dir: Path, *, manifest: dict[str, Any] | None = None,
    mode: TrainingMode | str = TrainingMode.TWO_PLAYER,
) -> dict[str, Any]:
    """Read-only CPU/mmap source -> immutable inference-only Policy+Layout.

    The source is trusted local PyTorch data, not a downloaded pickle. No
    optimizer/RNG/base-game-pool state is copied into evaluation snapshots.
    """
    mode = normalize_mode(mode).value
    before = source.stat()
    payload = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    if (payload.get("format_version") != CHECKPOINT_FORMAT_VERSION
            or payload.get("mode") != mode
            or type(payload.get("dead_rules_enabled")) is not bool
            or type(payload.get("update")) is not int or payload["update"] < 0):
        raise ValueError(f"invalid {mode} checkpoint payload (cannot mix modes)")
    config = ModelConfig(**payload["config"]["model"])
    if config.dead_rules_enabled != payload["dead_rules_enabled"]:
        raise ValueError("model config contradicts checkpoint dead rules")
    if manifest is not None and any(
        payload[key] != manifest[key] for key in ("update", "mode", "dead_rules_enabled")
    ):
        raise PublicationInProgress("latest.pt and manifest differ; retry later")
    for label in ("policy", "layout"):
        tensors = payload[label]
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError(f"missing {label} weights")
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"non-tensor {label}.{name}")
            if tensor.is_floating_point() or tensor.is_complex():
                # Finite-check one tensor at a time, not one full-model mask.
                if not torch.isfinite(tensor).all().item():
                    raise ValueError(f"NaN/Inf in {label}.{name}; evaluation refused")
    after = source.stat()
    if ((before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_ino, after.st_size, after.st_mtime_ns)):
        raise PublicationInProgress("checkpoint changed while being pinned")
    if manifest is not None and read_manifest(source.parent, mode=mode) != manifest:
        raise PublicationInProgress("manifest advanced while being pinned")
    compact = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": "inference_only_not_training_resume",
        "mode": payload["mode"], "dead_rules_enabled": payload["dead_rules_enabled"],
        "update": payload["update"], "config": {"model": asdict(config)},
        "policy": payload["policy"], "layout": payload["layout"],
    }
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    temporary = models_dir / ".snapshot.pt.tmp"
    torch.save(compact, temporary)
    digest = sha256_file(temporary)
    target = models_dir / f"u{payload['update']:09d}_{digest[:16]}.pt"
    if target.exists():
        if sha256_file(target) != digest:
            raise ValueError("snapshot hash collision or corrupt existing snapshot")
        temporary.unlink()
    else:
        temporary.replace(target)
    return {
        "path": target.relative_to(output_dir).as_posix(), "sha256": digest,
        "mode": mode,
        "update": payload["update"], "dead_rules_enabled": payload["dead_rules_enabled"],
        "source": str(source.resolve()), "bytes": target.stat().st_size,
    }


class ArenaStore:
    """Rank-0-only durable suite, pending jobs and bounded owned snapshots."""

    def __init__(self, root: Path, config: dict[str, Any], baseline: Path | None) -> None:
        self.root = root.resolve()
        self.mode = normalize_mode(config["match"].get("mode", "two_player")).value
        self.path = self.root / "state.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state.get("schema_version") != 1 or self.state.get("config") != config:
                raise ValueError("evaluation settings changed; use a new output directory")
            if baseline is not None and str(baseline.resolve()) != self.state["baseline"]["source"]:
                raise ValueError("baseline changed; use a new output directory")
        else:
            if baseline is None:
                raise ValueError("a new arena requires --baseline (a frozen older checkpoint)")
            reference = pin_checkpoint(baseline.resolve(), self.root, mode=self.mode)
            self.state = {
                "schema_version": 1, "config": config, "baseline": reference,
                "created_at": utc_now(), "rounds": [], "snapshots": [reference],
            }
            self.save()

    def save(self) -> None:
        atomic_json(self.path, self.state)

    def model_path(self, snapshot: dict[str, Any]) -> Path:
        path = (self.root / snapshot["path"]).resolve()
        if (path.parent != (self.root / "models").resolve()
                or not re.fullmatch(r"u\d{9,}_[0-9a-f]{16}\.pt", path.name)):
            raise ValueError("unsafe snapshot path in arena state")
        return path

    def verify(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("mode") != self.mode:
            raise ValueError("snapshot belongs to a different evaluation mode")
        path = self.model_path(snapshot)
        if not path.exists() or sha256_file(path) != snapshot["sha256"]:
            raise ValueError(f"missing or corrupt pinned checkpoint: {path}")

    def prepare(self, checkpoint_dir: Path) -> dict[str, Any]:
        for item in self.state["rounds"]:
            if item["status"] == "pending":
                return {"job": item, "resumed": True}
        completed = [r for r in self.state["rounds"] if r["status"] == "complete"]
        previous = completed[-1]["candidate"] if completed else self.state["baseline"]
        manifest = read_manifest(checkpoint_dir, mode=self.mode)
        if manifest["dead_rules_enabled"] != self.state["baseline"]["dead_rules_enabled"]:
            raise ValueError("training dead rules differ from the frozen baseline")
        if manifest["update"] < previous["update"]:
            return {"idle": "checkpoint_rolled_back", "latest_update": manifest["update"]}
        if manifest["update"] - previous["update"] < self.state["config"]["every_updates"]:
            return {"idle": "waiting_for_new_checkpoint", "latest_update": manifest["update"]}
        candidate = pin_checkpoint(checkpoint_dir / "latest.pt", self.root,
                                   manifest=manifest, mode=self.mode)
        recent_count = self.state["config"]["recent_opponents"]
        opponents = [self.state["baseline"]] + [
            r["candidate"] for r in completed[-recent_count:][::-1]
        ] if recent_count else [self.state["baseline"]]
        index = len(self.state["rounds"]) + 1
        settings = dict(self.state["config"]["match"])
        # Fresh non-overlapping seed blocks each round: no repeated peeking at
        # the same random games, while a resumed round replays exactly its seeds.
        settings["seed"] += (index - 1) * MatchSettings(**settings).seed_stride * settings["pairs"]
        MatchSettings(**settings)
        job = {
            "index": index, "status": "pending", "created_at": utc_now(),
            "candidate": candidate, "opponents": opponents, "settings": settings,
            # Sum over rounds 1/[r(r+1)] = 1. Bonferroni over all opponents
            # and rounds spends at most .05 for this unchanged suite.
            "alpha_per_match": 0.05 / (index * (index + 1) * len(opponents)),
        }
        self.state["rounds"].append(job)
        self.state["snapshots"].append(candidate)
        self.save()  # pin the full job before any games or result decisions
        return {"job": job, "resumed": False}

    def complete(self, job: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
        verdicts = [m["verdict"] for m in matches]
        if job["settings"]["smoke_test"]:
            verdict = "smoke_test_not_strength_evidence"
        elif any(v == "behind_opponent" for v in verdicts):
            verdict = "regression_against_historical_opponent"
        elif all(v == "ahead_of_opponent" for v in verdicts):
            verdict = "improvement_against_tested_opponents"
        else:
            verdict = "inconclusive"
        report = {
            "arena_version": ARENA_VERSION, "mode": self.mode, "round": job["index"],
            "completed_at": utc_now(), "candidate": job["candidate"],
            "opponents": job["opponents"], "verdict": verdict, "matches": matches,
            "scope": f"tested historical {self.mode} Policy+Layout teams only; not human Elo",
            "confidence_policy": "grouped Hoeffding; .05 total alpha spent across this mode-specific suite",
            "suite_config": self.state["config"],
        }
        atomic_json(self.root / "rounds" / f"round{job['index']:05d}.json", report)
        saved_job = self.state["rounds"][job["index"] - 1]
        saved_job["status"] = "complete"
        saved_job["completed_at"] = report["completed_at"]
        self.save()
        self.refresh_history()
        self.prune_snapshots()
        return report

    def refresh_history(self) -> None:
        """Rebuild, not append: restart cannot duplicate the historical curve."""
        reports = [json.loads((self.root / "rounds" / f"round{r['index']:05d}.json").read_text(
            encoding="utf-8"
        )) for r in self.state["rounds"] if r["status"] == "complete"]
        atomic_json(self.root / "history.json", reports)
        if reports:
            atomic_json(self.root / "latest.json", reports[-1])
        labels = {
            "ahead_of_opponent": "有领先证据", "behind_opponent": "有退步证据",
            "inconclusive": "证据不足", "smoke_test_not_strength_evidence": "仅工程冒烟",
        }
        title, unit = {
            "two_player": ("二人军棋", "两局换边"),
            "four_dark": ("四国军棋（四暗）", "四局整队轮转"),
            "double_open": ("四国军棋（双明）", "四局整队轮转"),
        }[self.mode]
        lines = [
            f"# {title}历史模型评测", "",
            f"得分 = (胜 + 0.5 × 和) / 总局数；{unit}组成一个统计样本。",
            "区间采用按轮次/对手修正的保守 Hoeffding 界；bootstrap 95% 区间只作描述。",
            "这里评估 Policy + Layout 组合，不是 loss，也不代表人类等级分。", "",
            "| 轮次 | 新版本 | 对手 | 局数 | 胜/和/负 | 得分 | 修正置信区间 | 结论 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for report in reports:
            for match in report["matches"]:
                lo, hi = match["score_ci"]
                lines.append(
                    f"| {report['round']} | u{match['candidate_update']} | u{match['opponent_update']} "
                    f"| {match['games']} | {match['wins']}/{match['draws']}/{match['losses']} "
                    f"| {match['score']:.2%} | {lo:.2%}–{hi:.2%} | {labels[match['verdict']]} |"
                )
        temporary = self.root / ".report.md.tmp"
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(self.root / "report.md")

    def prune_snapshots(self) -> None:
        # Only our content-addressed inference copies may be removed. Training
        # checkpoints and user-supplied baseline files are never deletion targets.
        rounds = self.state["rounds"]
        keep = {self.state["baseline"]["path"]}
        count = max(1, self.state["config"]["recent_opponents"])
        for item in rounds[-count:]:
            keep.add(item["candidate"]["path"])
        for item in rounds:
            if item["status"] == "pending":
                keep.update(s["path"] for s in [item["candidate"], *item["opponents"]])
        for snapshot in self.state["snapshots"]:
            path = self.model_path(snapshot)
            if snapshot["path"] not in keep and path.exists():
                self.verify(snapshot)  # don't delete a file somebody replaced
                path.unlink()


def primary_call(context: DistributedContext, function: Callable[[], Any]) -> Any:
    value, error = None, None
    if context.primary:
        try:
            value = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    synchronized_error(context, error)
    return context.broadcast_object(value)


def execute_round(store: ArenaStore | None, root: Path, job: dict[str, Any],
                  context: DistributedContext) -> dict[str, Any] | None:
    settings = MatchSettings(**job["settings"])
    matches = []
    for opponent in job["opponents"]:
        prefix = root / "matches" / f"round{job['index']:05d}_vs_{opponent['sha256'][:16]}"
        summary_path = prefix.with_suffix(".json")

        def existing_result() -> dict[str, Any] | None:
            if not summary_path.exists():
                store.verify(job["candidate"])
                store.verify(opponent)
                return None
            report = json.loads(summary_path.read_text(encoding="utf-8"))
            json.dumps(report, allow_nan=False)
            if (report.get("candidate_sha256") != job["candidate"]["sha256"]
                    or report.get("opponent_sha256") != opponent["sha256"]
                    or report.get("settings") != job["settings"]
                    or report.get("mode") != settings.mode
                    or report.get("alpha") != job["alpha_per_match"]
                    or report.get("games") != settings.pairs * settings.games_per_group):
                raise ValueError("saved match does not match the pending job")
            return report

        report = primary_call(context, existing_result)
        if report is None:
            primary_call(context, lambda: atomic_json(root / "status.json", {
                "state": "evaluating", "updated_at": utc_now(), "round": job["index"],
                "mode": settings.mode,
                "candidate_update": job["candidate"]["update"], "opponent_update": opponent["update"],
                "groups_per_opponent": settings.pairs, "games_per_group": settings.games_per_group,
            }))
            report = run_match(
                root / job["candidate"]["path"], root / opponent["path"],
                settings, context, prefix, alpha=job["alpha_per_match"],
                candidate_sha256=job["candidate"]["sha256"], opponent_sha256=opponent["sha256"],
            )

            def save_result() -> dict[str, Any]:
                report["candidate_sha256"] = job["candidate"]["sha256"]
                report["opponent_sha256"] = opponent["sha256"]
                atomic_json(summary_path, report)
                return report

            report = primary_call(context, save_result)
        matches.append(report)
    result = primary_call(context, lambda: store.complete(job, matches))
    primary_call(context, lambda: atomic_json(root / "status.json", {
        "state": "round_complete", "updated_at": utc_now(), "round": job["index"],
        "mode": settings.mode,
        "candidate_update": job["candidate"]["update"], "verdict": result["verdict"],
    }))
    return result


def build_parser(
    *, default_mode: TrainingMode | str = TrainingMode.TWO_PLAYER,
    allowed_modes: tuple[str, ...] = (TrainingMode.TWO_PLAYER.value,),
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=allowed_modes, default=normalize_mode(default_mode).value)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="required only for a new output directory")
    parser.add_argument("--device", default="cpu", help="explicitly choose reserved CUDA/NPU resources")
    parser.add_argument("--every-updates", type=int, default=25)
    parser.add_argument("--recent-opponents", type=int, default=2)
    if allowed_modes == (TrainingMode.TWO_PLAYER.value,):
        parser.add_argument("--pairs", type=int, default=200, help="two games per pair, per opponent")
    else:
        parser.add_argument("--groups", dest="pairs", metavar="GROUPS", type=int, default=200,
                            help="four games per team/seat rotation group, per opponent")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--layout-temperature", type=float, default=0.7)
    parser.add_argument("--max-plies", type=int, default=2000)
    parser.add_argument("--temporal-cache-entries", type=int, default=8)
    parser.add_argument("--parallel-games", type=int, default=32,
                        help="active games per rank; finished slots are refilled immediately")
    parser.add_argument("--inference-batch-size", type=int, default=32,
                        help="maximum requests batched for one version of the policy")
    parser.add_argument("--environment-workers", type=int, default=4,
                        help="environment threads per rank; 1 runs transitions on the calling thread")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true", help="never issue a strength verdict")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--once", action="store_true", help="default; run one eligible/resumable round then exit")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser


def evaluate_history(
    argv: list[str] | None = None, *, default_mode: TrainingMode | str = TrainingMode.TWO_PLAYER,
    allowed_modes: tuple[str, ...] = (TrainingMode.TWO_PLAYER.value,),
) -> dict[str, Any]:
    parser = build_parser(default_mode=default_mode, allowed_modes=allowed_modes)
    args = parser.parse_args(argv)
    if min(args.every_updates, args.poll_seconds, args.cpu_threads) < 1 or args.recent_opponents < 0:
        parser.error("intervals/threads must be positive; recent-opponents must be nonnegative")
    try:
        settings = MatchSettings(**{key: getattr(args, key) for key in MatchSettings.__dataclass_fields__})
    except ValueError as exc:
        parser.error(str(exc))
    checkpoint_dir, root = args.checkpoint_dir.resolve(), args.output_dir.resolve()
    if root == checkpoint_dir or root.is_relative_to(checkpoint_dir) or checkpoint_dir.is_relative_to(root):
        parser.error("output-dir and checkpoint-dir must be separate, non-nested directories")
    torch.set_num_threads(args.cpu_threads)
    context = DistributedContext.initialize(args.device)
    lock, store = None, None
    try:
        config = {
            "arena_version": ARENA_VERSION, "match": asdict(settings),
            "every_updates": args.every_updates, "recent_opponents": args.recent_opponents,
            "checkpoint_dir": str(checkpoint_dir), "device_type": context.device.type,
            "protocol_sha256": protocol_fingerprint(), "torch_version": str(torch.__version__),
        }
        primary_config = context.broadcast_object(config if context.primary else None)
        synchronized_error(context, None if primary_config == config else "rank evaluation code/settings differ")
        error = None
        if context.primary:
            try:
                lock = OutputLock(root)
                store = ArenaStore(root, config, args.baseline)
                store.refresh_history()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        synchronized_error(context, error)
        while True:
            def prepare() -> dict[str, Any]:
                try:
                    return store.prepare(checkpoint_dir)
                except PublicationInProgress as exc:
                    return {"idle": "checkpoint_publication_in_progress", "detail": str(exc)}

            prepared = primary_call(context, prepare)
            if "job" in prepared:
                result = execute_round(store, root, prepared["job"], context)
                if context.primary:
                    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
            else:
                result = prepared
                if context.primary:
                    print(json.dumps(prepared, ensure_ascii=False), flush=True)
                primary_call(context, lambda: atomic_json(root / "status.json", {
                    "state": "waiting", "mode": settings.mode, "updated_at": utc_now(), **prepared,
                }))
            if not args.watch:
                return result
            time.sleep(args.poll_seconds)
    except BaseException as exc:
        if context.primary and lock is not None:
            try:
                atomic_json(root / "status.json", {
                    "state": "stopped" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "mode": settings.mode,
                    "updated_at": utc_now(), "error": f"{type(exc).__name__}: {exc}",
                })
            except OSError:
                pass  # preserve the original error when the output disk is full
        raise
    finally:
        if lock is not None:
            lock.close()
        context.close()


def main() -> None:
    evaluate_history()


if __name__ == "__main__":
    main()
