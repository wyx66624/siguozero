"""Resumable candidate versus champion matches at environment-step or progress thresholds."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

import torch

from .accelerator import empty_cache
from .arena import MatchSettings, atomic_json, run_match, sha256_file, synchronized_error
from .checkpoint import CHECKPOINT_FORMAT_VERSION, capture_rng_state, restore_rng_state
from .distributed import DistributedContext
from .inference import InferenceEngine
from .models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from .settings import TrainingSettings


class ModelSelection:
    """state.json commits a round; best.pt and history.jsonl are repairable views.

    Snapshots contain Policy/Layout only. Optimizers, the critic, rollout state,
    and training counters remain exclusively in the trainer's latest.pt.
    """

    def __init__(self, settings: TrainingSettings, run_directory: str | Path,
                 context: DistributedContext) -> None:
        self.settings = settings
        self.context = context
        self.directory = Path(run_directory).resolve() / "model_selection"
        self.best_path = self.directory.parent / "checkpoints" / "best.pt"
        self.state_path = self.directory / "state.json"
        self.milestones = settings.arena_milestones
        self.state: dict[str, Any] = {}
        self._pending_baseline: dict[str, Any] | None = None

    @property
    def environment_schedule(self) -> bool:
        return self.settings.arena_interval_environment_plies is not None

    @property
    def milestone_key(self) -> str:
        return "milestone_environment_plies" if self.environment_schedule else "milestone_percent"

    @property
    def completed_key(self) -> str:
        return "last_completed_environment_plies" if self.environment_schedule else "last_completed_percent"

    def milestone_label(self, milestone: int) -> str:
        return f"{milestone} environment transitions" if self.environment_schedule else f"{milestone}%"

    def _round_tag(self, milestone: int) -> str:
        return f"env_{milestone:012d}" if self.environment_schedule else f"{milestone:03d}"

    def _contract(self) -> dict[str, Any]:
        settings = self.settings
        contract = {
            "mode": settings.mode.value, "algorithm": settings.algorithm,
            "dead_rules_enabled": settings.dead_rules_enabled,
            "total_updates": settings.total_updates,
            "step_budget_target": settings.step_budget_target,
            "step_budget_counter": settings.step_budget_counter,
            "milestones": None if self.environment_schedule else list(self.milestones),
            "games": settings.arena_games,
            "max_plies": settings.arena_max_plies, "seed": settings.arena_seed,
            "temporal_cache_entries": settings.arena_temporal_cache_entries,
        }
        if self.environment_schedule:
            contract["interval_environment_plies"] = settings.arena_interval_environment_plies
        return contract

    def progress_percent(self, *, update: int, cumulative: Mapping[str, int]) -> float:
        progress = update / self.settings.total_updates
        target = self.settings.step_budget_target
        if target is not None:
            key = self.settings.step_budget_counter
            progress = max(progress, cumulative.get(key, 0) / target)
        return min(100.0, 100 * progress)

    def _reached(self, milestone: int, update: int, cumulative: Mapping[str, int]) -> bool:
        if self.environment_schedule:
            return cumulative.get("environment_plies", 0) >= milestone
        # Integer comparisons avoid losing a boundary such as 29 / 100 * 100.
        if 100 * update >= milestone * self.settings.total_updates:
            return True
        target = self.settings.step_budget_target
        key = self.settings.step_budget_counter
        return target is not None and 100 * cumulative.get(key, 0) >= milestone * target

    def due_milestone(self, *, update: int, cumulative: Mapping[str, int]) -> int | None:
        if not self.settings.arena_enabled:
            return None
        if self.environment_schedule:
            interval = self.settings.arena_interval_environment_plies
            reached = min(cumulative.get("environment_plies", 0),
                          self.settings.target_environment_plies or 0)
            latest = reached // interval * interval
            return latest if latest > self.state.get(self.completed_key, 0) else None
        due = [p for p in self.milestones
               if p > self.state.get(self.completed_key, 0)
               and self._reached(p, update, cumulative)]
        # One update may cross several boundaries. Only its actual parameters
        # exist: evaluate once and explicitly record the unavailable milestones.
        return max(due) if due else None

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def _snapshot_payload(self, policy: GamePolicyTransformer,
                          layout: PieceConditionedLayoutPointerDecoder,
                          *, update: int) -> dict[str, Any]:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION, "update": update,
            "mode": self.settings.mode.value, "algorithm": self.settings.algorithm,
            "dead_rules_enabled": self.settings.dead_rules_enabled,
            "reason": "model_selection_inference_only",
            "config": self.settings.serializable(),
            # A deferred baseline must not alias CPU training parameters.
            "policy": {k: v.detach().to("cpu", copy=True) for k, v in policy.state_dict().items()},
            "layout": {k: v.detach().to("cpu", copy=True) for k, v in layout.state_dict().items()},
        }

    def _write_snapshot(self, payload: dict[str, Any], *, name: str) -> Path:
        path = self.directory / "snapshots" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def _snapshot(self, policy: GamePolicyTransformer,
                  layout: PieceConditionedLayoutPointerDecoder,
                  *, update: int, name: str) -> Path:
        return self._write_snapshot(self._snapshot_payload(policy, layout, update=update), name=name)

    def persist_baseline(self) -> None:
        """Write the initial opponent only when the first evaluation/save is due."""
        if not self.state or self.state["best_sha256"] is not None:
            return
        error = None
        if self.context.primary:
            try:
                if self._pending_baseline is None:
                    raise RuntimeError("initial model selection baseline is unavailable")
                baseline = self._write_snapshot(
                    self._pending_baseline, name=Path(self.state["best_snapshot"]).name,
                )
                next_state = {**self.state, "best_sha256": sha256_file(baseline)}
                atomic_json(self.state_path, next_state)
                self.state = next_state
                self._repair_views()
                self._pending_baseline = None
            except Exception as exc:
                error = f"model selection baseline save: {type(exc).__name__}: {exc}"
        synchronized_error(self.context, error)
        self.state = self.context.broadcast_object(self.state)

    def _repair_views(self) -> None:
        champion = self.directory / self.state["best_snapshot"]
        if not champion.is_file() or sha256_file(champion) != self.state["best_sha256"]:
            raise RuntimeError("best model snapshot is missing or has changed")
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.best_path.exists() or sha256_file(self.best_path) != self.state["best_sha256"]:
            temporary = self.best_path.with_name(".best.pt.tmp")
            shutil.copyfile(champion, temporary)
            temporary.replace(self.best_path)
        # Rebuild instead of appending: a restart after a commit cannot duplicate
        # a round or expose an uncommitted report in the history.
        records = [json.loads((self.directory / name).read_text(encoding="utf-8"))
                   for name in self.state["rounds"]]
        temporary = self.directory / ".history.jsonl.tmp"
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                                     for r in records), encoding="utf-8")
        temporary.replace(self.directory / "history.jsonl")

    def initialize(self, policy: GamePolicyTransformer,
                   layout: PieceConditionedLayoutPointerDecoder, *,
                   update: int, cumulative: Mapping[str, int]) -> None:
        if not self.settings.arena_enabled:
            return
        error = None
        if self.context.primary:
            try:
                if self.state_path.exists():
                    self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
                    contract = dict(self.state.get("contract", {}))
                    if "target_continuation_plies" in contract:
                        # v18 PPO already stopped on total environment plies;
                        # only its persisted target field used the GRPO name.
                        contract["step_budget_target"] = contract.pop("target_continuation_plies")
                        contract["step_budget_counter"] = (
                            "environment_plies" if contract.get("algorithm") == "ppo"
                            else "continuation_plies"
                        )
                    if self.state["last_evaluated_update"] > update:
                        raise ValueError("model selection is newer than the training checkpoint")
                    version = 2 if self.environment_schedule else 1
                    if self.state.get("version") == 1 and self.environment_schedule:
                        # Upgrade the old percentage schedule while retaining the
                        # champion and reports. The new match protocol is a separate
                        # statistical family; never reuse its predecessors' seeds.
                        excluded = {"milestones", "games", "interval_environment_plies"}
                        stable = lambda value: {k: v for k, v in value.items() if k not in excluded}
                        if ("interval_environment_plies" in contract
                                or stable(contract) != stable(self._contract())):
                            raise ValueError("model selection budget/protocol changed during schedule migration")
                        self._repair_views()  # Validate old immutable artifacts first.
                        group_size = 2 if self.settings.mode.value == "two_player" else 4
                        old_span = (2 + group_size) * (contract["games"] // group_size)
                        seed_base = contract["seed"] + (max(contract["milestones"]) + 1) * old_span
                        new_span = (2 + group_size) * (self.settings.arena_games // group_size)
                        if seed_base + (len(self.milestones) + 1) * new_span >= 2**63:
                            raise ValueError("migrated model selection seed range exceeds int64")
                        self.state["previous_schedule"] = {
                            "contract": contract,
                            "last_completed_percent": self.state["last_completed_percent"],
                            "rounds": list(self.state["rounds"]),
                            "migration_update": update,
                            "migration_environment_plies": cumulative.get("environment_plies", 0),
                        }
                        self.state.pop("last_completed_percent")
                        self.state.update(version=2, last_completed_environment_plies=0,
                                          schedule_seed_base=seed_base,
                                          statistical_family="environment_schedule_after_percentage")
                        contract = self._contract()
                    if self.state.get("version") != version or contract != self._contract():
                        raise ValueError("model selection schedule/budget changed; restore its original settings or use a new run directory")
                    if self.state["contract"] != contract:
                        self.state["contract"] = contract
                        atomic_json(self.state_path, self.state)
                else:
                    baseline = self.directory / "snapshots" / f"baseline_{update:09d}.pt"
                    if self.settings.checkpoint_policy == "evaluation":
                        self._pending_baseline = self._snapshot_payload(policy, layout, update=update)
                    else:
                        self._snapshot(policy, layout, update=update, name=baseline.name)
                    skipped = [p for p in self.milestones if self._reached(p, update, cumulative)]
                    self.state = {
                        "version": 2 if self.environment_schedule else 1, "contract": self._contract(),
                        "baseline_update": update,
                        "baseline_progress_percent": self.progress_percent(update=update, cumulative=cumulative),
                        "baseline_skipped_milestones": skipped,
                        self.completed_key: max(skipped, default=0),
                        "last_evaluated_update": update,
                        "best_update": update,
                        "best_snapshot": baseline.relative_to(self.directory).as_posix(),
                        "best_sha256": None if self._pending_baseline is not None else sha256_file(baseline),
                        "rounds": [],
                    }
                    if self._pending_baseline is None:
                        atomic_json(self.state_path, self.state)
                if self._pending_baseline is None:
                    self._repair_views()
            except Exception as exc:
                error = f"model selection initialization: {type(exc).__name__}: {exc}"
        synchronized_error(self.context, error)
        self.state = self.context.broadcast_object(self.state)

    def _validate_result(self, result: dict[str, Any]) -> None:
        games = self.settings.arena_games
        if result.get("games") != games:
            raise ValueError("incomplete model selection match")
        counts = [result.get(name) for name in ("wins", "draws", "losses")]
        if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != games:
            raise ValueError("invalid model selection result counts")
        score = result.get("score")
        if (not isinstance(score, (int, float)) or not math.isfinite(score)
                or not math.isclose(score, (counts[0] + 0.5 * counts[1]) / games,
                                    rel_tol=0, abs_tol=1e-12)):
            raise ValueError("model selection score does not match results")
        ci = result.get("score_ci")
        if (not isinstance(ci, (list, tuple)) or len(ci) != 2
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in ci)
                or not 0 <= ci[0] <= score <= ci[1] <= 1):
            raise ValueError("invalid model selection score interval")

    def evaluate(self, policy: GamePolicyTransformer,
                 layout: PieceConditionedLayoutPointerDecoder, *,
                 update: int, cumulative: Mapping[str, int],
                 amp_dtype: torch.dtype | None = None,
                 stop_requested: Callable[[], bool] | None = None) -> dict[str, Any] | None:
        milestone = self.due_milestone(update=update, cumulative=cumulative)
        if milestone is None:
            return None
        self.persist_baseline()
        tag = self._round_tag(milestone)
        candidate = self.directory / "snapshots" / f"candidate_{tag}_{update:09d}.pt"
        opponent = self.directory / self.state["best_snapshot"]
        report_path = self.directory / f"round_{tag}.json"
        error, candidate_hash, report = None, None, None
        if self.context.primary:
            try:
                self._snapshot(policy, layout, update=update, name=candidate.name)
                candidate_hash = sha256_file(candidate)
                # A fully written result before a failed state commit can be
                # reused only for these exact candidate/opponent parameters.
                if report_path.exists():
                    saved = json.loads(report_path.read_text(encoding="utf-8"))
                    if (saved.get("candidate_sha256") == candidate_hash
                            and saved.get("opponent_sha256") == self.state["best_sha256"]):
                        report = saved
            except Exception as exc:
                error = f"model selection snapshot: {type(exc).__name__}: {exc}"
        synchronized_error(self.context, error)
        candidate_hash = self.context.broadcast_object(candidate_hash)
        report = self.context.broadcast_object(report)
        if report is None:
            group_size = 2 if self.settings.mode.value == "two_player" else 4
            groups = self.settings.arena_games // group_size
            seed_index = (milestone // self.settings.arena_interval_environment_plies
                          if self.environment_schedule else milestone)
            match = MatchSettings(
                pairs=groups, mode=self.settings.mode.value,
                seed=self.state.get("schedule_seed_base", self.settings.arena_seed)
                     + seed_index * (2 + group_size) * groups,
                max_plies=self.settings.arena_max_plies,
                temporal_cache_entries=self.settings.arena_temporal_cache_entries,
                parallel_games=self.settings.arena_parallel_games,
                inference_batch_size=self.settings.arena_inference_batch_size,
                environment_workers=self.settings.arena_environment_workers,
            )
            rng = None
            modules = [(module, module.training) for model in (policy, layout) for module in model.modules()]
            flags = [(parameter, parameter.requires_grad) for model in (policy, layout) for parameter in model.parameters()]
            config = policy.config
            try:
                error = None
                try:
                    rng = capture_rng_state(self.context.device)
                    policy.clear_inference_board_cache()
                    policy.config = replace(config, inference_temporal_cache_entries=match.temporal_cache_entries,
                                            inference_board_cache_entries=2048)
                    engine = InferenceEngine(self.settings.mode, policy, layout, amp_dtype=amp_dtype)
                    engine.checkpoint_update = update
                    empty_cache(self.context.device)
                except Exception as exc:
                    error = f"rank {self.context.rank} model selection preparation: {type(exc).__name__}: {exc}"
                synchronized_error(self.context, error)
                report = run_match(
                    candidate, opponent, match, self.context,
                    self.directory / f"games_{tag}",
                    alpha=0.05 / len(self.milestones),
                    candidate_sha256=candidate_hash,
                    opponent_sha256=self.state["best_sha256"],
                    candidate_engine=engine,
                    stop_requested=stop_requested,
                )
            finally:
                policy.clear_inference_board_cache()
                policy.config = config
                for parameter, requires_grad in flags:
                    parameter.requires_grad_(requires_grad)
                for module, training in modules:
                    module.training = training
                if rng is not None:
                    restore_rng_state(rng, self.context.device)
                empty_cache(self.context.device)
        error = None
        if self.context.primary:
            try:
                if report is None:
                    raise ValueError("model selection match returned no result")
                self._validate_result(report)
                promoted = report["score"] > 0.5
                report.update({
                    self.milestone_key: milestone,
                    "schedule_unit": "environment_plies" if self.environment_schedule else "percent",
                    "environment_plies": cumulative.get("environment_plies", 0),
                    "statistical_family": self.state.get("statistical_family", "initial_schedule"),
                    "progress_percent": self.progress_percent(update=update, cumulative=cumulative),
                    "skipped_milestones": [p for p in self.milestones
                                           if self.state[self.completed_key] < p < milestone],
                    "candidate_update": update, "opponent_update": self.state["best_update"],
                    "candidate_sha256": candidate_hash, "opponent_sha256": self.state["best_sha256"],
                    "promoted": promoted, "decision": "promote" if promoted else "retain_best",
                    "best_update": update if promoted else self.state["best_update"],
                    "win_rate": report["wins"] / self.settings.arena_games,
                    "score_delta_vs_best": report["score"] - 0.5,
                    "promotion_rule": "score_strictly_greater_than_0.5",
                    "strength_verdict": report.get("verdict", "inconclusive"),
                })
                atomic_json(report_path, report)
                next_state = copy.deepcopy(self.state)
                next_state.update({self.completed_key: milestone, "last_evaluated_update": update})
                next_state["rounds"].append(report_path.name)
                if promoted:
                    next_state.update(best_update=update,
                                      best_snapshot=candidate.relative_to(self.directory).as_posix(),
                                      best_sha256=candidate_hash)
                atomic_json(self.state_path, next_state)
                self.state = next_state
                self._repair_views()
            except Exception as exc:
                error = f"model selection result commit: {type(exc).__name__}: {exc}"
        synchronized_error(self.context, error)
        self.state = self.context.broadcast_object(self.state)
        return self.context.broadcast_object(report)
