"""Resumable champion selection and observational historical-panel evaluation."""

from __future__ import annotations

import copy
from bisect import bisect_right
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

import torch

from .accelerator import empty_cache
from .arena import ARENA_VERSION, MatchSettings, atomic_json, run_match, sha256_file, synchronized_error
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
            "no_capture_draw_plies": settings.no_capture_draw_plies,
            "max_passes_per_player": settings.max_passes_per_player,
            "temporal_cache_entries": settings.arena_temporal_cache_entries,
        }
        if self.environment_schedule:
            contract["interval_environment_plies"] = settings.arena_interval_environment_plies
            if settings.arena_after_half_interval_environment_plies is not None:
                contract["after_half_interval_environment_plies"] = settings.arena_after_half_interval_environment_plies
        if settings.arena_after_half_historical_only:
            contract["after_half_historical_only"] = True
        if settings.arena_observational_only:
            contract["observational_only"] = True
        if settings.arena_champion_only:
            contract["champion_only"] = True
        if settings.arena_historical_teammate_fraction:
            contract["historical_teammate_fraction"] = settings.arena_historical_teammate_fraction
        if settings.historical_eval_total_games is not None:
            contract["historical_eval_total_games"] = settings.historical_eval_total_games
        return contract

    def _adopt_observational_evaluation(self, contract, *, update, cumulative):
        desired = self._contract()
        fields = {"observational_only", "historical_teammate_fraction"}
        stable = lambda value: {k: v for k, v in value.items() if k not in fields}
        if (self.state.get("version") != 2 or contract == desired
                or not desired.get("observational_only") or stable(contract) != stable(desired)):
            return contract
        if self.due_milestone(update=update, cumulative=cumulative) is not None:
            raise ValueError("finish the pending evaluation before changing its teammate protocol")
        self._repair_views()
        slots = len(self.state.get("evaluation_seed_milestones", self.milestones))
        groups = max(contract["games"], contract.get("historical_eval_total_games", 0),
                     self.settings.historical_eval_games) // 4
        old_base = self.state.get("schedule_seed_base", contract["seed"])
        new_base = old_base + (slots + 1) * 6 * groups
        if new_base + (slots + 1) * 6 * groups + 8 * 1_000_003 >= 2**63:
            raise ValueError("observational evaluation seed range exceeds int64")
        self.state.setdefault("round_alpha", self._round_alpha())
        self.state.setdefault("observational_evaluation_migrations", []).append(dict(
            previous_contract=contract, next_contract=desired, update=update,
            environment_plies=cumulative.get("environment_plies", 0),
            previous_rounds=list(self.state["rounds"]),
            fixed_reference_snapshot=self.state["best_snapshot"],
            fixed_reference_update=self.state["best_update"],
            previous_seed_base=old_base, next_seed_base=new_base))
        self.state["schedule_seed_base"] = new_base
        return desired

    def _adopt_champion_evaluation(self, contract, *, update, cumulative, adopt=False):
        """Migrate future challenges; keep completed reports and training weights intact."""
        desired = self._contract()
        fields = {"observational_only", "after_half_historical_only", "champion_only"}
        stable = lambda value: {k: v for k, v in value.items() if k not in fields}
        if (self.state.get("version") != 2 or contract == desired
                or not desired.get("champion_only") or stable(contract) != stable(desired)):
            return contract
        if not adopt:
            raise ValueError("changing to champion-only evaluation requires --adopt-champion-evaluation")
        if self.due_milestone(update=update, cumulative=cumulative) is not None:
            raise ValueError("finish the pending evaluation before changing to champion-only evaluation")
        self._repair_views()
        slots = len(self.state.get("evaluation_seed_milestones", self.milestones))
        groups = max(contract["games"], contract.get("historical_eval_total_games", 0),
                     self.settings.historical_eval_games) // 4
        old_base = self.state.get("schedule_seed_base", contract["seed"])
        new_base = old_base + (slots + 1) * 6 * groups
        if new_base + (slots + 1) * 6 * groups + 8 * 1_000_003 >= 2**63:
            raise ValueError("champion evaluation seed range exceeds int64")
        migration = dict(previous_contract=contract, next_contract=desired, update=update,
                         environment_plies=cumulative.get("environment_plies", 0),
                         previous_rounds=list(self.state["rounds"]),
                         previous_best_update=self.state["best_update"],
                         previous_best_sha256=self.state["best_sha256"],
                         previous_seed_base=old_base, next_seed_base=new_base)
        # Only the latest completed head-to-head result against this incumbent
        # may update it. Never pick a maximum from unrelated historical scores.
        if self.state["rounds"]:
            name = self.state["rounds"][-1]
            report = json.loads((self.directory / name).read_text(encoding="utf-8"))
            if (report.get("evaluation_type") == "fixed_reference"
                    and report.get("schedule_contract") == contract
                    and report.get("candidate_update") == self.state["last_evaluated_update"]
                    and report.get("candidate_snapshot") == self.state.get("latest_evaluated_snapshot")
                    and report.get("candidate_sha256") == self.state.get("latest_evaluated_sha256")
                    and report.get("opponent_update") == self.state["best_update"]
                    and report.get("opponent_sha256") == self.state["best_sha256"]):
                self._validate_historical_result(report, games=self.settings.arena_games)
                migration.update(source_report=name, source_score=report["score"])
                if report["score"] > .5:
                    self.state.update(best_update=report["candidate_update"],
                                      best_snapshot=report["candidate_snapshot"],
                                      best_sha256=report["candidate_sha256"])
        migration.update(best_update=self.state["best_update"], best_sha256=self.state["best_sha256"])
        self.state.setdefault("round_alpha", self._round_alpha())
        self.state.setdefault("champion_evaluation_migrations", []).append(migration)
        self.state["schedule_seed_base"] = new_base
        return desired

    def _adopt_evaluation_game_budget(self, contract, *, update, cumulative):
        """Change future sample counts without rewriting completed evaluations."""
        desired = self._contract()
        fields = {"games", "historical_eval_total_games"}
        stable = lambda value: {k: v for k, v in value.items() if k not in fields}
        if (self.state.get("version") != 2 or contract == desired
                or stable(contract) != stable(desired)):
            return contract
        if self.due_milestone(update=update, cumulative=cumulative) is not None:
            raise ValueError("finish the pending evaluation before changing its game budget")
        self._repair_views()
        group_size = 2 if self.settings.mode.value == "two_player" else 4
        slots = len(self.state.get("evaluation_seed_milestones", self.milestones))
        old_base = self.state.get("schedule_seed_base", contract["seed"])
        # Reserve the entire old seed range, including attempted but uncommitted
        # matches and the larger legacy per-opponent historical matches.
        old_games = max(contract["games"], contract.get("historical_eval_total_games", 0),
                        self.settings.historical_eval_games)
        new_base = old_base + (slots + 1) * (2 + group_size) * (old_games // group_size)
        new_games = max(desired["games"], desired.get("historical_eval_total_games", 0))
        new_end = new_base + (slots + 1) * (2 + group_size) * (new_games // group_size)
        if new_end + (1 + len(self.settings.historical_snapshot_fractions)) * 1_000_003 >= 2**63:
            raise ValueError("changed evaluation game budget exceeds the int64 seed range")
        self.state.setdefault("round_alpha", self._round_alpha())
        self.state.setdefault("evaluation_game_budget_migrations", []).append(dict(
            previous_contract=contract, next_contract=desired, update=update,
            environment_plies=cumulative.get("environment_plies", 0),
            previous_rounds=list(self.state["rounds"]),
            previous_seed_base=old_base, next_seed_base=new_base))
        self.state["schedule_seed_base"] = new_base
        return desired

    def _historical_game_counts(self, panel):
        total = self.settings.historical_eval_total_games
        if total is None:
            return [self.settings.historical_eval_games] * len(panel)
        if not panel:
            return []
        groups, remainder = divmod(total // 4, len(panel))
        if groups < 1 or total % 4:
            raise ValueError("historical panel budget cannot cover complete four-game rotations")
        return [4 * (groups + (index < remainder)) for index in range(len(panel))]

    def historical_only(self, cumulative: Mapping[str, int]) -> bool:
        target = self.settings.target_environment_plies
        return bool(self.settings.arena_after_half_historical_only and target is not None
                    and 2 * cumulative.get("environment_plies", 0) >= target)

    def _adopt_historical_only(self, contract, *, update, cumulative):
        desired = self._contract()
        previous = {k: v for k, v in desired.items() if k != "after_half_historical_only"}
        if desired.get("after_half_historical_only") and contract == previous:
            self._repair_views()
            self.state["historical_only_migration"] = dict(
                previous_contract=contract, update=update,
                environment_plies=cumulative.get("environment_plies", 0),
                previous_rounds=list(self.state["rounds"]))
            # Frequency and the spent confidence budget do not change.
            return desired
        return contract

    def _round_alpha(self) -> float:
        return self.state.get("round_alpha", .05 / len(self.milestones))

    def _adopt_reduced_evaluation_schedule(self, contract, *, update, cumulative):
        """Drop future boundaries while retaining reports, alpha and seed slots."""
        desired = self._contract()
        field = "after_half_interval_environment_plies"
        stable = lambda value: {k: v for k, v in value.items() if k != field}
        if (self.state.get("version") != 2 or contract == desired
                or stable(contract) != stable(desired)):
            return contract
        previous = replace(self.settings, arena_after_half_interval_environment_plies=contract.get(field))
        old_milestones = list(previous.arena_milestones)
        if (len(self.milestones) >= len(old_milestones)
                or not set(self.milestones).issubset(old_milestones)):
            return contract
        self._repair_views()
        # Coarsening does not earn a fresh confidence budget. Keep the original
        # slots so a retained later match cannot reuse an earlier match's seed.
        self.state.setdefault("round_alpha", .05 / len(old_milestones))
        self.state.setdefault("evaluation_seed_milestones", old_milestones)
        self.state.setdefault("evaluation_schedule_migrations", []).append(dict(
            previous_contract=contract, next_contract=desired, update=update,
            environment_plies=cumulative.get("environment_plies", 0),
            previous_rounds=list(self.state["rounds"]),
            previous_milestones=old_milestones, next_milestones=list(self.milestones)))
        return desired

    def _adopt_after_half_schedule(self, contract, *, update, cumulative):
        """Add a denser second half without replacing committed models/results."""
        desired = self._contract()
        if not contract.get("after_half_historical_only"):
            desired.pop("after_half_historical_only", None)
        old_contract = {k: v for k, v in desired.items() if k != "after_half_interval_environment_plies"}
        if ("after_half_interval_environment_plies" not in desired or contract != old_contract
                or self.state.get("version") != 2):
            return contract
        self._repair_views()
        old_interval = contract["interval_environment_plies"]
        target = self.settings.target_environment_plies
        reached = min(target, max(cumulative.get("environment_plies", 0),
                                  self.state[self.completed_key]))
        old_rounds = target // old_interval
        # Reserve every possibly attempted old round, including skipped or
        # interrupted rounds. A frequency increase cannot reset the CI budget.
        reserved = reached // old_interval
        remaining = len(self.milestones) - bisect_right(self.milestones, self.state[self.completed_key])
        budget = .05 * (1 - reserved / max(1, old_rounds))
        if remaining and budget <= 0:
            raise ValueError("finish the old final evaluation before changing an exhausted schedule")
        self.state["round_alpha"] = min(.05 / len(self.milestones), budget / remaining) if remaining else .05 / len(self.milestones)
        self.state["after_half_schedule_migration"] = dict(
            previous_contract=contract, update=update, environment_plies=reached,
            previous_rounds=list(self.state["rounds"]), reserved_old_rounds=reserved,
            remaining_rounds=remaining, remaining_alpha_budget=budget)
        return desired

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
            reached = min(cumulative.get("environment_plies", 0),
                          self.settings.target_environment_plies or 0)
            index = bisect_right(self.milestones, reached)
            latest = self.milestones[index - 1] if index else 0
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
        if self.state.get("latest_evaluated_snapshot"):
            snapshot = self.directory / self.state["latest_evaluated_snapshot"]
            if not snapshot.is_file() or sha256_file(snapshot) != self.state["latest_evaluated_sha256"]:
                raise RuntimeError("latest evaluated snapshot is missing or has changed")
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
        for report in reversed(records):
            if report.get("schedule_contract", {}).get("champion_only"):
                result = {key: report[key] for key in (
                    "opponent_update", "opponent_sha256", "games", "wins", "draws", "losses",
                    "score", "score_ci")}
                result.update(teammate_results=report.get("teammate_results", {}),
                              below_half=report["score"] < .5)
                atomic_json(self.directory.parent / "historical_opponents/latest_evaluation.json", {
                    "evaluation_type": "champion", "observational_only": False,
                    "candidate_sha256": report["candidate_sha256"],
                    "candidate_update": report["candidate_update"],
                    "environment_plies": report["environment_plies"],
                    "games": report["games"], "minimum_score": report["score"],
                    "promoted": report["promoted"], "best_update": report["best_update"],
                    "decision": report["decision"], "results": [result],
                })
                break
            if report.get("historical_panel"):
                atomic_json(self.directory.parent / "historical_opponents/latest_evaluation.json", {
                    "candidate_sha256": report["candidate_sha256"],
                    "candidate_update": report["candidate_update"],
                    "environment_plies": report["environment_plies"],
                    "evaluation_type": report.get("evaluation_type", "champion"),
                    **report["historical_panel"],
                })
                break

    def initialize(self, policy: GamePolicyTransformer,
                   layout: PieceConditionedLayoutPointerDecoder, *,
                   update: int, cumulative: Mapping[str, int],
                   adopt_current_draw_rules: bool = False, adopt_pass_rule: bool = False,
                   adopt_champion_evaluation: bool = False) -> None:
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
                    if contract.get("max_passes_per_player", 0) != 4 and adopt_pass_rule:
                        if self.state.get("rounds") or contract.get("max_passes_per_player", 0) != 0:
                            raise ValueError("pass-rule adoption requires an unevaluated no-pass protocol")
                        self._repair_views()
                        self.state["pass_rule_migration"] = {
                            "previous_contract": dict(contract), "update": update,
                            "environment_plies": cumulative.get("environment_plies", 0),
                        }
                        self.state["statistical_family"] = "no_capture_70_unlimited_pass4"
                        contract["max_passes_per_player"] = 4
                    if adopt_current_draw_rules and contract != self._contract():
                        excluded = {"max_plies", "no_capture_draw_plies"}
                        stable = lambda value: {k: v for k, v in value.items() if k not in excluded}
                        if (self.state.get("rounds") or stable(contract) != stable(self._contract())
                                or contract.get("max_plies") != 2000
                                or contract.get("no_capture_draw_plies", 60) != 60
                                or self.settings.arena_max_plies is not None
                                or self.settings.no_capture_draw_plies != 70):
                            raise ValueError("draw-rule adoption requires the unchanged, unevaluated 2000/60 protocol")
                        self._repair_views()
                        self.state["draw_rule_migration"] = {
                            "previous_contract": contract, "update": update,
                            "environment_plies": cumulative.get("environment_plies", 0),
                        }
                        self.state["statistical_family"] = "no_capture_70_unlimited_pass4"
                        contract = self._contract()
                    if self.state.get("version") == 1 and self.environment_schedule:
                        # Upgrade the old percentage schedule while retaining the
                        # champion and reports. The new match protocol is a separate
                        # statistical family; never reuse its predecessors' seeds.
                        excluded = {"milestones", "games", "interval_environment_plies", "after_half_interval_environment_plies",
                                    "after_half_historical_only"}
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
                    if contract != self._contract() and self.environment_schedule:
                        contract = self._adopt_after_half_schedule(contract, update=update, cumulative=cumulative)
                        contract = self._adopt_reduced_evaluation_schedule(contract, update=update, cumulative=cumulative)
                        contract = self._adopt_historical_only(contract, update=update, cumulative=cumulative)
                        contract = self._adopt_evaluation_game_budget(contract, update=update, cumulative=cumulative)
                        contract = self._adopt_observational_evaluation(contract, update=update, cumulative=cumulative)
                        contract = self._adopt_champion_evaluation(
                            contract, update=update, cumulative=cumulative, adopt=adopt_champion_evaluation)
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

    def _validate_result(self, result: dict[str, Any], *, games: int | None = None) -> None:
        games = self.settings.arena_games if games is None else games
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
                 stop_requested: Callable[[], bool] | None = None,
                 historical=None) -> dict[str, Any] | None:
        milestone = self.due_milestone(update=update, cumulative=cumulative)
        if milestone is None:
            return None
        self.persist_baseline()
        historical_only = self.historical_only(cumulative)
        observational = historical_only or self.settings.arena_observational_only
        reference_only = observational and not historical_only
        evaluation_type = "historical_only" if historical_only else "fixed_reference" if reference_only else "champion"
        panel = (historical.evaluation_panel() if not self.settings.arena_champion_only
                 and historical is not None and historical.active else [])
        panel_directory = historical.directory if historical is not None else self.directory
        if reference_only:
            panel = [dict(id=f"reference_u{self.state['best_update']:09d}",
                          file=self.state["best_snapshot"], sha256=self.state["best_sha256"],
                          update=self.state["best_update"], environment_plies=None)]
            panel_directory = self.directory
        synchronized_error(self.context, "historical-only evaluation requires a nonempty active frozen panel"
                           if historical_only and not panel else None)
        panel_contract = {"opponents": [e["sha256"] for e in panel],
                          "games": self.settings.historical_eval_games, "version": 1}
        panel_game_counts = self._historical_game_counts(panel)
        if reference_only:
            panel_game_counts = [self.settings.arena_games]
        if reference_only or self.settings.historical_eval_total_games is not None:
            panel_contract = {"opponents": [e["sha256"] for e in panel], "version": 2,
                              "total_games": (self.settings.arena_games if reference_only
                                              else self.settings.historical_eval_total_games),
                              "games_per_opponent": panel_game_counts}
        if observational:
            panel_contract["evaluation_type"] = evaluation_type
        if self.settings.arena_historical_teammate_fraction:
            panel_contract.update(historical_teammate_fraction=self.settings.arena_historical_teammate_fraction,
                                  teammate_protocol="paired_current_historical_v1")
        if historical is not None:
            historical.release_replica()
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
                            and saved.get("evaluation_type", "champion") == evaluation_type
                            and (observational or saved.get("opponent_sha256") == self.state["best_sha256"])
                            and saved.get("schedule_contract") == self._contract()
                            and saved.get("round_alpha") == self._round_alpha()
                            and (not panel or saved.get("historical_panel", {}).get("contract") == panel_contract)):
                        report = saved
            except Exception as exc:
                error = f"model selection snapshot: {type(exc).__name__}: {exc}"
        synchronized_error(self.context, error)
        candidate_hash = self.context.broadcast_object(candidate_hash)
        report = self.context.broadcast_object(report)
        if report is None:
            group_size = 2 if self.settings.mode.value == "two_player" else 4
            groups = self.settings.arena_games // group_size
            seed_groups = max(self.settings.arena_games,
                              self.settings.historical_eval_total_games or 0) // group_size
            seed_index = (bisect_right(self.state.get("evaluation_seed_milestones", self.milestones), milestone)
                          if self.environment_schedule else milestone)
            match = MatchSettings(
                pairs=groups, mode=self.settings.mode.value,
                seed=self.state.get("schedule_seed_base", self.settings.arena_seed)
                     + seed_index * (2 + group_size) * seed_groups,
                max_plies=self.settings.arena_max_plies,
                no_capture_draw_plies=self.settings.no_capture_draw_plies,
                temporal_cache_entries=self.settings.arena_temporal_cache_entries,
                parallel_games=self.settings.arena_parallel_games,
                inference_batch_size=self.settings.arena_inference_batch_size,
                environment_workers=self.settings.arena_environment_workers,
                historical_teammate_fraction=self.settings.arena_historical_teammate_fraction,
            )
            alpha = self._round_alpha() / (int(not observational) + len(panel))
            champion_cache = self.directory / f"champion_match_{tag}.json"
            champion_identity = dict(candidate_sha256=candidate_hash, opponent_sha256=self.state["best_sha256"],
                                     settings=asdict(match), alpha=alpha, arena_version=ARENA_VERSION)
            error = None
            if self.context.primary:
                try:
                    if not observational and champion_cache.exists():
                        saved = json.loads(champion_cache.read_text(encoding="utf-8"))
                        if saved.get("identity") == champion_identity:
                            report = saved["result"]
                            self._validate_historical_result(report, games=self.settings.arena_games)
                except Exception as exc:
                    error = f"champion match retry: {type(exc).__name__}: {exc}"
            synchronized_error(self.context, error)
            report = self.context.broadcast_object(report)
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
                if report is None and not observational:
                    report = run_match(
                        candidate, opponent, match, self.context,
                        self.directory / f"games_{tag}", alpha=alpha,
                        candidate_sha256=candidate_hash, opponent_sha256=self.state["best_sha256"],
                        candidate_engine=engine, stop_requested=stop_requested,
                    )
                    error = None
                    if self.context.primary:
                        try:
                            self._validate_historical_result(report, games=self.settings.arena_games)
                            atomic_json(champion_cache, {"identity": champion_identity, "result": report})
                        except Exception as exc:
                            error = f"champion match commit: {type(exc).__name__}: {exc}"
                    synchronized_error(self.context, error)
                if observational:
                    report = {}
                if panel:
                    panel_result = self._evaluate_historical_panel(
                        panel_directory, panel, panel_contract, candidate, candidate_hash, engine,
                        match, tag, stop_requested, alpha=alpha, observational_only=observational,
                        game_counts=panel_game_counts)
                    if self.context.primary:
                        report["historical_panel"] = panel_result
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
                panel_result = report.get("historical_panel")
                promoted = False
                if observational:
                    if (not panel_result or panel_result["contract"] != panel_contract
                            or [r["opponent_sha256"] for r in panel_result["results"]] != panel_contract["opponents"]):
                        raise ValueError("incomplete observational evaluation panel")
                    for row, games in zip(panel_result["results"], panel_game_counts):
                        self._validate_historical_result(row, games=games)
                    counts = {key: sum(row[key] for row in panel_result["results"])
                              for key in ("games", "wins", "draws", "losses")}
                    report.update(counts)
                    report.update(decision="use_latest", champion_evaluated=False,
                                   mode=self.settings.mode.value,
                                  score=(counts["wins"] + .5 * counts["draws"]) / counts["games"],
                                  win_rate=counts["wins"] / counts["games"],
                                   wall_seconds=sum(row["wall_seconds"] for row in panel_result["results"]))
                    if reference_only:
                        reference = panel_result["results"][0]
                        report.update(opponent_update=reference["opponent_update"],
                                      opponent_sha256=reference["opponent_sha256"],
                                      score_ci=reference["score_ci"], strength_verdict=reference["verdict"])
                        if "teammate_results" in reference:
                            report["teammate_results"] = reference["teammate_results"]
                else:
                    self._validate_historical_result(report, games=self.settings.arena_games)
                    promoted = report["score"] > 0.5 and (not panel_result or panel_result["promotion_allowed"])
                    report.update({
                        "champion_evaluated": True,
                        "opponent_update": self.state["best_update"], "opponent_sha256": self.state["best_sha256"],
                        "promoted": promoted, "decision": "promote" if promoted else "retain_best",
                        "best_update": update if promoted else self.state["best_update"],
                        "win_rate": report["wins"] / self.settings.arena_games,
                        "score_delta_vs_best": report["score"] - 0.5,
                        "promotion_rule": ("score_gt_0.5_and_every_historical_score_ge_0.5_and_no_confirmed_regression"
                                           if panel_result else "score_strictly_greater_than_0.5"),
                        "strength_verdict": report.get("verdict", "inconclusive"),
                    })
                report.update({
                    "evaluation_type": evaluation_type,
                    "schedule_contract": self._contract(), "round_alpha": self._round_alpha(),
                    self.milestone_key: milestone,
                    "schedule_unit": "environment_plies" if self.environment_schedule else "percent",
                    "environment_plies": cumulative.get("environment_plies", 0),
                    "statistical_family": self.state.get("statistical_family", "initial_schedule"),
                    "progress_percent": self.progress_percent(update=update, cumulative=cumulative),
                    "skipped_milestones": [p for p in self.milestones
                                           if self.state[self.completed_key] < p < milestone],
                    "candidate_update": update, "candidate_sha256": candidate_hash,
                    "candidate_snapshot": candidate.relative_to(self.directory).as_posix(),
                })
                atomic_json(report_path, report)
                next_state = copy.deepcopy(self.state)
                next_state.update({self.completed_key: milestone, "last_evaluated_update": update,
                                   "latest_evaluated_snapshot": report["candidate_snapshot"],
                                   "latest_evaluated_sha256": candidate_hash,
                                   "evaluation_type": evaluation_type})
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

    def _evaluate_historical_panel(self, panel_directory, panel, contract, candidate, candidate_hash,
                                   engine, match, tag, stop_requested, *, alpha, observational_only=False,
                                   game_counts=None):
        """One opponent at a time; persist complete matches for retry reuse."""
        rows = []
        previous = {}
        error = None
        if self.context.primary and self.state.get("rounds"):
            try:
                prior = json.loads((self.directory / self.state["rounds"][-1]).read_text(encoding="utf-8"))
                previous_panel = prior.get("historical_panel", {})
                if previous_panel.get("contract", {}).get("teammate_protocol") == contract.get("teammate_protocol"):
                    previous = {r["opponent_sha256"]: r for r in previous_panel.get("results", [])}
            except Exception as exc:
                error = f"historical prior results: {type(exc).__name__}: {exc}"
        synchronized_error(self.context, error)
        game_counts = self._historical_game_counts(panel) if game_counts is None else game_counts
        for index, entry in enumerate(panel):
            path = self.directory / f"historical_{tag}_{entry['id']}.json"
            games = game_counts[index]
            settings = replace(match, pairs=games // 4,
                               seed=match.seed + (index + 1) * 1_000_003)
            identity = {"candidate_sha256": candidate_hash, "opponent_sha256": entry["sha256"],
                        "contract": contract, "settings": asdict(settings), "alpha": alpha, "arena_version": ARENA_VERSION}
            result, error = None, None
            if self.context.primary:
                try:
                    if path.exists():
                        saved = json.loads(path.read_text(encoding="utf-8"))
                        if saved.get("identity") == identity:
                            result = saved["result"]
                            self._validate_historical_result(result, games=games)
                except Exception as exc:
                    error = f"historical evaluation retry: {type(exc).__name__}: {exc}"
            synchronized_error(self.context, error)
            result = self.context.broadcast_object(result)
            if result is None:
                empty_cache(self.context.device)
                result = run_match(candidate, panel_directory / entry["file"], settings, self.context,
                    self.directory / f"historical_games_{tag}_{entry['id']}", alpha=alpha,
                    candidate_sha256=candidate_hash, opponent_sha256=entry["sha256"],
                    candidate_engine=engine, stop_requested=stop_requested)
                error = None
                if self.context.primary:
                    try:
                        self._validate_historical_result(result, games=games)
                        atomic_json(path, {"identity": identity, "result": result})
                    except Exception as exc:
                        error = f"historical evaluation commit: {type(exc).__name__}: {exc}"
                synchronized_error(self.context, error)
            if self.context.primary:
                old = previous.get(entry["sha256"])
                confirmed = result["score_ci"][1] < .5 or bool(old and result["score_ci"][1] < old["score_ci"][0])
                rows.append(dict(opponent_id=entry["id"], opponent_sha256=entry["sha256"],
                    opponent_update=entry["update"], opponent_environment_plies=entry["environment_plies"],
                    games=result["games"], wins=result["wins"], draws=result["draws"], losses=result["losses"],
                    score=result["score"], score_ci=result["score_ci"],
                    previous_score=old["score"] if old else None,
                    score_change=result["score"] - old["score"] if old else None,
                    confirmed_regression=confirmed, below_half=result["score"] < .5,
                    verdict=result.get("verdict", "inconclusive"), wall_seconds=result.get("wall_seconds", 0.)))
                if "teammate_results" in result:
                    rows[-1]["teammate_results"] = result["teammate_results"]
        summary, error = None, None
        if self.context.primary:
            try:
                summary = dict(contract=contract, results=rows, games=sum(r["games"] for r in rows),
                               minimum_score=min(r["score"] for r in rows),
                               confirmed_regression=any(r["confirmed_regression"] for r in rows),
                               observational_only=observational_only,
                               all_confidently_above_half=all(r["score_ci"][0] > .5 for r in rows))
                if not observational_only:
                    summary["promotion_allowed"] = all(not r["below_half"] and not r["confirmed_regression"] for r in rows)
            except Exception as exc:
                error = f"historical panel commit: {type(exc).__name__}: {exc}"
        synchronized_error(self.context, error)
        return summary

    def _validate_historical_result(self, result, *, games=None):
        games = self.settings.historical_eval_games if games is None else games
        self._validate_result(result, games=games)
        if self.settings.arena_historical_teammate_fraction:
            splits = result.get("teammate_results", {})
            if set(splits) != {"current", "historical"}:
                raise ValueError("missing balanced teammate evaluation results")
            for split in splits.values():
                self._validate_result(split, games=games // 2)
            if any(sum(split[key] for split in splits.values()) != result[key]
                   for key in ("wins", "draws", "losses")):
                raise ValueError("teammate evaluation counts do not match the total")
