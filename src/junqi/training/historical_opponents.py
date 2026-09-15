"""Growing frozen training library and fixed evaluation panel for four-player PPO.

CPU weights use a bounded prepared RAM cache; the GPU keeps one inference
replica. Packed dtype slabs avoid thousands of parameter transfers. No optimizer
or critic is archived in the historical library.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import time
import uuid

import torch

from .arena import atomic_json, sha256_file, synchronized_error
from .checkpoint_format import CHECKPOINT_FORMAT_VERSION
from .inference_weights import install_packed_weights, matrix_parameter_names
from .historical_cache import HistoricalWeightCache
from .models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder, layout_sample_from_trace
from .rollout import FrozenPolicyActor


SCENARIOS = ("self_play", "historical_opponents", "historical_teammate", "historical_both")


def packed_weights(model):
    """Inference-compatible state dict backed by a few contiguous CPU slabs."""
    groups, state = {}, {}
    for name, value in model.state_dict().items():
        groups.setdefault(str(value.dtype), []).append((name, value))
    packs = {}
    for dtype, items in groups.items():
        # Coalesce on the source device before copying. A CUDA snapshot performs
        # one D2H transfer per dtype instead of one blocking copy per parameter.
        flat = torch.cat([value.detach().reshape(-1) for _, value in items]).cpu()
        offset = 0
        views = {}
        for name, value in items:
            end = offset + value.numel()
            state[name] = flat[offset:end].view(value.shape)
            views[name] = (offset, end, tuple(value.shape))
            offset = end
        packs[dtype] = {"flat": flat, "views": views}
    return state, packs


class HistoricalOpponents:
    def __init__(self, settings, run_directory, context):
        self.settings, self.context = settings, context
        self.directory = Path(run_directory).resolve() / "historical_opponents"
        self.catalog_path = self.directory / "catalog.json"
        self.entries = []
        self.progress = 0
        self.rng = random.Random(settings.seed + 7949 + 1_000_003 * context.rank)
        self.q, self.results = {}, {}
        self.active_id = None
        self.active_probability = 1.0
        self.active_panel_size = 0
        self.pending_cohort = None
        self.cohort_started = 0
        self.cohort_limit = settings.historical_cohort_games
        self.started = self.historical_started = self.completed = 0
        self.teammate_started = self.teammate_completed = self.mixed_completed = 0
        self.opponent_plies = self.teammate_plies = self.quality_update_games = 0
        self.role_credits = dict.fromkeys(SCENARIOS, 0.)
        self.scenario_started = dict.fromkeys(SCENARIOS, 0)
        self.scenario_results = {}
        self.teammate_results = {}
        self.seat_offsets = {}
        self.policy = self.layout = self.actor = None
        self.loaded_id = None
        self.device_packs = {}
        self.layout_queue = []
        self.load_count = self.upload_bytes = self.upload_calls = 0
        self.load_seconds = 0.0
        self._verified = set()
        self._catalog_dirty = False
        self.weight_cache = None
        self.archive_count = 0
        self.archive_seconds = 0.
        self._checkpoint_cpu_state = None
        if self.enabled and self.catalog_path.exists():
            catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            if not self._compatible_contract(catalog.get("contract"),
                    max((e["environment_plies"] for e in catalog["opponents"]), default=0)):
                raise ValueError("historical archive mode/rules/budget contract changed")
            self.entries = catalog["opponents"]
            self._catalog_dirty = catalog.get("contract") != self.contract()
            if len(self.evaluation_panel(include_future=True)) > 1 + len(settings.historical_snapshot_fractions):
                raise ValueError("historical archive exceeds its configured bound")

    @property
    def enabled(self):
        return self.settings.historical_enabled

    @property
    def threshold(self):
        return math.ceil((self.settings.target_environment_plies or 0) * self.settings.historical_start_fraction)

    @property
    def active(self):
        return self.enabled and self.progress >= self.threshold

    def contract(self):
        result = dict(mode=self.settings.mode.value, dead_rules_enabled=self.settings.dead_rules_enabled,
                    no_capture_draw_plies=self.settings.no_capture_draw_plies,
                    checkpoint_format=CHECKPOINT_FORMAT_VERSION,
                    target=self.settings.target_environment_plies,
                    start_fraction=self.settings.historical_start_fraction,
                    snapshot_fractions=list(self.settings.historical_snapshot_fractions))
        if self.settings.historical_checkpoint_start_fraction is not None:
            result["checkpoint_start_fraction"] = self.settings.historical_checkpoint_start_fraction
        if self.settings.historical_stage_mix_fraction:
            result["stage_mix_fraction"] = self.settings.historical_stage_mix_fraction
        return result

    def _compatible_contract(self, saved, progress):
        if saved == self.contract():
            return True
        previous = self.contract()
        previous.pop("checkpoint_start_fraction", None)
        previous.pop("stage_mix_fraction", None)
        # Adopt the extension before it can affect any archived checkpoint or
        # played mixed game. Other rules, budgets and protocols stay strict.
        return (saved == previous and progress < min(self.threshold, self.checkpoint_threshold))

    @property
    def checkpoint_threshold(self):
        fraction = self.settings.historical_checkpoint_start_fraction
        return math.ceil((self.settings.target_environment_plies or 0) * fraction) if fraction is not None else math.inf

    def evaluation_panel(self, *, include_future=False):
        return [e for e in self.entries if e.get("kind", "early") == "early"
                and (include_future or e["environment_plies"] <= self.progress)]

    def panel(self):
        return [e for e in self.entries if e["environment_plies"] <= self.progress]

    def update_progress(self, policy, layout, *, environment_plies, update):
        self.progress = int(environment_plies)
        if self._checkpoint_cpu_state is not None and self._checkpoint_cpu_state[:2] != (update, self.progress):
            self._checkpoint_cpu_state = None
        if not self.enabled:
            return
        self._configure_weight_cache(policy)
        # Archive at most one real set of parameters per completed update. A
        # late installation cannot fabricate versions at already passed steps.
        covered = {f for e in self.entries for f in e["milestones"]}
        due = [f for f in self.settings.historical_snapshot_fractions
               if self.progress >= math.ceil(f * self.settings.target_environment_plies) and f not in covered]
        archive = self.progress < self.threshold and (not self.entries or bool(due))
        if archive:
            self._archive_snapshot(policy, layout, update=update, kind="early", milestones=due)
        elif self._catalog_dirty:
            self._publish_catalog()
        if self.active and len(self.evaluation_panel()) < 2:
            raise RuntimeError("historical phase needs at least two real early snapshots; archive is incomplete")
        # Scheduling must not depend on cache size or I/O completion: cache-off
        # and cache-on runs consume identical RNG and select identical games.
        if (self.settings.historical_checkpoint_start_fraction is not None and self.active
                and self.cohort_started >= self.cohort_limit and self.pending_cohort is None):
            self.pending_cohort = self._choose_cohort()
        if self.weight_cache is not None:
            self.weight_cache.trim()
            if self.pending_cohort is not None and self.pending_cohort["id"] != self.loaded_id:
                entry = next(e for e in self.panel() if e["id"] == self.pending_cohort["id"])
                self.weight_cache.prefetch(entry)

    def _configure_weight_cache(self, policy):
        if self.weight_cache is None and self.settings.historical_cache_gib:
            matrix_dtype = ({"bfloat16": torch.bfloat16, "float16": torch.float16}.get(self.settings.amp)
                            if self.context.device.type == "cuda" else None)
            self.weight_cache = HistoricalWeightCache(self.directory, self.settings,
                matrix_names=matrix_parameter_names(policy), matrix_dtype=matrix_dtype,
                pin=self.context.device.type == "cuda")

    def _publish_catalog(self):
        error = None
        if self.context.primary:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                atomic_json(self.catalog_path, {"contract": self.contract(), "opponents": self.entries})
            except Exception as exc:
                error = f"historical catalog: {type(exc).__name__}: {exc}"
        synchronized_error(self.context, error)
        self.entries = self.context.broadcast_object(self.entries)
        self._catalog_dirty = False

    def _archive_snapshot(self, policy, layout, *, update, kind, milestones):
        started, error, payload = time.perf_counter(), None, None
        if self.context.primary:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                name = f"{kind}_{self.progress:012d}_u{update:09d}_{uuid.uuid4().hex[:8]}.pt"
                path = self.directory / name
                p, pp = packed_weights(policy)
                l, lp = packed_weights(layout)
                payload = dict(format_version=CHECKPOINT_FORMAT_VERSION, update=update,
                               mode=self.settings.mode.value, algorithm="ppo",
                               dead_rules_enabled=self.settings.dead_rules_enabled,
                               reason="historical_opponent_inference_only",
                               config=self.settings.serializable(), policy=p, layout=l,
                               packed_weights={"policy": pp, "layout": lp})
                temp = path.with_suffix(".pt.tmp")
                torch.save(payload, temp)
                temp.replace(path)
                entry = dict(id=path.stem, file=name, sha256=sha256_file(path), kind=kind,
                             environment_plies=self.progress, update=update, milestones=milestones,
                             bytes=path.stat().st_size)
                self.entries.append(entry)
                if self.weight_cache is not None:
                    self.weight_cache.remember_payload(entry, payload)
            except Exception as exc:
                error = f"historical snapshot: {type(exc).__name__}: {exc}"
        synchronized_error(self.context, error)
        self._publish_catalog()
        self.archive_count += 1
        self.archive_seconds += time.perf_counter() - started
        weights = {name: payload[name] for name in ("policy", "layout")} if payload is not None else None
        self._checkpoint_cpu_state = (update, self.progress, weights)
        return weights

    def archive_checkpoint(self, policy, layout, *, environment_plies, update):
        """Called before gathering checkpoint rank state, once per saved version.

        Return the same CPU weights for the full checkpoint to avoid a second
        GPU-to-CPU copy. A failed full save leaves only an orphan disk snapshot;
        exact resume prunes its metadata using the saved catalog lineage.
        """
        self.progress = int(environment_plies)
        if self._checkpoint_cpu_state is not None and self._checkpoint_cpu_state[:2] == (update, self.progress):
            return self._checkpoint_cpu_state[2]
        if not self.enabled or self.progress < self.checkpoint_threshold:
            return None
        if any(e["update"] == update and e["environment_plies"] == self.progress for e in self.entries):
            return None
        self._configure_weight_cache(policy)
        return self._archive_snapshot(policy, layout, update=update, kind="checkpoint", milestones=[])

    def probabilities(self):
        panel = self.panel()
        if not panel:
            return {}
        high = max(self.q.get(e["id"], 0.) for e in panel)
        weights = [math.exp(max(-60., self.q.get(e["id"], 0.) - high)) for e in panel]
        total, uniform = sum(weights), self.settings.historical_uniform_fraction
        probabilities = {e["id"]: (1 - uniform) * w / total + uniform / len(panel)
                         for e, w in zip(panel, weights, strict=True)}
        mix = self.settings.historical_stage_mix_fraction
        if mix:
            # Equal mass per occupied 10%-progress band, then uniform within a
            # band. Dense adjacent checkpoints cannot drown out earlier styles.
            bands = {}
            for e in panel:
                band = min(9, int(10 * e["environment_plies"] / self.settings.target_environment_plies))
                bands.setdefault(band, []).append(e["id"])
            for identifiers in bands.values():
                for identifier in identifiers:
                    probabilities[identifier] = ((1 - mix) * probabilities[identifier]
                                                  + mix / len(bands) / len(identifiers))
        return probabilities

    def _choose_cohort(self):
        probabilities = self.probabilities()
        identifier = self.rng.choices(list(probabilities), weights=list(probabilities.values()))[0]
        return dict(id=identifier, probability=probabilities[identifier], panel_size=len(probabilities))

    def role_targets(self):
        opponent, teammate = self.settings.historical_training_fraction, self.settings.historical_teammate_fraction
        return dict(zip(SCENARIOS, ((1 - opponent) * (1 - teammate), opponent * (1 - teammate),
                                    (1 - opponent) * teammate, opponent * teammate), strict=True))

    @property
    def credit(self):
        return self.started * self.settings.historical_training_fraction - self.historical_started

    def mix_contract(self):
        return dict(opponent_fraction=self.settings.historical_training_fraction,
                    teammate_fraction=self.settings.historical_teammate_fraction)

    def begin_rollout(self, pool, actor, behavior_version):
        if not self.active:
            return
        pinned = {identifier for s in pool.slots for identifier in (s.opponent_id, s.teammate_id)
                  if identifier is not None}
        if pinned and pinned != {self.active_id}:
            raise RuntimeError("unfinished games do not match the resident historical cohort")
        if self.active_id is None or (not pinned and self.cohort_started >= self.cohort_limit):
            choice = self.pending_cohort or self._choose_cohort()
            self.pending_cohort = None
            self.active_id, self.active_probability = choice["id"], choice["probability"]
            self.active_panel_size = choice["panel_size"]
            self.cohort_started = 0
            # Long unfinished games can delay a swap. Amortize that drain and
            # repay admission debt instead of repeatedly starving the 20% mix.
            debt = sum(max(0., self.role_credits[k]) for k in SCENARIOS[1:])
            self.cohort_limit = max(self.settings.historical_cohort_games, math.ceil(2 * debt))
            self.layout_queue.clear()
        self._load_active(actor)
        self.actor = FrozenPolicyActor(self.policy, amp_dtype=actor.amp_dtype, max_batch_size=actor.max_batch_size)

    def attach_cache(self, actor, behavior_version):
        if not self.active:
            return
        cache = actor.policy._fixed_kv_store
        if cache is not None:
            self.policy.start_ppo_inference_cache(capacity=cache.capacity, behavior_version=behavior_version)
            with torch.inference_mode(), torch.autocast(
                device_type=self.context.device.type, dtype=actor.amp_dtype or torch.float32,
                enabled=actor.amp_dtype is not None and self.context.device.type in ("cuda", "npu"),
            ):
                self.policy._fixed_kv_store.share_storage(cache)

    def clear_caches(self):
        if self.policy is not None:
            self.policy.clear_inference_board_cache()

    def release_replica(self):
        """Arena needs its own sequential opponent; keep only cohort metadata."""
        self.clear_caches()
        self.actor = self.policy = self.layout = None
        self.device_packs.clear()
        self.loaded_id = None

    def close(self):
        if self.weight_cache is not None:
            self.weight_cache.close()
        self._checkpoint_cpu_state = None

    def _load_active(self, actor):
        if self.loaded_id == self.active_id:
            return
        boundary = time.perf_counter()
        entry = next(e for e in self.panel() if e["id"] == self.active_id)
        cached = self.weight_cache is not None
        if cached:
            packs = self.weight_cache.get(entry)
        else:
            path = self.directory / entry["file"]
            if entry["sha256"] not in self._verified:
                if sha256_file(path) != entry["sha256"]:
                    raise ValueError("frozen historical snapshot hash changed")
                self._verified.add(entry["sha256"])
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            if (payload["mode"] != self.settings.mode.value or payload["format_version"] != CHECKPOINT_FORMAT_VERSION
                    or payload["config"]["no_capture_draw_plies"] != self.settings.no_capture_draw_plies):
                raise ValueError("historical opponent checkpoint contract mismatch")
            packs = payload["packed_weights"]
        self.clear_caches()
        if self.policy is None:
            # Preserve the learner's random stream on lazy initialization and resume.
            with torch.random.fork_rng(devices=[]):
                self.policy = GamePolicyTransformer(self.settings.model)
                self.layout = PieceConditionedLayoutPointerDecoder(self.settings.model)
        for name, module in (("policy", self.policy), ("layout", self.layout)):
            existing = {dtype: tensor for (model, dtype), tensor in self.device_packs.items() if model == name}
            slabs, byte_count, calls = install_packed_weights(module, packs[name],
                self.context.device, slabs=existing or None,
                matrix_dtype=actor.amp_dtype if not cached and name == "policy" and self.context.device.type == "cuda" else None,
                non_blocking=cached and self.context.device.type == "cuda")
            self.device_packs.update({(name, dtype): tensor for dtype, tensor in slabs.items()})
            self.upload_calls += calls
            self.upload_bytes += byte_count
            module.eval().requires_grad_(False)
        if cached:
            if self.context.device.type == "cuda":
                # One fence for all dtype slabs, not a synchronization for each
                # weight. Pinned buffers must survive until the copies finish.
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.context.device))
                event.synchronize()
            self.weight_cache.finish_upload()
        self.loaded_id = self.active_id
        self.load_count += 1
        self.load_seconds += time.perf_counter() - boundary

    def assignment(self):
        if not self.active:
            return None, None, 0
        # Product quotas give independent opponent/teammate marginals, including
        # overlap. Repay all three mixed categories after a cohort drain; using
        # the same 1-in-5 counter for both roles would incorrectly couple them.
        self.started += 1
        targets = self.role_targets()
        for key, fraction in targets.items():
            self.role_credits[key] += fraction
        allowed = (["self_play"] if self.cohort_started >= self.cohort_limit else
                   [k for k in SCENARIOS if targets[k] > 0])
        most = max(self.role_credits[k] for k in allowed)
        scenario = self.rng.choice([k for k in allowed if self.role_credits[k] >= most - 1e-9])
        self.role_credits[scenario] -= 1
        if scenario not in self.seat_offsets:
            self.seat_offsets[scenario] = self.rng.randrange(4)
        seat = (self.scenario_started[scenario] + self.seat_offsets[scenario]) % 4
        self.scenario_started[scenario] += 1
        opponent = scenario in ("historical_opponents", "historical_both")
        teammate = scenario in ("historical_teammate", "historical_both")
        if opponent or teammate:
            if self.active_id is None:
                raise RuntimeError("mixed game admission requires a resident cohort")
            self.cohort_started += 1
            self.historical_started += int(opponent)
            self.teammate_started += int(teammate)
        return self.active_id if opponent else None, self.active_id if teammate else None, seat

    def old_layouts(self, count=2):
        if not 1 <= count <= 3:
            raise ValueError("a mixed four-player game needs 1..3 frozen layouts")
        if len(self.layout_queue) < count:
            with torch.inference_mode():
                self.layout_queue.extend(self.layout.sample_layouts(
                    max(count, 2 * self.settings.layout_prefetch_games), self.settings.mode, temperature=.7))
        result = self.layout_queue[:count]
        del self.layout_queue[:count]
        return result

    def record_result(self, slot, reward):
        opponent, teammate = slot.opponent_id, getattr(slot, "teammate_id", None)
        scenario = ("historical_both" if opponent and teammate else "historical_opponents" if opponent
                    else "historical_teammate" if teammate else "self_play")
        if not opponent and not teammate and getattr(slot, "historical_scenario", None) is None:
            return
        if any(i != self.active_id for i in (opponent, teammate) if i is not None):
            raise RuntimeError("historical game changed opponents before completion")
        outcome = "wins" if reward > 0 else "losses" if reward < 0 else "draws"
        def record(table, key):
            table.setdefault(key, dict(wins=0, draws=0, losses=0))[outcome] += 1
        record(self.scenario_results, scenario)
        if opponent:
            record(self.results, opponent)
            self.completed += 1
        if teammate:
            record(self.teammate_results, teammate)
            self.teammate_completed += 1
        self.mixed_completed += int(bool(opponent or teammate))
        # A weak teammate must not make an easy opponent look challenging.
        # Only the fixed-current-teammate stratum informs opponent quality.
        if scenario != "historical_opponents":
            return
        score = 1. if reward > 0 else 0. if reward < 0 else .5
        # Appendix N inspired importance-weighted quality update, adapted to
        # draws and batched cohorts. Arena results never feed this sampler.
        self.q[opponent] = self.q.get(opponent, 0.) - self.settings.historical_learning_rate * score / (
            (self.active_panel_size or len(self.panel())) * self.active_probability)
        self.quality_update_games += 1

    def state_dict(self):
        return dict(version=2, contract=self.contract(), mix_contract=self.mix_contract(),
                    entries=copy.deepcopy(self.panel()), progress=self.progress,
                    rng=self.rng.getstate(), q=dict(self.q), results=copy.deepcopy(self.results),
                    active_id=self.active_id, active_probability=self.active_probability,
                    active_panel_size=self.active_panel_size, pending_cohort=copy.deepcopy(self.pending_cohort),
                    cohort_started=self.cohort_started, cohort_limit=self.cohort_limit, credit=self.credit, started=self.started,
                    historical_started=self.historical_started, completed=self.completed,
                    opponent_plies=self.opponent_plies, teammate_plies=self.teammate_plies,
                    teammate_started=self.teammate_started, teammate_completed=self.teammate_completed,
                    mixed_completed=self.mixed_completed, quality_update_games=self.quality_update_games,
                    role_credits=dict(self.role_credits), scenario_started=dict(self.scenario_started),
                    scenario_results=copy.deepcopy(self.scenario_results),
                    teammate_results=copy.deepcopy(self.teammate_results), seat_offsets=dict(self.seat_offsets),
                    layout_queue=[dict(mode=s.mode.value, position_indices=s.position_indices,
                                       old_log_probs=s.old_log_probs) for s in self.layout_queue])

    def load_state_dict(self, state):
        version = state.get("version")
        if not self.enabled or version not in (1, 2) or not self._compatible_contract(state.get("contract"), state["progress"]):
            raise ValueError("historical opponent resume contract changed")
        if version == 2 and state.get("mix_contract") != self.mix_contract():
            raise ValueError("historical role fractions changed across resume")
        if version == 1 and state["started"] and self.settings.historical_teammate_fraction:
            raise ValueError("legacy opponent-only active games require explicit role-schedule migration")
        disk = {e["id"]: e for e in self.entries}
        for entry in state["entries"]:
            if disk.get(entry["id"]) != entry or not (self.directory / entry["file"]).is_file():
                raise ValueError("historical checkpoint requires a missing or changed immutable opponent")
        # A disk snapshot from an update after latest.pt is on an abandoned
        # trajectory. Retain its file, but never silently admit it on replay.
        self._catalog_dirty = (self._catalog_dirty or self.entries != state["entries"]
                               or state.get("contract") != self.contract())
        self.entries = copy.deepcopy(state["entries"])
        for name in ("progress", "q", "results", "active_id", "active_probability", "cohort_started",
                     "started", "historical_started", "completed", "opponent_plies"):
            setattr(self, name, copy.deepcopy(state[name]))
        self.rng.setstate(state["rng"])
        self.active_panel_size = state.get("active_panel_size", len(self.panel()))
        self.pending_cohort = copy.deepcopy(state.get("pending_cohort"))
        if self.pending_cohort is not None:
            choice = self.pending_cohort
            if (choice["id"] not in {e["id"] for e in self.panel()}
                    or not 0 < choice["probability"] <= 1 or choice["panel_size"] < 1):
                raise ValueError("invalid saved historical prefetch choice")
        self.cohort_limit = state.get("cohort_limit", self.settings.historical_cohort_games)
        if version == 2:
            for name in ("teammate_started", "teammate_completed", "mixed_completed", "teammate_plies",
                         "quality_update_games", "role_credits", "scenario_started", "scenario_results",
                         "teammate_results", "seat_offsets"):
                setattr(self, name, copy.deepcopy(state[name]))
        else:
            self.scenario_started.update(self_play=self.started - self.historical_started,
                                         historical_opponents=self.historical_started)
            self.role_credits = {k: v * self.started - self.scenario_started[k] for k, v in self.role_targets().items()}
            self.scenario_results = {"historical_opponents": {
                k: sum(row[k] for row in self.results.values()) for k in ("wins", "draws", "losses")}}
            self.mixed_completed = self.quality_update_games = self.completed
        self.layout_queue = [layout_sample_from_trace(s["mode"], s["position_indices"], s["old_log_probs"])
                             for s in state["layout_queue"]]

    def metrics(self):
        return {"historical/active": int(self.active), "historical/threshold_environment_plies": self.threshold,
                "historical/evaluation_opponents": len(self.evaluation_panel()),
                "historical/checkpoint_start_environment_plies": (self.checkpoint_threshold
                    if math.isfinite(self.checkpoint_threshold) else None),
                "historical/stage_mix_fraction": self.settings.historical_stage_mix_fraction,
                "historical/archive_count": self.archive_count,
                "historical/archive_seconds": self.archive_seconds,
                "historical/archive_disk_bytes": sum(e.get("bytes", 0) for e in self.panel()),
                **(self.weight_cache.metrics() if self.weight_cache is not None else {}),
                "historical/opponents": len(self.panel()), "historical/games_started": self.historical_started,
                "historical/all_games_started_after_half": self.started, "historical/games_completed": self.completed,
                "historical/admission_fraction": self.historical_started / max(1, self.started),
                "historical/target_training_fraction": self.settings.historical_training_fraction,
                "historical/teammate_games_started": self.teammate_started,
                "historical/teammate_games_completed": self.teammate_completed,
                "historical/teammate_admission_fraction": self.teammate_started / max(1, self.started),
                "historical/target_teammate_fraction": self.settings.historical_teammate_fraction,
                "historical/both_games_started": self.scenario_started["historical_both"],
                "historical/mixed_games_completed": self.mixed_completed,
                "historical/mixed_admission_fraction": (self.historical_started + self.teammate_started
                    - self.scenario_started["historical_both"]) / max(1, self.started),
                "historical/teammate_admission_debt_games": self.started * self.settings.historical_teammate_fraction - self.teammate_started,
                "historical/quality_update_games": self.quality_update_games,
                "historical/cohort_limit": self.cohort_limit,
                "historical/admission_debt_games": self.credit,
                "historical/opponent_plies": self.opponent_plies,
                "historical/teammate_plies": self.teammate_plies,
                "historical/resident_models": int(self.policy is not None),
                "historical/weight_upload_count": self.load_count,
                "historical/weight_upload_calls": self.upload_calls,
                "historical/weight_upload_bytes": self.upload_bytes,
                "historical/weight_load_seconds": self.load_seconds,
                "historical/resident_weight_bytes": sum(v.numel() * v.element_size() for v in self.device_packs.values())}

    def write_status(self, pool):
        if self.enabled and self.context.primary:
            probabilities = self.probabilities()
            atomic_json(self.directory / "status.json", dict(
                **self.metrics(), progress=self.progress, active_id=self.active_id,
                pending_cohort=copy.deepcopy(self.pending_cohort),
                unfinished_historical_games=sum(s.opponent_id is not None or s.teammate_id is not None for s in pool.slots),
                scenarios=[dict(id=k, target_fraction=v, started=self.scenario_started[k],
                                actual_fraction=self.scenario_started[k] / max(1, self.started),
                                results=self.scenario_results.get(k, {})) for k, v in self.role_targets().items()],
                opponents=[dict(e, probability=probabilities.get(e["id"], 0.),
                                results=self.results.get(e["id"], {}),
                                teammate_results=self.teammate_results.get(e["id"], {})) for e in self.panel()],
                rank=self.context.rank, world_size=self.context.world_size))
