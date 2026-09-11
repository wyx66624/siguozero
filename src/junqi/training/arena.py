"""Shared infrastructure for separate two-player and four-player arenas.

This is independent evaluation, not an extra source of training rollouts.
The referee owns both hidden layouts; each actor sees only its PolicyState.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable, Iterator, Sequence
import uuid

import numpy as np
import torch

from .accelerator import is_accelerator, manual_seed_all
from .distributed import DistributedContext
from .encoding import GameHistory
from .inference import InferenceEngine
from .modes import TrainingMode, new_game, normalize_mode


ARENA_VERSION = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MatchSettings:
    # Historical API name: in four-player mode this counts four-game rotation
    # groups (--groups), NOT two-game pairs. All public stats label their unit.
    pairs: int = 200
    seed: int = 20260908
    temperature: float = 1.0
    layout_temperature: float = 0.7
    max_plies: int = 2000
    temporal_cache_entries: int = 8
    smoke_test: bool = False
    mode: str = TrainingMode.TWO_PLAYER.value
    parallel_games: int = 32
    inference_batch_size: int = 32
    environment_workers: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_mode(self.mode).value)
        for name in ("pairs", "max_plies", "temporal_cache_entries", "parallel_games",
                     "inference_batch_size", "environment_workers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("temperature", "layout_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (type(self.seed) is not int or self.seed < 0
                or self.seed + self.seed_stride * self.pairs >= 2**63):
            raise ValueError("seed range must fit a nonnegative signed 64-bit integer")

    @property
    def games_per_group(self) -> int:
        return 2 if self.mode == TrainingMode.TWO_PLAYER.value else 4

    @property
    def seed_stride(self) -> int:
        return 2 + self.games_per_group  # two independent Layout seeds + games

    @property
    def effective_temporal_cache_entries(self) -> int:
        # Each checkpoint owns half the seats. Retain a few recent prefixes for
        # each of its live players instead of thrashing an eight-entry cache.
        return max(self.temporal_cache_entries,
                   self.parallel_games * (self.games_per_group // 2) * 3)

    def board_cache_entries(self, max_transitions: int) -> int:
        return max(2048, self.parallel_games * (self.games_per_group // 2)
                   * (max_transitions + 1))


def protocol_for(mode: TrainingMode | str):
    """Keep team/seat schedules and aggregation in their own implementations."""
    if normalize_mode(mode) is TrainingMode.TWO_PLAYER:
        from . import arena_two_player
        return arena_two_player
    from . import arena_four_player
    return arena_four_player


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def synchronized_error(context: DistributedContext, error: str | None) -> None:
    """All ranks either advance or report a failed evaluation together."""
    if context.any(error is not None):
        errors = context.gather_object(error)
        errors = context.broadcast_object(errors)
        raise RuntimeError(f"arena failed (no result accepted): {errors}")


def seed_inference(seed: int, engine: InferenceEngine) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if is_accelerator(engine.policy.device):
        manual_seed_all(engine.policy.device.type, seed)


def paired_statistics(
    pair_scores: Sequence[float], *, alpha: float = 0.05, bootstrap_seed: int = 0
) -> dict[str, Any]:
    """Conservative bounded-mean CI; bootstrap is a descriptive companion.

    Hoeffding's bound remains non-degenerate for small all-win samples. It
    assumes independent pairs, NOT independent games within a seat pair.
    The caller may spend alpha across repeated rounds/opponents.
    """
    values = np.asarray(pair_scores, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("at least one pair score is required")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("pair scores must be finite and in [0, 1]")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    mean = float(values.mean())
    radius = math.sqrt(math.log(2 / alpha) / (2 * len(values)))
    lower, upper = max(0.0, mean - radius), min(1.0, mean + radius)
    # Bounded chunks keep large formal matches from allocating B * N arrays.
    rng = np.random.default_rng(bootstrap_seed)
    means = np.concatenate([
        values[rng.integers(len(values), size=(128, len(values)))].mean(axis=1)
        for _ in range(16)
    ])
    bootstrap = [float(x) for x in np.quantile(means, [0.025, 0.975])]
    verdict = (
        "ahead_of_opponent" if lower > 0.5 else
        "behind_opponent" if upper < 0.5 else "inconclusive"
    )
    return {
        "pairs": len(values), "score": mean,
        "score_ci": [lower, upper], "ci_method": "paired_hoeffding",
        "alpha": alpha, "bootstrap_ci95_descriptive": bootstrap,
        "verdict": verdict,
    }


def validate_engines(
    candidate: InferenceEngine, opponent: InferenceEngine,
    *, mode: TrainingMode | str = TrainingMode.TWO_PLAYER,
) -> None:
    expected = normalize_mode(mode)
    if candidate.mode is not expected or opponent.mode is not expected:
        raise ValueError(f"both checkpoints must be {expected.value}; cannot mix modes")
    if candidate.policy.config.dead_rules_enabled != opponent.policy.config.dead_rules_enabled:
        raise ValueError("cannot compare checkpoints with different dead rules")
    for engine in (candidate, opponent):
        for label, model in (("policy", engine.policy), ("layout", engine.layout)):
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError(f"{label} must be frozen in eval mode")


def play_pair(
    candidate: InferenceEngine, opponent: InferenceEngine,
    pair_index: int, settings: MatchSettings,
) -> list[dict[str, Any]]:
    """Backward-compatible two-player API; four-player uses play_group."""
    from .arena_two_player import play_group
    return play_group(candidate, opponent, pair_index, settings)


def play_game(
    candidate: InferenceEngine, opponent: InferenceEngine, settings: MatchSettings,
    *, setups, candidate_seats: tuple[int, ...], action_seed: int,
) -> dict[str, Any]:
    """Single-game reference using the same per-game RNG as parallel matches."""
    specs = [{"setups": setups, "candidate_seats": candidate_seats,
              "action_seed": action_seed, "metadata": {}}]
    single = replace(settings, parallel_games=1, environment_workers=1)
    return [row for batch in _iter_games(candidate, opponent, specs, single)
            for row in batch][0]


class _ArenaGame:
    def __init__(self, candidate: InferenceEngine, opponent: InferenceEngine,
                 settings: MatchSettings, spec: dict[str, Any]) -> None:
        self.settings = settings
        self.spec = spec
        self.rng = random.Random(spec["action_seed"])
        self.game = new_game(
            normalize_mode(settings.mode), setups=spec["setups"], seed=spec["action_seed"],
            max_plies=settings.max_plies,
            dead_rules_enabled=candidate.policy.config.dead_rules_enabled,
        )
        seats = spec["candidate_seats"]
        self.candidate_team = self.game.team_of(seats[0])
        if set(seats) != {seat for seat in range(self.game.config.player_count)
                          if self.game.team_of(seat) == self.candidate_team}:
            raise ValueError("candidate must control exactly one complete team")
        # Same-window histories are identical referee records. Only state_for
        # the player to act is passed to a model, preserving private information.
        by_window = {}
        self.histories = []
        for engine in (candidate, opponent):
            window = engine.policy.config.max_transitions
            if window not in by_window:
                by_window[window] = GameHistory.initialize(self.game, engine.mode,
                                                           max_transitions=window)
            self.histories.append(by_window[window])
        self.unique_histories = list(by_window.values())
        self.actions: list[dict[str, Any]] = []

    def observation(self):
        index = 0 if self.game.team_of(self.game.current_player) == self.candidate_team else 1
        return index, self.histories[index].state_for(self.game)

    def step(self, action: tuple[int, int]) -> None:
        result = self.game.step(action)  # illegal moves fail, never count as losses
        for history in self.unique_histories:
            history.append_after_step(self.game)
        self.actions.append({"ply": self.game.ply_count, "player": result.player,
                             "source": result.action[0], "destination": result.action[1],
                             "combat": result.combat.value})

    def record(self) -> dict[str, Any]:
        game = self.game
        if not game.is_terminal or game.result is None:
            raise ValueError("missing terminal result")
        reward = (0 if game.result.winner_team is None
                  else 1 if game.result.winner_team == self.candidate_team else -1)
        seats = self.spec["candidate_seats"]
        rewards = list(game.rewards())
        if rewards != [reward if seat in seats else -reward
                       for seat in range(game.config.player_count)]:
            raise ValueError("terminal rewards disagree with team outcome")
        return {
            "mode": self.settings.mode, "candidate_team": self.candidate_team,
            "candidate_seats": sorted(seats), "action_seed": self.spec["action_seed"],
            "candidate_reward": reward, "candidate_score": (reward + 1) / 2,
            "player_rewards": rewards, "active_players": list(game.active_players),
            "plies": game.ply_count, "terminal_reason": game.result.reason.value,
            "winner_team": game.result.winner_team, "actions": self.actions,
            **self.spec["metadata"],
        }


def _observe(game: _ArenaGame):
    return game.observation()


def _step_game(item):
    game, action = item
    game.step(action)


def _iter_games(candidate: InferenceEngine, opponent: InferenceEngine,
                specs: Iterable[dict[str, Any]], settings: MatchSettings,
                *, yield_every: int = 32) -> Iterator[list[dict[str, Any]]]:
    """Refill individual games immediately; batch inference by model owner.

    Yield regularly even with no terminal games so distributed ranks can
    exchange cancellation/failure signals without waiting for a long game.
    """
    validate_engines(candidate, opponent, mode=settings.mode)
    engines = (candidate, opponent)
    saved = [(engine.policy.config, engine.actor.max_batch_size) for engine in engines]
    executor = None
    source = iter(specs)
    active: list[_ArenaGame] = []
    exhausted = False
    completed = []
    try:
        for engine in engines:
            engine.policy.clear_inference_board_cache()
            engine.policy.config = replace(
                engine.policy.config,
                inference_temporal_cache_entries=settings.effective_temporal_cache_entries,
                inference_board_cache_entries=settings.board_cache_entries(
                    engine.policy.config.max_transitions),
            )
            engine.actor.max_batch_size = settings.inference_batch_size
            engine.policy.start_inference_board_cache()
        if settings.environment_workers > 1 and settings.parallel_games > 1:
            executor = ThreadPoolExecutor(max_workers=min(settings.environment_workers, settings.parallel_games))
        ticks = 0
        while active or not exhausted:
            while len(active) < settings.parallel_games and not exhausted:
                try:
                    spec = next(source)
                except StopIteration:
                    exhausted = True
                    break
                active.append(_ArenaGame(candidate, opponent, settings, spec))
            if not active:
                break
            observations = (list(executor.map(_observe, active)) if executor
                            else [_observe(game) for game in active])
            pending = []
            for index, engine in enumerate(engines):
                selected = [(game, state) for game, (owner, state) in zip(active, observations)
                            if owner == index]
                if not selected:
                    continue
                uniforms = torch.tensor([[[game.rng.random(), game.rng.random()]]
                                         for game, _ in selected], dtype=torch.float64)
                actions, _ = engine.actor.sample(
                    [state for _, state in selected], count=1,
                    temperature=settings.temperature, return_log_probs=False,
                    sampling_uniforms=uniforms,
                )
                if len(actions) != len(selected) or any(len(item) != 1 for item in actions):
                    raise ValueError("arena actor must return one action for every requested game")
                pending.extend((game, sampled[0]) for (game, _), sampled in zip(selected, actions))
            if executor:
                list(executor.map(_step_game, pending))
            else:
                for item in pending:
                    _step_game(item)
            survivors = []
            for game in active:
                if game.game.is_terminal:
                    completed.append(game.record())
                else:
                    survivors.append(game)
            active = survivors
            ticks += 1
            if ticks % yield_every == 0:
                yield completed
                completed = []
        if completed:
            yield completed
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for engine, (config, batch_size) in zip(engines, saved):
            engine.policy.clear_inference_board_cache()
            engine.policy.config = config
            engine.actor.max_batch_size = batch_size


def iter_play_groups(candidate: InferenceEngine, opponent: InferenceEngine,
                     group_indices: Iterable[int], settings: MatchSettings) -> Iterator[list[dict[str, Any]]]:
    protocol = protocol_for(settings.mode)
    specs = (spec for index in group_indices
             for spec in protocol.prepare_group(candidate, opponent, index, settings))
    yield from _iter_games(candidate, opponent, specs, settings)


def play_groups(candidate: InferenceEngine, opponent: InferenceEngine,
                group_indices: Iterable[int], settings: MatchSettings) -> list[dict[str, Any]]:
    """Execute any complete group shard on one device using a bounded game pool."""
    records = [row for batch in iter_play_groups(candidate, opponent, group_indices, settings)
               for row in batch]
    return sorted(records, key=lambda row: (row.get("group_index", row.get("pair_index")),
                                           row.get("rotation", row.get("candidate_seat"))))


def summarize_games(
    records: list[dict[str, Any]], settings: MatchSettings, *, alpha: float,
) -> dict[str, Any]:
    return protocol_for(settings.mode).summarize_games(records, settings, alpha=alpha)


def summarize_groups(
    records: list[dict[str, Any]], settings: MatchSettings, *, alpha: float,
    group_key: str, leg_key: str,
) -> dict[str, Any]:
    size = settings.games_per_group
    records = sorted(records, key=lambda row: (row[group_key], row[leg_key]))
    expected = [(i, leg) for i in range(settings.pairs) for leg in range(size)]
    if [(r[group_key], r[leg_key]) for r in records] != expected:
        raise ValueError("missing, duplicate, or unpaired evaluation games")
    for record in records:
        if record["candidate_reward"] not in (-1, 0, 1):
            raise ValueError("invalid reward")
        if record["candidate_score"] != (record["candidate_reward"] + 1) / 2:
            raise ValueError("score does not match reward")
    stats = paired_statistics([
        sum(record["candidate_score"] for record in records[i:i + size]) / size
        for i in range(0, len(records), size)
    ], alpha=alpha, bootstrap_seed=settings.seed)
    results = Counter(r["candidate_reward"] for r in records)
    reasons = dict(Counter(r["terminal_reason"] for r in records))
    stats.update({
        "games": len(records), "wins": results[1], "draws": results[0],
        "losses": results[-1], "draw_rate": results[0] / len(records),
        "terminal_reasons": reasons,
        "plies_mean": sum(r["plies"] for r in records) / len(records),
        "mode": settings.mode, "games_per_group": size,
        "warnings": [],
    })
    if reasons.get("max_plies_draw", 0) / len(records) > 0.1:
        stats["warnings"].append("more_than_10_percent_max_plies_draws")
    if stats["draw_rate"] > 0.8:
        stats["warnings"].append("more_than_80_percent_draws")
    if settings.smoke_test:
        stats["verdict"] = "smoke_test_not_strength_evidence"
    return stats


def run_match(
    candidate_path: Path, opponent_path: Path, settings: MatchSettings,
    context: DistributedContext, output_prefix: Path, *, alpha: float = 0.05,
    candidate_sha256: str | None = None, opponent_sha256: str | None = None,
    candidate_engine: InferenceEngine | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """Shard whole statistical groups, never split paired/rotated games.

    A trainer can lend its frozen candidate to avoid a second accelerator copy.
    The caller owns restoring its model configuration, flags, and RNG state.
    """
    start = time.monotonic()
    protocol = protocol_for(settings.mode)
    attempt = context.broadcast_object(uuid.uuid4().hex if context.primary else None)
    error = None
    try:
        # Identically named but different node-local files must not silently
        # become different opponents on different ranks.
        for path, expected in ((candidate_path, candidate_sha256), (opponent_path, opponent_sha256)):
            if expected is not None and sha256_file(path) != expected:
                raise ValueError(f"rank {context.rank}: checkpoint hash mismatch: {path}")
        if candidate_engine is None:
            candidate = InferenceEngine.from_checkpoint(
                candidate_path, device=str(context.device), mode=settings.mode,
                temporal_cache_entries=settings.temporal_cache_entries,
            )
        else:
            candidate = candidate_engine
            update = getattr(candidate, "checkpoint_update", None)
            if type(update) is not int or update < 0:
                raise ValueError("borrowed candidate must identify its checkpoint update")
            for model in (candidate.policy, candidate.layout):
                actual = model.device
                expected = context.device
                if actual.type != expected.type or (
                    actual.index is not None and expected.index is not None
                    and actual.index != expected.index
                ):
                    raise ValueError("borrowed candidate must use the arena device")
        opponent = InferenceEngine.from_checkpoint(
            opponent_path, device=str(context.device), mode=settings.mode,
            temporal_cache_entries=settings.temporal_cache_entries,
        )
        if candidate_engine is not None:
            # Training may explicitly use fp32/fp16 even on bf16-capable hardware.
            # Both contestants must use the same arithmetic for a fair match.
            opponent.actor.amp_dtype = candidate.actor.amp_dtype
        validate_engines(candidate, opponent, mode=settings.mode)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        error = f"rank {context.rank} load: {type(exc).__name__}: {exc}"
    synchronized_error(context, error)
    games = []
    shard = output_prefix.with_name(f"{output_prefix.name}.{attempt}.rank{context.rank:03d}.jsonl")
    temporary = shard.with_name(f".{shard.name}.tmp")
    stream = None
    rounds = iter_play_groups(candidate, opponent, range(context.rank, settings.pairs,
                                                        context.world_size), settings)
    local_done = False
    last_reported = -1
    try:
        error = None
        try:
            stream = temporary.open("w", encoding="utf-8")
        except OSError as exc:
            error = str(exc)
        synchronized_error(context, error)
        while context.any(not local_done):
            if stop_requested is not None and context.any(stop_requested()):
                raise InterruptedError("model selection interrupted before completing the match")
            error = None
            try:
                if not local_done:
                    try:
                        batch = next(rounds)
                    except StopIteration:
                        batch = []
                        local_done = True
                    for record in batch:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        games.append({k: v for k, v in record.items() if k != "actions"})
                    stream.flush()
            except Exception as exc:
                error = f"rank {context.rank} parallel games: {type(exc).__name__}: {exc}"
            synchronized_error(context, error)
            completed_games = context.reduce_int(len(games))
            if context.primary and completed_games != last_reported:
                last_reported = completed_games
                print(json.dumps({"mode": settings.mode,
                                  "arena_games_completed": completed_games,
                                  "arena_games_target": settings.pairs * settings.games_per_group,
                                  "parallel_games_per_rank": settings.parallel_games,
                                  "games_per_group": settings.games_per_group}), flush=True)
    finally:
        rounds.close()
        if stream is not None:
            stream.close()
        for engine in (candidate, opponent):
            engine.policy.clear_inference_board_cache()
            engine.actor.reset_temporal_prefixes()
    error = None
    try:
        temporary.replace(shard)
    except OSError as exc:
        error = str(exc)
    synchronized_error(context, error)
    shards = context.gather_object(games)
    seconds = context.reduce_float(time.monotonic() - start, operation="max")
    summary, error = None, None
    if context.primary:
        try:
            summary = summarize_games([r for part in shards for r in part], settings, alpha=alpha)
            summary.update({
                "arena_version": ARENA_VERSION, "evaluation_type": protocol.EVALUATION_TYPE,
                "layout_protocol": protocol.LAYOUT_PROTOCOL,
                "candidate_checkpoint": str(candidate_path), "opponent_checkpoint": str(opponent_path),
                "candidate_update": candidate.checkpoint_update,
                "opponent_update": opponent.checkpoint_update,
                "dead_rules_enabled": candidate.policy.config.dead_rules_enabled,
                "settings": asdict(settings), "world_size": context.world_size,
                "sampling_protocol": "per_game_uniform_inverse_cdf_v1",
                "effective_temporal_cache_entries": settings.effective_temporal_cache_entries,
                "device": str(context.device), "torch_version": str(torch.__version__),
                "game_shards": [str(output_prefix.with_name(
                    f"{output_prefix.name}.{attempt}.rank{rank:03d}.jsonl"
                )) for rank in range(context.world_size)],
                "wall_seconds": seconds, "games_per_second": summary["games"] / max(seconds, 1e-9),
                "plies_per_second": summary["plies_mean"] * summary["games"] / max(seconds, 1e-9),
            })
        except Exception as exc:
            error = f"result validation: {type(exc).__name__}: {exc}"
    synchronized_error(context, error)
    return summary
