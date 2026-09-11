"""Batched old-policy self-play and K=4, M=2 terminal continuations."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import random
import time
from typing import Any, Mapping, Sequence

import torch

from ..game import JunqiGame
from .accelerator import empty_cache, is_out_of_memory
from .encoding import GameHistory, PolicyState
from .models import (
    GamePolicyTransformer,
    LayoutSample,
    PieceConditionedLayoutPointerDecoder,
    _validate_sampling_uniforms,
    layout_sample_from_trace,
)
from .modes import TrainingMode, mode_spec, new_game, normalize_mode


ROOT_CANDIDATE_COUNT = 4
REPLICAS_PER_CANDIDATE = 2
TERMINAL_CONTINUATIONS_PER_ANCHOR = 8


@dataclass(slots=True)
class AnchorSnapshot:
    game: JunqiGame
    history: GameHistory
    state: PolicyState
    root_player: int


@dataclass(frozen=True, slots=True)
class PolicyGroup:
    state: PolicyState
    candidate_actions: tuple[tuple[int, int], ...]
    old_log_probs: tuple[float, ...]
    replica_rewards: tuple[tuple[float, float], ...]
    candidate_returns: tuple[float, ...]
    advantages: tuple[float, ...]
    continuation_plies: int
    behavior_version: int


@dataclass(frozen=True, slots=True)
class LayoutOutcome:
    sample: LayoutSample
    reward: float
    seat: int
    behavior_version: int


@dataclass(slots=True)
class RolloutMetrics:
    policy_samples: int = 0
    environment_plies: int = 0
    critic_inference_seconds: float = 0.0
    anchors: int = 0
    root_candidates: int = 0
    terminal_continuations: int = 0
    continuation_plies: int = 0
    base_plies: int = 0
    base_games_completed: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    wall_seconds: float = 0.0
    actor_inference_seconds: float = 0.0
    environment_step_seconds: float = 0.0

    def record_environment_steps(self, count: int = 1, *, continuation: bool = False) -> None:
        """Count executed state transitions, including every simulated branch.

        Copies, resets, model calls and replayed learner samples add no steps.
        A selected action executed again on the base game is another transition.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("environment step count must be a nonnegative integer")
        if continuation:
            self.continuation_plies += count
        else:
            self.base_plies += count
        self.environment_plies += count

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rollout/policy_samples": self.policy_samples,
            "rollout/environment_plies": self.environment_plies,
            "rollout/critic_inference_seconds": self.critic_inference_seconds,
            "rollout/anchors": self.anchors,
            "rollout/root_candidates": self.root_candidates,
            "rollout/terminal_continuations": self.terminal_continuations,
            "rollout/continuation_plies": self.continuation_plies,
            "rollout/base_plies": self.base_plies,
            "rollout/base_games_completed": self.base_games_completed,
            "rollout/wins": self.wins,
            "rollout/draws": self.draws,
            "rollout/losses": self.losses,
            "rollout/wall_seconds": self.wall_seconds,
            "rollout/actor_inference_seconds": self.actor_inference_seconds,
            "rollout/environment_step_work_seconds": self.environment_step_seconds,
            "rollout/continuations_per_second": (
                self.terminal_continuations / max(self.wall_seconds, 1e-9)
            ),
            "rollout/plies_per_second": (
                self.environment_plies / max(self.wall_seconds, 1e-9)
            ),
        }


@dataclass(slots=True)
class BaseGameSlot:
    game: JunqiGame
    history: GameHistory
    layouts: tuple[LayoutSample, ...]
    layout_behavior_version: int


class FrozenPolicyActor:
    """One shared policy for every seat; categorical sampling, never argmax.

    ``states`` may belong to different players and games.  They are batched into
    this single module instance after each state has been rotated to its owner's
    main perspective.  The trainer guarantees that parameters remain frozen for
    the complete collection phase.
    """

    def __init__(
        self,
        policy: GamePolicyTransformer,
        *,
        amp_dtype: torch.dtype | None = None,
        max_batch_size: int = 64,
    ) -> None:
        self.policy = policy
        self.amp_dtype = amp_dtype
        if max_batch_size <= 0:
            raise ValueError("actor max batch size must be positive")
        self.max_batch_size = max_batch_size
        self.oom_reductions = 0
        # A fresh actor is created for each frozen collection phase.  Historic
        # board globals are safe to reuse until the next optimizer update.
        self.policy.start_inference_board_cache()

    @property
    def device_type(self) -> str:
        return self.policy.device.type

    def sample(
        self,
        states: Sequence[PolicyState],
        *,
        count: int = 1,
        temperature: float = 1.0,
        return_log_probs: bool = True,
        sampling_uniforms: torch.Tensor | None = None,
    ) -> tuple[list[list[tuple[int, int]]], list[torch.Tensor]]:
        if sampling_uniforms is not None:
            _validate_sampling_uniforms(
                sampling_uniforms, state_count=len(states), sample_count=count
            )
        enabled = self.amp_dtype is not None and self.device_type in ("cuda", "npu")
        all_actions: list[list[tuple[int, int]]] = []
        all_logs: list[torch.Tensor] = []
        with torch.inference_mode(), torch.autocast(
            device_type=self.device_type,
            dtype=self.amp_dtype or torch.float32,
            enabled=enabled,
        ):
            start = 0
            while start < len(states):
                batch_size = min(self.max_batch_size, len(states) - start)
                try:
                    sampling_options = {}
                    if sampling_uniforms is not None:
                        # Retry the same draws when an OOM reduces the batch;
                        # advancing game RNGs here would alter match outcomes.
                        sampling_options["sampling_uniforms"] = sampling_uniforms[
                            start : start + batch_size
                        ]
                    actions, logs = self.policy.sample_action_groups(
                        states[start : start + batch_size],
                        count=count,
                        temperature=temperature,
                        return_log_probs=return_log_probs,
                        **sampling_options,
                    )
                except RuntimeError as error:
                    if (
                        not is_out_of_memory(error, self.device_type)
                        or self.device_type not in ("cuda", "npu")
                        or batch_size <= 1
                    ):
                        raise
                    self.max_batch_size = max(1, batch_size // 2)
                    self.oom_reductions += 1
                    empty_cache(self.policy.device)
                    continue
                all_actions.extend(actions)
                all_logs.extend(logs)
                start += batch_size
        return all_actions, all_logs

    def reset_temporal_prefixes(self) -> None:
        """Bound KV memory between independent anchor waves."""

        self.policy.reset_inference_temporal_cache()
        if self.device_type in ("cuda", "npu"):
            # The fixed paged arena remains allocated, while temporary SDPA
            # gathers and variable-length workspaces are no longer useful to
            # the next independent wave.  Returning those cached blocks here
            # prevents WDDM reserved memory from ratcheting toward device
            # capacity during long, diverse rollouts.
            empty_cache(self.policy.device)


class BaseGamePool:
    """Persistent games whose seats all share one passed Policy/Layout pair."""

    def __init__(
        self,
        mode: TrainingMode | str,
        *,
        pool_size: int,
        max_transitions: int,
        max_game_plies: int | None,
        dead_rules_enabled: bool = True,
        seed: int,
    ) -> None:
        if pool_size <= 0:
            raise ValueError("base game pool size must be positive")
        self.mode = normalize_mode(mode)
        self.pool_size = pool_size
        self.max_transitions = max_transitions
        self.max_game_plies = max_game_plies
        if not isinstance(dead_rules_enabled, bool):
            raise ValueError("dead_rules_enabled must be a boolean")
        self.dead_rules_enabled = dead_rules_enabled
        self.rng = random.Random(seed)
        self.slots: list[BaseGameSlot] = []

    def _new_slot(
        self,
        layout: PieceConditionedLayoutPointerDecoder,
        behavior_version: int,
    ) -> BaseGameSlot:
        spec = mode_spec(self.mode)
        samples = tuple(
            layout.sample_layouts(spec.player_count, self.mode, temperature=0.7)
        )
        game = new_game(
            self.mode,
            setups=[sample.setup for sample in samples],
            seed=self.rng.randrange(2**63),
            max_plies=self.max_game_plies,
            dead_rules_enabled=self.dead_rules_enabled,
        )
        history = GameHistory.initialize(
            game, self.mode, max_transitions=self.max_transitions
        )
        return BaseGameSlot(game, history, samples, behavior_version)

    def fill(
        self,
        layout: PieceConditionedLayoutPointerDecoder,
        behavior_version: int,
    ) -> None:
        while len(self.slots) < self.pool_size:
            self.slots.append(self._new_slot(layout, behavior_version))

    def state_dict(self) -> dict[str, Any]:
        """Serialize every unfinished base game and player-view history exactly."""

        return {
            # Version 4 records the explicit dead-rule architecture variant.
            "format_version": 4,
            "mode": self.mode.value,
            "dead_rules_enabled": self.dead_rules_enabled,
            "pool_size": self.pool_size,
            "max_transitions": self.max_transitions,
            "max_game_plies": self.max_game_plies,
            "rng_state": self.rng.getstate(),
            "slots": [self._slot_state_dict(slot) for slot in self.slots],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("format_version", -1)) != 4:
            raise ValueError("unsupported base-game-pool checkpoint format")
        if normalize_mode(state["mode"]) is not self.mode:
            raise ValueError("base-game-pool mode does not match trainer mode")
        restored_dead_rules = state["dead_rules_enabled"]
        if not isinstance(restored_dead_rules, bool):
            raise ValueError("invalid base-game-pool dead-rule marker")
        if restored_dead_rules != self.dead_rules_enabled:
            raise ValueError("base-game-pool dead-rule variant changed across resume")
        if int(state["pool_size"]) != self.pool_size:
            raise ValueError("base-game-pool size changed across resume")
        if int(state["max_transitions"]) != self.max_transitions:
            raise ValueError("history window changed across resume")
        if state["max_game_plies"] != self.max_game_plies:
            raise ValueError("maximum game plies changed across resume")
        slots = [self._slot_from_state_dict(item) for item in state["slots"]]
        if len(slots) > self.pool_size:
            raise ValueError("checkpoint contains too many base-game slots")
        self.slots = slots
        self.rng.setstate(state["rng_state"])

    @staticmethod
    def _slot_state_dict(slot: BaseGameSlot) -> dict[str, Any]:
        game = slot.game
        if game.is_terminal:
            raise ValueError("terminal games must be replaced before checkpointing")
        return {
            "game": {
                "config": game.config,
                "pieces": dict(game.pieces),
                "current_player": game.current_player,
                "active_players": game.active_players,
                "revealed_flags": game.revealed_flags,
                "ply_count": game.ply_count,
                "no_interaction_plies": game.no_interaction_plies,
                "public_candidates": dict(game.public_candidates),
                "known_identities": [
                    dict(known) for known in game.known_identities
                ],
                "known_casualties": [
                    [dict(counts) for counts in owner_tables]
                    for owner_tables in game.known_casualties
                ],
                "public_history": game.public_history,
            },
            "history": slot.history.state_dict(),
            "layouts": [
                {
                    "mode": sample.mode.value,
                    "position_indices": sample.position_indices,
                    "old_log_probs": sample.old_log_probs,
                }
                for sample in slot.layouts
            ],
            "layout_behavior_version": slot.layout_behavior_version,
        }

    def _slot_from_state_dict(self, state: Mapping[str, Any]) -> BaseGameSlot:
        raw_game = state["game"]
        game = JunqiGame.from_position(
            raw_game["config"],
            raw_game["pieces"],
            current_player=raw_game["current_player"],
            active_players=raw_game["active_players"],
            revealed_flags=raw_game["revealed_flags"],
            ply_count=int(raw_game["ply_count"]),
            no_interaction_plies=int(raw_game["no_interaction_plies"]),
            public_candidates=raw_game["public_candidates"],
            known_identities=raw_game["known_identities"],
            known_casualties=raw_game["known_casualties"],
            public_history=raw_game["public_history"],
        )
        if game.is_terminal:
            raise ValueError("checkpoint base-game slot unexpectedly terminal")
        history = GameHistory.from_state_dict(state["history"])
        if history.mode is not self.mode:
            raise ValueError("base-game history mode does not match pool")
        if (
            history.players[0].dead_rules_enabled
            != self.dead_rules_enabled
        ):
            raise ValueError("base-game history dead-rule variant does not match pool")
        layouts = tuple(
            layout_sample_from_trace(
                item["mode"], item["position_indices"], item["old_log_probs"]
            )
            for item in state["layouts"]
        )
        expected_players = mode_spec(self.mode).player_count
        if len(layouts) != expected_players:
            raise ValueError("base-game checkpoint has the wrong layout count")
        return BaseGameSlot(
            game=game,
            history=history,
            layouts=layouts,
            layout_behavior_version=int(state["layout_behavior_version"]),
        )

    def collect_anchors(
        self,
        actor: FrozenPolicyActor,
        layout: PieceConditionedLayoutPointerDecoder,
        *,
        count: int,
        behavior_version: int,
    ) -> tuple[list[AnchorSnapshot], list[LayoutOutcome], int, int]:
        if count <= 0:
            raise ValueError("anchor count must be positive")
        self.fill(layout, behavior_version)
        anchors: list[AnchorSnapshot] = []
        completed_layouts: list[LayoutOutcome] = []
        base_plies = 0
        completed_games = 0

        while len(anchors) < count:
            active_indices = list(
                range(min(len(self.slots), count - len(anchors)))
            )
            states: list[PolicyState] = []
            for index in active_indices:
                slot = self.slots[index]
                if slot.game.is_terminal:
                    raise RuntimeError("terminal base game remained in pool")
                root_player = slot.game.current_player
                if root_player is None:
                    raise RuntimeError("non-terminal base game has no current player")
                state = slot.history.state_for(slot.game, root_player)
                anchors.append(
                    AnchorSnapshot(
                        game=slot.game.clone(),
                        history=slot.history.clone(),
                        state=state,
                        root_player=root_player,
                    )
                )
                states.append(state)

            sampled, _logs = actor.sample(
                states,
                count=1,
                temperature=1.0,
                return_log_probs=False,
            )
            replacements: list[tuple[int, BaseGameSlot]] = []
            for index, action_group in zip(active_indices, sampled, strict=True):
                slot = self.slots[index]
                slot.game.step(action_group[0])
                slot.history.append_after_step(slot.game)
                base_plies += 1
                if slot.game.is_terminal:
                    completed_games += 1
                    rewards = slot.game.rewards()
                    for seat, sample in enumerate(slot.layouts):
                        completed_layouts.append(
                            LayoutOutcome(
                                sample=sample,
                                reward=rewards[seat],
                                seat=seat,
                                behavior_version=slot.layout_behavior_version,
                            )
                        )
                    replacements.append(
                        (index, self._new_slot(layout, behavior_version))
                    )
            for index, replacement in replacements:
                self.slots[index] = replacement
        return anchors, completed_layouts, base_plies, completed_games


@dataclass(slots=True)
class _Branch:
    anchor_index: int
    candidate_index: int
    replica_index: int
    root_player: int
    game: JunqiGame
    history: GameHistory
    starting_ply: int


def _step_branch_chunk(
    branches: Sequence[_Branch],
    sampled: Sequence[Sequence[tuple[int, int]]],
) -> float:
    started = time.perf_counter()
    for branch, actions in zip(branches, sampled, strict=True):
        branch.game.step(actions[0])
        branch.history.append_after_step(branch.game)
    return time.perf_counter() - started


def _advance_branches(
    active: Sequence[_Branch],
    actor: FrozenPolicyActor,
    executor: ThreadPoolExecutor | None,
) -> tuple[float, float]:
    """Overlap accelerator inference for chunk N+1 with CPU rules for chunk N."""

    inference_seconds = 0.0
    environment_seconds = 0.0
    futures: list[Future[float]] = []
    start = 0
    while start < len(active):
        stop = min(start + actor.max_batch_size, len(active))
        chunk = active[start:stop]
        states = [branch.history.state_for(branch.game) for branch in chunk]
        inference_started = time.perf_counter()
        sampled, _logs = actor.sample(
            states,
            count=1,
            temperature=1.0,
            return_log_probs=False,
        )
        inference_seconds += time.perf_counter() - inference_started
        if executor is None:
            environment_seconds += _step_branch_chunk(chunk, sampled)
        else:
            futures.append(executor.submit(_step_branch_chunk, chunk, sampled))
        start = stop
    for future in futures:
        environment_seconds += future.result()
    return inference_seconds, environment_seconds


def _standardize(values: Sequence[float], epsilon: float) -> tuple[float, ...]:
    tensor = torch.tensor(values, dtype=torch.float32)
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    if float(std) < epsilon:
        return tuple(0.0 for _ in values)
    return tuple(((tensor - mean) / (std + epsilon)).tolist())


def collect_policy_groups(
    anchors: Sequence[AnchorSnapshot],
    actor: FrozenPolicyActor,
    *,
    behavior_version: int,
    advantage_epsilon: float = 1e-4,
    anchor_wave_size: int = 8,
    environment_workers: int = 2,
) -> tuple[list[PolicyGroup], RolloutMetrics]:
    """Run K=4/M=2 terminal rollouts in bounded copy-on-write KV waves."""

    if anchor_wave_size <= 0:
        raise ValueError("anchor_wave_size must be positive")
    if environment_workers <= 0:
        raise ValueError("environment_workers must be positive")
    started = time.perf_counter()
    metrics = RolloutMetrics(
        anchors=len(anchors),
        root_candidates=len(anchors) * ROOT_CANDIDATE_COUNT,
        terminal_continuations=(
            len(anchors)
            * ROOT_CANDIDATE_COUNT
            * REPLICAS_PER_CANDIDATE
        ),
    )
    groups: list[PolicyGroup] = []
    use_pipeline = (
        actor.device_type in ("cuda", "npu") and environment_workers > 1
    )
    for wave_start in range(0, len(anchors), anchor_wave_size):
        wave = anchors[wave_start : wave_start + anchor_wave_size]
        # Independent waves bound retained K/V memory.  Inside a wave all
        # candidate/replica children reuse the immutable anchor prefix.
        actor.reset_temporal_prefixes()
        root_started = time.perf_counter()
        candidate_actions, old_log_probs = actor.sample(
            [anchor.state for anchor in wave],
            count=ROOT_CANDIDATE_COUNT,
            temperature=1.0,
        )
        metrics.actor_inference_seconds += time.perf_counter() - root_started

        branches: list[_Branch] = []
        for local_index, anchor in enumerate(wave):
            for candidate_index, action in enumerate(
                candidate_actions[local_index]
            ):
                for replica_index in range(REPLICAS_PER_CANDIDATE):
                    game = anchor.game.clone()
                    history = anchor.history.clone()
                    starting_ply = game.ply_count
                    game.step(action)
                    history.append_after_step(game)
                    branches.append(
                        _Branch(
                            anchor_index=local_index,
                            candidate_index=candidate_index,
                            replica_index=replica_index,
                            root_player=anchor.root_player,
                            game=game,
                            history=history,
                            starting_ply=starting_ply,
                        )
                    )

        executor = (
            ThreadPoolExecutor(
                max_workers=environment_workers,
                thread_name_prefix="junqi-env",
            )
            if use_pipeline
            else None
        )
        try:
            while True:
                active = [
                    branch for branch in branches if not branch.game.is_terminal
                ]
                if not active:
                    break
                inference_seconds, environment_seconds = _advance_branches(
                    active, actor, executor
                )
                metrics.actor_inference_seconds += inference_seconds
                metrics.environment_step_seconds += environment_seconds
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        reward_cube = [
            [
                [0.0 for _ in range(REPLICAS_PER_CANDIDATE)]
                for _ in range(ROOT_CANDIDATE_COUNT)
            ]
            for _ in wave
        ]
        continuation_plies = [0 for _ in wave]
        for branch in branches:
            reward = branch.game.rewards()[branch.root_player]
            reward_cube[branch.anchor_index][branch.candidate_index][
                branch.replica_index
            ] = reward
            continuation_plies[branch.anchor_index] += (
                branch.game.ply_count - branch.starting_ply
            )
            if reward > 0:
                metrics.wins += 1
            elif reward < 0:
                metrics.losses += 1
            else:
                metrics.draws += 1

        for local_index, anchor in enumerate(wave):
            replica_rewards = tuple(
                tuple(float(item) for item in rewards)
                for rewards in reward_cube[local_index]
            )
            candidate_returns = tuple(
                sum(rewards) / REPLICAS_PER_CANDIDATE
                for rewards in replica_rewards
            )
            advantages = _standardize(candidate_returns, advantage_epsilon)
            groups.append(
                PolicyGroup(
                    state=anchor.state,
                    candidate_actions=tuple(candidate_actions[local_index]),
                    old_log_probs=tuple(
                        float(value) for value in old_log_probs[local_index]
                    ),
                    replica_rewards=replica_rewards,
                    candidate_returns=candidate_returns,
                    advantages=advantages,
                    continuation_plies=continuation_plies[local_index],
                    behavior_version=behavior_version,
                )
            )
        metrics.record_environment_steps(sum(continuation_plies), continuation=True)
    metrics.wall_seconds = time.perf_counter() - started
    return groups, metrics
