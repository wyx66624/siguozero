"""Checkpoint-backed stochastic layout and move inference APIs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from ..game import JunqiGame, StepResult
from .accelerator import is_accelerator, is_bf16_supported, resolve_device
from .checkpoint import require_current_checkpoint
from .encoding import GameHistory
from .models import (
    GamePolicyTransformer,
    LayoutSample,
    ModelConfig,
    PieceConditionedLayoutPointerDecoder,
)
from .modes import TrainingMode, mode_spec, new_game, normalize_mode
from .rollout import FrozenPolicyActor


class InferenceEngine:
    """Run every seat through one shared Policy and one shared Layout instance."""

    def __init__(
        self,
        mode: TrainingMode | str,
        policy: GamePolicyTransformer,
        layout: PieceConditionedLayoutPointerDecoder,
        *,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.mode = normalize_mode(mode)
        self.policy = policy.eval().requires_grad_(False)
        self.layout = layout.eval().requires_grad_(False)
        self.actor = FrozenPolicyActor(self.policy, amp_dtype=amp_dtype)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        mode: TrainingMode | str | None = None,
        device: str = "auto",
        dead_rules_enabled: bool | None = None,
        temporal_cache_entries: int | None = None,
    ) -> InferenceEngine:
        resolved_device = resolve_device(device)
        payload = torch.load(
            Path(checkpoint), map_location="cpu", mmap=True, weights_only=False
        )
        require_current_checkpoint(payload)
        checkpoint_dead_rules = payload.get("dead_rules_enabled")
        if not isinstance(checkpoint_dead_rules, bool):
            raise ValueError("checkpoint has no valid dead-rule variant marker")
        if (
            dead_rules_enabled is not None
            and dead_rules_enabled != checkpoint_dead_rules
        ):
            raise ValueError("requested dead-rule variant does not match checkpoint")
        checkpoint_mode = normalize_mode(payload["mode"])
        requested_mode = checkpoint_mode if mode is None else normalize_mode(mode)
        if requested_mode is not checkpoint_mode:
            raise ValueError(
                f"checkpoint is {checkpoint_mode.value}, not {requested_mode.value}"
            )
        model_values = payload.get("config", {}).get("model", {})
        config = ModelConfig(**model_values) if model_values else ModelConfig()
        if config.dead_rules_enabled != checkpoint_dead_rules:
            raise ValueError("checkpoint model config contradicts its dead-rule marker")
        # Evaluation needs only Policy/Layout, never the reference or optimizer
        # tensors on an accelerator. mmap also avoids eagerly copying the full
        # multi-GB training checkpoint into host RAM.
        if temporal_cache_entries is not None:
            config = replace(
                config,
                inference_temporal_cache_entries=temporal_cache_entries,
                inference_board_cache_entries=2048,
            )
        policy = GamePolicyTransformer(config).to(resolved_device)
        layout = PieceConditionedLayoutPointerDecoder(config).to(resolved_device)
        policy.load_state_dict(payload["policy"], strict=True)
        layout.load_state_dict(payload["layout"], strict=True)
        amp_dtype = None
        if is_accelerator(resolved_device):
            amp_dtype = (
                torch.bfloat16
                if is_bf16_supported(resolved_device)
                else torch.float16
            )
        engine = cls(requested_mode, policy, layout, amp_dtype=amp_dtype)
        engine.checkpoint_update = int(payload.get("update", 0))
        return engine

    def sample_layouts(
        self, count: int | None = None, *, temperature: float = 0.7
    ) -> list[LayoutSample]:
        number = mode_spec(self.mode).player_count if count is None else count
        with torch.inference_mode():
            return self.layout.sample_layouts(
                number, self.mode, temperature=temperature
            )

    def new_game(
        self,
        *,
        seed: int | None = None,
        max_plies: int | None = None,
    ) -> tuple[JunqiGame, GameHistory]:
        samples = self.sample_layouts()
        game = new_game(
            self.mode,
            setups=[sample.setup for sample in samples],
            seed=seed,
            max_plies=max_plies,
            dead_rules_enabled=self.policy.config.dead_rules_enabled,
        )
        history = GameHistory.initialize(
            game,
            self.mode,
            max_transitions=self.policy.config.max_transitions,
        )
        return game, history

    def select_action(
        self,
        game: JunqiGame,
        history: GameHistory,
        *,
        temperature: float = 1.0,
    ) -> tuple[int, int]:
        if game.is_terminal:
            raise ValueError("cannot infer an action for a terminal game")
        state = history.state_for(game)
        actions, _logs = self.actor.sample(
            [state], count=1, temperature=temperature
        )
        return actions[0][0]

    def step(
        self,
        game: JunqiGame,
        history: GameHistory,
        *,
        temperature: float = 1.0,
    ) -> StepResult:
        action = self.select_action(game, history, temperature=temperature)
        result = game.step(action)
        history.append_after_step(game)
        return result

    def play_to_terminal(
        self,
        *,
        seed: int | None = None,
        max_plies: int | None = None,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        game, history = self.new_game(seed=seed, max_plies=max_plies)
        actions: list[dict[str, Any]] = []
        while not game.is_terminal:
            result = self.step(game, history, temperature=temperature)
            actions.append(
                {
                    "ply": game.ply_count,
                    "player": result.player,
                    "source": result.action[0],
                    "destination": result.action[1],
                    "combat": result.combat.value,
                }
            )
        return {
            "mode": self.mode.value,
            "dead_rules_enabled": self.policy.config.dead_rules_enabled,
            "seed": seed,
            "plies": game.ply_count,
            "result": game.result.reason.value if game.result else None,
            "winner_team": game.result.winner_team if game.result else None,
            "rewards": game.rewards(),
            "actions": actions,
        }


def build_parser(default_mode: TrainingMode | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SiguoZero checkpoint inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode",
        choices=[item.value for item in TrainingMode],
        default=None if default_mode is None else default_mode.value,
        required=default_mode is None,
    )
    parser.add_argument("--device", default="auto")
    dead_rules = parser.add_mutually_exclusive_group()
    dead_rules.add_argument(
        "--dead-rules",
        dest="dead_rules_enabled",
        action="store_true",
        help="require a checkpoint trained with deterministic dead rules",
    )
    dead_rules.add_argument(
        "--no-dead-rules",
        dest="dead_rules_enabled",
        action="store_false",
        help="require a checkpoint trained without deterministic dead rules",
    )
    parser.set_defaults(dead_rules_enabled=None)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--output", default=None, help="optional JSONL output path")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_mode: TrainingMode | None = None,
) -> None:
    args = build_parser(default_mode).parse_args(argv)
    engine = InferenceEngine.from_checkpoint(
        args.checkpoint,
        mode=args.mode,
        device=args.device,
        dead_rules_enabled=args.dead_rules_enabled,
    )
    records = [
        engine.play_to_terminal(
            seed=args.seed + index,
            max_plies=args.max_plies,
            temperature=args.temperature,
        )
        for index in range(args.games)
    ]
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    if args.output is None:
        print("\n".join(lines))
    else:
        Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
