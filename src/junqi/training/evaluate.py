"""Distributed checkpoint self-play evaluation with compact aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch

from .accelerator import is_accelerator, manual_seed_all
from .distributed import DistributedContext
from .inference import InferenceEngine


def _rank_output_path(path: Path, rank: int, world_size: int) -> Path:
    if world_size == 1:
        return path
    suffix = path.suffix or ".jsonl"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}.rank{rank:03d}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument(
        "--games-jsonl",
        type=Path,
        default=None,
        help="optional per-game JSONL; distributed ranks receive .rankNNN suffixes",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="optional aggregate JSON path written by rank 0",
    )
    return parser


def evaluate(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.games <= 0:
        raise ValueError("games must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    distributed = DistributedContext.initialize(args.device)
    try:
        engine = InferenceEngine.from_checkpoint(
            args.checkpoint,
            device=str(distributed.device),
        )
        output_path = None
        if args.games_jsonl is not None:
            # Rank-suffixed files may live on node-local storage, so every rank
            # creates its own parent instead of assuming a shared filesystem.
            args.games_jsonl.parent.mkdir(parents=True, exist_ok=True)
            output_path = _rank_output_path(
                args.games_jsonl, distributed.rank, distributed.world_size
            )

        local_counts = {"wins": 0, "draws": 0, "losses": 0, "plies": 0}
        local_min_plies: int | None = None
        local_max_plies = 0
        local_reasons: dict[str, int] = {}
        started = time.perf_counter()
        stream = (
            None
            if output_path is None
            else output_path.open("w", encoding="utf-8")
        )
        try:
            for game_index in range(distributed.rank, args.games, distributed.world_size):
                game_seed = args.seed + game_index
                random.seed(game_seed)
                torch.manual_seed(game_seed)
                if is_accelerator(distributed.device):
                    manual_seed_all(distributed.device.type, game_seed)
                record = engine.play_to_terminal(
                    seed=game_seed,
                    max_plies=args.max_plies,
                    temperature=args.temperature,
                )
                record["game_index"] = game_index
                reward = float(record["rewards"][0])
                outcome = "wins" if reward > 0 else "losses" if reward < 0 else "draws"
                plies = int(record["plies"])
                reason = str(record["result"])
                local_counts[outcome] += 1
                local_counts["plies"] += plies
                local_min_plies = (
                    plies if local_min_plies is None else min(local_min_plies, plies)
                )
                local_max_plies = max(local_max_plies, plies)
                local_reasons[reason] = local_reasons.get(reason, 0) + 1
                if stream is not None:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            if stream is not None:
                stream.close()

        elapsed = time.perf_counter() - started
        wins = distributed.reduce_int(local_counts["wins"])
        draws = distributed.reduce_int(local_counts["draws"])
        losses = distributed.reduce_int(local_counts["losses"])
        total_plies = distributed.reduce_int(local_counts["plies"])
        minimum = distributed.reduce_int(
            local_min_plies if local_min_plies is not None else 2**31 - 1,
            operation="min",
        )
        maximum = distributed.reduce_int(local_max_plies, operation="max")
        wall_seconds = distributed.reduce_float(elapsed, operation="max")
        gathered_reasons = distributed.gather_object(local_reasons)
        if not distributed.primary:
            return None

        reasons: dict[str, int] = {}
        for rank_reasons in gathered_reasons or []:
            for reason, count in rank_reasons.items():
                reasons[reason] = reasons.get(reason, 0) + int(count)
        summary: dict[str, Any] = {
            "evaluation/type": "shared_policy_self_play",
            "mode": engine.mode.value,
            "dead_rules_enabled": engine.policy.config.dead_rules_enabled,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "distributed/world_size": distributed.world_size,
            "games": args.games,
            "seat0/wins": wins,
            "seat0/draws": draws,
            "seat0/losses": losses,
            "seat0/win_rate": wins / args.games,
            "seat0/draw_rate": draws / args.games,
            "seat0/loss_rate": losses / args.games,
            "plies/mean": total_plies / args.games,
            "plies/min": minimum,
            "plies/max": maximum,
            "terminal_reasons": reasons,
            "timing/wall_seconds": wall_seconds,
            "throughput/games_per_second": args.games / max(wall_seconds, 1e-12),
            "throughput/plies_per_second": total_plies / max(wall_seconds, 1e-12),
            "seed": args.seed,
            "temperature": args.temperature,
        }
        rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        print(rendered)
        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.summary.with_name(f".{args.summary.name}.tmp")
            temporary.write_text(rendered + "\n", encoding="utf-8")
            temporary.replace(args.summary)
        return summary
    finally:
        distributed.close()


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
