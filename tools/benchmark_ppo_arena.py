"""Time a bounded four-game arena rotation using existing inference code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from junqi.training.accelerator import get_device_name, synchronize
from junqi.training.arena import MatchSettings
from junqi.training.arena_four_player import play_group
from junqi.training.inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.groups <= 0:
        parser.error("groups must be positive")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    candidate = InferenceEngine.from_checkpoint(args.checkpoint, device="cuda", temporal_cache_entries=8)
    opponent = InferenceEngine.from_checkpoint(args.checkpoint, device="cuda", temporal_cache_entries=8)
    settings = MatchSettings(pairs=args.groups, mode=candidate.mode.value, seed=20260911,
                             max_plies=2000, temporal_cache_entries=8, smoke_test=True)
    result = {"device": get_device_name(candidate.policy.device), "torch": torch.__version__,
              "mode": candidate.mode.value, "checkpoint": str(args.checkpoint),
              "purpose": "same random checkpoint against itself; timing only, no strength verdict",
              "max_game_plies": 2000, "groups": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for group in range(args.groups):
        synchronize(candidate.policy.device)
        started = time.perf_counter()
        games = play_group(candidate, opponent, group, settings)
        synchronize(candidate.policy.device)
        row = {"group": group, "seconds": time.perf_counter() - started,
               "games": [{key: value for key, value in game.items() if key != "actions"} for game in games]}
        result["groups"].append(row)
        print(json.dumps({"group": group, "seconds": row["seconds"],
                          "plies": [game["plies"] for game in games]}), flush=True)
        result["seconds_per_game"] = sum(item["seconds"] for item in result["groups"]) / (4 * len(result["groups"]))
        result["estimated_hours_15_rounds_1000_games"] = result["seconds_per_game"] * 15000 / 3600
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
