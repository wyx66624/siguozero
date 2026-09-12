"""Check actual checkpoint weights on identical histories, without training."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import fcntl
import gc
import json
from pathlib import Path
import random
import tempfile

import torch

from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig
from junqi.training.modes import TrainingMode
from junqi.training.rollout import BaseGamePool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lock = open(Path(tempfile.gettempdir()) / "siguozero-cuda-probe.lock", "a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = Path(args.checkpoint)
    before = source.stat()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["trainer_state"]
    pool_state = state.get("base_game_pool") or state["distributed"]["rank_states"][0]["base_game_pool"]
    pool_state = {**pool_state, "pool_size": 8, "slots": pool_state["slots"][:8]}
    config = replace(ModelConfig(**payload["config"]["model"]), ppo_cuda_graphs=True)
    result = {"checkpoint": str(source), "device": torch.cuda.get_device_name(), "checks": []}
    for dtype in (torch.float32, torch.bfloat16):
        for name, kind in (("policy", GamePolicyTransformer), ("critic", GameValueTransformer)):
            model = kind(config).cuda().eval()
            model.load_state_dict(payload[name])
            reference = copy.deepcopy(model)
            model.start_ppo_inference_cache(capacity=32, behavior_version=int(payload["update"]))
            pool = BaseGamePool(TrainingMode.FOUR_DARK, pool_size=8,
                                max_transitions=pool_state["max_transitions"],
                                max_game_plies=pool_state["max_game_plies"],
                                dead_rules_enabled=pool_state["dead_rules_enabled"], seed=1)
            pool.load_state_dict(pool_state)
            for slot in pool.slots:
                slot.history.enable_array_storage()
            rng = random.Random(621)
            max_context = max_output = 0.
            queries = 0
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                for wave in range(12):
                    slots = [slot for slot in pool.slots if not slot.game.is_terminal]
                    states = [slot.history.state_for(slot.game) for slot in slots]
                    if not states:
                        break
                    expected, actual = reference.encode(states), model.encode(states)
                    max_context = max(max_context, float((expected.context - actual.context).abs().max()))
                    if name == "policy":
                        expected_out = reference._all_legal_log_probs(expected, states)[0].float()
                        actual_out = model._all_legal_log_probs(actual, states)[0].float()
                    else:
                        expected_out = reference.value_head(expected.context).float()
                        actual_out = model.value_head(actual.context).float()
                    max_output = max(max_output, float((expected_out - actual_out).abs().max()))
                    tolerance = 5e-5 if dtype == torch.float32 else .04
                    torch.testing.assert_close(actual.context, expected.context, rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(actual_out, expected_out, rtol=tolerance, atol=tolerance)
                    queries += len(states)
                    for slot in slots:
                        slot.game.step(rng.choice(slot.game.legal_actions()))
                        slot.history.append_after_step(slot.game)
            result["checks"].append(dict(model=name, dtype=str(dtype), queries=queries,
                                         max_context_absolute_error=max_context, max_output_absolute_error=max_output,
                                         decode_tokens=model._fixed_kv_store.decode_tokens,
                                         graph_replays=model._fixed_kv_store.graph_replays))
            print(json.dumps(result["checks"][-1]), flush=True)
            model.clear_inference_board_cache()
            del model, reference, pool, expected, actual
            gc.collect()
            torch.cuda.empty_cache()
    result["checkpoint_stat_unchanged"] = (before.st_mtime_ns, before.st_size) == (source.stat().st_mtime_ns, source.stat().st_size)
    assert result["checkpoint_stat_unchanged"]
    result["complete"] = True
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
