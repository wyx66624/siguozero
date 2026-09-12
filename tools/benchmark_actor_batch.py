"""Compare cached actor+critic inference batches on fixed synthetic streams.

No environment is stepped. Rates are model decisions, not environment steps.
The fixed cache population controls context/memory across batch-size cases;
real G-game self-play must retain approximately 4*G player-view histories.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from benchmark_cuda import initialize_optimizer_state
from benchmark_ppo import diverse_states
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.ppo import FrozenValueActor
from junqi.training.rollout import FrozenPolicyActor
from junqi.training.settings import TrainingSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260911)
    torch.set_num_threads(4)
    settings = TrainingSettings.from_yaml(Path('configs/bootstrap.yaml'), 'four_dark', model_scale='main')
    config = replace(settings.model, inference_temporal_cache_entries=240)
    policy = GamePolicyTransformer(config).cuda()
    critic = GameValueTransformer(config).cuda()
    critic.initialize_from_policy(policy)
    layout = PieceConditionedLayoutPointerDecoder(config).cuda()
    reference = copy.deepcopy(layout).eval().requires_grad_(False)
    optimizers = []
    for module in (policy, critic, layout):
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
        initialize_optimizer_state(module, optimizer)
        optimizers.append(optimizer)
    full = diverse_states(settings.mode, config, 1001, 80)
    data = {'device': torch.cuda.get_device_name(), 'config_revision': settings.raw_config['config_revision'],
            'cached_streams': 80, 'prefill_batch': 20, 'measured_context_range': [985, 1000],
            'actual_environment_steps': 0, 'all_training_models_and_adam_states_resident': True,
            'post_prefill_cleanup': 'release unused allocator blocks, run four warm incremental waves, release unused blocks again',
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), Path('src/junqi/training/models.py'))}, 'rows': []}
    for batch in (20, 40, 80):
        policy.clear_inference_board_cache()
        critic.clear_inference_board_cache()
        torch.cuda.empty_cache()
        actor = FrozenPolicyActor(policy.eval(), amp_dtype=torch.bfloat16, max_batch_size=batch)
        value = FrozenValueActor(critic, amp_dtype=torch.bfloat16, max_batch_size=batch)
        prefix = [replace(state, records=state.records[:980]) for state in full]
        # Cold prefill is deliberately excluded: this isolates incremental inference.
        for start in range(0, 80, 20):
            actor.sample(prefix[start:start + 20])
            value.values(prefix[start:start + 20])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for length in range(981, 985):
            warm = [replace(state, records=state.records[:length]) for state in full]
            for start in range(0, 80, batch):
                actor.sample(warm[start:start + batch])
                value.values(warm[start:start + batch])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        waves = []
        for length in range(985, 1001):
            wave = [replace(state, records=state.records[:length]) for state in full]
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for start in range(0, 80, batch):
                actor.sample(wave[start:start + batch])
                value.values(wave[start:start + batch])
            torch.cuda.synchronize()
            waves.append(time.perf_counter() - begin)
        row = {'inference_batch': batch, 'model_decisions': 80 * len(waves),
               'seconds': sum(waves), 'median_80_stream_wave_seconds': statistics.median(waves),
               'model_decisions_per_second': 80 * len(waves) / sum(waves),
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
               'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
               'individual_wave_seconds': waves}
        data['rows'].append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    policy.clear_inference_board_cache()
    critic.clear_inference_board_cache()


if __name__ == '__main__':
    main()
