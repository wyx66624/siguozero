"""Bounded, isolated profiling of compact four-dark PPO from a saved probe.

Loads an existing checkpoint read-only, collects one rollout, and measures
backward passes without optimizer updates. Never saves model weights. CUDA
profiler service time is not tensor-core utilization or an unprofiled timeline.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict, replace
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import pstats
import tempfile
import time
from unittest.mock import patch

import torch

from benchmark_ppo_sequences import ProbeTrainer
from profile_ppo_compute import count_blocks, summarize_profiler
from junqi.training.checkpoint import restore_training_state
from junqi.training.metrics import MetricLogger
from junqi.training.ppo import FrozenValueActor, collect_ppo_samples
from junqi.training.rollout import FrozenPolicyActor
from junqi.training.settings import TrainingSettings
import junqi.training.models as loaded_models


def python_profile(profiler):
    rows = []
    for (filename, line, function), (primitive, calls, own, total, _) in pstats.Stats(profiler).stats.items():
        rows.append(dict(file=filename, line=line, function=function, calls=calls,
                         primitive_calls=primitive, self_seconds=own, cumulative_seconds=total))
    return dict(top_self=sorted(rows, key=lambda r: r['self_seconds'], reverse=True)[:30],
                top_cumulative=sorted(rows, key=lambda r: r['cumulative_seconds'], reverse=True)[:30],
                hot_calls={name: sum(row['calls'] for row in rows if name in row['function'])
                           for name in ('__hash__', 'builtins.hash', 'torch.as_tensor', 'torch.tensor', 'copy_')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--moves-per-game', type=int, default=256)
    parser.add_argument('--profile-waves', type=int, default=1)
    args = parser.parse_args()
    if min(args.moves_per_game, args.profile_waves) <= 0:
        parser.error('counts must be positive')
    lock = open(Path(tempfile.gettempdir()) / 'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.90)
    root = Path(__file__).resolve().parents[1]
    source_root = Path(loaded_models.__file__).resolve().parents[2]
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_stat = checkpoint.stat()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved_state = payload['trainer_state']
    pool_state = saved_state.get('base_game_pool')
    if pool_state is None:
        pool_state = saved_state['distributed']['rank_states'][0]['base_game_pool']
    games = int(pool_state['pool_size'])
    settings = TrainingSettings.from_yaml(
        root / 'configs/bootstrap.yaml', 'four_dark', model_scale='main',
        overrides=dict(device='cuda', base_game_pool_size=games,
                       actor_inference_batch=games, anchor_batch=games * args.moves_per_game,
                       policy_microbatch=8, ppo_minibatch_samples=512,
                       ppo_sequence_training=True, ppo_max_samples_per_sequence=64,
                       arena_enabled=False))
    settings = replace(settings, model=replace(settings.model, inference_temporal_cache_entries=12 * games))
    with patch.object(MetricLogger, 'start_resource_monitor'):
        trainer = ProbeTrainer(settings, run_directory=args.run_dir, auto_resume=False)
    trainer.timing_histories = []
    trainer.update, _ = restore_training_state(
        payload, expected_mode='four_dark', expected_dead_rules_enabled=settings.dead_rules_enabled,
        policy=trainer.policy, layout=trainer.layout, reference_policy=trainer.reference_policy,
        reference_layout=trainer.reference_layout, policy_optimizer=trainer.policy_optimizer,
        layout_optimizer=trainer.layout_optimizer, accelerator_device=trainer.device,
        expected_algorithm='ppo', critic=trainer.critic, critic_optimizer=trainer.critic_optimizer)
    trainer.pool.load_state_dict(pool_state)
    del payload, saved_state, pool_state
    gc.collect()
    result = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                  checkpoint=str(checkpoint), checkpoint_update=trainer.update,
                  config_revision=settings.raw_config['config_revision'], model_config=asdict(settings.model),
                  games=games, moves_per_game=args.moves_per_game, profile_waves=args.profile_waves,
                  parameters={name: sum(p.numel() for p in getattr(trainer, name).parameters())
                              for name in ('policy', 'critic', 'layout')},
                  source_sha256={name: hashlib.sha256((source_root / name[4:] if name.startswith('src/') else root / name).read_bytes()).hexdigest()
                                 for name in ('configs/bootstrap.yaml', 'src/junqi/training/models.py',
                                              'src/junqi/training/trainer.py', 'src/junqi/training/ppo.py',
                                              'src/junqi/training/paged_kv.py', 'src/junqi/training/encoding.py',
                                              'src/junqi/training/history_arrays.py', 'src/junqi/training/fixed_kv.py',
                                              'src/junqi/training/fixed_kv_kernels.py')
                                 if (source_root / name[4:] if name.startswith('src/') else root / name).is_file()},
                  imported_source_root=str(source_root),
                  limitations=['Diagnostic from a short random-initialized checkpoint, no strength inference.',
                               '40-game saved pool differs from the current 48-game default.',
                               'No optimizer step or checkpoint save is timed here.',
                               'Profiler overhead is separately timed; CUDA service is not utilization.'],
                  phases=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def measure(label, action, module=None):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if module is not None:
            module.reset_board_encoding_stats()
        started = time.perf_counter()
        extra = action() or {}
        torch.cuda.synchronize()
        phase = dict(phase=label, seconds=time.perf_counter() - started,
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, **extra)
        if module is not None:
            phase['encoding'] = module.board_encoding_metrics()
        result['phases'].append(phase)
        print(json.dumps({k: v for k, v in phase.items() if k not in ('encoding', 'blocks') }), flush=True)
        persist()
        return phase

    try:
        trainer.policy.eval()
        trainer.layout.eval()
        actor = FrozenPolicyActor(trainer.policy, amp_dtype=trainer.amp_dtype, max_batch_size=games)
        value_actor = FrozenValueActor(trainer.critic, amp_dtype=trainer.amp_dtype, max_batch_size=games)
        assert not trainer.policy.training and not trainer.critic.training
        collected = []

        def collect(count, keep=False):
            samples, outcomes, metrics = collect_ppo_samples(
                trainer.pool, actor, value_actor, trainer.layout, count=count,
                behavior_version=trainer.update, discount=settings.discount, gae_lambda=settings.gae_lambda)
            if keep:
                collected.extend(samples)
            return dict(metrics=metrics.as_dict(), history_mean=sum(len(s.state.records) for s in samples)/len(samples),
                        history_max=max(len(s.state.records) for s in samples),
                        actor_batch=actor.max_batch_size, critic_batch=value_actor.max_batch_size)

        measure('real_rollout_unprofiled', lambda: collect(games * args.moves_per_game, keep=True), trainer.policy)
        digest = hashlib.sha256()
        for sample in collected:
            record = sample.state.records[-1]
            digest.update(repr((sample.state.mode.value, record.board_codes, record.known_casualty_bits,
                                None if record.action is None else record.action.as_tuple(),
                                record.no_interaction_plies, record.active_mask, record.revealed_mask,
                                record.current_player, sample.action)).encode('utf-8'))
        result['observed_rollout_sha256'] = digest.hexdigest()
        result['critic_rollout_encoding'] = trainer.critic.board_encoding_metrics()
        # Use the still-warm caches and real environment progression, rather than
        # repeatedly querying one identical state that could hit a final-output cache.
        measure('warm_rollout_unprofiled', lambda: collect(games * args.profile_waves), trainer.policy)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, with_flops=False) as prof:
            measure('warm_rollout_torch_profiled', lambda: collect(games * args.profile_waves), trainer.policy)
        result['rollout_torch_profile'] = summarize_profiler(prof)
        del prof
        profiler = cProfile.Profile()
        profiler.enable()
        measure('warm_rollout_python_profiled', lambda: collect(games * args.profile_waves), trainer.policy)
        profiler.disable()
        result['rollout_python_profile'] = python_profile(profiler)
        persist()
        trainer.policy.train()
        trainer.critic.train()
        del actor, value_actor
        torch.cuda.empty_cache()
        samples = list(trainer._ppo_optimizer_batches(trainer._normalize_ppo_batch(collected))[0])
        result['learner_batch'] = dict(samples=len(samples),
            history_mean=sum(len(s.state.records) for s in samples) / len(samples),
            history_min=min(len(s.state.records) for s in samples),
            history_max=max(len(s.state.records) for s in samples))
        for name in ('policy', 'critic'):
            module = getattr(trainer, name)
            backward = getattr(trainer, '_backward_' + name + '_epoch')
            # First pass warms kernels/allocator; subsequent identical passes
            # change gradients only, without updating weights or optimizer state.
            measure(name + '_backward_warmup', lambda: dict(metrics=backward(samples)), module)
            measure(name + '_backward_unprofiled', lambda: dict(metrics=backward(samples)), module)
            with count_blocks(module) as counts:
                measure(name + '_backward_block_counted', lambda: dict(metrics=backward(samples)), module)
            result[name + '_blocks_including_recompute'] = dict(counts)
            getattr(trainer, name + '_optimizer').zero_grad(set_to_none=True)
        profile_samples = samples[:128]
        result['learner_profile_samples'] = len(profile_samples)
        measure('policy_profile_subset_unprofiled', lambda: dict(metrics=trainer._backward_policy_epoch(profile_samples)), trainer.policy)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, with_flops=True) as prof:
            measure('policy_backward_torch_profiled', lambda: dict(metrics=trainer._backward_policy_epoch(profile_samples)), trainer.policy)
        result['learner_torch_profile'] = summarize_profiler(prof)
        del prof
        profiler = cProfile.Profile()
        profiler.enable()
        measure('policy_backward_python_profiled', lambda: dict(metrics=trainer._backward_policy_epoch(samples)), trainer.policy)
        profiler.disable()
        result['learner_python_profile'] = python_profile(profiler)
        trainer.policy_optimizer.zero_grad(set_to_none=True)
        original_config = trainer.policy.config
        try:
            trainer.policy.config = replace(original_config, activation_checkpointing=False)
            measure('policy_backward_checkpoint_off_warmup', lambda: dict(metrics=trainer._backward_policy_epoch(samples)), trainer.policy)
            measure('policy_backward_checkpoint_off_unprofiled', lambda: dict(metrics=trainer._backward_policy_epoch(samples)), trainer.policy)
        except torch.OutOfMemoryError as error:
            result['checkpoint_off_error'] = str(error)
        finally:
            trainer.policy.config = original_config
        result['source_checkpoint_stat_unchanged'] = (checkpoint.stat().st_size == checkpoint_stat.st_size
                                                     and checkpoint.stat().st_mtime_ns == checkpoint_stat.st_mtime_ns)
        result['complete'] = True
        persist()
    finally:
        trainer.logger.close()


if __name__ == '__main__':
    main()
