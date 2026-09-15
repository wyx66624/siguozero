"""Bounded real PPO updates resumed from a read-only checkpoint.

PYTHONPATH can select a preserved source snapshot for the baseline. Restores
models, optimizers, unfinished games and RNG; never saves model parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import torch

import junqi.training.models as loaded_models
from junqi.training.checkpoint import CheckpointManager
from junqi.training.metrics import MetricLogger
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument('--config', default='configs/bootstrap.yaml')
    parser.add_argument('--microbatch', type=int)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--environment-workers', type=int)
    parser.add_argument('--samples-per-sequence', type=int)
    parser.add_argument('--layout-microbatch', type=int)
    parser.add_argument('--tensor-learner', action='store_true')
    parser.add_argument('--varlen-attention', action='store_true')
    parser.add_argument('--low-precision-residual', action='store_true')
    parser.add_argument('--compile-mode', default=None)
    parser.add_argument('--adaptive-learning-rate', choices=('on', 'off'), default=None,
                        help='compare final-policy KL probe overhead on identical optimizer work')
    parser.add_argument('--deferred-values', action='store_true')
    parser.add_argument('--pipeline-groups', type=int)
    parser.add_argument('--fused-optimizer', action='store_true')
    parser.add_argument('--learner-graphs', choices=('on', 'off'))
    parser.add_argument('--adaptive-clip', choices=('on', 'off'))
    parser.add_argument('--historical-library', help='isolated frozen library; force ONLY this probe to the half gate')
    parser.add_argument('--historical-cache-gib', type=float, default=0)
    parser.add_argument('--historical-cohort-games', type=int, default=1024)
    parser.add_argument('--sampling-graphs', action='store_true')
    parser.add_argument('--full-policy-epochs', action='store_true',
                        help='benchmark only: hold optimizer work fixed by disabling KL early exit')
    parser.add_argument("--moves-per-game", type=int, default=64)
    parser.add_argument('--transitions-per-update', type=int,
                        help='hold the exact global update size fixed for arbitrary pool sizes')
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--games", type=int,
                        help="optionally expand the saved pool in memory; preserves every saved game")
    args = parser.parse_args()
    if min(args.moves_per_game, args.updates) <= 0 or (args.games is not None and args.games <= 0):
        parser.error("moves-per-game, updates and games must be positive")
    lock = open(Path(tempfile.gettempdir()) / "siguozero-cuda-probe.lock", "a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.torch_threads <= 0:
        parser.error('torch-threads must be positive')
    torch.set_num_threads(args.torch_threads)
    torch.cuda.set_per_process_memory_fraction(.90)
    checkpoint = Path(args.checkpoint).resolve()
    before = checkpoint.stat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["trainer_state"]
    pool = state.get("base_game_pool")
    if pool is None:
        pool = state["distributed"]["rank_states"][0]["base_game_pool"]
    if any(slot.get('opponent_id') is not None or slot.get('teammate_id') is not None
           for slot in pool['slots']):
        parser.error('this bounded probe requires a pre-activation self-play pool')
    # The isolated probe has no historical snapshot files. Before activation,
    # remove only this inactive controller metadata from the in-memory payload.
    pool.pop('historical_opponents', None)
    saved_games, update = int(pool["pool_size"]), int(payload["update"])
    games = args.games or saved_games
    transitions = args.transitions_per_update or games * args.moves_per_game
    if transitions <= 0:
        parser.error('transitions-per-update must be positive')
    if games < saved_games:
        parser.error("this probe only expands a saved pool; it never discards saved games")
    if "distributed" in state and len(state["distributed"]["rank_states"]) != 1:
        parser.error("this single-device probe requires a single-rank checkpoint")
    # Isolated benchmarking only: keep all saved games and create extra games
    # with the restored layout/RNG before starting the measured training loop.
    pool["pool_size"] = games
    if "base_game_pool" in state:
        state["base_game_pool"]["pool_size"] = games
    root = Path(__file__).resolve().parents[1]
    source_root = Path(loaded_models.__file__).resolve().parents[2]
    # Capture provenance before model construction/compilation, not at the end
    # of a long run when unrelated edits might have changed the working tree.
    source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((source_root / 'junqi').rglob('*.py'))}
    benchmark_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_hash = hashlib.sha256((root / args.config).read_bytes()).hexdigest()
    overrides = {
        "device": "cuda", "base_game_pool_size": games,
        "actor_inference_batch": games, "anchor_batch": transitions,
        "ppo_minibatch_samples": 512,
        "arena_enabled": False, "checkpoint_policy": "evaluation",
        "inference_snapshot_every_updates": 0,
        "historical_enabled": False,
        "arena_after_half_historical_only": False,
    }
    if args.full_policy_epochs:
        overrides.update(early_stop_kl_multiple=1e12, early_stop_clip_fraction=1.0)
    if args.adaptive_learning_rate is not None:
        overrides['adaptive_learning_rate'] = args.adaptive_learning_rate == 'on'
    if args.adaptive_clip is not None:
        overrides['ppo_adaptive_clip'] = args.adaptive_clip == 'on'
    if args.microbatch is not None:
        overrides['policy_microbatch'] = args.microbatch
    if args.environment_workers is not None:
        overrides['rollout_environment_workers'] = args.environment_workers
    if args.samples_per_sequence is not None:
        overrides['ppo_max_samples_per_sequence'] = args.samples_per_sequence
    if args.layout_microbatch is not None:
        overrides['layout_microbatch_size'] = args.layout_microbatch
    if args.deferred_values:
        overrides['ppo_deferred_values'] = True
    if args.pipeline_groups is not None:
        overrides['ppo_pipeline_groups'] = args.pipeline_groups
    if args.fused_optimizer:
        overrides['ppo_fused_optimizer'] = True
    if args.learner_graphs is not None:
        overrides['ppo_learner_cuda_graphs'] = args.learner_graphs == 'on'
    settings = TrainingSettings.from_yaml(root / args.config, "four_dark", model_scale="main", overrides=overrides)
    if args.tensor_learner or args.varlen_attention or args.low_precision_residual or args.sampling_graphs or args.compile_mode is not None:
        from dataclasses import replace
        settings = replace(settings, model=replace(settings.model,
            ppo_tensor_learner=args.tensor_learner or args.varlen_attention or settings.model.ppo_tensor_learner,
            ppo_varlen_attention=args.varlen_attention or settings.model.ppo_varlen_attention,
            ppo_low_precision_residual=args.low_precision_residual or settings.model.ppo_low_precision_residual,
            ppo_sampling_graphs=args.sampling_graphs or settings.model.ppo_sampling_graphs,
            ppo_compile_mode=args.compile_mode or settings.model.ppo_compile_mode))
    # The real resume path consumes the checkpoint in memory. No model file is
    # copied into or mutated inside the temporary benchmark run directory.
    with patch.object(CheckpointManager, "load_latest", return_value=payload), \
            patch.object(CheckpointManager, "write_manifest"), \
            patch.object(MetricLogger, "start_resource_monitor"):
        trainer = SelfPlayTrainer(settings, run_directory=args.run_dir)
    del payload, state, pool
    if args.historical_library:
        from dataclasses import replace
        from junqi.training.historical_opponents import HistoricalOpponents
        # This local diagnostic changes counters only inside the probe. The
        # production checkpoint, RNG state and already-running games are intact.
        library_settings = replace(settings, historical_enabled=True,
            historical_cache_gib=args.historical_cache_gib,
            historical_cohort_games=args.historical_cohort_games)
        trainer.historical = HistoricalOpponents(library_settings, args.historical_library, trainer.distributed)
        trainer.pool.historical = trainer.historical
        trainer.cumulative['environment_plies'] = trainer.historical.threshold
        trainer.historical.update_progress(trainer.policy, trainer.layout,
            environment_plies=trainer.cumulative['environment_plies'], update=trainer.update)
    trainer.pool.fill(trainer.layout.eval(), trainer.update)
    gc.collect()
    counts_before = dict(trainer.cumulative)
    policy_steps = critic_steps = layout_steps = 0
    def count_policy(_optimizer, _args, _kwargs):
        nonlocal policy_steps
        policy_steps += 1
    def count_critic(_optimizer, _args, _kwargs):
        nonlocal critic_steps
        critic_steps += 1
    def count_layout(_optimizer, _args, _kwargs):
        nonlocal layout_steps
        layout_steps += 1
    trainer.policy_optimizer.register_step_post_hook(count_policy)
    trainer.critic_optimizer.register_step_post_hook(count_critic)
    trainer.layout_optimizer.register_step_post_hook(count_layout)
    history_rows = []
    original_normalize = trainer._normalize_ppo_batch
    def normalize(samples):
        lengths = [len(sample.state.records) for sample in samples]
        history_rows.append(dict(update=trainer.update + 1, samples=len(samples),
                                 mean_tokens=sum(lengths) / len(lengths),
                                 max_tokens=max(lengths),
                                 window_limit_fraction=sum(length == settings.model.max_sequence_tokens for length in lengths) / len(lengths)))
        return original_normalize(samples)
    trainer._normalize_ppo_batch = normalize
    original_evaluate = trainer._maybe_evaluate_model
    def bound_updates():
        result = original_evaluate()
        if trainer.update >= update + args.updates:
            trainer.stop_requested = True
        return result
    trainer._maybe_evaluate_model = bound_updates
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    original_log = trainer.logger.log
    def log(current_update, metrics):
        nonlocal last_boundary
        original_log(current_update, metrics)
        if "rollout/environment_plies" not in metrics:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        records.append(dict(update=current_update, wall_seconds=now - last_boundary,
                            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                            metrics=dict(metrics), history=history_rows[-1]))
        output.write_text(json.dumps(dict(complete=False, updates=records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(completed_probe_update=len(records), requested_updates=args.updates,
                              environment_steps=metrics["rollout/environment_plies"],
                              wall_seconds=records[-1]["wall_seconds"], history=history_rows[-1])), flush=True)
        last_boundary = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
    trainer.logger.log = log
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = last_boundary = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    steps = trainer.cumulative["environment_plies"] - counts_before["environment_plies"]
    result = dict(
        device=torch.cuda.get_device_name(), torch=torch.__version__, imported_source_root=str(source_root),
        torch_threads=torch.get_num_threads(),
        source_sha256=source_hashes, source_hashes_captured='before_trainer_construction',
        benchmark_sha256=benchmark_hash,
        config_sha256=config_hash,
        settings=settings.serializable(),
        checkpoint=str(checkpoint), checkpoint_update=update, completed_update=trainer.update,
        historical_probe=(dict(library=args.historical_library, cache_gib=args.historical_cache_gib,
            cohort_games=args.historical_cohort_games, forced_half_gate=True,
            metrics=trainer.historical.metrics()) if args.historical_library else None),
        games=games, saved_games=saved_games, added_games_before_timing=games - saved_games,
        transitions_per_update=transitions,
        requested_updates=args.updates, environment_steps=steps, model_config=asdict(settings.model),
        policy_epochs=settings.policy_epochs, critic_epochs=settings.critic_epochs,
        ppo_minibatch_samples=settings.ppo_minibatch_samples,
        configured_microbatch=settings.policy_microbatch,
        effective_microbatch=trainer.effective_policy_microbatch,
        environment_workers=settings.rollout_environment_workers,
        deferred_values=getattr(settings, 'ppo_deferred_values', False),
        pipeline_groups=getattr(settings, 'ppo_pipeline_groups', 1),
        fused_optimizer=getattr(settings, 'ppo_fused_optimizer', False),
        layout_prefetch_games=getattr(settings, 'layout_prefetch_games', 1),
        tensor_learner_calls={name: getattr(model, '_tensor_learner_calls', 0)
                              for name, model in (('policy', trainer.policy), ('critic', trainer.critic))},
        compile_active=(getattr(trainer.policy, '_tensor_learner_callable', None) is not None
                        or getattr(trainer.policy, '_packed_learner_callable', None) is not None),
        varlen_calls={name: getattr(model, '_varlen_calls', 0)
                      for name, model in (('policy', trainer.policy), ('critic', trainer.critic))},
        policy_optimizer_steps=policy_steps, critic_optimizer_steps=critic_steps,
        layout_optimizer_steps=layout_steps, layout_update_interval=settings.layout_update_interval,
        updates=records,
        wall_seconds=seconds, full_update_environment_steps_per_second=steps / seconds,
        peak_allocated_gib=max(row["peak_allocated_gib"] for row in records),
        peak_reserved_gib=max(row["peak_reserved_gib"] for row in records),
        checkpoint_stat_unchanged=(checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size)
                                  == (before.st_mtime_ns, before.st_size),
        limitations=["Bounded RTX 4090 training window, not a PRO 6000 or long-run measurement.",
                     "No evaluation match or parameter save; graph capture and any scheduled layout updates are included.",
                     "When expanding the pool, the saved games remain and additional games are initialized before timing."],
        complete=True,
    )
    assert result["checkpoint_stat_unchanged"]
    assert steps == transitions * args.updates
    assert trainer.update == update + args.updates and len(records) == args.updates
    assert not list(trainer.run_directory.rglob("*.pt"))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("device", "requested_updates", "environment_steps", "wall_seconds", "full_update_environment_steps_per_second", "policy_optimizer_steps", "critic_optimizer_steps", "layout_optimizer_steps", "complete")}), flush=True)


if __name__ == "__main__":
    main()
