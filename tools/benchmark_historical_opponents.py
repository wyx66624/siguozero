"""Bounded inference/KV diagnostic. Never starts training or writes a checkpoint.

Replays exactly the same legal game histories through one policy, or through
learner/frozen policies with 20% historical games. This is not a training ETA.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
import weakref

import torch

from junqi.training.encoding import GameHistory
from junqi.training.distributed import DistributedContext
from junqi.training.historical_opponents import HistoricalOpponents, packed_weights
from junqi.training.inference_weights import install_packed_weights
from junqi.training.models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import new_game
from junqi.training.rollout import FrozenPolicyActor
from junqi.training.settings import TrainingSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_4090_training.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--matrix-fp32", action="store_true", help="control: retain frozen linear weights in FP32")
    parser.add_argument("--teammate-fraction", type=float, help="override teammate mix; 0 reproduces opponent-only sampling")
    parser.add_argument("--waves", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (root / "src/junqi/training").glob("*.py")}
    if not 8 <= args.waves <= 64 or not 1 <= args.repeats <= 5:
        parser.error("waves must be 8..64 and repeats 1..5")
    lock = open(Path(tempfile.gettempdir()) / "siguozero-cuda-probe.lock", "a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(124)
    settings = TrainingSettings.from_yaml(args.config, "four_dark", model_scale="main")
    if args.teammate_fraction is not None:
        settings = replace(settings, historical_teammate_fraction=args.teammate_fraction)
        settings.validate()
    config, device = settings.model, torch.device("cuda")
    policy = GamePolicyTransformer(config).to(device).eval()
    layout = PieceConditionedLayoutPointerDecoder(config).to(device).eval()
    source = None
    snapshot_directory = None
    if args.checkpoint:
        original = Path(args.checkpoint).resolve()
        before = original.stat()
        snapshot_directory = tempfile.TemporaryDirectory(prefix="siguozero-history-probe-")
        path = Path(snapshot_directory.name) / "input.pt"
        shutil.copyfile(original, path)
        after_copy = original.stat()
        if (before.st_size, before.st_mtime_ns) != (after_copy.st_size, after_copy.st_mtime_ns):
            raise RuntimeError("source checkpoint changed during the isolated copy; retry after migration")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if payload["mode"] != "four_dark":
            raise ValueError("diagnostic checkpoint must be four_dark")
        policy.load_state_dict(payload["policy"])
        layout.load_state_dict(payload["layout"])
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        source = dict(path=str(original), update=payload["update"], format_version=payload["format_version"],
                      size=before.st_size, mtime_ns=before.st_mtime_ns, isolated_copy_sha256=digest)
        del payload
    old = GamePolicyTransformer(config)
    old_layout = PieceConditionedLayoutPointerDecoder(config)
    policy_state, policy_packs = packed_weights(policy)
    layout_state, layout_packs = packed_weights(layout)
    before_frozen = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    start = time.perf_counter()
    matrix_dtype = None if args.matrix_fp32 else torch.bfloat16
    ps, pb, pc = install_packed_weights(old, policy_packs, device, matrix_dtype=matrix_dtype)
    ls, lb, lc = install_packed_weights(old_layout, layout_packs, device)
    torch.cuda.synchronize()
    first_load = time.perf_counter() - start
    frozen_additional = torch.cuda.memory_allocated() - before_frozen
    old.eval().requires_grad_(False)
    old_layout.eval().requires_grad_(False)
    swaps = []
    pointers = [t.data_ptr() for t in (*ps.values(), *ls.values())]
    for _ in range(5):
        start = time.perf_counter()
        install_packed_weights(old, policy_packs, device, slabs=ps, matrix_dtype=matrix_dtype)
        install_packed_weights(old_layout, layout_packs, device, slabs=ls)
        torch.cuda.synchronize()
        swaps.append(time.perf_counter() - start)
    assert pointers == [t.data_ptr() for t in (*ps.values(), *ls.values())]
    del policy_packs, layout_packs, policy_state, layout_state
    gc.collect()
    count = settings.base_game_pool_size
    # Use the real joint quota scheduler, without loading/saving any archive.
    role_directory = tempfile.TemporaryDirectory(prefix="siguozero-role-probe-")
    roles = HistoricalOpponents(replace(settings, historical_cohort_games=count + 1),
                                role_directory.name, DistributedContext(0, 1, 0, device))
    roles.progress, roles.active_id = roles.threshold, "diagnostic_frozen"
    assignments = [roles.assignment() for _ in range(count)]
    role_counts = dict(roles.scenario_started)
    rng = random.Random(613)
    games = [new_game("four_dark", seed=i + 817, no_capture_draw_plies=settings.no_capture_draw_plies) for i in range(count)]
    histories = [GameHistory.initialize(g, "four_dark", max_transitions=config.max_transitions) for g in games]
    for history in histories:
        history.enable_array_storage()
    bank = []
    for _ in range(args.waves):
        wave = []
        for index, (game, history) in enumerate(zip(games, histories, strict=True)):
            if game.is_terminal:
                continue
            opponent, teammate, seat = assignments[index]
            frozen_turn = bool(opponent and game.current_player % 2 != seat % 2
                               or teammate and game.current_player == (seat + 2) % 4)
            wave.append((history.state_for(game), frozen_turn))
            game.step(rng.choice(game.legal_actions()))
            history.append_after_step(game)
        bank.append(wave)
    memory, timings = {"frozen_replica_additional_bytes": frozen_additional}, {}
    peaks = {}
    resident = torch.cuda.memory_allocated()
    # Both orders get an untimed warm-up for Triton and sampling compilation.
    for mixed in (False, True):
        results = []
        for iteration in range(args.repeats + 1):
            policy.clear_inference_board_cache()
            old.clear_inference_board_cache()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            actor = FrozenPolicyActor(policy, amp_dtype=torch.bfloat16, max_batch_size=settings.actor_inference_batch)
            frozen = FrozenPolicyActor(old, amp_dtype=torch.bfloat16, max_batch_size=settings.actor_inference_batch)
            policy.start_ppo_inference_cache(capacity=count * 4, behavior_version=iteration)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                policy._fixed_kv_store.allocate()
                allocated = torch.cuda.memory_allocated()
                if mixed:
                    old.start_ppo_inference_cache(capacity=count * 4, behavior_version=iteration)
                    old._fixed_kv_store.share_storage(policy._fixed_kv_store)
                    memory["shared_kv_additional_bytes"] = torch.cuda.memory_allocated() - allocated
                    assert old._fixed_kv_store.storage is policy._fixed_kv_store.storage
                store = policy._fixed_kv_store
                memory["kv_bytes"] = store.storage.numel() * store.storage.element_size() + store.contexts.numel() * store.contexts.element_size()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for wave in bank:
                    if not mixed:
                        actor.sample([state for state, _ in wave], count=1)
                    else:
                        for source_actor, is_old in ((actor, False), (frozen, True)):
                            states = [s for s, frozen_turn in wave if frozen_turn == is_old]
                            if states:
                                source_actor.sample(states, count=1)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if iteration:
                    results.append(elapsed)
                retired = weakref.ref(store)
                del store  # do not retain a retired 12 GB arena into the next repetition
            old.clear_inference_board_cache()
            policy.clear_inference_board_cache()
            peak = torch.cuda.max_memory_allocated()
            peaks["mixed" if mixed else "self_play"] = max(peaks.get("mixed" if mixed else "self_play", 0), peak)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated() > resident + 256 * 2**20:
                raise RuntimeError(f"diagnostic allocation retained: base={resident}, current={torch.cuda.memory_allocated()}, cache_alive={retired() is not None}")
        timings["mixed" if mixed else "self_play"] = results
    if snapshot_directory is not None:
        snapshot_directory.cleanup()
    role_directory.cleanup()
    after_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_hashes}
    if source_hashes != after_hashes:
        raise RuntimeError("source changed during the probe; retry on a stable source tree")
    result = dict(kind="matched_history_inference_and_resident_memory_diagnostic",
                  training_started=False, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  checkpoint=source, model=asdict(config), games=count, fixed_kv_capacity=count * 4,
                  role_counts=role_counts, role_target_fractions=roles.role_targets(),
                  decisions=sum(map(len, bank)), frozen_decisions=sum(old for wave in bank for _, old in wave),
                  maximum_history_tokens=max(len(s.records) for wave in bank for s, _ in wave),
                  memory=memory, peak_allocated_bytes=peaks,
                  frozen_policy_layout_bytes=pb + lb, upload_calls_per_version=pc + lc,
                  frozen_matrix_dtype=str(matrix_dtype),
                  first_load_seconds=first_load, reused_storage_upload_seconds=swaps,
                  inference_seconds=timings,
                  source_sha256=source_hashes, source_hashes_captured="before_model_construction_and_verified_after_probe",
                  limits="Same histories and decision counts; excludes optimizer, environment, archive I/O and full-window decode latency. Weights identical across contestants for timing only.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("memory", "frozen_policy_layout_bytes", "upload_calls_per_version", "inference_seconds", "reused_storage_upload_seconds")}), flush=True)


if __name__ == "__main__":
    main()
