from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from junqi.training.distributed import DistributedContext
from junqi.training.encoding import GameHistory
from junqi.training.historical_opponents import HistoricalOpponents, packed_weights
from junqi.training.model_selection import ModelSelection
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import TrainingMode, new_game
from junqi.training.ppo import FrozenValueActor, PPOSample, PPOTransition, collect_ppo_samples, generalized_advantages, policy_ppo_loss
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor
from junqi.training.settings import TrainingSettings


def settings(**overrides):
    model = replace(ModelConfig.tiny(), max_transitions=8, ppo_fixed_kv=True,
                    ppo_array_history=True, ppo_cuda_graphs=False, activation_checkpointing=False)
    base = TrainingSettings.from_yaml(Path(__file__).parents[1] / "configs/bootstrap.yaml", "four_dark", tiny=True,
        overrides=dict(device="cpu", total_updates=100, target_environment_plies=1000,
                       anchor_batch=32, base_game_pool_size=5, policy_microbatch=2,
                       ppo_deferred_values=True, ppo_pipeline_groups=2, rollout_environment_workers=2,
                       historical_enabled=True, historical_snapshot_fractions=(.1, .2),
                       historical_teammate_fraction=overrides.pop("historical_teammate_fraction", 0.),
                       historical_cohort_games=4, historical_eval_games=4, layout_prefetch_games=4,
                       arena_enabled=True, arena_games=4, arena_max_plies=8,
                       arena_interval_environment_plies=100, **overrides))
    return replace(base, model=model)


def distributed_mask_worker(rank, directory):
    from junqi.training.trainer import SelfPlayTrainer
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2,
        init_method="file://" + str(Path(directory) / "rendezvous"))
    context = DistributedContext(rank, 2, rank, torch.device("cpu"), backend="gloo")
    config = settings()
    trainer = SelfPlayTrainer(config, run_directory=Path(directory) / "run", distributed=context)
    try:
        trainer.historical.progress = 500
        game = new_game("four_dark", seed=713)
        state = GameHistory.initialize(game, "four_dark", max_transitions=8).state_for(game)
        sample = PPOSample(state, state.legal_actions[0], -3., 0., 1., 1., 0, learnable=rank == 1)
        reference = copy.deepcopy(trainer.policy).train()
        expected = policy_ppo_loss(reference, [replace(sample, learnable=True)],
                                  clip_epsilon=config.clip_epsilon, entropy_coefficient=config.entropy_coefficient,
                                  sequence_training=trainer._sequence_training_enabled())
        expected.loss.backward()
        trainer._backward_policy_epoch([sample])
        for p, q in zip(trainer.policy.parameters(), reference.parameters(), strict=True):
            if q.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=2e-6)
        original = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
        result = trainer._update_policy_batch([replace(sample, learnable=False)], epochs=1)
        assert result["optimizer/policy_steps"] == 0
        for key, value in trainer.policy.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    finally:
        trainer.logger.close()
        torch.distributed.destroy_process_group()


class HistoricalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = settings()
        self.context = DistributedContext(0, 1, 0, torch.device("cpu"))
        self.policy = GamePolicyTransformer(self.settings.model).eval()
        self.layout = PieceConditionedLayoutPointerDecoder(self.settings.model).eval()
        self.league = HistoricalOpponents(self.settings, self.root, self.context)
        self.pool = BaseGamePool("four_dark", pool_size=5, max_transitions=8, max_game_plies=8,
                                 seed=77, layout_prefetch_games=4)
        self.pool.historical = self.league

    def archive(self):
        for step in (0, 100, 200):
            with torch.no_grad():
                self.policy.action_encoder.projection.weight.add_(.01)
            self.league.update_progress(self.policy, self.layout, environment_plies=step, update=step // 10)

    def activate(self):
        self.archive()
        self.league.update_progress(self.policy, self.layout, environment_plies=500, update=50)
        actor = FrozenPolicyActor(self.policy, max_batch_size=8)
        self.league.begin_rollout(self.pool, actor, 50)
        return actor

    def test_half_gate_and_archive_are_bounded_and_inference_only(self):
        self.archive()
        for step in (201, 300, 499):
            self.league.update_progress(self.policy, self.layout, environment_plies=step, update=99_999)
            self.assertFalse(self.league.active)
            self.assertEqual(self.league.assignment(), (None, None, 0))
        self.assertIsNone(self.league.policy)
        self.assertEqual(len(list(self.league.directory.glob("*.pt"))), 3)
        payload = torch.load(self.league.directory / self.league.entries[0]["file"], mmap=True, weights_only=False)
        self.assertNotIn("policy_optimizer", payload)
        self.assertNotIn("critic", payload)
        storage = {v.untyped_storage().data_ptr() for v in payload["policy"].values()}
        self.assertLessEqual(len(storage), 2)
        self.league.update_progress(self.policy, self.layout, environment_plies=500, update=1)
        self.assertTrue(self.league.active)

    def test_distributed_empty_rank_has_correct_global_gradient_and_no_empty_adam_step(self):
        torch.multiprocessing.spawn(distributed_mask_worker, args=(str(self.root),), nprocs=2, join=True)

    def test_graph_warmup_stream_is_shared_across_captures_without_retaining_models(self):
        from junqi.training import cuda_graph_runtime
        with patch.object(cuda_graph_runtime, "_local", threading.local()), \
             patch.object(torch.cuda, "Stream", side_effect=lambda **_: object()) as create:
            first = cuda_graph_runtime.warmup_stream("cuda:0")
            self.assertIs(first, cuda_graph_runtime.warmup_stream("cuda:0"))
            self.assertIsNot(first, cuda_graph_runtime.warmup_stream("cuda:1"))
            self.assertEqual(create.call_count, 2)
            with self.assertRaises(ValueError):
                cuda_graph_runtime.warmup_stream("cpu")

    def test_cohort_admission_weights_and_resume(self):
        actor = self.activate()
        assignments = [self.league.assignment() for _ in range(20)]
        self.assertEqual(sum(i is not None for i, _, _ in assignments), 4)
        self.assertEqual([s % 2 for i, _, s in assignments if i is not None].count(0), 2)
        self.assertEqual([s % 2 for i, _, s in assignments if i is not None].count(1), 2)
        identifier = self.league.active_id
        before = self.league.probabilities()[identifier]
        slot = type("Slot", (), dict(opponent_id=identifier))()
        for _ in range(100):
            self.league.record_result(slot, 1.)
        self.assertLess(self.league.probabilities()[identifier], before)
        self.assertGreaterEqual(min(self.league.probabilities().values()), .05 / 3)
        self.assertAlmostEqual(sum(self.league.probabilities().values()), 1)
        state = self.league.state_dict()
        other = HistoricalOpponents(self.settings, self.root, self.context)
        other.load_state_dict(state)
        self.assertEqual([self.league.assignment() for _ in range(17)], [other.assignment() for _ in range(17)])
        # No unfinished historical game: choose a new cohort without one load per game.
        self.league.begin_rollout(self.pool, actor, 51)
        self.assertLessEqual(self.league.load_count, 2)

    def test_long_cohort_drain_is_repaid_without_per_game_model_swaps(self):
        actor = self.activate()
        for _ in range(120):
            self.league.assignment()
        self.assertEqual(self.league.historical_started, 4)
        self.assertAlmostEqual(self.league.credit, 20.)
        self.league.begin_rollout(self.pool, actor, 51)
        self.assertGreaterEqual(self.league.cohort_limit, 40)
        for _ in range(50):
            self.league.assignment()
        self.assertAlmostEqual(self.league.historical_started / self.league.started, .2)
        self.assertAlmostEqual(self.league.credit, 0.)

    def activate_independent_roles(self, cohort=1000):
        self.settings = replace(self.settings, historical_teammate_fraction=.2, historical_cohort_games=cohort)
        self.league = HistoricalOpponents(self.settings, self.root, self.context)
        self.pool.historical = self.league
        return self.activate()

    def test_independent_role_marginals_overlap_and_four_seat_rotation(self):
        self.activate_independent_roles()
        assigned = [self.league.assignment() for _ in range(100)]
        counts = Counter((bool(o), bool(t)) for o, t, _ in assigned)
        self.assertEqual(counts, {(False, False):64, (True, False):16, (False, True):16, (True, True):4})
        for role, count in counts.items():
            seats = Counter(s for o, t, s in assigned if (bool(o), bool(t)) == role)
            self.assertEqual(seats, dict.fromkeys(range(4), count // 4))
        self.assertEqual(self.league.historical_started, 20)
        self.assertEqual(self.league.teammate_started, 20)
        self.assertEqual(self.league.cohort_started, 36)
        # Both roles use one pinned version, never a second GPU replica.
        self.assertTrue(all(o == t for o, t, _ in assigned if o and t))
        saved = self.league.state_dict()
        other = HistoricalOpponents(self.settings, self.root, self.context)
        other.load_state_dict(saved)
        self.assertEqual([self.league.assignment() for _ in range(101)], [other.assignment() for _ in range(101)])

    def test_three_mixed_quotas_are_repaid_after_cohort_drain(self):
        actor = self.activate_independent_roles(cohort=4)
        for _ in range(120):
            self.league.assignment()
        self.assertEqual(self.league.cohort_started, 4)
        self.league.begin_rollout(self.pool, actor, 51)
        for _ in range(100):
            self.league.assignment()
        for key, target in self.league.role_targets().items():
            self.assertLess(abs(self.league.scenario_started[key] - 220 * target), 1.)
        self.assertLessEqual(self.league.load_count, 2)

    def test_pre_half_legacy_scheduler_upgrade_preserves_checkpoint_and_rng(self):
        self.archive()
        legacy = self.league.state_dict()
        legacy["version"] = 1
        config = replace(self.settings, historical_teammate_fraction=.2)
        other = HistoricalOpponents(config, self.root, self.context)
        other.load_state_dict(legacy)
        self.assertEqual(other.rng.getstate(), self.league.rng.getstate())
        self.assertEqual(other.panel(), self.league.panel())
        self.assertEqual(other.state_dict()["version"], 2)
        self.assertIsNone(other.policy)
        legacy["started"] = 10
        with self.assertRaisesRegex(ValueError, "role-schedule migration"):
            other.load_state_dict(legacy)

    def test_teammate_results_do_not_contaminate_opponent_difficulty(self):
        self.activate_independent_roles()
        identifier = self.league.active_id
        slot = type("Slot", (), dict(opponent_id=identifier, teammate_id=identifier))()
        self.league.record_result(slot, -1.)
        self.assertEqual(self.league.q, {})
        slot.opponent_id = None
        self.league.record_result(slot, 1.)
        self.assertEqual(self.league.q, {})
        self.assertEqual(self.league.teammate_results[identifier], dict(wins=1, draws=0, losses=1))
        slot.opponent_id, slot.teammate_id = identifier, None
        self.league.record_result(slot, 1.)
        self.assertLess(self.league.q[identifier], 0.)
        self.assertEqual(self.league.quality_update_games, 1)

    def test_all_role_layouts_actions_critic_views_and_resume_in_real_collector(self):
        actor = self.activate_independent_roles()
        seen = {}
        new_slot = self.pool._new_slot
        def remember(*args):
            slot = new_slot(*args)
            slot.history.enable_array_storage()
            seen[slot.history.players[0]._array_history.identity[0]] = slot
            return slot
        def check_gae(transitions, **kwargs):
            views = [(t.value_state or t.state).records.identity[1] % 2 for t in transitions]
            for a, b, t in zip(views, views[1:], transitions):
                if not t.terminal:
                    self.assertEqual(t.next_team_sign, 1 if a == b else -1)
            # Inject each possible final outcome on this real temporal/role
            # sequence. Both current teams must get their own return, and a
            # common draw penalty must never change sign at an opposing seat.
            for winner in (0, 1, None):
                utilities = [-.15 if winner is None else 1. if team == winner else -1. for team in views]
                synthetic = [replace(t, old_value=0., old_draw_value=0.,
                                     reward=utility if t.terminal else 0.,
                                     terminal_draw=t.terminal and winner is None)
                             for t, utility in zip(transitions, utilities)]
                estimated = generalized_advantages(synthetic, bootstrap_value=0., bootstrap_draw_value=0.,
                                                  discount=1., gae_lambda=1., behavior_version=50)
                through = max((i for i, t in enumerate(transitions) if t.terminal), default=-1)
                for sample, expected in zip(estimated[:through + 1], utilities):
                    self.assertAlmostEqual(sample.value_target, expected, places=6)
            return generalized_advantages(transitions, **kwargs)
        critic = GameValueTransformer(self.settings.model)
        with patch.object(self.pool, "_new_slot", side_effect=remember), \
             patch("junqi.training.ppo_pipeline.generalized_advantages", side_effect=check_gae):
            samples, layouts, metrics = collect_ppo_samples(self.pool, actor,
                FrozenValueActor(critic, amp_dtype=None, max_batch_size=8, deferred=True), self.layout,
                count=800, behavior_version=50, discount=1., gae_lambda=1., environment_workers=2, pipeline_groups=2)
        self.assertEqual({s.historical_scenario for s in seen.values()},
                         {"self_play", "historical_opponents", "historical_teammate", "historical_both"})
        self.assertEqual(len(layouts), metrics.base_games_completed * 4
                         - self.league.completed * 2 - self.league.teammate_completed)
        for sample in samples:
            game, seat = sample.state.records.identity
            slot = seen[game]
            frozen = (slot.opponent_id is not None and seat % 2 != slot.learner_team
                      or slot.teammate_id is not None and seat == (slot.primary_seat + 2) % 4)
            self.assertEqual(sample.learnable, not frozen)
            actual_view = (sample.value_state or sample.state).records.identity[1]
            self.assertEqual(actual_view, slot.primary_seat if frozen else seat)
        self.assertEqual(metrics.policy_samples, sum(s.learnable for s in samples))
        self.assertEqual(len(samples) - metrics.policy_samples, self.league.opponent_plies + self.league.teammate_plies)
        saved = self.pool.state_dict()
        other = BaseGamePool("four_dark", pool_size=5, max_transitions=8, max_game_plies=8, seed=0, layout_prefetch_games=4)
        other.historical = HistoricalOpponents(self.settings, self.root, self.context)
        other.load_state_dict(saved)
        self.assertEqual(other.state_dict(), saved)
        # Only a historical teammate still pins the shared frozen version.
        slot = next(s for s in seen.values() if s.teammate_id and not s.opponent_id)
        other.slots = [slot]
        other.historical.cohort_started = other.historical.cohort_limit
        identifier = other.historical.active_id
        other.historical.begin_rollout(other, actor, 51)
        self.assertEqual(other.historical.active_id, identifier)

    def test_packed_archive_is_usable_by_the_evaluation_engine(self):
        from junqi.training.inference import InferenceEngine
        self.archive()
        path = self.league.directory / self.league.entries[0]["file"]
        engine = InferenceEngine.from_checkpoint(path, device="cpu")
        expected = torch.load(path, mmap=True, weights_only=False)
        for name, value in engine.policy.state_dict().items():
            torch.testing.assert_close(value, expected["policy"][name], atol=0, rtol=0)
        game, history = engine.new_game(seed=13)
        self.assertIn(engine.select_action(game, history), game.legal_actions())

    def test_precast_frozen_matrices_match_amp_and_keep_norms_and_embeddings_fp32(self):
        from junqi.training.inference_weights import install_packed_weights
        frozen = copy.deepcopy(self.policy)
        _, packs = packed_weights(self.policy)
        slabs, _, _ = install_packed_weights(frozen, packs, "cpu", matrix_dtype=torch.bfloat16)
        self.assertEqual(frozen.temporal_norm.weight.dtype, torch.float32)
        self.assertEqual(frozen.position_embedding.weight.dtype, torch.float32)
        self.assertEqual(frozen.source_query[0].weight.dtype, torch.bfloat16)
        game = new_game("four_dark", seed=417)
        state = GameHistory.initialize(game, "four_dark", max_transitions=8).state_for(game)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            actual = frozen([state], [state.legal_actions])[0]
            expected = self.policy([state], [state.legal_actions])[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        pointers = {k: t.data_ptr() for k, t in slabs.items()}
        install_packed_weights(frozen, packs, "cpu", matrix_dtype=torch.bfloat16, slabs=slabs)
        self.assertEqual(pointers, {k: t.data_ptr() for k, t in slabs.items()})

    def test_weight_uploads_reuse_slabs_and_do_not_advance_learner_rng(self):
        actor = self.activate()
        pointers = {k: v.data_ptr() for k, v in self.league.device_packs.items()}
        self.assertEqual(self.league.upload_calls, len(pointers))
        for entry in self.league.entries:
            self.league.active_id = entry["id"]
            cpu_rng = torch.get_rng_state().clone()
            self.league._load_active(actor)
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
            self.assertEqual(pointers, {k: v.data_ptr() for k, v in self.league.device_packs.items()})
            expected = torch.load(self.league.directory / entry["file"], mmap=True, weights_only=False)
            for name, value in self.league.policy.state_dict().items():
                torch.testing.assert_close(value, expected["policy"][name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in self.league.policy.parameters()))

    def test_checkpoint_rewind_does_not_admit_future_disk_snapshot(self):
        self.league.update_progress(self.policy, self.layout, environment_plies=0, update=0)
        saved = self.league.state_dict()
        self.league.update_progress(self.policy, self.layout, environment_plies=100, update=10)
        discarded = self.league.entries[-1]["id"]
        restored = HistoricalOpponents(self.settings, self.root, self.context)
        restored.load_state_dict(saved)
        restored.update_progress(self.policy, self.layout, environment_plies=100, update=10)
        self.assertNotIn(discarded, [e["id"] for e in restored.panel()])
        self.assertEqual(len(restored.entries), 2)

    def test_real_optimizer_update_and_full_checkpoint_resume_with_frozen_opponents(self):
        from junqi.training.trainer import SelfPlayTrainer
        config = replace(self.settings, total_updates=1, max_game_plies=8, anchor_batch=200,
                         base_game_pool_size=25, policy_microbatch=8,
                         historical_teammate_fraction=.2, historical_cohort_games=1000)
        trainer = SelfPlayTrainer(config, run_directory=self.root / "trainer")
        for step in (100, 200, 500):
            trainer.historical.update_progress(trainer.policy, trainer.layout, environment_plies=step, update=0)
        trainer.cumulative["environment_plies"] = 500
        with patch.object(trainer, "_maybe_evaluate_model", return_value=None):
            trainer.train()
        self.assertEqual(trainer.cumulative["environment_plies"], 700)
        self.assertLess(trainer.cumulative["policy_samples"], 200)
        self.assertTrue(trainer.historical.teammate_completed)
        self.assertTrue(trainer.historical.scenario_results["historical_both"])
        pool_state = trainer.pool.state_dict()
        before = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
        resumed = SelfPlayTrainer(config, run_directory=self.root / "trainer")
        try:
            self.assertEqual(resumed.pool.state_dict(), pool_state)
            for key, value in resumed.policy.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertEqual(resumed.historical.active_id, trainer.historical.active_id)
            self.assertIsNone(resumed.historical.policy)  # lazy load, no resume-time RAM replica
        finally:
            resumed.logger.close()

    def test_mixed_collection_ownership_terminal_rewards_and_pool_resume(self):
        self.activate()
        critic = GameValueTransformer(self.settings.model)
        actor = FrozenPolicyActor(self.policy, max_batch_size=8)
        samples, layouts, metrics = collect_ppo_samples(self.pool, actor,
            FrozenValueActor(critic, amp_dtype=None, max_batch_size=8, deferred=True), self.layout,
            count=80, behavior_version=50, discount=1., gae_lambda=1., environment_workers=2, pipeline_groups=2)
        frozen = [s for s in samples if not s.learnable]
        self.assertTrue(frozen)
        self.assertEqual(len(samples), 80)
        self.assertEqual(metrics.environment_plies, 80)
        self.assertEqual(metrics.policy_samples, 80 - len(frozen))
        self.assertEqual(len(layouts), metrics.base_games_completed * 4 - self.league.completed * 2)
        self.assertIsNone(self.policy._fixed_kv_store)
        self.assertIsNone(self.league.policy._fixed_kv_store)
        for sample in frozen:
            self.assertIsNotNone(sample.value_state)
            self.assertNotEqual(sample.state.records.identity[1] % 2, sample.value_state.records.identity[1] % 2)
        saved = self.pool.state_dict()
        pool = BaseGamePool("four_dark", pool_size=5, max_transitions=8, max_game_plies=8, seed=1, layout_prefetch_games=4)
        pool.historical = HistoricalOpponents(self.settings, self.root, self.context)
        pool.load_state_dict(saved)
        self.assertEqual(pool.state_dict(), saved)
        # An unfinished cohort pins its checkpoint even after hitting the cohort cap.
        if any(s.opponent_id for s in pool.slots):
            identifier = pool.historical.active_id
            pool.historical.cohort_started = 4
            pool.historical.begin_rollout(pool, actor, 51)
            self.assertEqual(pool.historical.active_id, identifier)

    def test_historical_terminal_after_opponent_action_propagates_to_learner(self):
        game = new_game("four_dark", seed=9)
        state = GameHistory.initialize(game, "four_dark", max_transitions=8).state_for(game)
        for reward, draw in ((1., False), (-1., False), (-.15, True)):
            trace = [PPOTransition(state, state.legal_actions[0], 0., 0., 0., False, 1),
                     PPOTransition(state, state.legal_actions[0], 0., 0., reward, True, 1,
                                   terminal_draw=draw, learnable=False, value_state=state)]
            samples = generalized_advantages(trace, bootstrap_value=100., discount=1., gae_lambda=1., behavior_version=0)
            self.assertEqual([s.value_target for s in samples], [reward, reward])
            self.assertEqual([s.learnable for s in samples], [True, False])
            self.assertEqual(samples[0].draw_value_target, reward if draw else 0.)

    def test_policy_gradient_and_entropy_exclude_frozen_actions(self):
        game = new_game("four_dark", seed=9)
        state = GameHistory.initialize(game, "four_dark", max_transitions=8).state_for(game)
        good = PPOSample(state, state.legal_actions[0], -3., 0., 1., 1., 0)
        bad = replace(good, advantage=-1000., learnable=False)
        reference = copy.deepcopy(self.policy)
        args = dict(clip_epsilon=.2, entropy_coefficient=.1)
        actual = policy_ppo_loss(self.policy, [good, bad], **args)
        expected = policy_ppo_loss(reference, [good], **args)
        actual.loss.backward()
        expected.loss.backward()
        self.assertEqual(actual.metrics, expected.metrics)
        for p, q in zip(self.policy.parameters(), reference.parameters(), strict=True):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        self.policy.zero_grad(set_to_none=True)
        output = policy_ppo_loss(self.policy, [bad], **args)
        output.loss.backward()
        self.assertEqual(output.loss.item(), 0.)
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in self.policy.parameters()))

    def check_shared_fixed_kv(self, device="cpu", graphs=False, frozen_seats=(1, 3)):
        if device == "cuda":
            config = replace(self.settings.model, board_dim=128, temporal_dim=256, temporal_heads=8,
                             temporal_ffn_dim=128, temporal_layers=2, ppo_cuda_graphs=graphs)
            self.policy = GamePolicyTransformer(config).to(device).eval()
        policy, other = self.policy, copy.deepcopy(self.policy)
        with torch.no_grad():
            other.temporal_norm.weight.add_(.3)
        references = [copy.deepcopy(policy), copy.deepcopy(other)]
        if device == "cuda":
            from junqi.training.inference_weights import install_packed_weights
            _, packs = packed_weights(other)
            install_packed_weights(other, packs, device, matrix_dtype=torch.bfloat16)
            other.requires_grad_(False)
        for model in (policy, other):
            model.start_ppo_inference_cache(capacity=4, behavior_version=3)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            other._fixed_kv_store.share_storage(policy._fixed_kv_store)
        self.assertIs(policy._fixed_kv_store.storage, other._fixed_kv_store.storage)
        game = new_game("four_dark", seed=17, max_plies=70)
        history = GameHistory.initialize(game, "four_dark", max_transitions=8)
        history.enable_array_storage()
        rng = random.Random(55)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            for _ in range(24):
                state = history.state_for(game)
                owner = int(game.current_player in frozen_seats)
                model = (policy, other)[owner]
                tolerance = .02 if device == "cuda" else 4e-6
                torch.testing.assert_close(model.encode([state]).context, references[owner].encode([state]).context,
                                           atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(model([state], [state.legal_actions])[0],
                                           references[owner]([state], [state.legal_actions])[0],
                                           atol=tolerance, rtol=tolerance)
                self.assertLessEqual(len(model._fixed_kv_store.entries), 4)
                game.step(rng.choice(game.legal_actions()))
                history.append_after_step(game)
                if game.is_terminal:
                    break
            with self.assertRaisesRegex(RuntimeError, "another policy"):
                (other, policy)[owner].encode([state])
        policy._fixed_kv_store.release_game(state.records.identity[0])
        self.assertFalse(other._fixed_kv_store.entries)
        self.assertEqual(len(other._fixed_kv_store.free), 4)
        other.clear_inference_board_cache()
        policy.clear_inference_board_cache()

    def test_shared_fixed_kv_is_exact_and_does_not_allocate_a_second_arena(self):
        self.check_shared_fixed_kv()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_shared_cuda_graphs_use_the_correct_policy_weights(self):
        self.check_shared_fixed_kv("cuda", True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_shared_cuda_graphs_cover_teammate_only_and_both_roles(self):
        for frozen in ((2,), (1, 2, 3)):
            self.check_shared_fixed_kv("cuda", True, frozen)

    def test_panel_blocks_single_opponent_improvement_and_reuses_completed_matches(self):
        selection = ModelSelection(self.settings, self.root, self.context)
        selection.initialize(self.policy, self.layout, update=0, cumulative={"environment_plies": 0})
        self.archive()
        self.league.progress = 500
        def result(wins):
            return dict(games=4, wins=wins, draws=0, losses=4-wins, score=wins/4,
                        score_ci=[0., 1.], verdict="inconclusive")
        calls = []
        def run(*args, **kwargs):
            calls.append((args, kwargs))
            # Candidate beats champion, but loses to the first early opponent.
            return result(3 if len(calls) == 1 else 1 if len(calls) == 2 else 3)
        with patch("junqi.training.model_selection.run_match", side_effect=run):
            report = selection.evaluate(self.policy, self.layout, update=50,
                cumulative={"environment_plies": 500}, historical=self.league)
        self.assertEqual(len(calls), 4)
        self.assertFalse(report["promoted"])
        self.assertEqual(len(report["historical_panel"]["results"]), 3)
        self.assertFalse(report["historical_panel"]["promotion_allowed"])
        self.assertIs(calls[0][1]["candidate_engine"], calls[-1][1]["candidate_engine"])
        self.assertEqual({c[0][2].pairs for c in calls}, {1})
        self.assertTrue(all(c[1]["alpha"] == .05 / (len(selection.milestones) * 4) for c in calls))
        self.assertTrue((self.league.directory / "latest_evaluation.json").is_file())

    def test_interrupted_panel_reuses_champion_and_completed_opponents(self):
        selection = ModelSelection(self.settings, self.root, self.context)
        selection.initialize(self.policy, self.layout, update=0, cumulative={"environment_plies": 0})
        self.archive()
        self.league.progress = 500
        report = dict(games=4, wins=3, draws=0, losses=1, score=.75, score_ci=[0., 1.], verdict="inconclusive")
        args = dict(update=50, cumulative={"environment_plies": 500}, historical=self.league)
        with patch("junqi.training.model_selection.run_match", side_effect=[report.copy(), report.copy(), InterruptedError()]):
            with self.assertRaises(InterruptedError):
                selection.evaluate(self.policy, self.layout, **args)
        with patch("junqi.training.model_selection.run_match", return_value=report.copy()) as run:
            result = selection.evaluate(self.policy, self.layout, **args)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(len(result["historical_panel"]["results"]), 3)

    def historical_only_selection(self):
        self.settings = replace(self.settings, arena_after_half_historical_only=True,
                                arena_after_half_interval_environment_plies=50)
        selection = ModelSelection(self.settings, self.root, self.context)
        selection.initialize(self.policy, self.layout, update=0, cumulative={"environment_plies": 0})
        self.archive()
        return selection

    @staticmethod
    def losing_result():
        return dict(games=4, wins=0, draws=0, losses=4, score=0.,
                    score_ci=[0., .2], verdict="worse", wall_seconds=1.)

    def test_historical_only_keeps_advancing_models_despite_regression(self):
        from junqi.training.arena import sha256_file
        selection = self.historical_only_selection()
        with patch("junqi.training.model_selection.run_match", side_effect=lambda *a, **k: self.losing_result()) as run:
            first = selection.evaluate(self.policy, self.layout, update=40,
                                       cumulative={"environment_plies": 400}, historical=self.league)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(first["evaluation_type"], "champion")
        champion_hash = sha256_file(selection.best_path)
        self.assertFalse(selection.historical_only({"environment_plies": 499}))
        self.assertTrue(selection.historical_only({"environment_plies": 500}))
        saved_snapshots = []
        for update, progress in ((50, 500), (55, 550)):
            self.league.progress = progress
            with torch.no_grad():
                self.policy.action_encoder.projection.weight.add_(.01)
            with patch("junqi.training.model_selection.run_match", side_effect=lambda *a, **k: self.losing_result()) as run:
                report = selection.evaluate(self.policy, self.layout, update=update,
                    cumulative={"environment_plies": progress}, historical=self.league)
            self.assertEqual(run.call_count, 3)
            self.assertTrue(all(c.args[1].parent == self.league.directory for c in run.call_args_list))
            self.assertTrue(all(c.kwargs["alpha"] == selection._round_alpha() / 3 for c in run.call_args_list))
            self.assertIs(run.call_args_list[0].kwargs["candidate_engine"], run.call_args_list[-1].kwargs["candidate_engine"])
            self.assertEqual(report["decision"], "use_latest")
            self.assertFalse(report["champion_evaluated"])
            self.assertTrue(report["historical_panel"]["confirmed_regression"])
            self.assertTrue(report["historical_panel"]["observational_only"])
            self.assertNotIn("promotion_allowed", report["historical_panel"])
            for key in ("promoted", "score_ci", "opponent_sha256", "score_delta_vs_best"):
                self.assertNotIn(key, report)
            self.assertEqual(report["games"], 12)
            self.assertEqual(sha256_file(selection.best_path), champion_hash)
            self.assertEqual(selection.state["last_evaluated_update"], update)
            snapshot = selection.directory / selection.state["latest_evaluated_snapshot"]
            saved_snapshots.append((snapshot, sha256_file(snapshot)))
            payload = torch.load(snapshot, weights_only=False)
            self.assertEqual(payload["update"], update)
            self.assertNotIn("policy_optimizer", payload)
            for name, tensor in self.policy.state_dict().items():
                torch.testing.assert_close(tensor, payload["policy"][name], rtol=0, atol=0)
            published = json.loads((self.league.directory / "latest_evaluation.json").read_text())
            self.assertEqual(published["candidate_update"], update)
            self.assertTrue(published["observational_only"])
        self.assertNotEqual(saved_snapshots[0][1], saved_snapshots[1][1])
        self.assertEqual(sha256_file(saved_snapshots[0][0]), saved_snapshots[0][1])
        self.assertEqual(len(list(selection.directory.glob("champion_match_*.json"))), 1)

    def test_historical_only_retries_partial_panel_without_champion_or_duplicate_rounds(self):
        selection = self.historical_only_selection()
        self.league.progress = 500
        args = dict(update=50, cumulative={"environment_plies": 500}, historical=self.league)
        with patch("junqi.training.model_selection.run_match", side_effect=[self.losing_result(), InterruptedError()]):
            with self.assertRaises(InterruptedError):
                selection.evaluate(self.policy, self.layout, **args)
        self.assertFalse((self.league.directory / "latest_evaluation.json").exists())
        self.assertEqual(selection.state["rounds"], [])
        with patch("junqi.training.model_selection.run_match", side_effect=lambda *a, **k: self.losing_result()) as run:
            report = selection.evaluate(self.policy, self.layout, **args)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(report["decision"], "use_latest")
        resumed = ModelSelection(self.settings, self.root, self.context)
        resumed.initialize(self.policy, self.layout, update=50, cumulative=args["cumulative"])
        with patch("junqi.training.model_selection.run_match") as run:
            self.assertIsNone(resumed.evaluate(self.policy, self.layout, **args))
        run.assert_not_called()
        self.assertEqual(len(resumed.state["rounds"]), 1)

    def test_total_500_games_are_complete_rotations_and_retry_keeps_the_allocation(self):
        self.settings = replace(self.settings, arena_games=500, historical_eval_total_games=500)
        selection = self.historical_only_selection()
        self.assertEqual(selection._historical_game_counts([{}] * 6), [84, 84, 84, 84, 84, 80])
        self.league.progress = 500
        args = dict(update=50, cumulative={"environment_plies": 500}, historical=self.league)
        def result(*args, **kwargs):
            count = args[2].pairs * 4
            return dict(games=count, wins=count, draws=0, losses=0, score=1.,
                        score_ci=[0., 1.], verdict="inconclusive", wall_seconds=1.)
        attempted = []
        def interrupt(*args, **kwargs):
            attempted.append(args[2].pairs * 4)
            if len(attempted) == 2:
                raise InterruptedError()
            return result(*args, **kwargs)
        with patch("junqi.training.model_selection.run_match", side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                selection.evaluate(self.policy, self.layout, **args)
        self.assertEqual(attempted, [168, 168])
        with patch("junqi.training.model_selection.run_match", side_effect=result) as run:
            report = selection.evaluate(self.policy, self.layout, **args)
        self.assertEqual([call.args[2].pairs * 4 for call in run.call_args_list], [168, 164])
        self.assertEqual(report["games"], 500)
        self.assertEqual(report["historical_panel"]["contract"]["games_per_opponent"], [168, 168, 164])
        self.assertEqual(report["historical_panel"]["contract"]["total_games"], 500)
        self.assertEqual(sum(row["games"] for row in report["historical_panel"]["results"]), 500)
        self.assertFalse(report["champion_evaluated"])
        with patch("junqi.training.model_selection.run_match") as run:
            self.assertIsNone(selection.evaluate(self.policy, self.layout, **args))
        run.assert_not_called()

    def test_total_panel_budget_runs_real_games_and_restores_training_rng(self):
        self.settings = replace(self.settings, historical_eval_total_games=20)
        selection = self.historical_only_selection()
        self.league.progress = 500
        rng = torch.get_rng_state().clone()
        weights = {k: v.clone() for k, v in self.policy.state_dict().items()}
        report = selection.evaluate(self.policy, self.layout, update=50,
                                    cumulative={"environment_plies": 500}, historical=self.league)
        self.assertEqual(report["games"], 20)
        self.assertEqual([row["games"] for row in report["historical_panel"]["results"]], [8, 8, 4])
        self.assertEqual(report["decision"], "use_latest")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in self.policy.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)

    def test_historical_only_recovers_report_after_state_commit_failure(self):
        from junqi.training.arena import atomic_json
        selection = self.historical_only_selection()
        self.league.progress = 500
        args = dict(update=50, cumulative={"environment_plies": 500}, historical=self.league)
        def fail_state(path, data):
            if path == selection.state_path:
                raise OSError("state commit unavailable")
            return atomic_json(path, data)
        with patch("junqi.training.model_selection.atomic_json", side_effect=fail_state), \
             patch("junqi.training.model_selection.run_match", side_effect=lambda *a, **k: self.losing_result()):
            with self.assertRaisesRegex(RuntimeError, "state commit unavailable"):
                selection.evaluate(self.policy, self.layout, **args)
        self.assertFalse((self.league.directory / "latest_evaluation.json").exists())
        resumed = ModelSelection(self.settings, self.root, self.context)
        resumed.initialize(self.policy, self.layout, update=50, cumulative=args["cumulative"])
        with patch("junqi.training.model_selection.run_match") as run:
            report = resumed.evaluate(self.policy, self.layout, **args)
        run.assert_not_called()
        self.assertEqual(report["decision"], "use_latest")
        self.assertEqual(len(resumed.state["rounds"]), 1)

    def test_historical_only_migration_preserves_results_champion_and_confidence_budget(self):
        selection = ModelSelection(self.settings, self.root, self.context)
        selection.initialize(self.policy, self.layout, update=0, cumulative={"environment_plies": 0})
        with patch("junqi.training.model_selection.run_match", return_value=self.losing_result()):
            selection.evaluate(self.policy, self.layout, update=40, cumulative={"environment_plies": 400})
        old = selection.state_dict()
        updated = replace(self.settings, arena_after_half_historical_only=True)
        resumed = ModelSelection(updated, self.root, self.context)
        resumed.initialize(self.policy, self.layout, update=40, cumulative={"environment_plies": 400})
        for key in ("best_snapshot", "best_sha256", "best_update", "rounds", "last_evaluated_update"):
            self.assertEqual(resumed.state[key], old[key])
        self.assertEqual(resumed._round_alpha(), selection._round_alpha())
        self.assertIn("historical_only_migration", resumed.state)
        self.assertTrue(resumed.state["contract"]["after_half_historical_only"])
        with self.assertRaisesRegex(RuntimeError, "schedule/budget changed"):
            ModelSelection(self.settings, self.root, self.context).initialize(
                self.policy, self.layout, update=40, cumulative={"environment_plies": 400})

    def test_historical_only_missing_panel_cannot_fall_back_to_champion(self):
        selection = self.historical_only_selection()
        with patch("junqi.training.model_selection.run_match") as run:
            with self.assertRaisesRegex(RuntimeError, "nonempty active frozen panel"):
                selection.evaluate(self.policy, self.layout, update=50, cumulative={"environment_plies": 500})
        run.assert_not_called()
        self.assertEqual(selection.state["rounds"], [])

    def test_historical_only_real_matches_restore_training_parameters_rng_and_flags(self):
        selection = self.historical_only_selection()
        self.league.progress = 500
        self.policy.train()
        self.layout.eval()
        weights = {k: v.clone() for k, v in self.policy.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report = selection.evaluate(self.policy, self.layout, update=50,
                                    cumulative={"environment_plies": 500}, historical=self.league)
        self.assertEqual(report["games"], 12)
        self.assertEqual(report["decision"], "use_latest")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(self.policy.training)
        self.assertFalse(self.layout.training)
        self.assertTrue(all(p.requires_grad for p in self.policy.parameters()))
        for key, value in self.policy.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)

    def test_historical_only_training_loop_saves_then_resumes_current_model(self):
        from junqi.training.trainer import SelfPlayTrainer
        config = replace(self.settings, total_updates=6, target_environment_plies=144, anchor_batch=24,
                         arena_interval_environment_plies=48, arena_after_half_interval_environment_plies=24,
                         arena_after_half_historical_only=True, checkpoint_interval_environment_plies=24,
                         historical_teammate_fraction=.2, historical_cohort_games=1024, max_game_plies=8)
        trainer = SelfPlayTrainer(config, run_directory=self.root / "trainer")
        with patch("junqi.training.model_selection.run_match", side_effect=lambda *a, **k: self.losing_result()) as run:
            trainer.train()
        self.assertEqual(run.call_count, 1 + 4 * 3)
        state = trainer.model_selection.state
        reports = [json.loads((trainer.model_selection.directory / name).read_text()) for name in state["rounds"]]
        self.assertEqual([r["environment_plies"] for r in reports], [48, 72, 96, 120, 144])
        self.assertEqual([r["decision"] for r in reports[1:]], ["use_latest"] * 4)
        saved = torch.load(trainer.checkpoints.latest_path, weights_only=False)
        self.assertEqual(saved["update"], 6)
        self.assertEqual(saved["trainer_state"]["cumulative"]["environment_plies"], 144)
        resumed = SelfPlayTrainer(config, run_directory=self.root / "trainer")
        self.addCleanup(resumed.logger.close)
        self.assertEqual(resumed.update, 6)
        self.assertEqual(resumed.model_selection.state["last_completed_environment_plies"], 144)
        for key, value in resumed.policy.state_dict().items():
            torch.testing.assert_close(value, saved["policy"][key], rtol=0, atol=0)
        with patch("junqi.training.model_selection.run_match") as run:
            resumed.train()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
