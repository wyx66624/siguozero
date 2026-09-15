from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_historical_opponents import settings
from junqi.training.distributed import DistributedContext
from junqi.training.historical_cache import pack_bytes
from junqi.training.historical_opponents import HistoricalOpponents
from junqi.training.model_selection import ModelSelection
from junqi.training.models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor


class GrowingLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = settings(historical_checkpoint_start_fraction=.2,
            historical_stage_mix_fraction=.5, historical_cache_gib=.05,
            historical_cache_reserve_gib=0, historical_eval_total_games=500)
        self.context = DistributedContext(0, 1, 0, torch.device("cpu"))
        self.policy = GamePolicyTransformer(self.settings.model).eval()
        self.layout = PieceConditionedLayoutPointerDecoder(self.settings.model).eval()
        self.league = self.make_league()
        self.actor = FrozenPolicyActor(self.policy, max_batch_size=8)
        self.pool = BaseGamePool("four_dark", pool_size=5, max_transitions=8, max_game_plies=8,
                                 seed=77, layout_prefetch_games=4)
        self.pool.historical = self.league

    def make_league(self, config=None):
        league = HistoricalOpponents(config or self.settings, self.root, self.context)
        self.addCleanup(league.close)
        return league

    def progress(self, step):
        self.league.update_progress(self.policy, self.layout, environment_plies=step, update=step)

    def archive(self, step):
        return self.league.archive_checkpoint(self.policy, self.layout, environment_plies=step, update=step)

    def early(self):
        for step in (0, 100, 200):
            self.progress(step)

    def test_all_complete_checkpoints_after_gate_and_duplicate_saves(self):
        self.progress(0)
        self.assertIsNone(self.archive(199))
        weights = self.archive(200)
        self.assertEqual(set(weights), {"policy", "layout"})
        for name in weights["policy"]:
            torch.testing.assert_close(weights["policy"][name], self.policy.state_dict()[name], rtol=0, atol=0)
        with patch("junqi.training.historical_opponents.packed_weights", side_effect=AssertionError("duplicate D2H")):
            self.assertIs(self.archive(200), weights)
        for step in (250, 501, 700, 1000):
            self.archive(step)
        self.assertEqual([e["environment_plies"] for e in self.league.entries], [0, 200, 250, 501, 700, 1000])
        self.assertEqual(len(self.league.evaluation_panel()), 1)
        payload = torch.load(self.league.directory / self.league.entries[-1]["file"], mmap=True, weights_only=False)
        self.assertNotIn("critic", payload)
        self.assertNotIn("policy_optimizer", payload)

    def test_milestone_and_complete_save_share_one_immutable_version(self):
        self.early()
        with patch("junqi.training.historical_opponents.packed_weights", side_effect=AssertionError("duplicate D2H")):
            self.assertEqual(set(self.archive(200)), {"policy", "layout"})
        self.assertEqual(len(self.league.panel()), 3)
        self.assertEqual(len(self.league.evaluation_panel()), 3)

    def test_hundreds_of_training_versions_do_not_expand_500_game_evaluation(self):
        self.early()
        for i in range(240):
            self.league.entries.append(dict(self.league.entries[-1], id=f"later_{i}",
                                           kind="checkpoint", milestones=[], environment_plies=201+i))
        self.progress(500)
        panel = self.league.evaluation_panel()
        selector = ModelSelection(self.settings, self.root, self.context)
        counts = selector._historical_game_counts(panel)
        self.assertEqual(sum(counts), 500)
        self.assertTrue(all(n % 4 == 0 for n in counts))
        self.assertEqual(len(counts), 3)
        self.assertEqual(len(self.league.panel()), 243)
        self.league._publish_catalog()
        restored = self.make_league()
        self.assertEqual(len(restored.entries), 243)

    def test_safe_legacy_migration_and_future_snapshot_rewind(self):
        old_settings = replace(self.settings, historical_checkpoint_start_fraction=None,
                               historical_stage_mix_fraction=0)
        old = self.make_league(old_settings)
        old.update_progress(self.policy, self.layout, environment_plies=0, update=0)
        saved = old.state_dict()
        self.league = self.make_league()
        self.league.load_state_dict(saved)
        self.assertEqual(self.league.rng.getstate(), old.rng.getstate())
        self.progress(100)
        current = self.league.state_dict()
        self.archive(300)
        future = self.league.entries[-1]["id"]
        restored = self.make_league()
        restored.load_state_dict(current)
        self.assertNotIn(future, [e["id"] for e in restored.entries])
        saved["progress"] = 201
        with self.assertRaisesRegex(ValueError, "resume contract"):
            restored.load_state_dict(saved)

    def test_stage_coverage_survives_many_nearly_adjacent_checkpoints(self):
        self.early()
        self.league.progress = 900
        for i in range(100):
            self.league.entries.append(dict(self.league.entries[-1], id=f"recent_{i}",
                                           kind="checkpoint", environment_plies=801+i%90))
        self.league.q = {e["id"]: -100 for e in self.league.entries[:3]}
        probabilities = self.league.probabilities()
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertGreaterEqual(probabilities[self.league.entries[0]["id"]], .5/4)
        self.assertTrue(all(p > 0 for p in probabilities.values()))

    def test_quality_correction_uses_selection_time_pool_size(self):
        self.early()
        self.progress(500)
        self.league.begin_rollout(self.pool, self.actor, 500)
        identifier = self.league.active_id
        denominator = self.league.active_panel_size * self.league.active_probability
        self.archive(510)
        self.league.record_result(type("Slot", (), dict(opponent_id=identifier, teammate_id=None))(), 1.)
        self.assertAlmostEqual(self.league.q[identifier], -self.settings.historical_learning_rate/denominator)

    def test_prefetched_choice_and_rng_survive_full_scheduler_resume(self):
        self.early()
        self.progress(500)
        self.league.begin_rollout(self.pool, self.actor, 500)
        for _ in range(20):
            self.league.assignment()
        self.progress(510)
        pending = copy.deepcopy(self.league.pending_cohort)
        self.assertIsNotNone(pending)
        saved = self.league.state_dict()
        restored = self.make_league()
        restored.load_state_dict(saved)
        restored.update_progress(self.policy, self.layout, environment_plies=510, update=510)
        self.league.begin_rollout(self.pool, self.actor, 510)
        restored.begin_rollout(self.pool, self.actor, 510)
        self.assertEqual(restored.active_id, pending["id"])
        self.assertEqual(restored.active_probability, pending["probability"])
        self.assertEqual(restored.rng.getstate(), self.league.rng.getstate())

    def test_ram_hit_does_not_reload_or_reconvert_and_lru_respects_pressure(self):
        self.early()
        cache = self.league.weight_cache
        entry = self.league.entries[0]
        with patch.object(cache, "_prepare", side_effect=AssertionError("reconversion")):
            first = cache.get(entry)
            second = cache.get(entry)
        self.assertIs(first, second)
        cache.finish_upload()
        self.assertEqual(cache.reads, 0)  # populated directly from archive CPU slabs
        cache.maximum = pack_bytes(first)
        cache.trim()
        self.assertLessEqual(cache.bytes, cache.maximum)
        self.assertEqual(len(cache.items), 1)
        cache.reserve = 1024
        with patch("junqi.training.historical_cache.available_memory_bytes", return_value=0):
            cache.trim()
        self.assertEqual(len(cache.items), 0)

    def test_cache_capacity_does_not_change_scheduling_or_rng(self):
        self.early()
        self.progress(500)
        other = self.make_league(replace(self.settings, historical_cache_gib=0))
        other.load_state_dict(self.league.state_dict())
        for step in range(501, 508):
            for league in (self.league, other):
                league.begin_rollout(self.pool, self.actor, step)
            self.assertEqual(self.league.active_id, other.active_id)
            self.assertEqual([self.league.assignment() for _ in range(30)],
                             [other.assignment() for _ in range(30)])
            for league in (self.league, other):
                league.update_progress(self.policy, self.layout, environment_plies=step, update=step)
            self.assertEqual(self.league.rng.getstate(), other.rng.getstate())
            self.assertEqual(self.league.pending_cohort, other.pending_cohort)

    def test_cold_load_verifies_content_before_caching(self):
        self.early()
        cache = self.league.weight_cache
        cache.items.clear()
        cache.bytes = 0
        entry = self.league.entries[0]
        path = self.league.directory / entry["file"]
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            cache.get(entry)

    def test_cached_installs_reuse_storage_and_match_original_weights(self):
        self.early()
        self.progress(500)
        self.league.begin_rollout(self.pool, self.actor, 500)
        pointers = {k: p.data_ptr() for k, p in self.league.device_packs.items()}
        for entry in self.league.entries:
            self.league.active_id = entry["id"]
            self.league._load_active(self.actor)
            self.assertEqual(pointers, {k: p.data_ptr() for k, p in self.league.device_packs.items()})
            payload = torch.load(self.league.directory / entry["file"], mmap=True, weights_only=False)
            for key, value in self.league.policy.state_dict().items():
                torch.testing.assert_close(value, payload["policy"][key], rtol=0, atol=0)

    def test_real_mixed_update_saves_library_and_exact_optimizer_resume(self):
        from junqi.training.trainer import SelfPlayTrainer
        config = replace(self.settings, total_updates=1, max_game_plies=8, anchor_batch=200,
                         base_game_pool_size=25, policy_microbatch=8, historical_teammate_fraction=.2,
                         historical_cohort_games=1000)
        trainer = SelfPlayTrainer(config, run_directory=self.root / "trainer")
        for step in (100, 200, 500):
            trainer.historical.update_progress(trainer.policy, trainer.layout, environment_plies=step, update=0)
        trainer.cumulative["environment_plies"] = 500
        with patch.object(trainer, "_maybe_evaluate_model", return_value=None):
            trainer.train()
        self.assertEqual(trainer.cumulative["environment_plies"], 700)
        self.assertEqual(trainer.historical.entries[-1]["kind"], "checkpoint")
        saved = trainer.pool.state_dict()
        resumed = SelfPlayTrainer(config, run_directory=self.root / "trainer")
        self.addCleanup(resumed.logger.close)
        self.addCleanup(resumed.historical.close)
        self.assertEqual(resumed.pool.state_dict(), saved)
        for key, value in trainer.policy.state_dict().items():
            torch.testing.assert_close(value, resumed.policy.state_dict()[key], rtol=0, atol=0)
        original = trainer.policy_optimizer.state_dict()["state"]
        actual = resumed.policy_optimizer.state_dict()["state"]
        for key in original:
            for name in original[key]:
                torch.testing.assert_close(original[key][name], actual[key][name], rtol=0, atol=0)

    def test_invalid_resource_limits_are_rejected(self):
        for kwargs in (dict(historical_cache_gib=-1), dict(historical_cache_gib=float("nan")),
                       dict(historical_pinned_mib=2048), dict(historical_checkpoint_start_fraction=1),
                       dict(historical_stage_mix_fraction=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                settings(**kwargs)


if __name__ == "__main__":
    unittest.main()
