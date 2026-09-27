"""Numeric contracts for entropy trends reconstructed from existing logs."""
import math
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from junqi.web.exploration import (SourceEntropyStore, entropy_history,
                                  normalized_action_entropy, source_entropy_observation)


def record(update, *, opening=.2, other=.6, steps=100):
    # Ten percent opening, eighty percent later play, ten percent forced moves.
    return {"update": update, "rollout/environment_plies": steps,
            "policy/opening_entropy_ratio_mass": .1 * opening,
            "policy/opening_entropy_fraction": .1,
            "policy/other_entropy_ratio_mass": .8 * other,
            "policy/other_entropy_fraction": .8}


class ExplorationHistoryTests(unittest.TestCase):
    def test_phase_sample_weights_and_forced_moves(self):
        self.assertAlmostEqual(normalized_action_entropy(record(1)), 5 / 9)
        # An absent phase is valid when its moments explicitly record zero.
        row = record(1)
        row["policy/opening_entropy_ratio_mass"] = 0
        row["policy/opening_entropy_fraction"] = 0
        self.assertAlmostEqual(normalized_action_entropy(row), .6)

    def test_window_uses_environment_steps_not_epochs_and_expires_old_rows(self):
        rows = [record(1, opening=.9, other=.9),
                record(2, opening=.2, other=.2, steps=100),
                record(3, opening=.6, other=.6, steps=300)]
        rows[1]["optimizer/policy_epochs"] = 3
        result = entropy_history(rows, window_updates=2)
        self.assertAlmostEqual(result["moving_average"], .5)
        self.assertEqual(result["environment_plies"], 400)
        self.assertEqual((result["start_update"], result["end_update"]), (2, 3))
        self.assertEqual(result["covered_updates"], 2)

    def test_missing_nonfinite_and_forced_only_are_not_zero_observations(self):
        forced = record(2)
        for key in list(forced):
            if key.startswith("policy/"):
                forced[key] = 0
        invalid = record(3)
        invalid["policy/other_entropy_ratio_mass"] = math.nan
        result = entropy_history([record(1), forced, invalid], window_updates=2)
        self.assertIsNone(result["moving_average"])
        self.assertEqual(result["covered_updates"], 0)
        self.assertIsNone(normalized_action_entropy({}))
        zero = entropy_history([record(4, opening=0, other=0)])
        self.assertEqual(zero["moving_average"], 0)
        self.assertEqual(zero["covered_updates"], 1)


def source_probe(update, probabilities=(.25, .25, .5), actions=((1, 2), (1, 3), (4, 5))):
    return {"update": update, "learnable": True, "legal_actions": list(actions),
            "after": {"legal_probabilities": list(probabilities)}}


class SourceEntropyHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.store = SourceEntropyStore()

    def archive(self, update, rows, rank=0):
        path = self.run / 'tactical_diagnostics/details' / f'update_{update:09d}.rank{rank:03d}.jsonl.gz'
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        return path

    def test_marginal_entropy_uses_all_destinations_and_keeps_measured_zero(self):
        observation = source_entropy_observation(source_probe(1))
        self.assertAlmostEqual(observation['entropy'], math.log(2))
        self.assertAlmostEqual(observation['ratio'], 1.)
        concentrated = source_entropy_observation(source_probe(1, (.5, .5, 0)))
        self.assertEqual(concentrated['ratio'], 0.)
        forced = source_entropy_observation(source_probe(1, (.5, .5), ((1, 2), (1, 3))))
        self.assertEqual(forced['entropy'], 0.)
        self.assertIsNone(forced['ratio'])

    def test_invalid_incomplete_and_frozen_probabilities_are_not_observations(self):
        for row in [{}, {**source_probe(1), 'learnable': False},
                    source_probe(1, (math.nan, .25, .5)), source_probe(1, (.25, .25)),
                    source_probe(1, (.1, .1, .1)), source_probe(1, (.5, .5), ((1, 2), (1, 2)))]:
            self.assertIsNone(source_entropy_observation(row), row)

    def test_rank_counts_and_environment_weights_exclude_forced_and_expire_missing(self):
        # Three eligible uniform sources on rank 0, one deterministic on rank 1.
        self.archive(1, [source_probe(1) for _ in range(3)] +
                     [source_probe(1, (.5, .5), ((1, 2), (1, 3)))])
        self.archive(1, [source_probe(1, (.5, .5, 0))], rank=1)
        self.archive(2, [source_probe(2, (.5, .5, 0))])
        self.archive(2, [source_probe(2, (.5, .5, 0))], rank=1)
        rows = [record(1, steps=100), record(2, steps=300)]
        result = self.store.snapshot(self.run, rows, world_size=2, window_updates=2)
        self.assertAlmostEqual(result['moving_average'], .75 * 100 / 400)
        self.assertEqual(result['probe_samples'], 7)
        self.assertEqual(result['eligible_samples'], 6)
        self.assertEqual(result['latest_ratio'], 0.)
        self.assertEqual(result['measurement'], 'tactical_probe_after')
        # Missing rank 1 makes update 3 incomplete, never a zero-valued update.
        self.archive(3, [source_probe(3)])
        result = self.store.snapshot(self.run, [*rows, record(3)], world_size=2, window_updates=2)
        self.assertEqual(result['covered_updates'], 1)
        self.assertEqual(result['moving_average'], 0.)
        self.assertIsNone(result['latest_ratio'])
        self.assertEqual((result['start_update'], result['end_update']), (2, 3))

    def test_cache_reloads_replaced_files_and_rejects_old_resume_branch(self):
        path = self.archive(1, [source_probe(1)])
        result = self.store.snapshot(self.run, [record(1)])
        self.assertAlmostEqual(result['moving_average'], 1.)
        with patch('junqi.web.exploration.gzip.open', side_effect=AssertionError('uncached read')):
            self.assertEqual(self.store.snapshot(self.run, [record(1)]), result)
        self.archive(1, [source_probe(1, (.5, .5, 0)) for _ in range(2)])
        self.assertEqual(self.store.snapshot(self.run, [record(1)])['moving_average'], 0.)
        os.utime(path, (10, 10))
        resumed = self.store.snapshot(self.run, [record(1)], session_started=100, resumed_update=0)
        self.assertIsNone(resumed['moving_average'])
        self.assertEqual(resumed['covered_updates'], 0)

    def test_corrupt_archive_is_unknown_and_probe_update_must_match(self):
        path = self.archive(1, [source_probe(99)])
        self.assertIsNone(self.store.snapshot(self.run, [record(1)])['moving_average'])
        path.write_bytes(b'not a gzip archive')
        self.assertIsNone(self.store.snapshot(self.run, [record(1)])['moving_average'])


if __name__ == "__main__":
    unittest.main()
