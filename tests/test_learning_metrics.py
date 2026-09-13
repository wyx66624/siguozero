from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from junqi.training.learning_metrics import ppo_signal_metrics, ppo_signal_sums


def samples(values, targets):
    return [SimpleNamespace(old_value=value, value_target=target, advantage=target - value,
                            old_draw_value=0., draw_value_target=0.)
            for value, target in zip(values, targets, strict=True)]


class LearningMetricsTests(unittest.TestCase):
    def metrics(self, values, targets):
        return ppo_signal_metrics(ppo_signal_sums(samples(values, targets)))

    def test_zero_targets_are_no_signal_and_not_perfect_predictions(self):
        metrics = self.metrics([0, 0], [0, 0])
        self.assertEqual(metrics["ppo/no_advantage_signal"], 1)
        self.assertEqual(metrics["critic/rollout_value_mse"], 0)
        self.assertEqual(metrics["critic/rollout_explained_variance_defined"], 0)
        self.assertNotIn("critic/rollout_explained_variance", metrics)
        json.dumps(metrics, allow_nan=False)

    def test_zero_mean_does_not_hide_nonzero_learning_signal(self):
        metrics = self.metrics([0, 0], [-1, 1])
        self.assertEqual(metrics["ppo/raw_advantage_mean"], 0)
        self.assertEqual(metrics["ppo/raw_advantage_std"], 1)
        self.assertEqual(metrics["ppo/raw_nonzero_advantage_fraction"], 1)
        self.assertEqual(metrics["ppo/no_advantage_signal"], 0)
        self.assertEqual(metrics["critic/rollout_nonzero_target_fraction"], 1)
        self.assertEqual(metrics["critic/rollout_explained_variance"], 0)

    def test_perfect_and_reversed_predictions(self):
        perfect = self.metrics([-1, 1], [-1, 1])
        reversed_ = self.metrics([1, -1], [-1, 1])
        self.assertEqual(perfect["critic/rollout_explained_variance"], 1)
        self.assertEqual(perfect["critic/rollout_value_mse"], 0)
        self.assertEqual(reversed_["critic/rollout_explained_variance"], -3)

    def test_constant_bias_requires_mse_even_when_ev_is_one(self):
        metrics = self.metrics([1, 3], [-1, 1])
        self.assertEqual(metrics["critic/rollout_explained_variance"], 1)
        self.assertEqual(metrics["critic/rollout_value_mse"], 4)

    def test_constant_nonzero_advantage_is_still_a_signal(self):
        metrics = self.metrics([0, 0], [1, 1])
        self.assertEqual(metrics["ppo/raw_advantage_std"], 0)
        self.assertEqual(metrics["ppo/no_advantage_signal"], 0)
        self.assertEqual(metrics["critic/rollout_value_mse"], 1)
        self.assertEqual(metrics["critic/rollout_explained_variance_defined"], 0)

    def test_global_moments_handle_unequal_and_locally_constant_batches(self):
        first = samples([0, 0, 0], [1, 1, 1])
        second = samples([0], [-1])
        parts = [ppo_signal_sums(batch) for batch in (first, second, [])]
        merged = {key: sum(part[key] for part in parts) for key in parts[0]}
        metrics = ppo_signal_metrics(merged)
        self.assertEqual(metrics, ppo_signal_metrics(ppo_signal_sums(first + second)))
        self.assertEqual(metrics["critic/rollout_explained_variance_defined"], 1)
        self.assertEqual(metrics["critic/rollout_explained_variance"], 0)
        self.assertEqual(metrics["critic/rollout_target_mean"], .5)

    def test_empty_global_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "global sample"):
            ppo_signal_metrics(ppo_signal_sums([]))


if __name__ == "__main__":
    unittest.main()
