from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class SamplingUniformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def policy_and_states(self, mode="two_player", count=4):
        from junqi.training.encoding import GameHistory
        from junqi.training.models import GamePolicyTransformer, ModelConfig
        from junqi.training.modes import new_game

        torch.manual_seed(314)
        policy = GamePolicyTransformer(ModelConfig.tiny()).eval()
        states = []
        for seed in range(count):
            game = new_game(mode, seed=seed + 400)
            history = GameHistory.initialize(game, mode, max_transitions=16)
            states.append(history.state_for(game))
        return policy, states

    def test_batched_and_individual_sampling_preserve_actions_and_logs(self):
        from junqi.training.rollout import FrozenPolicyActor

        for mode in ("two_player", "four_dark", "double_open"):
            with self.subTest(mode=mode):
                policy, states = self.policy_and_states(mode)
                actor = FrozenPolicyActor(policy, max_batch_size=4)
                uniforms = torch.rand((len(states), 7, 2), generator=torch.Generator().manual_seed(71))
                original_rng = torch.random.get_rng_state().clone()
                with patch("torch.multinomial", side_effect=AssertionError("global RNG used")):
                    actions, logs = actor.sample(states, count=7, sampling_uniforms=uniforms)
                    actor.max_batch_size = 1
                    individual_actions, individual_logs = actor.sample(
                        states, count=7, sampling_uniforms=uniforms
                    )
                    permutation = [3, 0, 2, 1]
                    reordered_actions, reordered_logs = actor.sample(
                        [states[index] for index in permutation],
                        count=7,
                        sampling_uniforms=uniforms[permutation],
                    )
                self.assertEqual(actions, individual_actions)
                for index, old_index in enumerate(permutation):
                    self.assertEqual(reordered_actions[index], actions[old_index])
                    torch.testing.assert_close(reordered_logs[index], logs[old_index])
                for state, state_actions, log, individual_log in zip(
                    states, actions, logs, individual_logs, strict=True
                ):
                    self.assertTrue(all(action in state.legal_actions for action in state_actions))
                    torch.testing.assert_close(log, individual_log)
                torch.testing.assert_close(torch.random.get_rng_state(), original_rng, rtol=0, atol=0)
                with torch.inference_mode():
                    expected_logs = policy.log_probs_for_action_groups(states, actions)
                for actual, expected in zip(logs, expected_logs, strict=True):
                    torch.testing.assert_close(actual, expected)

    def test_uniform_cdf_skips_masked_and_zero_probability_endpoints(self):
        from junqi.training.models import _categorical_from_uniforms

        probabilities = torch.tensor([[0, 2, 0, 8, 0], [0, 0, 3, 0, 0]], dtype=torch.float32)
        uniforms = torch.tensor(
            [[0, 0.19, 0.2, 0.999, 1 - 2**-53], [0, 0.19, 0.2, 0.999, 1 - 2**-53]],
            dtype=torch.float64,
        )
        sampled = _categorical_from_uniforms(probabilities, uniforms)
        self.assertEqual(sampled.tolist(), [[1, 1, 3, 3, 3], [2, 2, 2, 2, 2]])
        self.assertTrue((probabilities.gather(-1, sampled) > 0).all())

    def test_uniform_cdf_uses_float32_accumulation_for_low_precision_weights(self):
        from junqi.training.models import _categorical_from_uniforms

        probabilities = torch.tensor([[0, 1, 3, 0], [0, 3, 1, 0]], dtype=torch.bfloat16)
        uniforms = torch.tensor([[0, 0.249, 0.25, 0.99999999], [0, 0.749, 0.75, 0.99999999]], dtype=torch.float64)
        self.assertEqual(
            _categorical_from_uniforms(probabilities, uniforms).tolist(),
            [[1, 1, 2, 2], [1, 1, 2, 2]],
        )

    def test_conditional_destination_masks_hold_at_both_uniform_extremes(self):
        policy, states = self.policy_and_states(count=1)
        # A single legal source and destination exercises leading/trailing
        # masks even when the supplied float64 uniform rounds up to float32 1.
        action = states[0].legal_actions[len(states[0].legal_actions) // 2]
        state = replace(states[0], legal_actions=(action,))
        uniforms = torch.tensor([[[0, 0], [1 - 2**-53, 1 - 2**-53]]], dtype=torch.float64)
        actions, logs = policy.sample_action_groups([state], count=2, sampling_uniforms=uniforms)
        self.assertEqual(actions, [[action, action]])
        torch.testing.assert_close(logs[0], torch.zeros(2), rtol=0, atol=0)
        self.assertEqual(
            policy.sample_action_groups([state], count=2, sampling_uniforms=uniforms, return_log_probs=False),
            ([[action, action]], []),
        )

    def test_invalid_uniform_shapes_and_values_rejected_before_encoding(self):
        from junqi.training.rollout import FrozenPolicyActor

        policy, states = self.policy_and_states(count=2)
        actor = FrozenPolicyActor(policy, max_batch_size=1)
        invalid = [
            torch.zeros(2, 2), torch.zeros(1, 1, 2), torch.zeros(2, 2, 2),
            torch.zeros(2, 1, 3), torch.zeros(2, 1, 2, dtype=torch.int64),
            [[[0.1, 0.2]], [[0.2, 0.3]]],
        ]
        for value in (float("nan"), float("inf"), float("-inf"), -0.0001, 1.0, 1.0001):
            uniforms = torch.zeros(2, 1, 2, dtype=torch.float64)
            uniforms[1, 0, 1] = value
            invalid.append(uniforms)
        with patch.object(policy, "encode", side_effect=AssertionError("invalid input reached model")):
            for uniforms in invalid:
                for sampler in (policy.sample_action_groups, actor.sample):
                    with self.subTest(uniforms=uniforms, sampler=sampler.__name__):
                        with self.assertRaisesRegex(ValueError, "sampling_uniforms"):
                            sampler(states, sampling_uniforms=uniforms)

    def test_default_path_keeps_multinomial_and_rng_behavior(self):
        from junqi.training.rollout import FrozenPolicyActor

        policy, states = self.policy_and_states(count=2)
        actor = FrozenPolicyActor(policy, max_batch_size=2)
        torch.manual_seed(132)
        with patch("torch.multinomial", wraps=torch.multinomial) as multinomial:
            actions, logs = actor.sample(states, count=3)
        self.assertEqual(multinomial.call_count, 2)
        torch.manual_seed(132)
        explicit_actions, explicit_logs = actor.sample(states, count=3, sampling_uniforms=None)
        self.assertEqual(actions, explicit_actions)
        for actual, expected in zip(logs, explicit_logs, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_oom_retry_reuses_draws_when_batch_shrinks(self):
        from junqi.training.rollout import FrozenPolicyActor

        observed = []

        def sample(states, *, count, temperature, return_log_probs, sampling_uniforms):
            observed.append((tuple(states), sampling_uniforms.clone()))
            if len(states) > 2:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return [[(state, sample) for sample in range(count)] for state in states], []

        policy = SimpleNamespace(
            device=torch.device("cuda"), start_inference_board_cache=Mock(),
            sample_action_groups=sample,
        )
        actor = FrozenPolicyActor(policy, max_batch_size=4)
        uniforms = torch.rand((5, 2, 2), generator=torch.Generator().manual_seed(25))
        original_uniforms = uniforms.clone()
        with patch("junqi.training.rollout.empty_cache") as empty_cache:
            actions, logs = actor.sample(list(range(5)), count=2, sampling_uniforms=uniforms, return_log_probs=False)
        self.assertEqual(actions, [[(state, sample) for sample in range(2)] for state in range(5)])
        self.assertEqual(logs, [])
        self.assertEqual(actor.max_batch_size, 2)
        self.assertEqual(actor.oom_reductions, 1)
        empty_cache.assert_called_once_with(policy.device)
        self.assertEqual([states for states, _ in observed], [(0, 1, 2, 3), (0, 1), (2, 3), (4,)])
        for states, supplied in observed:
            torch.testing.assert_close(supplied, uniforms[list(states)], rtol=0, atol=0)
        torch.testing.assert_close(uniforms, original_uniforms, rtol=0, atol=0)

    def test_empty_actor_batch_accepts_only_matching_uniform_shape(self):
        from junqi.training.rollout import FrozenPolicyActor

        policy, _ = self.policy_and_states(count=0)
        actor = FrozenPolicyActor(policy)
        self.assertEqual(actor.sample([], sampling_uniforms=torch.empty(0, 1, 2)), ([], []))
        with self.assertRaisesRegex(ValueError, "sampling_uniforms"):
            actor.sample([], sampling_uniforms=torch.empty(1, 1, 2))

    @unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), "CUDA is not available")
    def test_accelerator_uniform_validation_keeps_context_usable(self):
        from junqi.training.rollout import FrozenPolicyActor

        policy, states = self.policy_and_states(count=1)
        actor = FrozenPolicyActor(policy.cuda())
        uniforms = torch.zeros((1, 1, 2), device="cuda")
        for value in (float("nan"), float("inf"), -0.1, 1.0):
            uniforms.fill_(value)
            with self.assertRaisesRegex(ValueError, "sampling_uniforms"):
                actor.sample(states, sampling_uniforms=uniforms)
        uniforms.zero_()
        actions, _ = actor.sample(states, sampling_uniforms=uniforms)
        self.assertIn(actions[0][0], states[0].legal_actions)


if __name__ == "__main__":
    unittest.main()
