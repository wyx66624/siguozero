from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from junqi.board import PASS_ACTION
from junqi.training.encoding import GameHistory
from junqi.training.inference import InferenceEngine
from junqi.training.models import GamePolicyTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import new_game


class InferenceArgmaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, mode="four_dark"):
        torch.manual_seed(314)
        config = ModelConfig.tiny()
        engine = InferenceEngine(mode, GamePolicyTransformer(config),
                                 PieceConditionedLayoutPointerDecoder(config))
        game = new_game(mode, seed=400)
        return engine, game, GameHistory.initialize(game, mode, max_transitions=16)

    def test_argmax_compares_joint_moves_and_masks_illegal_high_logits(self):
        engine, game, history = self.fixture()
        state = history.state_for(game)
        groups = {}
        for action in state.legal_actions:
            groups.setdefault(action[0], []).append(action)
        source = next(source for source, actions in groups.items()
                      if source != PASS_ACTION[0] and len(actions) >= 2)
        actions = (*groups[source][:2], PASS_ACTION)
        state = replace(state, legal_actions=actions)
        history = SimpleNamespace(state_for=lambda game: state)
        with torch.no_grad():
            for query in (engine.policy.source_query, engine.policy.destination_query):
                for parameter in query.parameters():
                    parameter.zero_()
                query[-1].bias.fill_(100)
            engine.policy.source_query[-1].bias[source] = math.log(0.6)
            engine.policy.source_query[-1].bias[PASS_ACTION[0]] = math.log(0.4)
            for _, target in actions:
                engine.policy.destination_query[-1].bias[target] = 0
        with torch.inference_mode():
            _, probabilities = engine.policy.distributions([state])
        torch.testing.assert_close(probabilities[0], torch.tensor([0.3, 0.3, 0.4]))
        # The most likely source has two moves of probability .3 each;
        # the complete move with probability .4 must win, including legal passes.
        with patch("torch.multinomial", side_effect=AssertionError("argmax sampled")):
            for seed in (1, 2, 100):
                torch.manual_seed(seed)
                rng = torch.get_rng_state().clone()
                self.assertEqual(engine.select_action(game, history, temperature=0), PASS_ACTION)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_tied_maxima_and_single_legal_move_are_deterministic(self):
        engine, game, history = self.fixture()
        original = history.state_for(game)
        for parameter in engine.policy.source_query.parameters():
            parameter.zero_()
        for parameter in engine.policy.destination_query.parameters():
            parameter.zero_()
        moves = [action for action in original.legal_actions if action[0] != PASS_ACTION[0]]
        for actions in ((moves[0], PASS_ACTION), (PASS_ACTION, moves[0]), (PASS_ACTION,)):
            state = replace(original, legal_actions=actions)
            with self.subTest(actions=actions):
                self.assertEqual(engine.select_action(
                    game, SimpleNamespace(state_for=lambda game: state), temperature=0
                ), actions[0])

    def test_real_rule_steps_choose_maximum_in_every_mode_without_sampling(self):
        for mode in ("two_player", "four_dark", "double_open"):
            with self.subTest(mode=mode):
                engine, game, history = self.fixture(mode)
                for _ in range(8):
                    state = history.state_for(game)
                    with torch.inference_mode():
                        _, probabilities = engine.policy.distributions([state])
                    expected = state.legal_actions[int(probabilities[0].argmax())]
                    rng = torch.get_rng_state().clone()
                    with patch("torch.multinomial", side_effect=AssertionError("argmax sampled")):
                        result = engine.step(game, history, temperature=0)
                    self.assertEqual(result.action, expected)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertEqual(len(history.state_for(game).records), game.ply_count + 1)

    def test_positive_temperatures_continue_sampling(self):
        engine, game, history = self.fixture()
        for temperature in (0.3, 0.7, 1.0):
            with self.subTest(temperature=temperature):
                rng = torch.get_rng_state().clone()
                action = engine.select_action(game, history, temperature=temperature)
                self.assertIn(action, game.legal_actions())
                self.assertFalse(torch.equal(rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
