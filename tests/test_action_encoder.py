from __future__ import annotations

import copy
from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import yaml
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch/PyYAML training extras are not installed") from error

from junqi.board import FourPlayerBoard, TwoPlayerBoard
from junqi.game import CombatOutcome, ObservedEvent
from junqi.training.checkpoint import CHECKPOINT_FORMAT_VERSION, require_current_checkpoint
from junqi.training.encoding import (
    ACTION_ENCODER_TYPE, ActionFeatures, GameHistory, PlayerHistory,
    action_point_coordinates,
)
from junqi.training.models import (
    GamePolicyTransformer, GameValueTransformer, ModelConfig, PublicActionEncoder,
    collate_policy_states,
)
from junqi.training.modes import TrainingMode, mode_spec, new_game
from junqi.training.settings import TrainingSettings


class ActionEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_coordinates_are_unique_and_follow_existing_board_view_rotations(self):
        for mode in TrainingMode:
            spec = mode_spec(mode)
            coords = [action_point_coordinates(code, mode) for code in range(spec.point_count)]
            self.assertEqual(len(set(coords)), spec.point_count)
            board_type = TwoPlayerBoard if mode is TrainingMode.TWO_PLAYER else FourPlayerBoard
            base = board_type()
            for viewer in range(spec.player_count):
                viewed = board_type(viewer)
                for code, (x, y) in enumerate(coords):
                    physical = base.decode(code)
                    expected = (x, y)
                    rotations = 2 * viewer if mode is TrainingMode.TWO_PLAYER else viewer
                    for _ in range(rotations):
                        expected = (-expected[1], expected[0])
                    self.assertEqual(action_point_coordinates(viewed.encode(physical), mode), expected)
        self.assertEqual(action_point_coordinates(0, "four_dark"), (-2, -3))
        self.assertEqual(action_point_coordinates(30, "four_dark"), (-3, 2))
        self.assertEqual(action_point_coordinates(120, "four_dark"), (-2, -2))
        self.assertEqual(action_point_coordinates(124, "four_dark"), (0, 0))
        self.assertEqual(action_point_coordinates(0, "two_player"), (-2, -1))
        self.assertEqual(action_point_coordinates(34, "two_player"), (-2, 1))

    def test_outcome_fields_cannot_change_the_action_input(self):
        event = ObservedEvent(1, 3, (0, 120), False, CombatOutcome.MOVE, None, (), ())
        changed = replace(event, was_attack=True, combat=CombatOutcome.BOTH_REMOVED,
                          flag_captured_owner=1, newly_revealed_flags=(0, 1, 2),
                          eliminated_players=(0, 1))
        left, right = ActionFeatures.from_event(event), ActionFeatures.from_event(changed)
        self.assertEqual([field.name for field in fields(left)], ["source", "destination", "actor"])
        self.assertEqual(left, right)
        self.assertEqual(left.as_vector("four_dark"), (-2., -3., -2., -2., 3., 70.))
        encoder = PublicActionEncoder(256)
        values = torch.tensor([left.as_vector("four_dark"), right.as_vector("four_dark")])
        output = encoder(values, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(output[0], output[1])

    def test_single_linear_layer_and_absent_actions_are_zero_even_with_bias(self):
        encoder = PublicActionEncoder(256)
        self.assertEqual(list(encoder.children()), [encoder.projection])
        self.assertEqual(sum(p.numel() for p in encoder.parameters()), 1792)
        with torch.no_grad():
            encoder.projection.bias.fill_(5.)
        values = torch.tensor([[[-2., -3., -2., -2., 0., 69.], [float("nan")] * 6]], requires_grad=True)
        present = torch.tensor([[True, False]])
        output = encoder(values, present)
        torch.testing.assert_close(output[0, 0], encoder.projection(values[0, 0]))
        self.assertEqual(torch.count_nonzero(output[0, 1]), 0)
        output.sum().backward()
        self.assertEqual(torch.count_nonzero(values.grad[0, 1]), 0)
        self.assertTrue(torch.isfinite(encoder.projection.weight.grad).all())
        self.assertGreater(float(encoder.projection.weight.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "6 coordinate"):
            encoder(torch.zeros(2, 8), torch.ones(2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "presence"):
            encoder(torch.zeros(2, 6), torch.ones(2, 1, dtype=torch.bool))

    def test_all_viewers_and_modes_collate_six_public_values(self):
        for mode in TrainingMode:
            states = []
            game = new_game(mode, seed=710)
            history = GameHistory.initialize(game, mode)
            game.step(game.legal_actions()[0])
            history.append_after_step(game)
            for viewer, player in enumerate(history.players):
                observed = game.observe(viewer)
                raw = observed.history[-1]
                record = player.records[-1]
                self.assertEqual(record.action.as_tuple(), (*raw.action, raw.actor))
                state = player.as_policy_state(((0, 1),))
                states.append(state)
            batch = collate_policy_states(states, device="cpu")
            self.assertEqual(batch.action_fields.shape, (mode_spec(mode).player_count, 2, 6))
            self.assertEqual(batch.action_fields.dtype, torch.float32)
            self.assertFalse(batch.action_present[:, 0].any())
            self.assertTrue(batch.action_present[:, 1].all())
            self.assertEqual(torch.count_nonzero(batch.action_fields[:, 0]), 0)
            for row, state in enumerate(states):
                torch.testing.assert_close(batch.action_fields[row, 1],
                                           torch.tensor(state.records[-1].action.as_vector(
                                               state.mode, no_capture_plies=state.records[-1].no_interaction_plies)))

    def test_invalid_points_and_players_fail_before_encoding(self):
        for mode, point in (("four_dark", -1), ("four_dark", 129), ("two_player", 60),
                            ("two_player", True), ("four_dark", 1.5)):
            with self.assertRaises(ValueError):
                action_point_coordinates(point, mode)
        for mode, actor in (("four_dark", 4), ("four_dark", True), ("two_player", 2)):
            with self.assertRaises(ValueError):
                ActionFeatures(0, 1, actor).as_vector(mode)

    def test_replay_round_trip_and_legacy_history_checkpoint_rejection(self):
        for mode in TrainingMode:
            game = new_game(mode, seed=711)
            history = GameHistory.initialize(game, mode)
            game.step(game.legal_actions()[0])
            history.append_after_step(game)
            for player in history.players:
                saved = player.state_dict()
                self.assertEqual(saved["action_encoding"], ACTION_ENCODER_TYPE)
                self.assertEqual(len(saved["records"][1]["action"]), 3)
                self.assertEqual(PlayerHistory.from_state_dict(saved).records, player.records)
                old = copy.deepcopy(saved)
                del old["action_encoding"]
                with self.assertRaisesRegex(ValueError, "action encoding"):
                    PlayerHistory.from_state_dict(old)
                old = copy.deepcopy(saved)
                old["records"][1]["action"] = (0, 1, 0, 0, 0, 4, 0, 0)
                with self.assertRaisesRegex(ValueError, "source, destination and actor"):
                    PlayerHistory.from_state_dict(old)
        require_current_checkpoint({"format_version": CHECKPOINT_FORMAT_VERSION})
        for old in (4, 5):
            with self.assertRaisesRegex(ValueError, "version 6"):
                require_current_checkpoint({"format_version": old})

    def test_cached_and_full_history_policy_critic_agree_after_multiple_actions(self):
        for mode in TrainingMode:
            for model_type in (GamePolicyTransformer, GameValueTransformer):
                config = ModelConfig.tiny()
                cached = model_type(config).eval()
                full = copy.deepcopy(cached)
                game = new_game(mode, seed=712)
                history = GameHistory.initialize(game, mode)
                cached.start_inference_board_cache()
                for _ in range(8):
                    for player in history.players:
                        state = player.as_policy_state(((0, 1),))
                        with torch.inference_mode():
                            actual, expected = cached.encode([state]), full.encode([state])
                            torch.testing.assert_close(actual.context, expected.context, atol=2e-6, rtol=2e-6)
                    if game.is_terminal:
                        break
                    game.step(game.legal_actions()[0])
                    history.append_after_step(game)

    def test_yaml_action_width_and_architecture_are_enforced(self):
        config_path = Path(__file__).parents[1] / "configs/bootstrap.yaml"
        for scale in ("bootstrap", "main", "extended"):
            settings = TrainingSettings.from_yaml(config_path, TrainingMode.FOUR_DARK, model_scale=scale)
            self.assertEqual(settings.model.action_encoder_type, ACTION_ENCODER_TYPE)
        for field, value, error in (("input_dim", 8, "exactly six"),
                                    ("output_dim", 128, "equal output"),
                                    ("architecture", "field_embeddings", "unsupported action encoder")):
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            data["models"]["policy"]["action_encoder"][field] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.yaml"
                path.write_text(yaml.safe_dump(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    TrainingSettings.from_yaml(path, TrainingMode.FOUR_DARK)


if __name__ == "__main__":
    unittest.main()
