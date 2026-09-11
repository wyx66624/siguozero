from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from test_arena import ArenaFixtures, TORCH_AVAILABLE


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class ParallelArenaTests(ArenaFixtures, unittest.TestCase):
    def test_serial_batched_workers_and_group_order_preserve_games_past_history_window(self):
        from junqi.training.arena import MatchSettings, play_groups, protocol_for

        for mode in ("two_player", "four_dark", "double_open"):
            with self.subTest(mode=mode):
                candidate, opponent = self.engines(mode)
                window = candidate.policy.config.max_transitions
                settings = MatchSettings(mode=mode, pairs=2, max_plies=window + 8,
                                         smoke_test=True, parallel_games=1,
                                         inference_batch_size=1, environment_workers=1)
                protocol = protocol_for(mode)
                expected = [row for index in range(settings.pairs)
                            for row in protocol.play_group(candidate, opponent, index, settings)]
                self.assertTrue(any(row["plies"] > window for row in expected))
                for parallel, batch, workers, indices in (
                    (1, 1, 1, [0, 1]),
                    (8, 8, 1, [0, 1]),
                    (8, 3, 4, [0, 1]),
                    (8, 8, 4, [1, 0]),
                ):
                    with self.subTest(parallel=parallel, batch=batch, workers=workers,
                                      indices=indices):
                        actual = play_groups(candidate, opponent, indices, replace(
                            settings, parallel_games=parallel, inference_batch_size=batch,
                            environment_workers=workers,
                        ))
                        self.assertEqual(actual, expected)

    def test_actual_batches_share_the_original_policy_and_restore_caches(self):
        from junqi.training.arena import MatchSettings, play_groups

        candidate, opponent = self.engines("four_dark")
        original = [(engine.policy, engine.layout, engine.policy.config,
                     engine.actor.max_batch_size) for engine in (candidate, opponent)]
        batches = {"candidate": [], "opponent": []}

        def observed_sample(label, engine, states, **kwargs):
            index = 0 if label == "candidate" else 1
            self.assertIs(engine.policy, original[index][0])
            self.assertIs(engine.layout, original[index][1])
            batches[label].append(len(states))
            self.assertEqual(engine.policy.config.inference_temporal_cache_entries, 48)
            return real_sample[label](states, **kwargs)

        real_sample = {"candidate": candidate.actor.sample, "opponent": opponent.actor.sample}
        with patch.object(candidate.actor, "sample", side_effect=lambda states, **kw:
                          observed_sample("candidate", candidate, states, **kw)), \
                patch.object(opponent.actor, "sample", side_effect=lambda states, **kw:
                             observed_sample("opponent", opponent, states, **kw)), \
                patch.object(candidate.policy, "sample_action_groups",
                             wraps=candidate.policy.sample_action_groups) as candidate_forward, \
                patch.object(opponent.policy, "sample_action_groups",
                             wraps=opponent.policy.sample_action_groups) as opponent_forward:
            records = play_groups(candidate, opponent, [0, 1], MatchSettings(
                mode="four_dark", pairs=2, max_plies=4, parallel_games=8,
                inference_batch_size=3, environment_workers=4,
            ))
        self.assertEqual(len(records), 8)
        for forward in (candidate_forward, opponent_forward):
            sizes = [len(call.args[0]) for call in forward.call_args_list]
            self.assertGreater(max(sizes), 1)
            self.assertLessEqual(max(sizes), 3)
        for index, (label, engine) in enumerate((("candidate", candidate), ("opponent", opponent))):
            self.assertGreater(max(batches[label]), 1)
            self.assertIs(engine.policy.config, original[index][2])
            self.assertEqual(engine.actor.max_batch_size, original[index][3])
            self.assertFalse(engine.policy._inference_board_cache)
            self.assertFalse(engine.policy._inference_temporal_cache)
            self.assertTrue(all(parameter.grad is None for parameter in engine.policy.parameters()))

    def test_illegal_batched_action_fails_instead_of_becoming_a_loss(self):
        from junqi.training.arena import MatchSettings, play_groups

        candidate, opponent = self.engines("two_player")
        original = candidate.policy.config

        def illegal(states, **kwargs):
            return [[(-1, -1)] for _ in states], []

        with patch.object(candidate.actor, "sample", side_effect=illegal), \
                patch.object(opponent.actor, "sample", side_effect=illegal):
            with self.assertRaises(ValueError):
                play_groups(candidate, opponent, [0, 1], MatchSettings(
                    pairs=2, max_plies=4, parallel_games=4,
                    inference_batch_size=4, environment_workers=4,
                ))
        self.assertIs(candidate.policy.config, original)
        self.assertFalse(candidate.policy._inference_temporal_cache)

    def test_finished_game_is_replaced_before_its_long_peer_finishes(self):
        from junqi.training.arena import MatchSettings, new_game, play_groups

        candidate, opponent = self.engines("two_player")
        games = []
        long_peer_was_running = []

        def create(mode, **kwargs):
            if len(games) == 2:
                long_peer_was_running.append(not games[0].is_terminal)
            kwargs["max_plies"] = 8 if not games else 1
            game = new_game(mode, **kwargs)
            games.append(game)
            return game

        with patch("junqi.training.arena.new_game", side_effect=create):
            records = play_groups(candidate, opponent, [0, 1], MatchSettings(
                pairs=2, parallel_games=2, inference_batch_size=2,
                environment_workers=2, max_plies=8,
            ))
        self.assertEqual(long_peer_was_running, [True])
        self.assertEqual(len(records), 4)
        self.assertEqual(sorted(record["plies"] for record in records), [1, 1, 1, 8])

    def test_parallel_settings_defaults_cache_scaling_and_strict_validation(self):
        from junqi.training.arena import MatchSettings
        from junqi.training.settings import TrainingSettings

        yaml_path = Path(__file__).resolve().parents[1] / "configs" / "bootstrap.yaml"
        training = TrainingSettings.from_yaml(yaml_path, "four_dark", tiny=True)
        match = MatchSettings()
        for name, default in (("parallel_games", 32), ("inference_batch_size", 32),
                              ("environment_workers", 4)):
            self.assertEqual(getattr(match, name), default)
            self.assertEqual(getattr(training, "arena_" + name), default)
            for value in (0, -1, True, 1.5, "4"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        replace(match, **{name: value})
                    with self.assertRaisesRegex(ValueError, "arena_" + name):
                        replace(training, **{"arena_" + name: value}).validate()
        self.assertEqual(match.effective_temporal_cache_entries, 96)
        self.assertEqual(replace(match, mode="four_dark").effective_temporal_cache_entries, 192)
        self.assertEqual(replace(match, temporal_cache_entries=200).effective_temporal_cache_entries, 200)

    def test_training_and_all_evaluator_parsers_accept_resource_overrides(self):
        from junqi.training.cli import build_parser as training_parser
        from junqi.training.evaluate_history import build_parser as evaluator_parser

        args = training_parser().parse_args([
            "--mode", "four_dark", "--arena-parallel-games", "12",
            "--arena-inference-batch", "6", "--arena-environment-workers", "2",
        ])
        self.assertEqual((args.arena_parallel_games, args.arena_inference_batch_size,
                          args.arena_environment_workers), (12, 6, 2))
        for mode in ("two_player", "four_dark", "double_open"):
            with self.subTest(mode=mode):
                modes = ("two_player",) if mode == "two_player" else ("four_dark", "double_open")
                args = evaluator_parser(default_mode=mode, allowed_modes=modes).parse_args([
                    "--checkpoint-dir", "checkpoints", "--output-dir", "arena",
                    "--parallel-games", "12", "--inference-batch-size", "6",
                    "--environment-workers", "2",
                ])
                self.assertEqual((args.parallel_games, args.inference_batch_size,
                                  args.environment_workers), (12, 6, 2))


if __name__ == "__main__":
    unittest.main()
