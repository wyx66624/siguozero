from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from junqi import JunqiGame, PieceType


try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class ObservationEncodingTests(unittest.TestCase):
    def test_model_scale_selects_validated_microbatch_profile(self) -> None:
        from junqi.training.modes import TrainingMode
        from junqi.training.settings import TrainingSettings

        config_path = Path(__file__).parents[1] / "configs" / "bootstrap.yaml"
        for scale, expected in (("bootstrap", 16), ("main", 8), ("extended", 6)):
            settings = TrainingSettings.from_yaml(
                config_path,
                TrainingMode.FOUR_DARK,
                model_scale=scale,
            )
            self.assertEqual(settings.policy_microbatch, expected)

    def test_dead_rule_switch_changes_model_and_run_directory(self) -> None:
        from junqi.training.models import GamePolicyTransformer, parameter_count
        from junqi.training.modes import TrainingMode
        from junqi.training.settings import TrainingSettings

        config_path = Path(__file__).parents[1] / "configs" / "bootstrap.yaml"
        enabled = TrainingSettings.from_yaml(
            config_path,
            TrainingMode.TWO_PLAYER,
            tiny=True,
            dead_rules_enabled=True,
        )
        disabled = TrainingSettings.from_yaml(
            config_path,
            TrainingMode.TWO_PLAYER,
            tiny=True,
            dead_rules_enabled=False,
        )
        self.assertEqual(
            enabled.resolve_run_directory("runs/two_player").name,
            "with_dead_rules",
        )
        self.assertEqual(
            disabled.resolve_run_directory("runs/two_player").name,
            "without_dead_rules",
        )
        with self.assertRaisesRegex(ValueError, "contradicts"):
            enabled.resolve_run_directory("runs/two_player/without_dead_rules")

        enabled_policy = GamePolicyTransformer(enabled.model)
        disabled_policy = GamePolicyTransformer(disabled.model)
        self.assertIsNotNone(enabled_policy.board_encoder.casualty_projection)
        self.assertIsNone(disabled_policy.board_encoder.casualty_projection)
        self.assertIsNone(disabled_policy.board_encoder.board_casualty_fusion)
        self.assertGreater(
            parameter_count(enabled_policy), parameter_count(disabled_policy)
        )

    def test_disabled_dead_rules_build_no_casualty_tensor(self) -> None:
        from junqi.training.encoding import GameHistory
        from junqi.training.models import (
            GamePolicyTransformer,
            ModelConfig,
            collate_policy_states,
        )
        from junqi.training.modes import TrainingMode, new_game

        model = GamePolicyTransformer(
            ModelConfig.tiny(dead_rules_enabled=False)
        ).eval()
        states = []
        for mode in TrainingMode:
            game = new_game(mode, seed=91, dead_rules_enabled=False)
            history = GameHistory.initialize(game, mode)
            state = history.state_for(game)
            states.append(state)
            self.assertIsNone(state.records[0].known_casualty_bits)
            batch = collate_policy_states(
                [state], device="cpu", dead_rules_enabled=False
            )
            self.assertIsNone(batch.casualty_bits)
            self.assertEqual(model.encode([state]).context.shape, (1, 64))
        with self.assertRaisesRegex(ValueError, "requires casualty"):
            collate_policy_states(
                [states[-1]], device="cpu", dead_rules_enabled=True
            )

    def test_configuration_rejects_per_seat_model_instances(self) -> None:
        import yaml

        from junqi.training.modes import TrainingMode
        from junqi.training.settings import TrainingSettings

        config_path = Path(__file__).parents[1] / "configs" / "bootstrap.yaml"
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data["models"]["common"]["player_parameter_sharing"] = "one_model_per_seat"
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "invalid.yaml"
            invalid_path.write_text(
                yaml.safe_dump(data, allow_unicode=True), encoding="utf-8"
            )
            for scale in ("bootstrap", "main", "extended"):
                with self.subTest(scale=scale), self.assertRaisesRegex(
                    ValueError, "all seats must share"
                ):
                    TrainingSettings.from_yaml(
                        invalid_path,
                        TrainingMode.FOUR_DARK,
                        model_scale=scale,
                    )

    def test_three_mode_piece_code_contracts(self) -> None:
        from junqi.training.encoding import encode_visible_board
        from junqi.training.modes import TrainingMode, new_game

        for mode in TrainingMode:
            game = new_game(mode, seed=17)
            codes = encode_visible_board(game.observe(0, history_limit=0))
            expected = 60 if mode is TrainingMode.TWO_PLAYER else 129
            self.assertEqual(len(codes), expected)
            self.assertTrue(any(30 <= code <= 41 for code in codes))
            if mode is TrainingMode.TWO_PLAYER:
                self.assertNotIn(1, codes)
                self.assertNotIn(3, codes)
                self.assertIn(2, codes)
            elif mode is TrainingMode.FOUR_DARK:
                self.assertIn(1, codes)
                self.assertIn(2, codes)
                self.assertIn(3, codes)
                self.assertFalse(any(code >= 60 for code in codes))
            else:
                self.assertIn(1, codes)
                self.assertIn(3, codes)
                self.assertTrue(any(94 <= code <= 105 for code in codes))

    def test_exact_piece_codes_use_collision_free_relative_seat_blocks(self) -> None:
        from junqi import PieceType
        from junqi.training.encoding import (
            BOARD_CODE_VOCAB_SIZE,
            BOARD_PAD_CODE,
            exact_piece_code,
        )
        from junqi.training.modes import TrainingMode

        self.assertEqual(
            tuple(
                exact_piece_code(
                    PieceType.ENGINEER,
                    relative_owner,
                    TrainingMode.FOUR_DARK,
                )
                for relative_owner in range(4)
            ),
            (32, 64, 96, 128),
        )
        self.assertEqual(
            exact_piece_code(
                PieceType.ENGINEER, 1, TrainingMode.TWO_PLAYER
            ),
            96,
        )

        all_four_player_codes = {
            exact_piece_code(kind, owner, TrainingMode.FOUR_DARK)
            for owner in range(4)
            for kind in PieceType
        }
        self.assertEqual(len(all_four_player_codes), 4 * len(PieceType))
        self.assertEqual(max(all_four_player_codes), 137)
        self.assertEqual(BOARD_PAD_CODE, 138)
        self.assertEqual(BOARD_CODE_VOCAB_SIZE, 139)

    def test_inferred_identity_is_written_into_future_board_tokens(self) -> None:
        from junqi import (
            ArmPoint,
            GameConfig,
            GameVariant,
            InformationMode,
            Piece,
            PieceType,
        )
        from junqi.training.encoding import GameHistory
        from junqi.training.modes import TrainingMode

        pieces = {
            ArmPoint(0, 6, 2): Piece(0, PieceType.FLAG),
            ArmPoint(1, 6, 2): Piece(1, PieceType.FLAG),
            ArmPoint(0, 2, 3): Piece(0, PieceType.COMMANDER),
            ArmPoint(0, 1, 3): Piece(1, PieceType.ARMY_COMMANDER),
            ArmPoint(1, 2, 3): Piece(1, PieceType.COMMANDER),
        }
        game = JunqiGame.from_position(
            GameConfig(
                variant=GameVariant.TWO_PLAYER,
                information_mode=InformationMode.DARK,
            ),
            pieces,
            revealed_flags=(False, False),
        )
        history = GameHistory.initialize(game, TrainingMode.TWO_PLAYER)

        game.step((7, 2))
        history.append_after_step(game)
        survivor = ArmPoint(0, 1, 3)
        observer_code = game.board_for(1).encode(survivor)
        self.assertEqual(history.players[1].records[-1].board_codes[observer_code], 104)

    def test_exact_casualties_are_written_and_survive_history_roundtrip(self) -> None:
        from junqi import (
            ArmPoint,
            GameConfig,
            GameVariant,
            InformationMode,
            Piece,
            PieceType,
        )
        from junqi.training.encoding import GameHistory, PlayerHistory
        from junqi.training.modes import TrainingMode

        pieces = {
            ArmPoint(0, 6, 2): Piece(0, PieceType.FLAG),
            ArmPoint(1, 6, 2): Piece(1, PieceType.FLAG),
            ArmPoint(1, 5, 1): Piece(0, PieceType.ENGINEER),
            ArmPoint(0, 3, 2): Piece(0, PieceType.COMMANDER),
            ArmPoint(1, 5, 2): Piece(1, PieceType.MINE),
            ArmPoint(1, 2, 3): Piece(1, PieceType.COMMANDER),
        }
        game = JunqiGame.from_position(
            GameConfig(
                variant=GameVariant.TWO_PLAYER,
                information_mode=InformationMode.DARK,
            ),
            pieces,
            revealed_flags=(False, False),
        )
        history = GameHistory.initialize(game, TrainingMode.TWO_PLAYER)
        game.step((50, 51))
        history.append_after_step(game)

        record = history.players[0].records[-1]
        self.assertEqual(len(record.known_casualty_bits), 25)
        self.assertEqual(record.known_casualty_bits[1], 1)
        restored = PlayerHistory.from_state_dict(history.players[0].state_dict())
        self.assertEqual(restored.records, history.players[0].records)

    def test_collator_places_mode_specific_casualties_in_relative_seat_blocks(self) -> None:
        from junqi.training.encoding import GameHistory, PolicyState
        from junqi.training.models import (
            GamePolicyTransformer,
            ModelConfig,
            collate_policy_states,
        )
        from junqi.training.modes import TrainingMode, new_game

        model = GamePolicyTransformer(ModelConfig.tiny()).eval()
        expected = {
            TrainingMode.FOUR_DARK: (0, 25, 50),
            TrainingMode.DOUBLE_OPEN: (0, 50),
            TrainingMode.TWO_PLAYER: (25,),
        }
        for mode, expected_indices in expected.items():
            game = new_game(mode, seed=71)
            original = GameHistory.initialize(game, mode).state_for(game)
            raw = [0] * len(original.records[0].known_casualty_bits)
            for index in range(0, len(raw), 25):
                raw[index] = 1
            record = replace(
                original.records[0], known_casualty_bits=tuple(raw)
            )
            state = PolicyState(mode, (record,), original.legal_actions)
            batch = collate_policy_states([state], device="cpu")
            actual_indices = tuple(
                batch.casualty_bits[0].nonzero().flatten().tolist()
            )
            self.assertEqual(actual_indices, expected_indices)
            contexts = model.encode([original, state]).context
            self.assertFalse(torch.allclose(contexts[0], contexts[1]))

    def test_initial_token_is_pinned_when_history_slides(self) -> None:
        from junqi.training.encoding import GameHistory
        from junqi.training.modes import TrainingMode, new_game

        game = new_game(TrainingMode.TWO_PLAYER, seed=23, max_plies=10)
        history = GameHistory.initialize(
            game, TrainingMode.TWO_PLAYER, max_transitions=2
        )
        initial = history.players[0].records[0]
        for _ in range(3):
            game.step(game.legal_actions()[0])
            history.append_after_step(game)
        records = history.players[0].records
        self.assertEqual(len(records), 3)
        self.assertIs(records[0], initial)
        self.assertIsNone(records[0].action)
        self.assertTrue(all(record.action is not None for record in records[1:]))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class NeuralModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)

    def test_layout_pointer_always_generates_legal_inventory(self) -> None:
        from junqi.training.models import (
            ModelConfig,
            PieceConditionedLayoutPointerDecoder,
        )
        from junqi.training.modes import TrainingMode

        model = PieceConditionedLayoutPointerDecoder(ModelConfig.tiny()).eval()
        for mode in TrainingMode:
            samples = model.sample_layouts(3, mode)
            self.assertEqual(len(samples), 3)
            self.assertTrue(all(len(sample.setup) == 25 for sample in samples))
            self.assertTrue(
                all(len(sample.position_indices) == 25 for sample in samples)
            )

    def test_policy_samples_only_legal_actions_in_all_modes(self) -> None:
        from junqi.training.encoding import GameHistory
        from junqi.training.models import GamePolicyTransformer, ModelConfig
        from junqi.training.modes import TrainingMode, new_game

        model = GamePolicyTransformer(ModelConfig.tiny()).eval()
        for mode in TrainingMode:
            game = new_game(mode, seed=31)
            history = GameHistory.initialize(game, mode, max_transitions=16)
            state = history.state_for(game)
            action_groups, logs = model.sample_action_groups([state], count=4)
            self.assertEqual(len(action_groups[0]), 4)
            self.assertTrue(all(action in game.legal_actions() for action in action_groups[0]))
            self.assertTrue(torch.isfinite(logs[0]).all())

    def test_frozen_actor_deduplicates_and_caches_history_boards_exactly(self) -> None:
        from junqi.training.encoding import GameHistory
        from junqi.training.models import GamePolicyTransformer, ModelConfig
        from junqi.training.modes import TrainingMode, new_game

        config = ModelConfig.tiny()
        uncached = GamePolicyTransformer(config).eval()
        cached = copy.deepcopy(uncached).eval()
        game = new_game(TrainingMode.TWO_PLAYER, seed=33, max_plies=8)
        history = GameHistory.initialize(
            game, TrainingMode.TWO_PLAYER, max_transitions=16
        )
        initial = history.state_for(game, 0)

        with torch.inference_mode():
            expected_initial = uncached.encode([initial, initial])
            cached.start_inference_board_cache(max_entries=64)
            actual_initial = cached.encode([initial, initial])
        torch.testing.assert_close(actual_initial.context, expected_initial.context)
        torch.testing.assert_close(
            actual_initial.current_points, expected_initial.current_points
        )

        game.step(game.legal_actions()[0])
        history.append_after_step(game)
        advanced = history.state_for(game, 0)
        with torch.inference_mode():
            expected_advanced = uncached.encode([advanced])
            actual_advanced = cached.encode([advanced])
        torch.testing.assert_close(actual_advanced.context, expected_advanced.context)
        torch.testing.assert_close(
            actual_advanced.current_points, expected_advanced.current_points
        )

        metrics = cached.board_encoding_metrics()
        self.assertEqual(metrics["encoding/raw_board_tokens"], 4.0)
        self.assertEqual(metrics["encoding/board_input_tokens"], 2.0)
        self.assertEqual(metrics["encoding/board_unique_tokens"], 2.0)
        self.assertEqual(metrics["encoding/board_encoder_tokens"], 2.0)
        self.assertGreater(
            metrics["encoding/within_batch_history_saved_fraction"], 0.0
        )
        self.assertGreater(metrics["encoding/temporal_cache_hits"], 0.0)
        self.assertGreater(
            metrics["encoding/temporal_attention_saved_fraction"], 0.0
        )
        self.assertEqual(metrics["encoding/board_encoder_saved_fraction"], 0.5)

    def test_four_seats_reuse_the_exact_same_inference_model(self) -> None:
        from junqi.training.inference import InferenceEngine
        from junqi.training.models import (
            GamePolicyTransformer,
            ModelConfig,
            PieceConditionedLayoutPointerDecoder,
        )
        from junqi.training.modes import TrainingMode

        config = ModelConfig.tiny()
        policy = GamePolicyTransformer(config)
        layout = PieceConditionedLayoutPointerDecoder(config)
        engine = InferenceEngine(TrainingMode.FOUR_DARK, policy, layout)

        self.assertIs(engine.policy, policy)
        self.assertIs(engine.layout, layout)
        self.assertIs(engine.actor.policy, policy)

        game, history = engine.new_game(seed=29, max_plies=4)
        acting_seats: list[int] = []
        policy_instance_ids: list[int] = []
        while not game.is_terminal:
            self.assertIsNotNone(game.current_player)
            acting_seats.append(game.current_player)
            policy_instance_ids.append(id(engine.actor.policy))
            engine.step(game, history)

        self.assertEqual(len(acting_seats), 4)
        self.assertEqual(len(set(acting_seats)), 4)
        self.assertEqual(set(policy_instance_ids), {id(policy)})

    def test_rollout_is_exactly_four_by_two_and_reaches_terminal(self) -> None:
        from junqi.training.encoding import GameHistory
        from junqi.training.models import GamePolicyTransformer, ModelConfig
        from junqi.training.modes import TrainingMode, new_game
        from junqi.training.rollout import (
            AnchorSnapshot,
            FrozenPolicyActor,
            collect_policy_groups,
        )

        game = new_game(TrainingMode.TWO_PLAYER, seed=37, max_plies=4)
        history = GameHistory.initialize(
            game, TrainingMode.TWO_PLAYER, max_transitions=16
        )
        state = history.state_for(game)
        anchor = AnchorSnapshot(game, history, state, game.current_player)
        actor = FrozenPolicyActor(GamePolicyTransformer(ModelConfig.tiny()).eval())
        groups, metrics = collect_policy_groups([anchor], actor, behavior_version=0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].candidate_actions), 4)
        self.assertTrue(all(len(pair) == 2 for pair in groups[0].replica_rewards))
        self.assertEqual(metrics.terminal_continuations, 8)
        self.assertEqual(metrics.wins + metrics.draws + metrics.losses, 8)

    def test_policy_and_layout_losses_backpropagate(self) -> None:
        from junqi.training.encoding import GameHistory
        from junqi.training.losses import layout_grpo_loss, policy_grpo_loss
        from junqi.training.models import (
            GamePolicyTransformer,
            ModelConfig,
            PieceConditionedLayoutPointerDecoder,
        )
        from junqi.training.modes import TrainingMode, new_game
        from junqi.training.rollout import LayoutOutcome, PolicyGroup

        config = ModelConfig.tiny()
        policy = GamePolicyTransformer(config)
        reference_policy = copy.deepcopy(policy).eval()
        game = new_game(TrainingMode.TWO_PLAYER, seed=41)
        history = GameHistory.initialize(game, TrainingMode.TWO_PLAYER, max_transitions=16)
        state = history.state_for(game)
        actions = state.legal_actions[:4]
        with torch.no_grad():
            old_logs = policy.log_probs_for_action_groups([state], [actions])[0]
        group = PolicyGroup(
            state=state,
            candidate_actions=actions,
            old_log_probs=tuple(old_logs.tolist()),
            replica_rewards=((1.0, 1.0), (0.0, 0.0), (-1.0, -1.0), (1.0, 0.0)),
            candidate_returns=(1.0, 0.0, -1.0, 0.5),
            advantages=(1.0, -0.2, -1.4, 0.6),
            continuation_plies=4,
            behavior_version=0,
        )
        output = policy_grpo_loss(
            policy,
            reference_policy,
            [group],
            clip_epsilon=0.2,
            kl_coefficient=0.02,
            entropy_coefficient=0.01,
        )
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(any(parameter.grad is not None for parameter in policy.parameters()))

        layout = PieceConditionedLayoutPointerDecoder(config)
        reference_layout = copy.deepcopy(layout).eval()
        samples = layout.sample_layouts(4, TrainingMode.TWO_PLAYER)
        outcomes = [
            LayoutOutcome(sample, reward, index % 2, 0)
            for index, (sample, reward) in enumerate(
                zip(samples, (1.0, -1.0, 0.0, 1.0), strict=True)
            )
        ]
        layout_output = layout_grpo_loss(
            layout,
            reference_layout,
            outcomes,
            clip_epsilon=0.2,
            kl_coefficient=0.02,
            entropy_coefficient=0.01,
        )
        layout_output.loss.backward()
        self.assertTrue(torch.isfinite(layout_output.loss))
        self.assertTrue(any(parameter.grad is not None for parameter in layout.parameters()))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class CheckpointTests(unittest.TestCase):
    def test_train_update_does_not_clone_a_behavior_model(self) -> None:
        from unittest.mock import patch

        from junqi.training.modes import TrainingMode
        from junqi.training.settings import TrainingSettings
        from junqi.training.trainer import SelfPlayTrainer

        config_path = Path(__file__).parents[1] / "configs" / "bootstrap.yaml"
        with tempfile.TemporaryDirectory() as directory:
            settings = TrainingSettings.from_yaml(
                config_path,
                TrainingMode.FOUR_DARK,
                tiny=True,
                overrides={"device": "cpu"},
            )
            with patch("junqi.training.trainer.copy", wraps=copy) as copy_module:
                trainer = SelfPlayTrainer(
                    settings,
                    run_directory=directory,
                    auto_resume=False,
                )
                # Exactly one Policy/Layout copy is retained for KL reference.
                # A training update must not clone another behavior pair.
                self.assertEqual(copy_module.deepcopy.call_count, 2)
                trainer.train()
                self.assertEqual(copy_module.deepcopy.call_count, 2)

    def test_atomic_checkpoint_restores_models_optimizers_and_rng(self) -> None:
        from junqi.training.checkpoint import (
            CheckpointManager,
            restore_training_state,
        )
        from junqi.training.models import (
            GamePolicyTransformer,
            ModelConfig,
            PieceConditionedLayoutPointerDecoder,
        )

        config = ModelConfig.tiny()
        policy = GamePolicyTransformer(config)
        layout = PieceConditionedLayoutPointerDecoder(config)
        ref_policy = copy.deepcopy(policy)
        ref_layout = copy.deepcopy(layout)
        policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        layout_optimizer = torch.optim.AdamW(layout.parameters(), lr=5e-5)
        expected = next(policy.parameters()).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, keep_archives=2)
            manager.save(
                update=7,
                mode="two_player",
                dead_rules_enabled=True,
                policy=policy,
                layout=layout,
                reference_policy=ref_policy,
                reference_layout=ref_layout,
                policy_optimizer=policy_optimizer,
                layout_optimizer=layout_optimizer,
                trainer_state={"marker": 9},
                config={"model": {}},
                archive=True,
                reason="test",
            )
            with torch.no_grad():
                next(policy.parameters()).add_(10)
            payload = manager.load_latest(map_location="cpu")
            self.assertIsNotNone(payload)
            self.assertIs(payload["dead_rules_enabled"], True)
            update, state = restore_training_state(
                payload,
                expected_mode="two_player",
                expected_dead_rules_enabled=True,
                policy=policy,
                layout=layout,
                reference_policy=ref_policy,
                reference_layout=ref_layout,
                policy_optimizer=policy_optimizer,
                layout_optimizer=layout_optimizer,
            )
            self.assertEqual(update, 7)
            self.assertEqual(state["marker"], 9)
            torch.testing.assert_close(next(policy.parameters()), expected)
            self.assertTrue((Path(directory) / "checkpoints" / "latest.pt").exists())

            with self.assertRaisesRegex(RuntimeError, "dead-rule variant"):
                restore_training_state(
                    payload,
                    expected_mode="two_player",
                    expected_dead_rules_enabled=False,
                    policy=policy,
                    layout=layout,
                    reference_policy=ref_policy,
                    reference_layout=ref_layout,
                    policy_optimizer=policy_optimizer,
                    layout_optimizer=layout_optimizer,
                )

            from junqi.training.inference import InferenceEngine

            with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
                InferenceEngine.from_checkpoint(
                    manager.latest_path,
                    mode="two_player",
                    device="cpu",
                    dead_rules_enabled=False,
                )

    def test_trainer_resumes_nonempty_layout_outcome_buffer(self) -> None:
        from junqi.training.modes import TrainingMode
        from junqi.training.rollout import LayoutOutcome
        from junqi.training.settings import TrainingSettings
        from junqi.training.trainer import SelfPlayTrainer

        config_path = Path(__file__).parents[1] / "configs" / "bootstrap.yaml"
        with tempfile.TemporaryDirectory() as directory:
            settings = TrainingSettings.from_yaml(
                config_path,
                TrainingMode.TWO_PLAYER,
                tiny=True,
                overrides={"device": "cpu"},
            )
            first = SelfPlayTrainer(
                settings,
                run_directory=directory,
                auto_resume=False,
            )
            sample = first.layout.sample_layouts(1, TrainingMode.TWO_PLAYER)[0]
            first.layout_buffer.append(
                LayoutOutcome(
                    sample=sample,
                    reward=1.0,
                    seat=0,
                    behavior_version=3,
                )
            )
            first.pool.fill(first.layout, behavior_version=3)
            original_slot = first.pool.slots[0]
            original_slot.game.step(original_slot.game.legal_actions()[0])
            original_slot.history.append_after_step(original_slot.game)
            expected_observations = tuple(
                original_slot.game.observe(player, history_limit=None)
                for player in range(original_slot.game.config.player_count)
            )
            expected_history = original_slot.history.state_dict()
            first.update = 3
            first.save_checkpoint(reason="buffer-test", archive=False)
            first.logger.close()

            resumed = SelfPlayTrainer(
                settings,
                run_directory=directory,
                auto_resume=True,
            )
            try:
                self.assertEqual(resumed.update, 3)
                self.assertEqual(len(resumed.layout_buffer), 1)
                restored = resumed.layout_buffer[0]
                self.assertEqual(restored.sample.position_indices, sample.position_indices)
                self.assertEqual(restored.sample.setup, sample.setup)
                self.assertEqual(restored.reward, 1.0)
                self.assertEqual(restored.behavior_version, 3)
                self.assertEqual(len(resumed.pool.slots), settings.base_game_pool_size)
                resumed_slot = resumed.pool.slots[0]
                self.assertEqual(
                    tuple(
                        resumed_slot.game.observe(player, history_limit=None)
                        for player in range(resumed_slot.game.config.player_count)
                    ),
                    expected_observations,
                )
                self.assertEqual(resumed_slot.history.state_dict(), expected_history)
            finally:
                resumed.logger.close()

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                SelfPlayTrainer(
                    settings,
                    run_directory=directory,
                    auto_resume=False,
                )

    def test_base_game_pool_restores_persistent_exact_knowledge(self) -> None:
        from junqi.training.encoding import GameHistory, exact_piece_code
        from junqi.training.models import (
            ModelConfig,
            PieceConditionedLayoutPointerDecoder,
        )
        from junqi.training.modes import TrainingMode
        from junqi.training.rollout import BaseGamePool

        layout = PieceConditionedLayoutPointerDecoder(ModelConfig.tiny()).eval()
        pool = BaseGamePool(
            TrainingMode.TWO_PLAYER,
            pool_size=1,
            max_transitions=16,
            max_game_plies=32,
            seed=19,
        )
        pool.fill(layout, behavior_version=4)
        slot = pool.slots[0]
        dead_point, dead_piece = next(
            (point, piece)
            for point, piece in slot.game.pieces.items()
            if piece.owner == 1
            and piece.kind not in (PieceType.FLAG, PieceType.COMMANDER)
        )
        pieces = dict(slot.game.pieces)
        del pieces[dead_point]
        public_candidates = dict(slot.game.public_candidates)
        del public_candidates[dead_point]
        point, piece = next(
            (candidate_point, candidate_piece)
            for candidate_point, candidate_piece in pieces.items()
            if candidate_piece.owner == 1
        )
        known = ({point: piece.kind}, {})
        casualties = (
            ({}, {dead_piece.kind: 1}),
            ({}, {dead_piece.kind: 1}),
        )
        slot.game = JunqiGame.from_position(
            slot.game.config,
            pieces,
            current_player=slot.game.current_player,
            active_players=slot.game.active_players,
            revealed_flags=slot.game.revealed_flags,
            ply_count=slot.game.ply_count,
            no_interaction_plies=slot.game.no_interaction_plies,
            public_candidates=public_candidates,
            known_identities=known,
            known_casualties=casualties,
            public_history=slot.game.public_history,
        )
        slot.history = GameHistory.initialize(
            slot.game,
            TrainingMode.TWO_PLAYER,
            max_transitions=16,
        )

        state = pool.state_dict()
        legacy_state = copy.deepcopy(state)
        legacy_state["format_version"] = 2
        incompatible = BaseGamePool(
            TrainingMode.TWO_PLAYER,
            pool_size=1,
            max_transitions=16,
            max_game_plies=32,
            seed=999,
        )
        with self.assertRaisesRegex(ValueError, "unsupported.*format"):
            incompatible.load_state_dict(legacy_state)

        resumed = BaseGamePool(
            TrainingMode.TWO_PLAYER,
            pool_size=1,
            max_transitions=16,
            max_game_plies=32,
            seed=999,
        )
        resumed.load_state_dict(state)
        restored_slot = resumed.slots[0]
        self.assertEqual(restored_slot.game.known_identities[0][point], piece.kind)
        self.assertEqual(
            restored_slot.game.known_casualties[0][1][dead_piece.kind], 1
        )
        code = restored_slot.game.board_for(0).encode(point)
        self.assertEqual(
            restored_slot.history.players[0].records[-1].board_codes[code],
            exact_piece_code(piece.kind, 1, TrainingMode.TWO_PLAYER),
        )
        self.assertEqual(
            sum(restored_slot.history.players[0].records[-1].known_casualty_bits),
            1,
        )


if __name__ == "__main__":
    unittest.main()
