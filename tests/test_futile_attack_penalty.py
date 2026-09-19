"""Public evidence, real PPO workers, decay, and lossless checkpoint adoption."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
from junqi.training.encoding import GameHistory
from junqi.training.futile_attack import EVIDENCE_ATTRIBUTE, FutileAttackPenalty, step_with_evidence
from junqi.training.metrics import MetricLogger
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import TrainingMode
from junqi.training.ppo import FrozenValueActor, PPOTransition, collect_ppo_samples, generalized_advantages
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor, RolloutMetrics
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


ROOT = Path(__file__).resolve().parents[1]
MODE = TrainingMode.FOUR_DARK
TARGET = ArmPoint(1, 5, 1)
FIRST = ArmPoint(1, 4, 1)
SECOND = ArmPoint(1, 4, 2)  # A camp has a diagonal road to the target.


def action(game, start, end):
    board = game.board_for(game.current_player)
    move = (board.encode(start), board.encode(end))
    assert move in game.legal_actions(), move
    return move


def position(*, target=PieceType.MINE, second=PieceType.PLATOON_COMMANDER,
             victim=PieceType.ARMY_COMMANDER, dead=True, terminal=False):
    pieces = {ArmPoint(seat, 6, 4): Piece(seat, PieceType.FLAG) for seat in range(4)}
    pieces.update({ArmPoint(seat, 2, 3): Piece(seat, PieceType.ENGINEER) for seat in range(4)})
    pieces.update({FIRST: Piece(0, victim), TARGET: Piece(1, target),
                   SECOND: Piece(0, second)})
    return JunqiGame.from_position(GameConfig(variant=GameVariant.FOUR_PLAYER,
        information_mode=InformationMode.FOUR_DARK, dead_rules_enabled=dead,
        max_plies=5 if terminal else None), pieces, current_player=0)


def prepare(**kwargs):
    game = position(**kwargs)
    history = GameHistory.initialize(game, MODE, max_transitions=16)
    result, penalized = step_with_evidence(game, action(game, FIRST, TARGET), enabled=True)
    assert result.combat.value == 'defender_wins' and not penalized
    history.append_after_step(game)
    while game.current_player != 0:
        step_with_evidence(game, (0, 0), enabled=True)
        history.append_after_step(game)
    return game, history


class FutileAttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_first_unknown_encounter_and_hidden_truth_do_not_trigger_or_leak(self):
        observations = []
        for target in (PieceType.MINE, PieceType.ARMY_COMMANDER):
            game, _ = prepare(target=target, victim=PieceType.DIVISION_COMMANDER)
            observations.append(game.observe(0))
            result, penalized = step_with_evidence(game, action(game, SECOND, TARGET), enabled=True)
            self.assertTrue(penalized)
            self.assertEqual(result.combat.value, 'defender_wins')
        self.assertEqual(observations[0], observations[1])

    def test_bombs_and_possible_mine_engineers_are_exempt_and_death_clears_evidence(self):
        for kind in (PieceType.BOMB, PieceType.ENGINEER):
            game, _ = prepare(second=kind)
            result, penalized = step_with_evidence(game, action(game, SECOND, TARGET), enabled=True)
            self.assertFalse(penalized)
            self.assertIn(result.combat.value, ('both_removed', 'attacker_wins'))
            self.assertNotIn(TARGET, getattr(game, EVIDENCE_ATTRIBUTE)[0])

    def test_equal_victim_rank_is_also_known_weaker_and_stronger_attack_is_exempt(self):
        for kind, expected in ((PieceType.ARMY_COMMANDER, True), (PieceType.COMMANDER, False)):
            # A second army commander is illegal inventory, so use a division
            # commander for both the witness and the equal-rank follow-up.
            game = position(second=PieceType.DIVISION_COMMANDER if expected else kind)
            pieces = dict(game.pieces)
            pieces[FIRST] = Piece(0, PieceType.DIVISION_COMMANDER)
            game = JunqiGame.from_position(game.config, pieces, current_player=0)
            step_with_evidence(game, action(game, FIRST, TARGET), enabled=True)
            while game.current_player != 0:
                step_with_evidence(game, (0, 0), enabled=True)
            _, penalized = step_with_evidence(game, action(game, SECOND, TARGET), enabled=True)
            self.assertEqual(penalized, expected)

    def test_evidence_follows_a_moving_survivor_and_is_private_to_each_viewer(self):
        game = position(target=PieceType.COMMANDER)
        step_with_evidence(game, action(game, FIRST, TARGET), enabled=True)
        self.assertIn(TARGET, getattr(game, EVIDENCE_ATTRIBUTE)[0])
        self.assertNotIn(TARGET, getattr(game, EVIDENCE_ATTRIBUTE)[2])
        while game.current_player != 1:
            step_with_evidence(game, (0, 0), enabled=True)
        step_with_evidence(game, action(game, TARGET, FIRST), enabled=True)
        self.assertNotIn(TARGET, getattr(game, EVIDENCE_ATTRIBUTE)[0])
        self.assertIn(FIRST, getattr(game, EVIDENCE_ATTRIBUTE)[0])
        _, penalized = step_with_evidence(game, action(game, SECOND, FIRST), enabled=True)
        self.assertTrue(penalized)

    def test_attacking_survivor_also_supplies_public_evidence(self):
        game = position(target=PieceType.COMMANDER)
        while game.current_player != 1:
            step_with_evidence(game, (0, 0), enabled=True)
        step_with_evidence(game, action(game, TARGET, FIRST), enabled=True)
        _, penalized = step_with_evidence(game, action(game, SECOND, FIRST), enabled=True)
        self.assertTrue(penalized)

    def test_tracking_works_without_exact_identity_rules_and_disabled_path_is_plain_rules(self):
        game = position(dead=False)
        step_with_evidence(game, action(game, FIRST, TARGET), enabled=True)
        while game.current_player != 0:
            step_with_evidence(game, (0, 0), enabled=True)
        self.assertTrue(step_with_evidence(game, action(game, SECOND, TARGET), enabled=True)[1])
        game = position()
        step_with_evidence(game, action(game, FIRST, TARGET), enabled=False)
        self.assertFalse(hasattr(game, EVIDENCE_ATTRIBUTE))
        game, _ = prepare()
        step_with_evidence(game, (0, 0), enabled=False)
        self.assertFalse(hasattr(game, EVIDENCE_ATTRIBUTE))

    def test_pass_elimination_clears_evidence_for_the_removed_player(self):
        original = position()
        pieces = dict(original.pieces)
        del pieces[ArmPoint(1, 2, 3)]
        game = JunqiGame.from_position(original.config, pieces, current_player=0,
                                       passes_remaining=[4, 1, 4, 4])
        step_with_evidence(game, action(game, FIRST, TARGET), enabled=True)
        self.assertIn(TARGET, getattr(game, EVIDENCE_ATTRIBUTE)[0])
        for _ in range(8):
            if game.piece_at(TARGET) is None:
                break
            step_with_evidence(game, (0, 0), enabled=True)
        self.assertIsNone(game.piece_at(TARGET))
        self.assertNotIn(TARGET, getattr(game, EVIDENCE_ATTRIBUTE)[0])

    def test_penalty_and_draw_follow_separate_gae_signs(self):
        game, history = prepare()
        state = history.state_for(game)
        for draw, expected in ((False, [.01, -.01]), (True, [-.14, -.16])):
            trace = [PPOTransition(state, (0, 0), -1., 0., 0., False, -1),
                     PPOTransition(state, (0, 0), -1., 0., -.16 if draw else -.01, draw, -1,
                                   terminal_draw=draw, futile_attack_reward=-.01)]
            samples = generalized_advantages(trace, bootstrap_value=0., discount=1.,
                                            gae_lambda=1., behavior_version=0)
            for sample, value in zip(samples, expected):
                self.assertAlmostEqual(sample.value_target, value)
                self.assertAlmostEqual(sample.draw_value_target, -.15 if draw else 0.)
        metrics = RolloutMetrics()
        self.assertEqual(metrics.record_futile_attack(True, .01, learnable=False), 0)
        self.assertEqual(metrics.futile_attacks, 0)

    def test_all_collectors_apply_once_and_preserve_evidence_on_resume(self):
        for arrays, workers, groups in ((False, 1, 1), (True, 1, 1), (False, 2, 1), (True, 2, 1), (True, 2, 2)):
            for terminal in (False, True):
                with self.subTest(arrays=arrays, workers=workers, groups=groups, terminal=terminal):
                    config = replace(ModelConfig.tiny(), ppo_array_history=arrays, ppo_fixed_kv=False)
                    actor = FrozenPolicyActor(GamePolicyTransformer(config).eval(), max_batch_size=4)
                    critic = FrozenValueActor(GameValueTransformer(config), amp_dtype=None,
                                              max_batch_size=4, deferred=groups > 1)
                    layout = PieceConditionedLayoutPointerDecoder(config).eval()
                    pool = BaseGamePool(MODE, pool_size=4, max_transitions=16, max_game_plies=None, seed=37)
                    pool.fill(layout, 0)
                    for slot in pool.slots:
                        slot.game, slot.history = prepare(terminal=terminal)
                    move = action(pool.slots[0].game, SECOND, TARGET)
                    def sample(states, *, count, return_log_probs):
                        return [[move if move in state.legal_actions else (0, 0)] for state in states], [[-1.]] * len(states)
                    with patch.object(actor, 'sample', side_effect=sample):
                        samples, outcomes, metrics = collect_ppo_samples(pool, actor, critic, layout,
                            count=4, behavior_version=0, discount=1., gae_lambda=1.,
                            environment_workers=workers, pipeline_groups=groups, futile_attack_penalty=.01)
                    self.assertEqual(metrics.futile_attacks, 4)
                    self.assertAlmostEqual(metrics.futile_attack_reward_abs_sum, .04)
                    for item in samples:
                        self.assertAlmostEqual(item.value_target, -.16 if terminal else -.01, places=6)
                    if terminal:
                        self.assertTrue(outcomes)
                        self.assertTrue(all(item.reward == -.15 for item in outcomes))
                    else:
                        saved = pool.state_dict()
                        pool.load_state_dict(saved)
                        for slot in pool.slots:
                            self.assertEqual(getattr(slot.game, EVIDENCE_ATTRIBUTE)[0][TARGET], 8)

    def test_decay_uses_progress_and_smoothed_rate_never_rebounds_and_resumes(self):
        controller = FutileAttackPenalty(.01, 200, .5, start_environment_plies=20)
        self.assertEqual(controller.coefficient(20), .01)
        controller.observe(10, 100)
        controller.observe(0, 100)
        self.assertAlmostEqual(controller.coefficient(60), .0025)
        controller.observe(100, 100)
        self.assertAlmostEqual(controller.coefficient(60), .0025)
        restored = FutileAttackPenalty(.01, 200, .5)
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())
        self.assertEqual(restored.coefficient(60), controller.coefficient(60))
        for step in (100, 101, 200):
            self.assertEqual(restored.coefficient(step), 0.)

    def test_collection_splits_at_halfway_and_ddp_cannot_penalize_past_cutoff(self):
        trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
        trainer.settings = SimpleNamespace(step_budget_target=200, futile_attack_penalty=.01)
        trainer.local_anchor_batch = 32
        trainer.cumulative = {'environment_plies': 97}
        trainer.futile_attack_controller = FutileAttackPenalty(.01, 200)
        trainer.distributed = SimpleNamespace(world_size=1)
        self.assertEqual(trainer._ppo_collection_count(), 3)
        self.assertGreater(trainer._futile_attack_coefficient(), 0)
        trainer.cumulative['environment_plies'] = 100
        self.assertEqual(trainer._futile_attack_coefficient(), 0)
        trainer.cumulative['environment_plies'] = 97
        trainer.distributed.world_size = 4
        self.assertEqual(trainer._futile_attack_coefficient(), 0)

    def test_checkpoint_adoption_preserves_weights_games_optimizers_and_halfway_cutoff(self):
        from test_flag_capture_reward import FlagCaptureRewardTests
        settings = TrainingSettings.from_yaml(ROOT / 'configs/bootstrap.yaml', MODE, tiny=True, overrides={
            'device':'cpu', 'anchor_batch':16, 'policy_microbatch':4, 'base_game_pool_size':4,
            'actor_inference_batch':4, 'total_updates':1, 'max_game_plies':7, 'arena_enabled':False})
        with tempfile.TemporaryDirectory() as directory, patch.object(MetricLogger, 'start_resource_monitor'):
            old = SelfPlayTrainer(settings, run_directory=directory, auto_resume=False)
            old.train()
            before = torch.load(old.checkpoints.latest_path, weights_only=False)
            target = replace(settings, futile_attack_penalty=.01, target_environment_plies=128, total_updates=8)
            with self.assertRaisesRegex(RuntimeError, 'adopt-futile-attack-penalty'):
                SelfPlayTrainer(target, run_directory=directory)
            trainer = SelfPlayTrainer(target, run_directory=directory, adopt_futile_attack_penalty=True)
            try:
                migrated = torch.load(trainer.checkpoints.latest_path, weights_only=False)
                compare = FlagCaptureRewardTests().assert_tree_equal
                for key in ('policy','layout','critic','policy_optimizer','layout_optimizer','critic_optimizer','rng_state'):
                    compare(migrated[key], before[key])
                for key in ('base_game_pool','layout_buffer','cumulative','model_selection'):
                    compare(migrated['trainer_state'][key], before['trainer_state'][key])
                self.assertEqual(trainer._futile_attack_coefficient(), .01)
                trainer.train()
                self.assertEqual(trainer.cumulative['environment_plies'], 128)
                latest = json.loads((trainer.run_directory / 'latest_metrics.json').read_text())
                self.assertEqual(latest['training/futile_attack_penalty'], 0)
                saved = trainer.futile_attack_controller.state_dict()
            finally:
                trainer.logger.close()
            resumed = SelfPlayTrainer(target, run_directory=directory)
            try:
                self.assertEqual(resumed.futile_attack_controller.state_dict(), saved)
                self.assertEqual(resumed._futile_attack_coefficient(), 0)
            finally:
                resumed.logger.close()


if __name__ == '__main__':
    unittest.main()
