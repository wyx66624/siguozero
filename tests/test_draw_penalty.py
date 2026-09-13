from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from junqi.training.encoding import ActionFeatures, GameHistory, PolicyState
from junqi.training.history_arrays import ArrayHistory
from junqi.training.inference import InferenceEngine
from junqi.training.models import GamePolicyTransformer, ModelConfig, PublicActionEncoder, collate_policy_states
from junqi.training.modes import TrainingMode, new_game
from junqi.training.packed_observation import observation_rows
from junqi.training.ppo import FrozenValueActor, PPOTransition, collect_ppo_samples, generalized_advantages
from junqi.training.rollout import FrozenPolicyActor
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


CONFIG = Path(__file__).parents[1] / 'configs/bootstrap.yaml'


def settings(**overrides):
    return TrainingSettings.from_yaml(CONFIG, 'four_dark', tiny=True, overrides={
        'device': 'cpu', 'anchor_batch': 16, 'policy_microbatch': 2,
        'base_game_pool_size': 4, 'actor_inference_batch': 4, 'total_updates': 1, **overrides})


class DrawPenaltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_draw_gae_is_negative_for_every_actor_with_resets_and_bootstrap(self):
        game = new_game('four_dark', seed=2)
        state = GameHistory.initialize(game, TrainingMode.FOUR_DARK).state_for(game)
        def t(reward=0., terminal=False, sign=-1, old=0., common=0., draw=False):
            return PPOTransition(state, state.legal_actions[0], -1., old, reward, terminal, sign,
                                 old_draw_value=common, terminal_draw=draw)
        trace = [t(), t(sign=1), t(), t(-.15, True, draw=True)]
        samples = generalized_advantages(trace, bootstrap_value=999., bootstrap_draw_value=-999.,
                                        discount=1., gae_lambda=1., behavior_version=9)
        for s in samples:
            self.assertAlmostEqual(s.value_target, -.15)
            self.assertAlmostEqual(s.draw_value_target, -.15)
            self.assertLess(s.advantage, 0.)
        discounted = generalized_advantages(trace, bootstrap_value=999., discount=.9,
                                            gae_lambda=.95, behavior_version=9)
        for i, s in enumerate(discounted):
            self.assertAlmostEqual(s.value_target, -.15 * (.9 * .95) ** (3 - i))
        mixed = generalized_advantages([*trace, t(1., True)], bootstrap_value=999.,
                                       discount=1., gae_lambda=1., behavior_version=9)
        self.assertEqual([round(s.value_target, 6) for s in mixed], [-.15] * 4 + [1.])
        for sign, expected in ((-1, -.45), (1, .27)):
            cut = generalized_advantages([t(sign=sign, old=.2, common=-.04)],
                bootstrap_value=.3, bootstrap_draw_value=-.1, discount=.9, gae_lambda=.95, behavior_version=0)
            self.assertAlmostEqual(cut[0].value_target, expected)
            self.assertAlmostEqual(cut[0].draw_value_target, -.09)

    def test_all_collectors_penalize_draws_but_count_them_as_draws(self):
        for workers, deferred, groups in ((1, False, 1), (1, True, 1), (2, False, 1),
                                           (2, True, 1), (2, True, 2)):
            with self.subTest(workers=workers, deferred=deferred, groups=groups), tempfile.TemporaryDirectory() as directory:
                trainer = SelfPlayTrainer(settings(), run_directory=directory, auto_resume=False)
                try:
                    actor = FrozenPolicyActor(trainer.policy.eval(), max_batch_size=4)
                    critic = FrozenValueActor(trainer.critic, amp_dtype=None, max_batch_size=4, deferred=deferred)
                    samples, outcomes, metrics = collect_ppo_samples(trainer.pool, actor, critic,
                        trainer.layout.eval(), count=16, behavior_version=0, discount=1., gae_lambda=1.,
                        environment_workers=workers, pipeline_groups=groups, draw_reward=-.15)
                    self.assertEqual((len(samples), metrics.environment_plies), (16, 16))
                    self.assertEqual((metrics.wins, metrics.draws, metrics.losses), (0, 4, 0))
                    self.assertEqual(len(outcomes), 16)
                    self.assertTrue(all(s.value_target == -.15 and s.draw_value_target == -.15 for s in samples))
                    self.assertTrue(all(o.reward == -.15 for o in outcomes))
                    trainer._update_critic(samples)
                    self.assertGreater(trainer.critic.draw_value_head.weight.count_nonzero(), 0)
                finally:
                    trainer.logger.close()

    def test_countdown_all_modes_replay_arrays_packed_and_model_paths(self):
        for mode in TrainingMode:
            game = new_game(mode, seed=7)
            history = GameHistory.initialize(game, mode, max_transitions=16)
            game.step(game.legal_actions()[0])
            history.append_after_step(game)
            for count, remaining in ((0, 70), (1, 69), (60, 10), (69, 1), (70, 0)):
                game.no_interaction_plies = count
                for viewer, player in enumerate(history.players):
                    record = replace(player.records[-1], no_interaction_plies=count)
                    states = [PolicyState(mode, (record,), ((0, 1),))]
                    batch = collate_policy_states(states, device='cpu')
                    self.assertEqual(float(batch.action_fields[0, 0, -1]), remaining)
                    model = GamePolicyTransformer(ModelConfig.tiny()).eval()
                    encoded = []
                    hook = model.action_encoder.register_forward_hook(lambda _module, _args, output: encoded.append(output))
                    model._record_temporal_tokens([record], mode, torch.zeros(1, model.config.board_dim), 0)
                    hook.remove()
                    torch.testing.assert_close(encoded[0], model.action_encoder(batch.action_fields, batch.action_present))
                    arrays = ArrayHistory(mode, (record,), 16, (100, viewer))
                    array_state = PolicyState(mode, arrays.view(), ((0, 1),))
                    array_batch = collate_policy_states([array_state], device='cpu')
                    torch.testing.assert_close(batch.action_fields, array_batch.action_fields)
                # Production workers store the same counter; model expands it without duplicating replay data.
                rows = observation_rows(game, mode)
                self.assertTrue((rows[:, -8] == count).all())
        encoder = PublicActionEncoder(8)
        old_weight = torch.randn(8, 5)
        bias = torch.randn(8)
        encoder.load_state_dict({'projection.weight': old_weight, 'projection.bias': bias})
        values = torch.randn(3, 6)
        torch.testing.assert_close(encoder(values, torch.ones(3, dtype=torch.bool)),
                                  torch.nn.functional.linear(values[:, :5], old_weight, bias))
        self.assertEqual(encoder.projection.weight[:, 5].count_nonzero(), 0)

    def test_v7_migration_preserves_weights_moments_progress_and_histories(self):
        with tempfile.TemporaryDirectory() as directory:
            old_settings = settings(draw_reward=0., max_game_plies=3)
            original = SelfPlayTrainer(old_settings, run_directory=directory, auto_resume=False)
            original.train()
            path = original.checkpoints.latest_path
            payload = torch.load(path, weights_only=False)
            payload['format_version'] = 7
            for kind, model in (('policy', original.policy), ('critic', original.critic)):
                weights = payload[kind]
                weights['action_encoder.projection.weight'] = weights['action_encoder.projection.weight'][:, :5].clone()
                optimizer = payload[kind + '_optimizer']
                names = [name for name, _ in model.named_parameters()]
                ids = optimizer['param_groups'][0]['params']
                state = optimizer['state'][ids[names.index('action_encoder.projection.weight')]]
                for moment in ('exp_avg', 'exp_avg_sq'):
                    state[moment] = state[moment][:, :5].clone()
                if kind == 'critic':
                    for name in ('draw_value_head.weight', 'draw_value_head.bias'):
                        weights.pop(name)
                    for parameter_id in ids[-2:]:
                        optimizer['state'].pop(parameter_id, None)
                    optimizer['param_groups'][0]['params'] = ids[:-2]
            torch.save(payload, path)
            expected_pool = copy.deepcopy(payload['trainer_state']['base_game_pool'])
            target = replace(old_settings, draw_reward=-.15, total_updates=2)
            with self.assertRaisesRegex(RuntimeError, 'adopt-draw-penalty'):
                SelfPlayTrainer(target, run_directory=directory, adopt_pass_rule=True)
            # Frozen old inference snapshots are still usable without rewriting their files.
            engine = InferenceEngine.from_checkpoint(path, device='cpu')
            self.assertEqual(engine.policy.action_encoder.projection.weight[:, 5].count_nonzero(), 0)
            resumed = SelfPlayTrainer(target, run_directory=directory, adopt_draw_penalty=True, adopt_pass_rule=True)
            try:
                self.assertEqual(resumed.update, 1)
                self.assertEqual(resumed.cumulative, payload['trainer_state']['cumulative'])
                self.assertEqual(resumed.pool.state_dict(), expected_pool)
                self.assertEqual(resumed.policy_lr_scale, original.policy_lr_scale)
                for kind, model, optimizer in (('policy', resumed.policy, resumed.policy_optimizer),
                                               ('critic', resumed.critic, resumed.critic_optimizer)):
                    for name, weight in payload[kind].items():
                        actual = model.state_dict()[name]
                        if name == 'action_encoder.projection.weight':
                            actual = actual[:, :5]
                        torch.testing.assert_close(actual, weight, rtol=0, atol=0)
                    parameter = model.action_encoder.projection.weight
                    ids = payload[kind + '_optimizer']['param_groups'][0]['params']
                    names = [name for name, _ in model.named_parameters()]
                    saved = payload[kind + '_optimizer']['state'][ids[names.index('action_encoder.projection.weight')]]
                    for moment in ('exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(optimizer.state[parameter][moment][:, :5], saved[moment], rtol=0, atol=0)
                        self.assertEqual(optimizer.state[parameter][moment][:, 5].count_nonzero(), 0)
                self.assertEqual(resumed.critic.draw_value_head.weight.count_nonzero(), 0)
                resumed.train()
                self.assertEqual(resumed.update, 2)
                saved = torch.load(path, weights_only=False)
                self.assertEqual(saved['format_version'], 9)
                self.assertEqual(saved['config']['draw_reward'], -.15)
                self.assertEqual(saved['trainer_state']['draw_objective_migration']['update'], 1)
            finally:
                resumed.logger.close()


if __name__ == '__main__':
    unittest.main()
