"""Fixed ACPO widening, legacy compatibility, graph replay and resume state."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from unittest.mock import patch

import pytest
import torch

from junqi.training.clipping import DEFAULT_CLIP_SCHEDULE, advantage_clip_bounds, clipped_surrogate, scheduled_policy_clip
from junqi.training.learner_graph import PackedLearnerGraph
from junqi.training.models import GamePolicyTransformer, ModelConfig, PolicyFeatures
from junqi.training.ppo import policy_ppo_loss
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
import test_learner_graph as graph_fixtures
from test_ppo import training_settings


def production_settings():
    return TrainingSettings.from_yaml(Path(__file__).parents[1] / 'configs/expand_mappo_training.yaml',
                                      'four_dark', model_scale='main')


@pytest.mark.parametrize('steps', [0, 536_346_624, 1_000_000_000, 1_500_000_000,
                                  2_250_000_000, 3_000_000_000])
def test_production_acpo_bounds_are_fixed_at_every_training_stage(steps):
    settings = production_settings()
    assert settings.ppo_clip_mode == 'acpo' and settings.ppo_clip_delta == .05
    assert settings.ppo_clip_schedule is settings.ppo_clip_minimum is settings.ppo_clip_maximum is None
    bounds = scheduled_policy_clip(settings, steps)
    assert bounds.lower == bounds.upper == .2
    assert bounds.minimum is bounds.maximum is None
    assert bounds.bonus == .05
    assert bounds.metrics()['policy/clip_delta'] == .05
    assert bounds.metrics()['policy/clip_progress_scheduled'] == 0
    assert bounds.metrics()['policy/clip_hard_bounds_enabled'] == 0
    assert 'policy/clip_minimum' not in bounds.metrics() and 'policy/clip_maximum' not in bounds.metrics()
    assert not bounds.absolute_advantage and bounds.unclipped_advantage_threshold == 0
    advantage = torch.tensor([-100., -2., 0., 2., 100.])
    lower, upper = advantage_clip_bounds(advantage, bounds.lower, bounds.upper, bounds.bonus,
                                         bounds.minimum, bounds.maximum, bounds.absolute_advantage)
    expected = torch.tensor([bounds.upper + .05 * .5 * (1 + math.erf(a / math.sqrt(2)))
                             for a in advantage.tolist()])
    torch.testing.assert_close(upper, expected)
    torch.testing.assert_close(lower, torch.full_like(lower, bounds.lower))
    assert float(upper[2]) == pytest.approx(bounds.upper + .025)
    assert float(upper[-1]) == pytest.approx(bounds.upper + .05)


def test_legacy_scaling_is_preserved_and_explicit_zero_delta_wins():
    settings = training_settings(ppo_adaptive_clip=True, target_environment_plies=3_000_000_000)
    old = scheduled_policy_clip(settings, 0)
    assert old.bonus == .15
    zero = scheduled_policy_clip(replace(settings, ppo_clip_delta=0.), 0)
    assert zero.bonus == 0.
    fixed = scheduled_policy_clip(replace(settings, ppo_clip_delta=.05), 0)
    assert fixed.bonus == .05
    assert zero.lower == fixed.lower == old.lower
    assert zero.upper == fixed.upper == old.upper


@pytest.mark.parametrize('upper', [.04, .6])
def test_acpo_has_no_legacy_hard_floor_or_ceiling(upper):
    settings = replace(production_settings(), ppo_clip_lower=.05, ppo_clip_upper=upper)
    settings.validate()
    bounds = scheduled_policy_clip(settings, 3_000_000_000)
    advantage = torch.tensor([-100., 0., 100.])
    low, high = advantage_clip_bounds(advantage, bounds.lower, bounds.upper, bounds.bonus,
                                      bounds.minimum, bounds.maximum)
    torch.testing.assert_close(low, torch.full_like(low, .05))
    torch.testing.assert_close(high, torch.tensor([upper, upper + .025, upper + .05]))


@pytest.mark.parametrize('extension', [dict(ppo_clip_minimum=.1), dict(ppo_clip_maximum=.45),
                                      dict(ppo_clip_schedule=DEFAULT_CLIP_SCHEDULE),
                                      dict(ppo_clip_absolute_advantage=True),
                                      dict(ppo_unclipped_advantage_threshold=2.), dict(ppo_clip_delta=None)])
def test_acpo_rejects_project_specific_clipping_extensions(extension):
    with pytest.raises(ValueError, match='ACPO clipping'):
        replace(production_settings(), **extension).validate()


@pytest.mark.parametrize('delta', [-.01, 1.01, float('nan'), float('inf'), True, '0.05'])
def test_invalid_delta_is_rejected(delta):
    with pytest.raises(ValueError, match='clip delta'):
        replace(production_settings(), ppo_clip_delta=delta).validate()


def test_fixed_delta_clips_previously_wide_positive_update_but_keeps_negative_rule():
    settings = training_settings(ppo_adaptive_clip=True, target_environment_plies=3_000_000_000)
    advantage = torch.tensor([3., -3., -3.])
    gradients = []
    for delta in (None, .05):
        bounds = scheduled_policy_clip(replace(settings, ppo_clip_delta=delta), 536_346_624)
        lower, upper = advantage_clip_bounds(advantage, bounds.lower, bounds.upper, bounds.bonus,
                                             bounds.minimum, bounds.maximum, bounds.absolute_advantage)
        ratio = torch.tensor([1.35, .7, 1.35], requires_grad=True)
        objective, _ = clipped_surrogate(ratio, advantage, lower, upper)
        gradients.append(torch.autograd.grad(-objective.sum(), ratio)[0])
    torch.testing.assert_close(gradients[0], torch.tensor([-3., 0., 3.]))
    torch.testing.assert_close(gradients[1], torch.tensor([0., 0., 3.]))


def test_resume_adopts_fixed_delta_and_preserves_learning_state(tmp_path):
    torch.set_num_threads(1)
    settings = training_settings(ppo_adaptive_clip=True, target_environment_plies=3_000_000_000)
    trainer = SelfPlayTrainer(settings, run_directory=tmp_path)
    trainer.train()
    saved = torch.load(trainer.checkpoints.latest_path, map_location='cpu', weights_only=False)
    trainer.logger.close()
    current = replace(settings, ppo_clip_mode='acpo', ppo_clip_lower=.2, ppo_clip_upper=.2,
                      ppo_clip_delta=.05, ppo_clip_schedule=None, ppo_clip_minimum=None, ppo_clip_maximum=None)
    resumed = SelfPlayTrainer(current, run_directory=tmp_path)
    try:
        assert resumed.update == saved['update']
        assert resumed.cumulative == saved['trainer_state']['cumulative']
        for name in ('policy', 'critic', 'layout', 'reference_layout',
                     'policy_optimizer', 'critic_optimizer', 'layout_optimizer'):
            torch.testing.assert_close(getattr(resumed, name).state_dict(), saved[name], rtol=0, atol=0)
        assert random.getstate() == saved['rng_state']['python']
        torch.testing.assert_close(torch.get_rng_state(), saved['rng_state']['torch_cpu'], rtol=0, atol=0)
        assert resumed.pool.state_dict() == saved['trainer_state']['base_game_pool']
        assert resumed._policy_clip().bonus == .05
        assert resumed._policy_clip().lower == resumed._policy_clip().upper == .2
        assert resumed._policy_clip().minimum is resumed._policy_clip().maximum is None
        resolved = json.loads((resumed.run_directory / 'resolved_config.json').read_text())
        assert resolved['ppo_clip_delta'] == .05
        assert resolved['ppo_clip_mode'] == 'acpo' and resolved['ppo_clip_schedule'] is None
    finally:
        resumed.logger.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA capture verification')
def test_cuda_replay_adopts_fixed_delta_and_matches_eager_loss_and_gradients():
    torch.manual_seed(73)
    model = GamePolicyTransformer(ModelConfig.tiny()).cuda()
    reference = copy.deepcopy(model)
    graph = PackedLearnerGraph(model, critic=False, amp_dtype=torch.bfloat16, tokens=64, sources=256)
    graph.SEQUENCES, graph.SAMPLES, graph.ACTIONS = 8, 8, 2048
    samples = [replace(s, advantage=a) for s, a in zip(graph_fixtures.LearnerGraphTests().batch(), (3., -3., .2))]
    contexts = torch.randn(8, model.config.temporal_dim, device='cuda')
    features = PolicyFeatures(contexts[:3], torch.ones(3, 129, dtype=torch.bool, device='cuda'))
    upper_means = []
    with patch.object(model, '_packed_temporal_forward', lambda *args: contexts):
        current = production_settings()
        legacy = replace(current, ppo_clip_mode='scheduled', ppo_clip_delta=None,
                         ppo_clip_minimum=.1, ppo_clip_maximum=.45, ppo_clip_schedule=DEFAULT_CLIP_SCHEDULE)
        for settings in (legacy, current, replace(current, ppo_clip_lower=.05, ppo_clip_upper=.04),
                         replace(current, ppo_clip_lower=.05, ppo_clip_upper=.6)):
            bounds = scheduled_policy_clip(settings, 536_346_624)
            assert graph.stage(model, samples, clip=.2, coefficient=.005, source_coefficient=.005,
                               opening_coefficient=.005, opening_plies=16, policy_clip=bounds)
            actual = {key: value.detach().clone() for key, value in graph.run(model).items()}
            reference.zero_grad(set_to_none=True)
            with patch.object(reference, '_encode_full', return_value=features), torch.autocast('cuda', dtype=torch.bfloat16):
                expected = policy_ppo_loss(reference, samples, clip_epsilon=.2, entropy_coefficient=.005,
                                           source_entropy_coefficient=.005, opening_entropy_coefficient=.005,
                                           entropy_opening_plies=16, sequence_training=True,
                                           defer_metrics=True, policy_clip=bounds)
            expected.loss.backward()
            for key in actual:
                torch.testing.assert_close(actual[key], expected.metrics[key], rtol=2e-5, atol=2e-6, msg=key)
            for parameter, wanted in zip(model.parameters(), reference.parameters(), strict=True):
                if wanted.grad is not None:
                    assert torch.isfinite(parameter.grad).all()
                    torch.testing.assert_close(parameter.grad, wanted.grad, rtol=2e-4, atol=2e-6)
            upper_means.append(float(actual['policy/clip_upper_mean']))
    assert upper_means[0] - upper_means[1] > .02
    assert .04 < upper_means[2] < .09 and .6 < upper_means[3] < .65
    assert graph.captures == 1 and graph.replays == 4
