"""First-pass coverage, ACPO clipping and retained history sampling."""
import copy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from junqi.training.clipping import advantage_clip_bounds, clipped_surrogate, scheduled_policy_clip
from junqi.training.distributed import DistributedContext
from junqi.training.historical_opponents import HistoricalOpponents
from junqi.training.models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder, PolicyFeatures, ModelConfig
from junqi.training.ppo import policy_ppo_loss
from junqi.training.settings import TrainingSettings
from junqi.training.modes import TrainingMode, new_game
from junqi.training.trainer import SelfPlayTrainer
from test_historical_opponents import settings as league_settings
import test_learner_graph as graph_test
from test_ppo import training_settings


def config():
    return TrainingSettings.from_yaml(Path(__file__).parents[1]/'configs/expand_mappo_training.yaml',
                                     'four_dark', model_scale='main')


def test_production_steps_mix_and_kl_contract():
    settings = config()
    assert settings.target_environment_plies == 3_000_000_000
    assert settings.checkpoint_interval_environment_plies == 10_000_000
    assert settings.arena_interval_environment_plies == 50_000_000
    assert settings.arena_after_half_interval_environment_plies == 50_000_000
    assert settings.arena_champion_only
    assert settings.lr_epoch_completion
    assert settings.lr_kl_band_lower is settings.lr_kl_band_upper is None
    assert settings.lr_minimum_completed_epochs == 2 and settings.lr_increase_kl_threshold == .08
    assert settings.ppo_kl_probe_interval_steps == 16
    assert settings.early_stop_kl_threshold == .10
    assert settings.policy_epochs == settings.critic_epochs == 3
    assert settings.ppo_min_policy_epochs == 1 and settings.early_stop_clip_fraction == 1.
    assert not settings.lr_increase_on_full_epochs
    assert settings.lr_increase_factor == 1.05 and settings.lr_decrease_factor == .95
    assert settings.lr_stable_updates == 1 and settings.lr_cooldown_updates == 0
    assert settings.minimum_learning_rate == 1e-6 and settings.maximum_learning_rate == 1e-4
    assert math.ceil(settings.historical_start_fraction * settings.target_environment_plies) == 1_000_000_000
    assert math.ceil(settings.historical_checkpoint_start_fraction * settings.target_environment_plies) == 50_000_000
    assert settings.historical_training_fraction == .5 and settings.historical_teammate_fraction == 0
    assert settings.historical_sampling == 'strength'
    assert settings.historical_initial_loss_rate == .5


def test_new_games_share_geometry_but_keep_rule_state_independent():
    first = new_game(TrainingMode.FOUR_DARK, seed=5)
    second = new_game(TrainingMode.FOUR_DARK, seed=5)
    before = second.state_key()
    assert first._boards is second._boards
    assert first._pieces is not second._pieces
    first.step(first.legal_actions()[0])
    assert second.state_key() == before


@pytest.mark.parametrize('artifact', [None, 'checkpoints/orphan.pt', 'historical_opponents/catalog.json', 'other.txt'])
def test_only_empty_preprovisioned_storage_allows_fresh_training(tmp_path, artifact):
    settings = training_settings()
    run = settings.resolve_run_directory(tmp_path)
    for name in ('checkpoints', 'historical_opponents'):
        (run/name).mkdir(parents=True)
    if artifact:
        (run/artifact).write_text('existing state')
        with pytest.raises(RuntimeError, match='prior artifacts'):
            SelfPlayTrainer(settings, run_directory=run)
    else:
        trainer = SelfPlayTrainer(settings, run_directory=run)
        try:
            assert trainer.update == 0
            assert (run/'checkpoints/latest.pt').is_file()
        finally:
            trainer.logger.close()


@pytest.mark.parametrize('critic', [False, True])
def test_high_kl_cannot_drop_first_pass_samples(critic):
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings = SimpleNamespace(policy_epochs=3, critic_epochs=3, ppo_min_policy_epochs=1,
        target_kl=.02, early_stop_kl_multiple=5., early_stop_clip_fraction=1.)
    samples = list(range(13))
    seen = []
    trainer._ppo_optimizer_batches = lambda rows: [rows[i:i+5] for i in range(0, len(rows), 5)]
    def update(batch, **kwargs):
        seen.extend(batch)
        return {'policy/approx_kl_old': .8, 'policy/clip_fraction': .9,
                'optimizer/policy_steps': 1., 'optimizer/critic_steps': 1.}
    trainer._update_policy_batch = trainer._update_critic_batch = update
    metrics = trainer._update_ppo_minibatches(samples, critic=critic)
    assert seen == samples * (3 if critic else 1)
    if not critic:
        assert metrics['optimizer/policy_first_pass_coverage'] == 1
        assert metrics['optimizer/policy_early_stopped'] == 1


@pytest.mark.parametrize('probes,expected_epochs', [([.12], 1), ([.03, .12], 2), ([.03, .07, .09], 3)])
def test_epoch_endpoint_kl_stops_only_after_full_first_pass(probes, expected_epochs):
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings = SimpleNamespace(policy_epochs=3, critic_epochs=3, ppo_min_policy_epochs=1,
        target_kl=.02, early_stop_kl_multiple=5., early_stop_clip_fraction=1.,
        lr_increase_on_full_epochs=True)
    trainer.update = 12
    trainer.logger = SimpleNamespace(event=lambda text: None)
    samples = list(range(13))
    seen = []
    trainer._ppo_optimizer_batches = lambda rows: [rows[i:i+5] for i in range(0, len(rows), 5)]
    def update(batch, **kwargs):
        seen.extend(batch)
        # Clip is controlled by LR, not an extra early-stop condition here.
        return {'policy/approx_kl_old': .01, 'policy/clip_fraction': .3,
                'policy/entropy': 2.5, 'optimizer/policy_steps': 1.}
    trainer._update_policy_batch = update
    with patch.object(trainer, '_final_policy_diagnostics', side_effect=[{
        'policy/final_kl': kl, 'policy/final_clip_fraction': .2, 'policy/final_entropy': 2.4,
        'policy/final_probe_samples': 8., 'timing/lr_probe_seconds': .1} for kl in probes]) as probe:
        result = trainer._update_ppo_minibatches(samples, critic=False)
    assert seen == samples * expected_epochs
    assert result['optimizer/policy_first_pass_coverage'] == 1
    assert result['optimizer/policy_epochs'] == expected_epochs
    assert result['optimizer/policy_early_stopped'] == (expected_epochs < 3)
    assert probe.call_count == expected_epochs
    assert result['policy/final_kl'] == probes[-1]
    assert result[f'policy/epoch_{expected_epochs}/coverage'] == 1
    assert result[f'policy/epoch_{expected_epochs}/final_entropy'] == 2.4


def test_second_epoch_can_stop_mid_pass_after_minibatch_kl_exceeds_limit():
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings = SimpleNamespace(policy_epochs=3, critic_epochs=3, ppo_min_policy_epochs=1,
        target_kl=.02, early_stop_kl_multiple=5., early_stop_clip_fraction=1.,
        lr_increase_on_full_epochs=True)
    trainer.update = 12
    trainer.logger = SimpleNamespace(event=lambda text: None)
    samples = list(range(13))
    seen = []
    trainer._ppo_optimizer_batches = lambda rows: [rows[i:i+5] for i in range(0, len(rows), 5)]
    def update(batch, **kwargs):
        seen.extend(batch)
        return {'policy/approx_kl_old': .2 if len(seen) > 13 else .01,
                'policy/clip_fraction': .01, 'policy/entropy': 2.5, 'optimizer/policy_steps': 1.}
    trainer._update_policy_batch = update
    with patch.object(trainer, '_final_policy_diagnostics', return_value={
        'policy/final_kl': .03, 'policy/final_clip_fraction': .01,
        'policy/final_probe_samples': 8., 'timing/lr_probe_seconds': .1}):
        result = trainer._update_ppo_minibatches(samples, critic=False)
    assert seen == samples + samples[:5]
    assert result['optimizer/policy_first_pass_coverage'] == 1
    assert result['optimizer/policy_epochs'] == pytest.approx(18 / 13)
    assert result['policy/epoch_2/coverage'] == pytest.approx(5 / 13)
    assert result['optimizer/policy_stop_kl'] == 1


def test_legacy_optional_absolute_bounds_and_unclipped_tail_gradients():
    advantage = torch.tensor([3., -3., .1, -.1, 1., -1.])
    ratio = torch.tensor([1.8, .2, 1.8, .2, 1.4, .6], requires_grad=True)
    lower, upper = advantage_clip_bounds(advantage, .25, .3, .15, .1, .45, True)
    torch.testing.assert_close(lower, upper)
    assert upper[0] == upper[1] and upper[0] > upper[4] > upper[2]
    objective, outside = clipped_surrogate(ratio, advantage, lower, upper, 2.)
    (-objective.sum()).backward()
    torch.testing.assert_close(ratio.grad, torch.tensor([-3., 3., 0., 0., -1., 1.]))
    assert outside.tolist() == [0., 0., 1., 1., 0., 0.]


@pytest.mark.parametrize('environment_plies', [0, 175_000_000, 1_500_000_000, 3_000_000_000])
def test_production_acpo_bounds_are_signed_smooth_and_clip_extreme_advantages(environment_plies):
    bounds = scheduled_policy_clip(config(), environment_plies)
    assert not bounds.absolute_advantage and bounds.unclipped_advantage_threshold == 0
    advantage = torch.tensor([-100., -3., -2.0001, -2., -1.9999, -1., 0.,
                               1., 1.9999, 2., 2.0001, 3., 100.], requires_grad=True)
    lower, upper = advantage_clip_bounds(advantage, bounds.lower, bounds.upper, bounds.bonus,
                                         bounds.minimum, bounds.maximum, bounds.absolute_advantage)
    expected = torch.tensor([bounds.upper + bounds.bonus * .5 * (1 + math.erf(a / math.sqrt(2)))
                             for a in advantage.detach().tolist()]).clamp(bounds.minimum, bounds.maximum)
    torch.testing.assert_close(upper, expected)
    torch.testing.assert_close(lower, torch.full_like(lower, bounds.lower))
    assert not lower.requires_grad and not upper.requires_grad
    assert upper[1] < upper[6] < upper[-2]
    assert upper[10] - upper[8] < 1e-4
    assert upper[4] - upper[2] < 1e-4
    # Even extreme advantages must plateau after a beneficial ratio change
    # exceeds its bound. Worsening the objective must retain a corrective gradient.
    for beneficial in (True, False):
        ratio = torch.where(advantage.detach() >= 0,
                            1.8 if beneficial else .2, .2 if beneficial else 1.8).requires_grad_()
        objective, outside = clipped_surrogate(ratio, advantage, lower, upper,
                                               bounds.unclipped_advantage_threshold)
        gradient = torch.autograd.grad(-objective.sum(), ratio)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(advantage) if beneficial else -advantage.detach())
        torch.testing.assert_close(outside, torch.ones_like(outside))


def test_production_acpo_graph_and_eager_losses_and_gradients_match():
    torch.set_num_threads(1)
    torch.manual_seed(27)
    fixture = graph_test.LearnerGraphTests()
    model = GamePolicyTransformer(ModelConfig.tiny())
    reference = copy.deepcopy(model)
    samples = [replace(s, advantage=a) for s, a in zip(fixture.batch(), (3., -3., .2))]
    contexts = torch.randn(8, model.config.temporal_dim)
    graph = fixture.graph(model, False)
    graph.callable = lambda *args: contexts
    bounds = scheduled_policy_clip(config(), 0)
    assert not bounds.absolute_advantage and bounds.unclipped_advantage_threshold == 0
    assert graph.stage(model, samples, clip=.2, coefficient=.01, policy_clip=bounds,
                       opening_coefficient=.01, opening_plies=16)
    loss, metrics = graph._loss(model)
    with patch.object(reference, '_encode_full', return_value=PolicyFeatures(
            contexts[:3], torch.ones(3, 129, dtype=torch.bool))):
        eager = policy_ppo_loss(reference, samples, clip_epsilon=.2, entropy_coefficient=.01,
                                opening_entropy_coefficient=.01, entropy_opening_plies=16,
                                sequence_training=True, defer_metrics=True, policy_clip=bounds)
    torch.testing.assert_close(metrics, torch.stack(list(eager.metrics.values())), rtol=2e-6, atol=2e-7)
    assert eager.metrics['policy/unclipped_advantage_fraction'] == 0
    assert float(eager.metrics['policy/clip_lower_mean']) == pytest.approx(bounds.lower)
    loss.backward()
    eager.loss.backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        if q.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=3e-7)


def test_resume_adopts_acpo_bounds_without_restoring_old_tail_exemption(tmp_path):
    old = training_settings(ppo_adaptive_clip=True, ppo_clip_absolute_advantage=True,
                           ppo_unclipped_advantage_threshold=2., target_environment_plies=3_000_000_000)
    trainer = SelfPlayTrainer(old, run_directory=tmp_path)
    trainer.train()
    saved = torch.load(trainer.checkpoints.latest_path, map_location='cpu', weights_only=False)
    trainer.logger.close()
    current = replace(old, ppo_clip_absolute_advantage=False, ppo_unclipped_advantage_threshold=0.)
    resumed = SelfPlayTrainer(current, run_directory=tmp_path)
    try:
        assert resumed.update == saved['update']
        assert resumed.cumulative == saved['trainer_state']['cumulative']
        for name in ('policy', 'critic', 'layout', 'reference_layout',
                     'policy_optimizer', 'critic_optimizer', 'layout_optimizer'):
            torch.testing.assert_close(getattr(resumed, name).state_dict(), saved[name], rtol=0, atol=0)
        bounds = resumed._policy_clip()
        assert bounds == scheduled_policy_clip(current, resumed.cumulative['environment_plies'])
        assert not bounds.absolute_advantage and bounds.unclipped_advantage_threshold == 0
    finally:
        resumed.logger.close()


def test_strength_sampling_matches_main_formula_and_survives_resume(tmp_path):
    torch.set_num_threads(1)
    settings = league_settings(historical_sampling='strength', historical_initial_loss_rate=.5,
        historical_uniform_fraction=0., historical_checkpoint_start_fraction=.05,
        historical_training_fraction=.5)
    context = DistributedContext(0, 1, 0, torch.device('cpu'))
    league = HistoricalOpponents(settings, tmp_path, context)
    policy, layout = GamePolicyTransformer(settings.model), PieceConditionedLayoutPointerDecoder(settings.model)
    for progress in (0, 100, 200):
        league.update_progress(policy, layout, environment_plies=progress, update=progress)
    early, left, right = [e['id'] for e in league.entries]
    assert early not in league.probabilities()
    assert league.probabilities() == {left: .5, right: .5}
    league.loss_rates = {left: .2, right: .8}
    probabilities = league.probabilities()
    assert probabilities[right] / probabilities[left] == pytest.approx(math.exp(6.))
    league.active_id = left
    slot = SimpleNamespace(opponent_id=left, teammate_id=None)
    league.record_result(slot, 0.)
    assert league.loss_rates[left] == .2
    league.record_result(slot, -1.)
    assert league.loss_rates[left] == pytest.approx(.208)
    saved = league.state_dict()
    restored = HistoricalOpponents(settings, tmp_path, context)
    restored.load_state_dict(saved)
    assert restored.probabilities() == league.probabilities()
    assert restored._choose_cohort() == league._choose_cohort()
    assert restored.role_targets() == dict(self_play=.5, historical_opponents=.5,
                                          historical_teammate=0., historical_both=0.)
    bad = copy.deepcopy(saved)
    bad['loss_rates'][left] = float('nan')
    with pytest.raises(ValueError, match='loss rate'):
        restored.load_state_dict(bad)
    league.close()
    restored.close()


def test_all_periodic_model_versions_retained_before_opponent_admission(tmp_path):
    settings = league_settings(historical_sampling='strength', historical_initial_loss_rate=.5,
        historical_uniform_fraction=0., historical_checkpoint_start_fraction=.05,
        checkpoint_interval_environment_plies=10)
    league = HistoricalOpponents(settings, tmp_path, DistributedContext(0, 1, 0, torch.device('cpu')))
    policy, layout = GamePolicyTransformer(settings.model), PieceConditionedLayoutPointerDecoder(settings.model)
    league.update_progress(policy, layout, environment_plies=0, update=0)
    for steps in (10, 20, 30, 40, 50):
        league.update_progress(policy, layout, environment_plies=steps, update=steps)
        assert league.archive_checkpoint(policy, layout, environment_plies=steps, update=steps) is not None
        if steps < 50:
            assert league.probabilities() == {}
    saved_steps = [entry['environment_plies'] for entry in league.entries]
    assert saved_steps == [0, 10, 20, 30, 40, 50]
    assert all((league.directory/entry['file']).is_file() for entry in league.entries)
    assert list(league.probabilities()) == [league.entries[-1]['id']]
    league.close()
