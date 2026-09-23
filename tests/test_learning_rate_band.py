"""KL-band hysteresis, checkpoint continuity, and the matching PPO guard."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from junqi.training.learning_rate import BoundedLearningRate
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from test_ppo import training_settings


def controller(**overrides):
    return BoundedLearningRate(**dict(minimum=5e-6, maximum=1e-4, target_kl=.02,
        kl_band_lower=.08, kl_band_upper=.14, increase_factor=1.05, decrease_factor=.95,
        stable_updates=3, cooldown_updates=3, scale=.1) | overrides)


def observe(c, kl):
    return c.observe(scheduled=1e-4, kl=kl, early_stopped=kl is not None and kl > .14)


@pytest.mark.parametrize('kl,factor,reason', [(.07, 1.05, 'low_kl'), (.15, .95, 'high_kl')])
def test_symmetric_confirmation_and_three_full_observation_updates(kl, factor, reason):
    c = controller()
    for count in (1, 2):
        assert observe(c, kl) == (1e-5, 0)
        assert c.stable_count == count
    rate, direction = observe(c, kl)
    assert rate == pytest.approx(1e-5 * factor)
    assert direction == (1 if factor > 1 else -1)
    assert c.reason == reason and c.cooldown == 3
    for remaining in (2, 1, 0):
        assert observe(c, kl) == (rate, 0)
        assert c.reason == 'cooldown'
        assert c.cooldown == remaining and c.stable_count == 0
    for count in (1, 2):
        assert observe(c, kl) == (rate, 0)
        assert c.stable_count == count
    assert observe(c, kl)[0] == pytest.approx(rate * factor)


@pytest.mark.parametrize('kl', [.08, .11, .14])
def test_inclusive_band_holds_and_clears_pending_direction(kl):
    c = controller()
    observe(c, .07)
    observe(c, .07)
    assert observe(c, kl) == (1e-5, 0)
    assert c.reason == 'hold' and c.stable_count == c.stable_direction == 0
    assert observe(c, .07) == (1e-5, 0)
    assert c.stable_count == 1


def test_direction_reversal_and_missing_samples_break_consecutive_evidence():
    c = controller()
    for kl in [.07, .07, .15, .15, .07, .15, None, .15, .15]:
        assert observe(c, kl) == (1e-5, 0)
    assert c.stable_count == 2 and c.stable_direction == -1
    assert observe(c, .15)[0] == pytest.approx(.95e-5)


def test_band_state_resume_preserves_streak_cooldown_and_next_rate():
    c = controller()
    for kl in [.15, .15]:
        observe(c, kl)
    for kl in [.15, .07, .07, .07, .07, .07, .07, .11, .15]:
        clone = controller()
        clone.load_state_dict(c.state_dict(), legacy_scale=1, scheduled=1e-4)
        assert observe(c, kl) == observe(clone, kl)
        assert c.state_dict() == clone.state_dict()


def test_legacy_peak_history_is_reset_when_adopting_mean_band():
    c = controller()
    c.load_state_dict(dict(version=1, scale=.1, kl_ema=.8, stable_count=2,
        cooldown=2, observations=197), legacy_scale=1, scheduled=1e-4)
    assert c.rate(1e-4) == 1e-5
    assert c.kl_ema is None and c.stable_count == c.stable_direction == c.cooldown == 0
    assert c.observations == 197
    assert observe(c, .07) == (1e-5, 0)


def test_band_bounds_have_no_hidden_multiplier_accumulation():
    c = controller(scale=.05, cooldown_updates=0)
    for _ in range(30):
        assert observe(c, .2) == (5e-6, 0)
    assert c.reason == 'lower_bound'
    for _ in range(3):
        rate, direction = observe(c, .01)
    assert rate == pytest.approx(5.25e-6) and direction == 1


@pytest.mark.parametrize('changes', [dict(lr_kl_band_lower=.14, lr_kl_band_upper=.08),
    dict(lr_kl_band_lower=.08), dict(lr_kl_band_upper=.14),
    dict(lr_kl_band_lower=True, lr_kl_band_upper=.14),
    dict(lr_kl_band_lower=.08, lr_kl_band_upper=float('nan')),
    dict(lr_kl_band_lower=.08, lr_kl_band_upper=.14, lr_increase_on_full_epochs=True),
    dict(lr_kl_band_lower=.08, lr_kl_band_upper=.14, adaptive_learning_rate=False)])
def test_invalid_band_settings_are_rejected(changes):
    with pytest.raises(ValueError, match='KL band'):
        replace(training_settings(), **changes).validate()


@pytest.mark.parametrize('probes,expected_epochs', [([.15], 1), ([.09, .15], 2),
    ([.08, .12, .14], 3)])
def test_band_early_stop_uses_epoch_probe_and_preserves_first_pass(probes, expected_epochs):
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings = SimpleNamespace(policy_epochs=3, critic_epochs=3, ppo_min_policy_epochs=1,
        target_kl=.02, early_stop_kl_multiple=5, early_stop_clip_fraction=1,
        lr_increase_on_full_epochs=False, lr_kl_band_lower=.08, lr_kl_band_upper=.14)
    trainer.update = 0
    trainer.logger = SimpleNamespace(event=lambda message: None)
    samples, seen = list(range(13)), []
    trainer._ppo_optimizer_batches = lambda rows: [rows[i:i+5] for i in range(0, len(rows), 5)]
    def update(batch, **kwargs):
        seen.extend(batch)
        return {'policy/approx_kl_old': .2, 'policy/clip_fraction': .1,
                'policy/entropy': 2., 'optimizer/policy_steps': 1.}
    trainer._update_policy_batch = update
    with patch.object(trainer, '_final_policy_diagnostics', side_effect=[{
        'policy/final_kl': kl, 'policy/final_clip_fraction': .1,
        'policy/final_probe_samples': 8., 'timing/lr_probe_seconds': .1} for kl in probes]):
        result = trainer._update_ppo_minibatches(samples, critic=False)
    assert seen == samples * expected_epochs
    assert result['optimizer/policy_first_pass_coverage'] == 1
    assert result['optimizer/policy_epochs'] == expected_epochs
    assert result['optimizer/policy_stop_kl'] == (probes[-1] > .14)
    assert result['policy/minibatch_kl_max'] == .2


def test_band_trainer_ignores_peak_for_lr_and_reuses_existing_probe(tmp_path):
    torch.set_num_threads(1)
    trainer = SelfPlayTrainer(training_settings(lr_kl_band_lower=.08, lr_kl_band_upper=.14,
        lr_stable_updates=3, lr_cooldown_updates=3, lr_increase_factor=1.05,
        lr_decrease_factor=.95, minimum_learning_rate=5e-6, maximum_learning_rate=1e-4),
        run_directory=tmp_path)
    try:
        trainer.lr_controller.scale = .1
        assert trainer._set_policy_lr(1) == 1e-5
        with patch.object(trainer, '_final_policy_diagnostics', side_effect=AssertionError('duplicate probe')):
            for _ in range(3):
                result = trainer._finish_learning_rate([], {
                    'optimizer/policy_steps': 3., 'optimizer/policy_epochs': 3.,
                    'policy/final_probe_samples': 8., 'policy/final_kl': .07,
                    'policy/minibatch_kl_max': .3, 'policy/epoch_kl_max': .07})
        assert result['optimizer/policy_lr_next'] == pytest.approx(1.05e-5)
        assert result['policy/lr_kl_feedback'] == .07
        assert result['optimizer/lr_cooldown'] == 3
        assert trainer._learning_rate(1_000_000) == pytest.approx(1.05e-5)
    finally:
        trainer.logger.close()


def test_adoption_preserves_actual_lr_then_resume_preserves_band_next_lr(tmp_path):
    torch.set_num_threads(1)
    old = training_settings(lr_increase_on_full_epochs=True, minimum_learning_rate=5e-6,
        maximum_learning_rate=1e-4, lr_increase_factor=1.5, lr_decrease_factor=.6,
        lr_stable_updates=1, lr_cooldown_updates=0, ppo_minibatch_samples=2, lr_probe_samples=8)
    trainer = SelfPlayTrainer(old, run_directory=tmp_path)
    trainer.train()
    trainer.policy_optimizer.param_groups[0]['lr'] = 1e-5
    trainer.critic_optimizer.param_groups[0]['lr'] = 1e-5
    trainer.lr_controller.scale = .15  # Old next-LR decision must not be applied.
    trainer.save_checkpoint(reason='test_band_adoption', archive=False)
    original = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    new = replace(old, lr_increase_on_full_epochs=False, lr_kl_band_lower=.08,
        lr_kl_band_upper=.14, lr_stable_updates=3, lr_cooldown_updates=3,
        lr_increase_factor=1.05, lr_decrease_factor=.95)
    resumed = SelfPlayTrainer(new, run_directory=tmp_path)
    try:
        assert resumed.update == 1
        assert resumed._set_policy_lr(2) == 1e-5
        assert all(torch.equal(v, resumed.policy.state_dict()[k]) for k, v in original.items())
        for _ in range(3):
            observe(resumed.lr_controller, .15)
        resumed.save_checkpoint(reason='test_band_state', archive=False)
        expected = resumed.lr_controller.state_dict()
    finally:
        resumed.logger.close()
    again = SelfPlayTrainer(new, run_directory=tmp_path)
    try:
        assert again.lr_controller.state_dict() == expected
        assert again._set_policy_lr(2) == pytest.approx(.95e-5)
    finally:
        again.logger.close()


def test_disabling_adaptation_from_expand_yaml_clears_band():
    settings = TrainingSettings.from_yaml('configs/expand_mappo_training.yaml', 'four_dark',
        model_scale='main', overrides={'adaptive_learning_rate': False})
    assert settings.lr_kl_band_lower is settings.lr_kl_band_upper is None


@pytest.mark.parametrize('value', [0, -.1, True, float('nan'), float('inf'), '0.1'])
def test_invalid_explicit_kl_guard_is_rejected(value):
    with pytest.raises(ValueError, match='KL early-stop threshold'):
        replace(training_settings(), early_stop_kl_threshold=value).validate()


def test_legacy_band_separates_epoch_guard_from_lr_decrease_threshold():
    settings = replace(TrainingSettings.from_yaml('configs/expand_mappo_training.yaml', 'four_dark',
        model_scale='main'), lr_epoch_completion=False, lr_kl_band_lower=.08, lr_kl_band_upper=.12)
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings = settings
    assert trainer._policy_kl_stop_threshold() == .10
    assert (settings.lr_kl_band_lower, settings.lr_kl_band_upper) == (.08, .12)
    c = controller(kl_band_upper=.12)
    for kl in [.11, .12, .11, .12]:
        assert c.observe(scheduled=1e-4, kl=kl, early_stopped=True) == (1e-5, 0)
        assert c.reason == 'hold' and c.stable_count == 0
    for count in (1, 2):
        assert c.observe(scheduled=1e-4, kl=.120001, early_stopped=True) == (1e-5, 0)
        assert c.stable_count == count
    assert c.observe(scheduled=1e-4, kl=.120001, early_stopped=True)[0] == pytest.approx(.95e-5)
    assert c.cooldown == 3


@pytest.mark.parametrize('probes,expected_epochs', [([.100001], 1), ([.10, .11], 2),
    ([.08, .09, .10], 3)])
def test_independent_guard_checks_complete_epochs_and_preserves_critic(probes, expected_epochs):
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings = SimpleNamespace(policy_epochs=3, critic_epochs=3, ppo_min_policy_epochs=1,
        target_kl=.02, early_stop_kl_multiple=5, early_stop_clip_fraction=1,
        early_stop_kl_threshold=.10, lr_increase_on_full_epochs=False,
        lr_kl_band_lower=.08, lr_kl_band_upper=.12)
    trainer.update = 0
    trainer.logger = SimpleNamespace(event=lambda message: None)
    samples, seen = list(range(13)), []
    trainer._ppo_optimizer_batches = lambda rows: [rows[i:i+5] for i in range(0, len(rows), 5)]
    def update(batch, **kwargs):
        seen.extend(batch)
        return {'policy/approx_kl_old': .3, 'policy/clip_fraction': .1,
                'policy/entropy': 2., 'optimizer/policy_steps': 1.}
    trainer._update_policy_batch = update
    trainer._update_critic_batch = lambda batch, **kw: {'optimizer/critic_steps': 1.}
    with patch.object(trainer, '_final_policy_diagnostics', side_effect=[{
        'policy/final_kl': kl, 'policy/final_clip_fraction': .1,
        'policy/final_probe_samples': 8., 'timing/lr_probe_seconds': .1} for kl in probes]):
        result = trainer._update_ppo_minibatches(samples, critic=False)
        critic = trainer._update_ppo_minibatches(samples, critic=True)
    assert seen == samples * expected_epochs
    assert result['optimizer/policy_first_pass_coverage'] == 1
    assert result['optimizer/policy_epochs'] == expected_epochs
    assert result['optimizer/policy_stop_kl'] == (probes[-1] > .10)
    assert critic['optimizer/critic_epochs'] == 3


def test_band_threshold_edit_resets_streak_but_preserves_rate_ema_and_cooldown():
    old = controller()
    for _ in range(3):
        observe(old, .15)
    assert old.cooldown == 3
    new = controller(kl_band_upper=.12)
    new.load_state_dict(old.state_dict(), legacy_scale=1, scheduled=1e-4)
    assert new.rate(1e-4) == old.rate(1e-4)
    assert new.cooldown == old.cooldown and new.kl_ema == old.kl_ema
    for remaining in (2, 1, 0):
        new.observe(scheduled=1e-4, kl=.13, early_stopped=True)
        assert new.cooldown == remaining and new.stable_count == 0
    old.cooldown, old.stable_count, old.stable_direction = 0, 2, -1
    new.load_state_dict(old.state_dict(), legacy_scale=1, scheduled=1e-4)
    assert new.stable_count == new.stable_direction == 0
