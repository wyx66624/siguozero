"""Coverage-based LR decisions and post-update fixed-probe partial epochs."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from junqi.training.learning_rate import BoundedLearningRate
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from test_ppo import training_settings


@pytest.mark.parametrize('threshold', [.05, .08, .09])
@pytest.mark.parametrize('epochs,kl,expected', [
    (1., .15, .95), (1.5, .101, .95), (1.999, .08, 1.),
    (1.5, .1, 1.), (2., .1, 1.), (2., .100001, .95),
    (2., .101, .95), (2.000001, .101, 1.), (2.5, .11, 1.), (3., 'equal', 1.),
    (3., 'above', 1.), (3., 'below', 1.05), (3., .04, 1.05)])
def test_immediate_decisions_and_strict_boundaries(epochs, kl, expected, threshold):
    if isinstance(kl, str):
        kl = threshold + {'equal': 0., 'above': .00001, 'below': -.00001}[kl]
    c = BoundedLearningRate(1e-6, 1e-4, .02, epoch_completion=True,
        increase_kl_threshold=threshold,
        increase_factor=1.05, decrease_factor=.95, scale=.05,
        stable_updates=3, cooldown_updates=3, cooldown=3)
    rate, direction = c.observe(scheduled=1e-4, kl=kl, early_stopped=epochs < 3,
        completed_epochs=epochs, full_epochs_completed=epochs == 3)
    assert rate == pytest.approx(5e-6 * expected)
    assert direction == (expected > 1) - (expected < 1)
    assert c.cooldown == c.stable_count == 0


@pytest.mark.parametrize('probes,expected_samples,expected_calls', [
    ([.11], 12, 1),                           # First pass always complete.
    ([.08, .101], 16, 2),                    # Stop within epoch 2.
    ([.08, .09, .1, .101], 24, 4),           # Stop exactly at the end of epoch 2.
    ([.08, .08, .08, .08, .101], 28, 5),    # Stop within epoch 3.
    ([.1] * 7, 36, 7),                      # Equality does not stop.
])
def test_probe_after_updates_stops_partial_epochs_without_duplicate_probes(probes, expected_samples, expected_calls):
    t = SelfPlayTrainer.__new__(SelfPlayTrainer)
    t.settings = SimpleNamespace(policy_epochs=3, critic_epochs=3, ppo_min_policy_epochs=1,
        early_stop_kl_threshold=.1, target_kl=.02, early_stop_kl_multiple=5.,
        early_stop_clip_fraction=1., lr_epoch_completion=True, ppo_kl_probe_interval_steps=1)
    t.update = 0
    t.logger = SimpleNamespace(event=lambda _: None)
    samples, seen = list(range(12)), []
    t._ppo_optimizer_batches = lambda rows: [rows[i:i+4] for i in range(0, len(rows), 4)]
    def update(rows, **kwargs):
        seen.extend(rows)
        return {'policy/approx_kl_old': 10., 'policy/clip_fraction': .9,
                'policy/entropy': 1., 'optimizer/policy_steps': 1.}
    t._update_policy_batch = update
    t._update_critic_batch = lambda rows, **kw: {'optimizer/critic_steps': 1.}
    values = [{'policy/final_kl': k, 'policy/final_probe_samples': 8,
               'policy/final_clip_fraction': .1, 'timing/lr_probe_seconds': .1} for k in probes]
    with patch.object(t, '_final_policy_diagnostics', side_effect=values) as probe:
        result = t._update_ppo_minibatches(samples, critic=False)
        assert probe.call_count == expected_calls
        critic = t._update_ppo_minibatches(samples, critic=True)
        assert probe.call_count == expected_calls
    assert seen == (samples * 3)[:expected_samples]
    assert result['optimizer/policy_epochs'] == expected_samples / 12
    assert result['optimizer/policy_first_pass_coverage'] == 1.
    assert result['timing/lr_probe_seconds'] == pytest.approx(.1 * expected_calls)
    assert critic['optimizer/critic_epochs'] == 3.


def test_new_config_and_legacy_switch_off():
    s = TrainingSettings.from_yaml('configs/expand_mappo_training.yaml', 'four_dark', model_scale='main')
    assert s.lr_epoch_completion and s.minimum_learning_rate == 1e-6
    assert s.policy_learning_rate == s.critic_learning_rate == 1e-5
    assert s.maximum_learning_rate == 1e-4
    tiny = TrainingSettings.from_yaml('configs/expand_mappo_training.yaml', 'four_dark', tiny=True)
    assert tiny.policy_epochs == 1 and not tiny.lr_epoch_completion
    assert s.lr_minimum_completed_epochs == 2 and s.lr_increase_kl_threshold == .05
    assert s.lr_decrease_kl_threshold == .1
    assert s.ppo_kl_probe_interval_steps == 16 and s.early_stop_kl_threshold == .1
    assert s.lr_kl_band_lower is s.lr_kl_band_upper is None
    disabled = TrainingSettings.from_yaml('configs/expand_mappo_training.yaml', 'four_dark',
        model_scale='main', overrides={'adaptive_learning_rate': False})
    assert not disabled.lr_epoch_completion


@pytest.mark.parametrize('algorithm', ['ppo', 'mappo'])
def test_fresh_initial_rate_and_resume_keep_saved_feedback_rate(tmp_path, algorithm):
    torch.set_num_threads(1)
    source = ('configs/expand_mappo_training.yaml' if algorithm == 'mappo'
              else 'configs/bootstrap.yaml')
    settings = TrainingSettings.from_yaml(source, 'four_dark', tiny=True, overrides={
        'device': 'cpu', 'anchor_batch': 8, 'policy_microbatch': 2,
        'adaptive_learning_rate': True, 'policy_learning_rate': 1e-5,
        'critic_learning_rate': 1e-5, 'minimum_learning_rate': 1e-6,
        'maximum_learning_rate': 1e-4, 'lr_epoch_completion': True,
        'lr_increase_on_full_epochs': False, 'lr_kl_band_lower': None,
        'lr_kl_band_upper': None, 'policy_epochs': 3, 'lr_increase_factor': 1.05,
        'early_stop_kl_threshold': .1, 'ppo_kl_probe_interval_steps': 1,
        'lr_increase_kl_threshold': .09})
    trainer = SelfPlayTrainer(settings, run_directory=tmp_path)
    try:
        assert trainer._set_policy_lr(1) == pytest.approx(1e-5)
        assert trainer.policy_optimizer.param_groups[0]['lr'] == pytest.approx(1e-5)
        assert trainer.critic_optimizer.param_groups[0]['lr'] == pytest.approx(1e-5)
        next_rate, direction = trainer.lr_controller.observe(scheduled=1e-4,
            kl=.04, early_stopped=False, completed_epochs=3, full_epochs_completed=True)
        assert next_rate == pytest.approx(1.05e-5) and direction == 1
        trainer.save_checkpoint(reason='initial_lr_resume_test', archive=False)
    finally:
        trainer.logger.close()
    # A later config edit must not reset an existing adaptive schedule on resume.
    resumed = SelfPlayTrainer(replace(settings, policy_learning_rate=2e-5,
        critic_learning_rate=2e-5, lr_increase_kl_threshold=.05), run_directory=tmp_path)
    try:
        assert resumed._set_policy_lr(1) == pytest.approx(1.05e-5)
        assert resumed.critic_optimizer.param_groups[0]['lr'] == pytest.approx(1.05e-5)
        assert resumed.lr_controller.increase_kl_threshold == .05
        assert resumed.lr_controller.decrease_kl_threshold == .1
        # Retuning a restored checkpoint preserves its next LR and applies the
        # new strict rise threshold, including KL inside the former rise range.
        next_rate, direction = resumed.lr_controller.observe(scheduled=1e-4,
            kl=.06, early_stopped=False, completed_epochs=3, full_epochs_completed=True)
        assert next_rate == pytest.approx(1.05e-5) and direction == 0
    finally:
        resumed.logger.close()


@pytest.mark.parametrize('epochs,kl,expected,reason', [
    (1.5, .1, 1., 'hold'),
    (2., .100001, .95, 'early_high_kl'),
    (2., .1, 1., 'hold'),
    (2.5, .11, 1., 'hold'),
    (3., .05, 1., 'hold'),
    (3., .049999, 1.05, 'completed_low_kl'),
])
def test_trainer_reports_strict_epoch_kl_feedback(tmp_path, epochs, kl, expected, reason):
    settings = training_settings(lr_epoch_completion=True, policy_epochs=3,
        lr_increase_on_full_epochs=False, lr_kl_band_lower=None, lr_kl_band_upper=None,
        lr_increase_kl_threshold=.05, lr_decrease_kl_threshold=.1,
        lr_increase_factor=1.05, lr_decrease_factor=.95, maximum_learning_rate=1e-4,
        early_stop_kl_threshold=.1, ppo_kl_probe_interval_steps=16)
    trainer = SelfPlayTrainer(settings, run_directory=tmp_path)
    trainer.lr_controller.scale = .05
    try:
        result = trainer._finish_learning_rate([], {
            'optimizer/policy_steps': 1., 'optimizer/policy_epochs': epochs,
            'optimizer/policy_first_pass_coverage': 1.,
            'optimizer/policy_stop_kl': float(kl > .1),
            'policy/final_probe_samples': 8., 'policy/final_kl': kl})
        assert result['optimizer/policy_lr_next'] == pytest.approx(5e-6 * expected)
        assert result['optimizer/lr_reason'] == reason
        assert result['optimizer/lr_increase_kl_threshold'] == .05
        assert result['optimizer/lr_decrease_kl_threshold'] == .1
    finally:
        trainer.logger.close()


@pytest.mark.parametrize('threshold', [0., -.1, float('inf'), float('nan'), True, '0.1'])
def test_invalid_decrease_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match='epoch-completion'):
        replace(training_settings(), lr_decrease_kl_threshold=threshold).validate()


def test_partial_final_update_still_uses_epoch_endpoint_guard(tmp_path):
    settings = training_settings(lr_epoch_completion=True, policy_epochs=3,
        lr_increase_on_full_epochs=False, lr_kl_band_lower=None, lr_kl_band_upper=None,
        ppo_minibatch_samples=512, early_stop_kl_threshold=.1, ppo_kl_probe_interval_steps=16)
    trainer = SelfPlayTrainer(settings, run_directory=tmp_path)
    trainer.lr_controller.scale = .05
    try:
        probe_values = {'policy/final_kl': .11, 'policy/final_probe_samples': 8,
            'policy/final_clip_fraction': .1, 'timing/lr_probe_seconds': 0.}
        with patch.object(trainer, '_final_policy_diagnostics', return_value=probe_values):
            trainer.train()
        metrics = json.loads((trainer.run_directory / 'latest_metrics.json').read_text())
        assert metrics['optimizer/policy_epochs'] == 1.
        assert metrics['optimizer/policy_first_pass_coverage'] == 1.
        assert metrics['optimizer/lr_direction'] == -1.
    finally:
        trainer.logger.close()


def test_switch_and_normal_resume_preserve_model_optimizer_and_actual_then_next_rate(tmp_path):
    torch.set_num_threads(1)
    old = training_settings(lr_kl_band_lower=.08, lr_kl_band_upper=.12,
        minimum_learning_rate=5e-6, maximum_learning_rate=1e-4, ppo_minibatch_samples=2)
    t = SelfPlayTrainer(old, run_directory=tmp_path)
    t.train()
    t.policy_optimizer.param_groups[0]['lr'] = 5e-6
    t.lr_controller.scale = .05
    t.save_checkpoint(reason='before_completion', archive=False)
    weights = {k:v.clone() for k,v in t.policy.state_dict().items()}
    steps = [v['step'].clone() for v in t.policy_optimizer.state.values()]
    new = replace(old, lr_epoch_completion=True, lr_kl_band_lower=None, lr_kl_band_upper=None,
        minimum_learning_rate=1e-6, lr_increase_factor=1.05, lr_decrease_factor=.95,
        early_stop_kl_threshold=.1, ppo_kl_probe_interval_steps=1, policy_epochs=3)
    resumed = SelfPlayTrainer(new, run_directory=tmp_path)
    try:
        assert resumed._set_policy_lr(2) == pytest.approx(5e-6)
        assert all(torch.equal(v, resumed.policy.state_dict()[k]) for k,v in weights.items())
        assert all(torch.equal(a,b['step']) for a,b in zip(steps, resumed.policy_optimizer.state.values(), strict=True))
        resumed.lr_controller.observe(scheduled=1e-4, kl=.11, early_stopped=True,
            completed_epochs=1.5, full_epochs_completed=False)
        resumed.save_checkpoint(reason='after_completion', archive=False)
    finally:
        resumed.logger.close()
    again = SelfPlayTrainer(new, run_directory=tmp_path)
    try:
        assert again._set_policy_lr(2) == pytest.approx(4.75e-6)
    finally:
        again.logger.close()
