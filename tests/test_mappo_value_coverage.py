"""Every value queried by team GAE must have a matching critic target."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from junqi.training.mappo import MAPPOTransition, critic_training_samples, team_generalized_advantages
from junqi.training.ppo import critic_ppo_loss, normalize_advantages
from junqi.training.trainer import SelfPlayTrainer
from test_mappo import histories


def transitions():
    game, private, central = histories()
    state = private.state_for(game)
    perspectives = tuple(central.value_state_for(team) for team in (0, 1))
    return [MAPPOTransition(state, state.legal_actions[0], -1., perspectives,
                            rewards, terminal, team, learnable, old)
            for rewards, terminal, team, learnable, old in (
                ((0., .03), False, 1, True, (.2, -.4)),
                ((.2, -.2), False, 0, False, (.1, .6)),
                ((1., -1.), True, 0, True, (.8, .5)),
                ((-.15, -.15), True, 1, True, (-.3, .1)),
                ((-.01, 0.), False, 0, True, (.4, -.2)),
            )]


@pytest.mark.parametrize('discount,lam', [(1., .95), (.9, .7), (1., 1.), (1., 0.)])
def test_both_targets_match_independent_returns_without_changing_actor(discount, lam):
    trace = transitions()
    bootstrap = (.7, -.3)
    samples = team_generalized_advantages(trace, bootstrap_values=bootstrap,
        discount=discount, gae_lambda=lam, behavior_version=9)
    normalized = normalize_advantages(samples, mean=.2, std=.4, epsilon=1e-8)
    assert len(samples) == len(trace)
    paired = critic_training_samples(normalized)
    assert len(paired) == 2 * len(trace)
    for i, (sample, item) in enumerate(zip(samples, trace, strict=True)):
        assert sample.state is item.state and not sample.state.centralized
        assert (sample.action, sample.learnable) == (item.action, item.learnable)
        assert normalized[i].other_team_value is sample.other_team_value
        for team in (0, 1):
            expected, weight = 0., 1.
            for j in range(i, len(trace)):
                later = trace[j]
                next_v = (0. if later.terminal else trace[j+1].old_values[team]
                          if j+1 < len(trace) else bootstrap[team])
                expected += weight * (later.rewards[team] + discount * next_v - later.old_values[team])
                if later.terminal:
                    break
                weight *= discount * lam
            value = paired[2*i + int(team != item.value_team)]
            assert value.state is item.value_states[team]
            assert value.state.centralized and value.value_state is None
            assert value.other_team_value is None
            assert value.old_value == item.old_values[team]
            assert value.value_target == pytest.approx(item.old_values[team] + expected)
            assert value.advantage == pytest.approx(expected)
            if team == item.value_team:
                assert sample.advantage == pytest.approx(expected)


def test_losing_team_on_winners_turn_receives_negative_terminal_supervision():
    terminal = transitions()[2]
    sample = team_generalized_advantages([terminal], bootstrap_values=(999., 999.),
        discount=1., gae_lambda=.95, behavior_version=0)[0]
    # This label was previously absent, although this value bootstrapped the
    # losing team's immediately preceding decision.
    assert sample.value_target == 1.
    assert sample.other_team_value.target == -1.
    with pytest.raises(ValueError, match='both centralized team targets'):
        critic_training_samples([replace(sample, other_team_value=None)])

    class TwoValues(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor([.8, .5]))

        def forward(self, states):
            return torch.stack([self.values[int(state is terminal.value_states[1])] for state in states])

    critic = TwoValues()
    loss = critic_ppo_loss(critic, critic_training_samples([sample]),
                          clip_epsilon=.2, value_coefficient=.5)
    loss.loss.backward()
    assert critic.values.grad[0] < 0  # Raise the winner's value.
    assert critic.values.grad[1] > 0  # Lower the loser's off-turn value.


@pytest.mark.parametrize('sequence_training', [False, True])
def test_both_perspectives_train_every_epoch_without_doubling_adam_steps(sequence_training):
    samples = team_generalized_advantages(transitions(), bootstrap_values=(.7, -.3),
        discount=1., gae_lambda=.95, behavior_version=0)
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings = SimpleNamespace(algorithm='mappo', critic_epochs=3, policy_epochs=3,
        ppo_minibatch_samples=2, ppo_max_samples_per_sequence=256,
        early_stop_kl_threshold=.1, lr_kl_band_lower=.08)
    trainer.distributed = SimpleNamespace(sum_metrics=lambda values: values)
    trainer._sequence_training_enabled = lambda: sequence_training
    seen = []

    def update(batch, *, epochs):
        assert epochs == 1 and len(batch) <= 4
        seen.extend(batch)
        return {'optimizer/critic_steps': 1.}

    trainer._update_critic_batch = update
    metrics = trainer._update_critic(samples)
    assert metrics['optimizer/critic_epochs'] == 3
    assert metrics['optimizer/critic_steps'] == 9  # ceil(5 / 2) * 3, unchanged.
    assert metrics['optimizer/critic_value_samples'] == 10
    assert metrics['optimizer/critic_opposing_value_samples'] == 5
    assert metrics['optimizer/critic_perspectives_per_transition'] == 2
    expected = critic_training_samples(samples)
    assert len(seen) == 3 * len(expected)
    for item in expected:
        assert sum(value.state is item.state and value.old_value == item.old_value
                   and value.value_target == item.value_target for value in seen) == 3
    assert 'critic/both_teams_explained_variance' in metrics
