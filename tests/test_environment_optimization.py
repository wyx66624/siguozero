"""Public-information and replay parity for environment work elimination."""
import random
import gc
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from junqi.training.mappo_encoding import centralized_observation_rows
from junqi.training.modes import TrainingMode, new_game
from junqi.training.packed_observation import observation_rows
from junqi.training.mappo import collect_mappo_samples
from junqi.training.rollout import RolloutMetrics


@pytest.mark.parametrize('mode', list(TrainingMode))
def test_reused_public_rows_keep_private_data_and_match_standalone_critic(mode):
    game = new_game(mode, seed=8237, max_plies=160)
    game.enable_critic_inventory_tracking()
    rng = random.Random(271)
    for _ in range(160):
        private = observation_rows(game, mode)
        retained = private.copy()
        expected = centralized_observation_rows(game, mode)
        actual = centralized_observation_rows(game, mode, public_rows=private)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(private, retained)
        assert not np.shares_memory(actual, private)
        if game.is_terminal:
            break
        game.step(rng.choice(game.legal_actions()))


@pytest.mark.parametrize('mode', list(TrainingMode))
@pytest.mark.parametrize('seed', [913, 22419])
def test_quiet_moves_match_inventory_recomputed_after_every_step(mode, seed):
    fast = new_game(mode, seed=seed, max_plies=350)
    fast.enable_critic_inventory_tracking()
    full = fast.clone()
    rng = random.Random(seed + 1)
    battles = 0
    for _ in range(350):
        assert fast.legal_actions() == full.legal_actions()
        action = rng.choice(fast.legal_actions())
        left, right = fast.step(action), full.step(action)
        # Independently force the full fixed-point solver, even on moves where
        # the optimized path considers all knowledge unchanged.
        full._inventory.propagate(full)
        assert left == right
        assert fast._inventory.state_dict() == full._inventory.state_dict()
        assert fast.known_casualties == full.known_casualties
        assert fast.known_identities == full.known_identities
        np.testing.assert_array_equal(observation_rows(fast, mode), observation_rows(full, mode))
        np.testing.assert_array_equal(centralized_observation_rows(fast, mode),
                                      centralized_observation_rows(full, mode))
        battles += left.defender is not None
        if fast.is_terminal:
            break
    assert battles > 0


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('fail', [True, False])
def test_collection_restores_gc_state_on_success_and_failure(enabled, fail):
    original = gc.isenabled()
    pool = SimpleNamespace(centralized_critic=True)
    actor = SimpleNamespace(policy=SimpleNamespace(config=SimpleNamespace(ppo_array_history=True)))
    critic = SimpleNamespace(deferred=True, critic=object())
    result = ([], [], RolloutMetrics())
    def collect(*args, **kwargs):
        assert not gc.isenabled()
        if fail:
            raise RuntimeError('injected worker failure')
        return result
    try:
        (gc.enable if enabled else gc.disable)()
        with patch('junqi.training.mappo._collect', side_effect=collect), \
                patch('junqi.training.mappo.gc.collect', return_value=0) as collect_cycles:
            def run():
                return collect_mappo_samples(pool, actor, critic, None, count=1,
                    behavior_version=0, discount=1., gae_lambda=.95, environment=object())
            if fail:
                with pytest.raises(RuntimeError, match='injected worker failure'):
                    run()
            else:
                assert run() is result
            assert gc.isenabled() == enabled
            assert collect_cycles.call_count == int(enabled)
    finally:
        (gc.enable if original else gc.disable)()
