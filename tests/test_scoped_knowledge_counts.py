"""Owner-scoped scans must preserve the old all-army inference semantics."""
import random
from types import MethodType

import numpy as np
import pytest

from junqi.game import JunqiGame
from junqi.training.modes import TrainingMode, new_game
from junqi.training.packed_observation import observation_rows
from junqi.training.mappo_encoding import centralized_observation_rows


@pytest.mark.parametrize('mode', list(TrainingMode))
def test_scoped_counts_replay_matches_full_board_scans(mode):
    fast = new_game(mode, seed=29415, max_plies=250)
    fast.enable_critic_inventory_tracking()
    reference = fast.clone()
    # Exercise the old scan for every viewer, including viewers who know both
    # participants. Later inference must agree after moves, captures and passes.
    reference._known_live_counts = MethodType(
        lambda game, viewer, owners=None: JunqiGame._known_live_counts(game, viewer), reference)
    rng = random.Random(73211)
    battles = 0
    for _ in range(250):
        assert fast.legal_actions() == reference.legal_actions()
        action = rng.choice(fast.legal_actions())
        actual, expected = fast.step(action), reference.step(action)
        assert actual == expected
        assert fast._inventory.state_dict() == reference._inventory.state_dict()
        assert fast.known_casualties == reference.known_casualties
        assert fast.known_identities == reference.known_identities
        np.testing.assert_array_equal(observation_rows(fast, mode), observation_rows(reference, mode))
        np.testing.assert_array_equal(centralized_observation_rows(fast, mode),
                                      centralized_observation_rows(reference, mode))
        battles += actual.defender is not None
        if fast.is_terminal:
            break
    assert battles > 0
