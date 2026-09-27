"""The fast observation path preserves public masks through game changes."""
import random

import pytest

from junqi.training.modes import TrainingMode, new_game


@pytest.mark.parametrize('mode', list(TrainingMode))
def test_bulk_masks_match_point_queries_after_moves_and_restore(mode):
    game = new_game(mode, seed=289, max_plies=64)
    game.enable_inventory_tracking()
    rng = random.Random(9023)
    for ply in range(64):
        for viewer in range(game.config.player_count):
            assert game._inventory.masks(game, viewer) == {
                point: game.inventory_mask(viewer, point) for point in game._pieces
            }
        if game.is_terminal:
            break
        game.step(rng.choice(game.legal_actions()))
        if ply == 31:
            game = game.clone()
