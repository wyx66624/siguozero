"""Public observation parity and ownership across optimized environment paths."""
import os
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from junqi import inventory
from junqi.training.fast_observation import public_rows, centralized_rows
from junqi.training.mappo_encoding import _centralized_observation_rows_reference
from junqi.training.modes import TrainingMode, new_game
from junqi.training.packed_observation import _observation_rows_reference
from junqi.training.ppo_environment import ParallelPPOEnvironment, _advance


class EnvironmentFastPathTests(unittest.TestCase):
    def test_two_player_rows_preserve_public_and_referee_views(self):
        # Inventory observations require deterministic dead rules by contract.
        for dead in (True,):
            for seed in (41, 289, 831):
                with self.subTest(dead_rules=dead, seed=seed):
                    game = new_game(TrainingMode.TWO_PLAYER, seed=seed,
                                    dead_rules_enabled=dead, max_plies=72)
                    rng = random.Random(seed + 900)
                    for _ in range(72):
                        expected = _observation_rows_reference(game, TrainingMode.TWO_PLAYER)
                        actual = public_rows(game, TrainingMode.TWO_PLAYER)
                        np.testing.assert_array_equal(actual, expected)
                        before = actual.copy()
                        np.testing.assert_array_equal(
                            centralized_rows(game, TrainingMode.TWO_PLAYER, actual),
                            _centralized_observation_rows_reference(
                                game, TrainingMode.TWO_PLAYER, public_rows=expected))
                        np.testing.assert_array_equal(actual, before)
                        if game.is_terminal:
                            break
                        game.step(rng.choice(game.legal_actions()))

    def test_worker_transport_preserves_rows_snapshots_and_old_messages(self):
        # Linux exercises both transports; other hosts exercise the portable pipe.
        for shared in ((False, True) if os.name == "posix" else (False,)):
            for mode in TrainingMode:
                with self.subTest(shared=shared, mode=mode), patch.dict(
                    os.environ, {"SIGUOZERO_SHARED_OBSERVATIONS": str(int(shared))}
                ):
                    games = [new_game(mode, seed=831 + i, max_plies=16) for i in range(4)]
                    for game in games:
                        game.enable_critic_inventory_tracking()
                    reference = [game.clone() for game in games]
                    pool = SimpleNamespace(mode=mode, slots=[
                        SimpleNamespace(game=game) for game in games])
                    rng = random.Random(714)
                    saved = []
                    with ParallelPPOEnvironment(2) as environment:
                        environment.begin(pool, centralized_critic=True, track_flag_defense=True)
                        self.assertEqual(environment.shared_observations is not None, shared)
                        for ply in range(24):
                            indices, actions, expected = [], [], {}
                            for index, game in enumerate(reference):
                                if game.is_terminal:
                                    game = new_game(mode, seed=10100 + ply * 100 + index,
                                                    max_plies=16)
                                    game.enable_critic_inventory_tracking()
                                    reference[index] = game
                                    environment.replace(index, game.clone())
                                action = rng.choice(game.legal_actions())
                                indices.append(index)
                                actions.append(action)
                                expected[index] = _advance(
                                    index, game, action, mode,
                                    centralized_critic=True, track_flag_defense=True)
                            environment.submit(indices, actions)
                            rows = environment.receive()
                            for row in rows:
                                other = expected[row.index]
                                for name in row._fields:
                                    if name in ("records", "centralized_records"):
                                        np.testing.assert_array_equal(
                                            getattr(row, name), getattr(other, name))
                                    else:
                                        self.assertEqual(getattr(row, name), getattr(other, name))
                            if ply == 0:
                                saved = [(r, r.records.copy(), r.centralized_records.copy())
                                         for r in rows]
                            for row, actor, critic in saved:
                                np.testing.assert_array_equal(row.records, actor)
                                np.testing.assert_array_equal(row.centralized_records, critic)
                        environment.synchronize(pool)
                        for slot, game in zip(pool.slots, reference):
                            self.assertEqual(slot.game._inventory.state_dict(),
                                             game._inventory.state_dict())

    @unittest.skipIf(inventory._native_inventory is None, "optional native solver not built")
    def test_native_matching_and_casualties_match_python(self):
        native = inventory._native_inventory
        rng = random.Random(62103)
        try:
            for _ in range(150):
                truth = list(inventory.INVENTORY_SLOTS)
                rng.shuffle(truth)
                masks = tuple(inventory.TYPE_BITS[kind] | sum(
                    bit for bit in inventory.TYPE_BITS.values() if rng.random() < .25
                ) for kind in truth)
                split = rng.randrange(26)
                results = []
                for implementation in (None, native):
                    inventory._native_inventory = implementation
                    inventory._supported_domains.cache_clear()
                    inventory.guaranteed_casualties.cache_clear()
                    supported = inventory.supported_domains(masks)
                    bits = tuple(inventory.domain_bits(domain) for domain in supported)
                    results.append((supported, inventory.guaranteed_casualties(
                        tuple(sorted(bits[:split])), tuple(sorted(bits[split:])))))
                    with self.assertRaises(ValueError):
                        inventory.supported_domains((0,) + masks[1:])
                self.assertEqual(*results)
        finally:
            inventory._native_inventory = native
            inventory._supported_domains.cache_clear()
            inventory.guaranteed_casualties.cache_clear()


if __name__ == "__main__":
    unittest.main()
