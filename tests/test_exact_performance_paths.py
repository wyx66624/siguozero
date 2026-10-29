"""Exact discrete results, bounded ownership and immutable action metadata."""
import os
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from junqi import inventory
from junqi.training.modes import TrainingMode, new_game
from junqi.training.ppo_environment import _advance
from junqi.training.queued_environment import QueuedPPOEnvironment


class ExactPerformancePathsTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'native pipe readiness requires POSIX')
    def test_queue_handles_out_of_order_reads_without_overwriting_histories(self):
        for mode in TrainingMode:
            with self.subTest(mode=mode), patch.dict(os.environ, {'SIGUOZERO_SHARED_OBSERVATIONS': '1'}):
                games = [new_game(mode, seed=804+i, max_plies=100) for i in range(4)]
                for game in games:
                    game.enable_critic_inventory_tracking()
                reference = [game.clone() for game in games]
                pool = SimpleNamespace(mode=mode, slots=[SimpleNamespace(game=g) for g in games])
                old = []
                with QueuedPPOEnvironment(2) as env:
                    env.begin(pool, centralized_critic=True)
                    for step in range(16):
                        expected = {}
                        tickets = []
                        for indices in ((0, 1), (2, 3)):
                            actions = [reference[i].legal_actions()[step % len(reference[i].legal_actions())]
                                       for i in indices]
                            for i, action in zip(indices, actions):
                                expected[i] = _advance(i, reference[i], action, mode, centralized_critic=True)
                            tickets.append(env.submit(indices, actions))
                        with self.assertRaises(RuntimeError):
                            env.submit([0], [(0, 0)])
                        for ticket in reversed(tickets):
                            for row in env.receive(ticket):
                                other = expected[row.index]
                                np.testing.assert_array_equal(row.records, other.records)
                                np.testing.assert_array_equal(row.centralized_records, other.centralized_records)
                                self.assertEqual(row.state, other.state)
                                self.assertEqual(row.rewards, other.rewards)
                                if step == 0:
                                    old.append((row.records, row.records.copy()))
                        for original, saved in old:
                            np.testing.assert_array_equal(original, saved)
                    env.synchronize(pool)
                    for slot, game in zip(pool.slots, reference):
                        self.assertEqual(slot.game._inventory.state_dict(), game._inventory.state_dict())

    @unittest.skipUnless(inventory._native_inventory is not None and
                         hasattr(inventory._native_inventory, 'legal60'), 'optional native geometry not built')
    def test_integer_actions_preserve_stable_order_after_moves(self):
        rng = random.Random(67123)
        for seed in range(8):
            game = new_game(TrainingMode.TWO_PLAYER, seed=seed, max_plies=100)
            for _ in range(64):
                for player in (0, 1):
                    values = []
                    for native in ('0', '1'):
                        with patch.dict(os.environ, {'SIGUOZERO_NATIVE_RULES': native}):
                            game._invalidate_legal_actions()
                            values.append(game.legal_actions(player))
                    self.assertEqual(*values)
                if game.is_terminal:
                    break
                game.step(rng.choice(game.legal_actions()))


if __name__ == '__main__':
    unittest.main()
