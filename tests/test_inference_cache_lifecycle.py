from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class InferenceCacheLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def board_key(state, index=-1):
        record = state.records[index]
        return state.mode, record.board_codes, record.known_casualty_bits

    def board_fixture(self):
        from junqi.training.encoding import GameHistory
        from junqi.training.models import GamePolicyTransformer, ModelConfig
        from junqi.training.modes import new_game

        policy = GamePolicyTransformer(replace(ModelConfig.tiny(), incremental_inference=False)).eval()
        game = new_game("two_player", seed=821, max_plies=8)
        history = GameHistory.initialize(game, "two_player", max_transitions=16)
        initial = history.state_for(game, 0)
        game.step(game.legal_actions()[0])
        history.append_after_step(game)
        advanced = history.state_for(game, 0)
        unrelated_game = new_game("two_player", seed=829, max_plies=8)
        unrelated = GameHistory.initialize(unrelated_game, "two_player").state_for(unrelated_game)
        policy.start_inference_board_cache(max_entries=2)
        return policy, initial, advanced, unrelated

    def test_full_board_cache_refreshes_recent_games_and_touches_hits(self):
        policy, initial, advanced, unrelated = self.board_fixture()
        keys = [self.board_key(state) for state in (initial, unrelated, advanced)]
        self.assertEqual(len(set(keys)), 3)
        with torch.inference_mode():
            policy.encode([initial])
            policy.encode([unrelated])
            self.assertEqual(list(policy._inference_board_cache), keys[:2])
            previous_hits = policy._board_cache_hits
            previous_encoded = policy._board_encoder_tokens
            policy.encode([advanced])
            # The historic initial board was reused and moved to the recent
            # end before the new current board evicted the unrelated game.
            self.assertEqual(list(policy._inference_board_cache), [keys[0], keys[2]])
            self.assertEqual(policy._board_cache_hits - previous_hits, 1)
            self.assertEqual(policy._board_encoder_tokens - previous_encoded, 1)
            previous = policy._inference_board_cache[keys[0]]
            policy.encode([initial])
            self.assertEqual(list(policy._inference_board_cache), [keys[2], keys[0]])
            self.assertIs(policy._inference_board_cache[keys[0]], previous)
            policy.encode([unrelated])
            self.assertEqual(list(policy._inference_board_cache), [keys[0], keys[1]])
            self.assertLessEqual(len(policy._inference_board_cache), 2)

    def test_eviction_preserves_features_already_borrowed_by_current_batch(self):
        policy, initial, advanced, _ = self.board_fixture()
        reference = copy.deepcopy(policy)
        reference.clear_inference_board_cache()
        policy.start_inference_board_cache(max_entries=1)
        with torch.inference_mode():
            policy.encode([initial])
            expected = reference.encode([advanced])
            actual = policy.encode([advanced])
        self.assertNotIn(self.board_key(initial), policy._inference_board_cache)
        self.assertIn(self.board_key(advanced), policy._inference_board_cache)
        torch.testing.assert_close(actual.context, expected.context)
        torch.testing.assert_close(actual.point_mask, expected.point_mask)

    def test_enabled_gradients_bypass_detached_board_cache(self):
        policy, initial, advanced, _ = self.board_fixture()
        with torch.inference_mode():
            policy.encode([initial])
        original = dict(policy._inference_board_cache)
        features = policy.encode([advanced])
        self.assertTrue(features.context.requires_grad)
        features.context[:, 0].sum().backward()
        self.assertTrue(any(
            parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
            for parameter in policy.board_encoder.parameters()
        ))
        self.assertEqual(list(policy._inference_board_cache), list(original))
        for key, value in original.items():
            self.assertIs(policy._inference_board_cache[key], value)

    def paged_fixture(self, batch_size=1):
        from junqi.training.models import GamePolicyTransformer, ModelConfig
        from junqi.training.modes import TrainingMode
        from junqi.training.paged_kv import PagedKVCache

        policy = GamePolicyTransformer(replace(ModelConfig.tiny(), max_transitions=32)).half().eval()
        # Exercise the real allocator and copy-on-write bookkeeping on CPU.
        # Only its device eligibility check is mocked; no accelerator is used.
        with patch("junqi.training.paged_kv.is_accelerator", return_value=True):
            store = PagedKVCache(
                device=torch.device("cpu"), dtype=torch.float16,
                num_layers=1, num_heads=4, head_dim=16,
                max_tokens=33, max_entries=16,
            )
        policy._paged_kv_store = store
        policy._inference_temporal_cache_limit = 16
        states, prefixes = [], []
        for index in range(batch_size):
            records = tuple(range(index * 100, index * 100 + 18))
            state = SimpleNamespace(mode=TrainingMode.TWO_PLAYER, records=records)
            prefix = store.from_contiguous(
                [torch.zeros(4, 16, 16, dtype=torch.float16)],
                [torch.zeros(4, 16, 16, dtype=torch.float16)],
                length=16, context=torch.zeros(64, dtype=torch.float16),
            )
            policy._put_temporal_cache((state.mode, records[:16]), prefix)
            states.append(state)
            prefixes.append(prefix)
        policy._record_board_batch = lambda records, mode: (
            torch.zeros(len(records), 32, dtype=torch.float16),
            torch.ones(len(records), 60, dtype=torch.bool),
        )
        policy._record_temporal_tokens = lambda records, mode, globals_, positions: (
            torch.zeros(len(records), 1, 64, dtype=torch.float16)
        )
        return policy, store, states, prefixes

    def test_paged_oom_releases_intermediates_and_preserves_borrowed_prefixes(self):
        for target_name in ("_record_board_batch", "_record_temporal_tokens", "fork_for_append", "make_page_table", "decode"):
            for failure_call in (1, 2):
                with self.subTest(target=target_name, failure_call=failure_call):
                    policy, store, states, prefixes = self.paged_fixture()
                    target = policy if target_name.startswith("_record") else store
                    original = getattr(target, target_name)
                    previous_references = list(store._references)
                    previous_used = store.used_pages
                    calls = 0

                    def injected(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == failure_call:
                            raise torch.OutOfMemoryError("injected paged append OOM")
                        return original(*args, **kwargs)

                    with torch.inference_mode(), patch.object(target, target_name, side_effect=injected):
                        with self.assertRaises(torch.OutOfMemoryError):
                            policy._advance_paged_temporal_group(states, prefixes, missing=2)
                    self.assertEqual(store._references, previous_references)
                    self.assertEqual(store.used_pages, previous_used)
                    self.assertEqual(list(policy._inference_temporal_cache.values()), prefixes)
                    with torch.inference_mode():
                        features = policy._advance_paged_temporal_group(states, prefixes, missing=2)
                    self.assertTrue(torch.isfinite(features.context).all())
                    self.assertEqual(store.used_pages, 2)  # Shared prefix plus final tail.
                    self.assertEqual(len(policy._inference_temporal_cache), 2)

    def test_partial_fork_allocation_failure_reclaims_all_temporary_generations(self):
        policy, store, states, prefixes = self.paged_fixture(batch_size=2)
        previous_references = list(store._references)
        original_allocate = store._allocate_page
        calls = 0

        def allocate():
            nonlocal calls
            calls += 1
            if calls == 4:  # Second game's allocation on the second append.
                raise torch.OutOfMemoryError("injected partial fork OOM")
            return original_allocate()

        with torch.inference_mode(), patch.object(store, "_allocate_page", side_effect=allocate):
            with self.assertRaises(torch.OutOfMemoryError):
                policy._advance_paged_temporal_group(states, prefixes, missing=2)
        self.assertEqual(store._references, previous_references)
        self.assertEqual(store.used_pages, 2)
        with torch.inference_mode():
            policy._advance_paged_temporal_group(states, prefixes, missing=2)
        self.assertEqual(store.used_pages, 4)

    def test_installing_more_tips_than_lru_capacity_does_not_double_release(self):
        policy, store, states, prefixes = self.paged_fixture(batch_size=2)
        policy._inference_temporal_cache_limit = 1
        with torch.inference_mode():
            policy._advance_paged_temporal_group(states, prefixes, missing=2)
        self.assertEqual(len(policy._inference_temporal_cache), 1)
        self.assertIn((states[-1].mode, states[-1].records), policy._inference_temporal_cache)
        self.assertEqual(store.used_pages, 2)
        self.assertTrue(all(reference >= 0 for reference in store._references))


if __name__ == "__main__":
    unittest.main()
