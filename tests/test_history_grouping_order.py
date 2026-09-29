"""Metadata reuse must preserve the exact prefix and optimizer ordering."""
import random
from types import SimpleNamespace

from junqi.training.encoding import history_prefix_groups


class History:
    def __init__(self, identity, start, length):
        self.key = identity, start, length

    @property
    def cache_key(self):
        return tuple(self.key)

    def __len__(self):
        return self.key[2]


def test_metadata_grouping_preserves_ties_modes_windows_and_all_samples():
    rng = random.Random(385)
    samples = [SimpleNamespace(mode=rng.choice(('two_player', 'four_dark')),
        records=History((rng.randrange(12), rng.randrange(2)),
                        rng.choice((0, 32, 64)), rng.randrange(1, 302)))
        for _ in range(2048)]
    samples += samples[::7]
    expected, windows = [], {}
    for index in sorted(range(len(samples)), key=lambda i: len(samples[i].records), reverse=True):
        state = samples[index]
        identity, start, _ = state.records.cache_key
        key = state.mode, identity, start
        if key not in windows:
            windows[key] = len(expected)
            expected.append([])
        expected[windows[key]].append(index)
    assert history_prefix_groups(samples) == expected
    assert sorted(i for group in expected for i in group) == list(range(len(samples)))
    assert history_prefix_groups([]) == []
