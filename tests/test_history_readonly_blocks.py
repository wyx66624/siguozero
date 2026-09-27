"""Appending through the private writer must preserve published snapshots."""
from dataclasses import replace

import numpy as np
import pytest

from junqi.training.encoding import StateTokenRecord
from junqi.training.history_arrays import ArrayHistory
from junqi.training.modes import TrainingMode


@pytest.mark.parametrize('window', [17, 1000])
def test_readonly_prefixes_survive_appends_and_window_retirement(window):
    initial = StateTokenRecord((0,)*129, (0,)*75, None, 0, 15, 0, 0, (0,)*129)
    history = ArrayHistory(TrainingMode.FOUR_DARK, (initial,), window, (111, 0))
    held = []
    for step in range(1300):
        history.append(replace(initial, no_interaction_plies=step))
        if step in (0, 255, 256, 999, 1000, 1200):
            view = history.view()
            assert len(view) == min(step+1, window)+1
            for length in (1, max(1, len(view)//2), len(view)):
                prefix = view[:length]
                assert len(prefix) == length
                assert prefix.cache_key == (view.identity, view.window_start, length)
                assert all(not chunk.flags.writeable for chunk in prefix.chunks)
                expected = np.concatenate(prefix.chunks).copy()
                held.append((prefix, expected))
                with pytest.raises(ValueError):
                    prefix.chunks[-1][-1, 0] = 1
    for prefix, expected in held:
        actual = np.empty_like(expected)
        prefix.copy_rows(actual)
        np.testing.assert_array_equal(actual, expected)
    assert all(not block.flags.writeable for _, block in history.blocks)
    assert len(history.blocks) <= (window+255)//256+1
