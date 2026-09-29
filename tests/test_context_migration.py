"""A shorter window must preserve the complete training state and remain resumable."""
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from junqi.training.context_migration import reconcile_context_window
from junqi.training.inference import InferenceEngine
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


CONFIG = Path(__file__).parents[1] / "configs/expand_mappo_training.yaml"


def settings():
    return TrainingSettings.from_yaml(CONFIG, "two_player", tiny=True, overrides=dict(
        device="cpu", anchor_batch=16, base_game_pool_size=2, policy_microbatch=2,
        max_game_plies=64, policy_epochs=1, critic_epochs=1, inference_snapshot_every_updates=0))


def test_shrink_preserves_models_adam_rng_and_games_and_trains_again(tmp_path):
    torch.set_num_threads(1)
    old = settings()
    original = SelfPlayTrainer(old, run_directory=tmp_path)
    original.train()
    path = original.checkpoints.latest_path
    before = torch.load(path, map_location="cpu", weights_only=False)
    modern = replace(old, model=replace(old.model, max_transitions=3,
        position_embedding_capacity=old.model.max_sequence_tokens, paged_kv_length_bucket_tokens=4,
        history_window_stride=2),
        total_updates=2)
    with pytest.raises(ValueError, match="shrink-context-window"):
        SelfPlayTrainer(modern, run_directory=tmp_path)
    resumed = SelfPlayTrainer(modern, run_directory=tmp_path, shrink_context_window=True)
    try:
        after = torch.load(path, map_location="cpu", weights_only=False)
        assert after["reason"] == "context_window_migration"
        assert after["update"] == before["update"] == 1
        assert after["trainer_state"]["cumulative"] == before["trainer_state"]["cumulative"]
        for key in ("policy", "critic", "layout", "reference_layout", "policy_optimizer",
                    "critic_optimizer", "layout_optimizer", "rng_state"):
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
        old_pool = before["trainer_state"]["base_game_pool"]
        pool = after["trainer_state"]["base_game_pool"]
        assert pool["rng_state"] == old_pool["rng_state"]
        assert pool["layout_queue"] == old_pool["layout_queue"]
        assert pool["max_transitions"] == 3
        assert pool["history_window_stride"] == 2
        removed = 0
        for old_slot, slot in zip(old_pool["slots"], pool["slots"], strict=True):
            assert slot["game"] == old_slot["game"]
            for get_history in (lambda s: s["history"], lambda s: s["critic_history"]["history"]):
                for a, b in zip(get_history(old_slot)["players"], get_history(slot)["players"], strict=True):
                    assert b["max_transitions"] == 3
                    assert b["history_window_stride"] == 2
                    expected = a["records"] if len(a["records"]) <= 4 else [a["records"][0], *a["records"][-3:]]
                    assert b["records"] == expected
                    removed += len(a["records"]) - len(b["records"])
        assert removed > 0
        assert after["trainer_state"]["context_window_migration"]["removed_history_records"] == removed
        engine = InferenceEngine.from_checkpoint(path, device="cpu")
        assert engine.policy.config.max_transitions == 3
        assert engine.policy.position_embedding.num_embeddings == 17
        resumed.train()  # Exercises attention-window rollover and restored Adam moments.
        assert resumed.update == 2
        assert resumed.cumulative["environment_plies"] == 32
        ordinary = SelfPlayTrainer(modern, run_directory=tmp_path)
        assert ordinary.update == 2
        assert not ordinary.context_shrunk_on_resume
        assert ordinary.context_window_migration["from"] == 16
        ordinary.historical.close()
        ordinary.logger.close()
    finally:
        resumed.historical.close()
        resumed.logger.close()


def test_all_rank_histories_shrink_and_invalid_migrations_do_not_mutate():
    old = settings()
    modern = replace(old, model=replace(old.model, max_transitions=3, position_embedding_capacity=17))
    def pool():
        player = {"max_transitions": 16, "records": list(range(10))}
        return {"max_transitions": 16, "slots": [{"history": {"players": [copy.deepcopy(player)]},
            "critic_history": {"history": {"players": [copy.deepcopy(player)]}}}]}
    payload = {"config": {"model": {"max_transitions": 16}}, "update": 8,
        "trainer_state": {"distributed": {"rank_states": [{"base_game_pool": pool()} for _ in range(2)]}}}
    for target in (old, replace(old, model=replace(old.model, max_transitions=20)),
                   replace(old, model=replace(old.model, max_transitions=3))):
        saved = copy.deepcopy(payload)
        if target is old:
            assert not reconcile_context_window(saved, target, adopt=True)
        else:
            with pytest.raises(ValueError):
                reconcile_context_window(saved, target, adopt=True)
        assert saved == payload
    broken = copy.deepcopy(payload)
    broken["trainer_state"]["distributed"]["rank_states"][1]["base_game_pool"]["max_transitions"] = 5
    saved = copy.deepcopy(broken)
    with pytest.raises(ValueError, match="saved pool context"):
        reconcile_context_window(broken, modern, adopt=True)
    assert broken == saved
    assert reconcile_context_window(payload, modern, adopt=True)
    for rank in payload["trainer_state"]["distributed"]["rank_states"]:
        p = rank["base_game_pool"]
        assert p["max_transitions"] == 3
        for h in (p["slots"][0]["history"], p["slots"][0]["critic_history"]["history"]):
            assert h["players"][0]["records"] == [0, 7, 8, 9]


@pytest.mark.parametrize("scale", ["bootstrap", "main", "extended"])
def test_two_player_context_only(scale):
    two = TrainingSettings.from_yaml(CONFIG, "two_player", model_scale=scale)
    four = TrainingSettings.from_yaml(CONFIG, "four_dark", model_scale=scale)
    assert two.model.max_transitions == 300
    assert two.model.max_sequence_tokens == two.model.paged_kv_length_bucket_tokens == 301
    assert two.model.position_embedding_capacity == 1001
    assert two.model.history_window_stride == 32
    assert four.model.max_transitions == 1000
    with pytest.raises(ValueError, match="capacity"):
        replace(two.model, position_embedding_capacity=300)


def test_block_window_keeps_latest_rows_and_resumes_at_the_same_boundary():
    import numpy as np
    from junqi.training.encoding import GameHistory
    from junqi.training.history_arrays import observation_width, record_array
    from junqi.training.modes import TrainingMode, new_game

    mode = TrainingMode.TWO_PLAYER
    game = new_game(mode, seed=71)
    linked = GameHistory.initialize(game, mode, max_transitions=300, history_window_stride=32)
    array = linked.clone()
    array.enable_array_storage()
    width = observation_width(mode, True)
    rows = np.stack([record_array(p.records[0], mode, True) for p in linked.players])
    rows[:, width + 5] = 1  # A post-action record; the pinned initial remains unchanged.
    held = None
    resumed = None
    for step in range(1, 901):
        rows[:, width + 6] = step  # Distinct test records, never fed into a model.
        for history in (linked, array, resumed):
            if history is not None:
                history.append_encoded_rows(rows)
        discarded = ((max(0, step - 300) + 31) // 32) * 32
        expected = [0, *range(discarded + 1, step + 1)]
        for history in (linked, array, resumed):
            if history is not None:
                assert [r.no_interaction_plies for r in history.players[0].records] == expected
                assert len(history.players[0].records) <= 301
        if step == 311:
            held = array.players[0].records
            held_expected = list(held)
            resumed = GameHistory.from_state_dict(array.state_dict())
            resumed.enable_array_storage()
        if held is not None:
            assert list(held) == held_expected
