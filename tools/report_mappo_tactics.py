"""Summarize real training logs or verify a locally generated sampled replay.

PYTHONPATH=src python tools/report_mappo_tactics.py --run-dir RUN --last-updates 20
PYTHONPATH=src python tools/report_mappo_tactics.py --replay FILE.pt.gz --verify
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from junqi.training.mappo import (PlayerMAPPOTransition, player_generalized_advantages,
                                 player_step_rewards, virtual_player_turns)
from junqi.training.ppo_environment import _advance
from junqi.training.rollout import BaseGamePool
from junqi.training.tactical_events import TACTIC_NAMES, public_tactical_event
from junqi.training.tactical_rewards import BOMB_REWARD_FIELDS


def verify_replay(payload):
    """Check legal moves, public inputs, all player rewards/targets and personal GAE."""
    objective = payload["objective"]
    if objective["advantage_mode"] != "player_decisions":
        raise ValueError("this verifier requires personal MAPPO advantages")
    checked = 0
    for saved in payload["games"]:
        initial = saved["initial_slot"]
        config = initial["game"]["config"]
        players = config.player_count
        pool = BaseGamePool(payload["mode"], pool_size=1,
            max_transitions=initial["history"]["players"][0]["max_transitions"],
            history_window_stride=initial["history"]["players"][0].get("history_window_stride", 1),
            max_game_plies=config.max_plies, dead_rules_enabled=config.dead_rules_enabled,
            no_capture_draw_plies=config.no_interaction_draw_plies, seed=0, centralized_critic=True)
        slot = pool._slot_from_state_dict(initial)
        slot.history.enable_array_storage()
        slot.critic_history.enable_array_storage()
        trace = []
        for step in saved["steps"]:
            record = step["row"]
            actor, action = record["event"]["actor"], tuple(record["action"])
            assert slot.game.current_player == actor, "actor mismatch"
            assert list(slot.game.legal_actions()) == record["legal_actions"], "legal moves mismatch"
            state = slot.history.state_for(slot.game)
            values = tuple(slot.critic_history.value_state_for(p) for p in range(players))
            public = public_tactical_event(slot.game, action)
            assert public == {k: v for k, v in record["event"].items()
                              if k not in ("diagnostic_seconds", "referee_outcome")}, "public labels mismatch"
            row = _advance(0, slot.game, action, pool.mode,
                centralized_critic=True, track_futile_attacks=objective["futile_attack_penalty"] > 0,
                tactical_logging=True,
                track_known_nonflag_hq=bool(objective.get("known_nonflag_hq_penalty", 0)),
                track_flag_defense=bool(objective.get("flag_defense_missed_penalty", 0)
                                       or objective.get("flag_defense_success_reward", 0)))
            np.testing.assert_array_equal(row.records, step["public_rows"])
            np.testing.assert_array_equal(row.centralized_records, step["critic_rows"])
            assert row.tactical["referee_outcome"] == record["event"]["referee_outcome"], "outcome mismatch"
            futile = -objective["futile_attack_penalty"] if row.futile_attack and record["learnable"] else 0.
            rewards = player_step_rewards(row, player=actor, player_count=players,
                draw_reward=objective["draw_reward"], flag_capture_reward=objective["flag_capture_reward"],
                bomb_rewards={kind: objective[field] for kind, field in BOMB_REWARD_FIELDS.items()},
                futile_reward=futile, reward_scope=objective["reward_scope"],
                known_nonflag_hq_reward=(-objective.get("known_nonflag_hq_penalty", 0.)
                    if row.known_nonflag_hq and record["learnable"] else 0.),
                flag_defense_reward=((objective.get("flag_defense_success_reward", 0.) if row.flag_defense == "success"
                    else -objective.get("flag_defense_missed_penalty", 0.) if row.flag_defense == "missed" else 0.)
                    if record["learnable"] else 0.))
            np.testing.assert_allclose(rewards, record["rewards"], atol=1e-10, rtol=0)
            old_values = {v["player"]: v["old_value"] for v in record["critic_values"]}
            virtual_turns = ()
            if "actor_gae_clock" in record:
                assert record["actor_gae_clock"] == "player_turns_including_eliminated", "unknown Actor GAE clock"
                assert record["critic_gae_clock"] in ("player_decisions", "player_turns_including_eliminated"), "unknown Critic GAE clock"
                virtual_turns = virtual_player_turns(player=actor, next_player=row.state.player,
                    player_count=players, turn_step=config.turn_step)
                assert list(virtual_turns) == record["virtual_player_turns"], "virtual turns mismatch"
            trace.append(PlayerMAPPOTransition(state, action, record["old_log_probability"], values,
                rewards, row.terminal, actor, record["learnable"], tuple(old_values[p] for p in range(players)),
                virtual_turns=virtual_turns))
            slot.history.append_encoded_rows(row.records)
            slot.critic_history.append_encoded_rows(row.centralized_records)
            checked += 1
        gae_options = dict(bootstrap_values=tuple(saved.get("bootstrap_values", [0.]*players)),
            discount=objective["discount"], gae_lambda=objective["gae_lambda"], behavior_version=payload["update"]-1)
        samples = player_generalized_advantages(trace, **gae_options)
        # Actor-only archives used virtual turns for policy advantages but the
        # old real-decision clock for all Critic targets. Keep both replayable.
        legacy_values = (player_generalized_advantages(
            [replace(t, virtual_turns=()) for t in trace], **gae_options)
            if any(step["row"].get("critic_gae_clock", "player_decisions") == "player_decisions"
                   for step in saved["steps"]) else samples)
        for sample, legacy_value, step in zip(samples, legacy_values, saved["steps"], strict=True):
            record = step["row"]
            critic_sample = (sample if record.get("critic_gae_clock") == "player_turns_including_eliminated"
                             else legacy_value)
            np.testing.assert_allclose([sample.advantage, critic_sample.value_target],
                [record["raw_advantage"], record["value_target"]], rtol=0, atol=1e-8)
            for other in critic_sample.additional_values:
                expected = next(v for v in record["critic_values"] if v["player"] == other.player)
                np.testing.assert_allclose(other.target, expected["target"], rtol=0, atol=1e-8)
            if sample.learnable:
                norm = payload["normalization"]
                expected = ((sample.advantage - norm["mean"]) / (norm["std"] + norm["epsilon"])
                            if norm["std"] >= norm["epsilon"] else sample.advantage)
                np.testing.assert_allclose(expected, record["normalized_advantage"], rtol=0, atol=1e-8)
    return {"verified_games_or_segments": len(payload["games"]), "verified_decisions": checked,
            "checks": ["legal moves", "all public histories", "all critic histories", "public tactical labels",
                       "realized rewards", "personal raw and normalized advantages", "all player value targets"]}


def summarize(run_dir, last_updates):
    files = sorted((Path(run_dir)/"tactical_diagnostics/updates").glob("update_*.rank000.json"))[-last_updates:]
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    if not reports:
        raise ValueError("no completed tactical logs found")
    print(f"update {reports[0]['update']}–{reports[-1]['update']}；{len(reports)} 轮抽样统计，不是全部对局成功率")
    print(f"抽样棋步 {sum(r['sampled_decisions'] for r in reports)}；学习动作 {sum(r['learner_decisions'] for r in reports)}")
    for name, title in TACTIC_NAMES.items():
        rows = [r["categories"][name] for r in reports]
        totals = {k: sum(r[k] for r in rows) for k in ("opportunities", "selected", "raw_positive", "raw_negative", "optimizer_uses")}
        print(f"{title}: 选中 {totals['selected']}/{totals['opportunities']}，"
              f"原始正/负优势 {totals['raw_positive']}/{totals['raw_negative']}，实际优化器使用 {totals['optimizer_uses']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--last-updates", type=int, default=20)
    parser.add_argument("--replay", type=Path, help="trusted local training .pt.gz only")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--game", type=int, default=0)
    parser.add_argument("--export-json", type=Path)
    args = parser.parse_args()
    if args.replay:
        with gzip.open(args.replay, "rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=False)
        if args.verify:
            print(json.dumps(verify_replay(payload), ensure_ascii=False, indent=2))
        if not payload["games"]:
            print("本轮所选环境没有采到棋步。")
            return
        game = payload["games"][args.game]
        print(f"game {args.game}/{len(payload['games'])}: {game['game_id']}, "
              f"start_ply={game['initial_ply']}, terminal={game['terminal']}")
        rows = [s["row"] for s in game["steps"]]
        if args.export_json:
            initial = game["initial_slot"]["game"]
            # Human-readable static board uses absolute seat-0 coordinates.
            from junqi.game import _shared_game_boards
            board = _shared_game_boards(initial["config"].variant)[0]
            args.export_json.write_text(json.dumps(dict(update=payload["update"], game_id=game["game_id"],
                initial_ply=game["initial_ply"], coordinates="initial board: seat 0; actions: acting player's board",
                referee_initial_pieces=[dict(code=board.encode(p), owner=v.owner, kind=v.kind.value)
                                        for p, v in initial["pieces"].items()], steps=rows),
                ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            for row in rows:
                print(f"ply={row['event']['ply']} player={row['event']['actor']} action={row['action']} "
                      f"A={row['raw_advantage']:+.5f} normalized={row['normalized_advantage']} "
                      f"V={row['old_value']:+.5f} target={row['value_target']:+.5f} "
                      f"uses={row['optimizer_uses']} tactics={','.join(row['event']['taken'])}")
    elif args.run_dir and args.last_updates > 0:
        summarize(args.run_dir, args.last_updates)
    else:
        parser.error("provide --run-dir and a positive --last-updates, or --replay")


if __name__ == "__main__":
    main()
