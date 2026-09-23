"""Bounded MAPPO tactical logs and lossless sampled training replay segments."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import gzip
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import torch

from .tactical_events import TACTIC_NAMES


@dataclass(frozen=True)
class TacticalLogSettings:
    enabled: bool = False
    environment_lanes: int = 2
    probe_samples: int = 96
    keep_updates: int = 100
    max_detail_mib: int = 1024

    @classmethod
    def from_config(cls, raw):
        values = raw.get("runtime", {}).get("tactical_logging", {})
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise ValueError("invalid runtime.tactical_logging fields")
        result = cls(**values)
        if type(result.enabled) is not bool:
            raise ValueError("tactical logging enabled must be boolean")
        for field in ("environment_lanes", "keep_updates", "max_detail_mib"):
            if type(getattr(result, field)) is not int or getattr(result, field) < 1:
                raise ValueError(f"tactical logging {field} must be a positive integer")
        if type(result.probe_samples) is not int or not 0 <= result.probe_samples <= 512:
            raise ValueError("tactical probe_samples must be in 0..512")
        return result


def _atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class MAPPOTrainingAudit:
    def __init__(self, settings, *, update, rank=0):
        self.settings, self.update, self.rank = settings, update, rank
        self.report_id = uuid.uuid4().hex
        self.games, self.live, self.samples, self.probes = {}, {}, [], []
        self.indices = ()
        self.total_lanes = self.total_decisions = self.optimizer_steps = 0
        self.parent_seconds = self.probe_seconds = 0.

    def begin(self, pool):
        self.total_lanes = len(pool.slots)
        count = min(self.settings.environment_lanes, self.total_lanes)
        # Rotate evenly spaced lanes. This consumes no self-play RNG state.
        self.indices = tuple(sorted({(self.update + i * self.total_lanes // count) % self.total_lanes
                                     for i in range(count)}))
        for index in self.indices:
            self.new_game(index, pool.slots[index])

    def new_game(self, index, slot):
        if index not in self.indices:
            return
        from .rollout import BaseGamePool
        started = time.perf_counter()
        game_id = f"{self.report_id}.{index}.{len(self.games)}"
        initial = copy.deepcopy(BaseGamePool._slot_state_dict(slot))
        self.games[game_id] = dict(game_id=game_id, lane=index, initial_slot=initial, steps=[],
            initial_ply=slot.game.ply_count, starts_at_initial_position=slot.game.ply_count == 0,
            terminal=False, layout_behavior_version=slot.layout_behavior_version,
            opponent_id=slot.opponent_id, teammate_id=slot.teammate_id,
            learner_team=slot.learner_team, learner_seat=slot.primary_seat,
            scenario=slot.historical_scenario or "self_play")
        self.live[index] = game_id
        self.parent_seconds += time.perf_counter() - started

    def record(self, index, row, state, *, learnable, behavior_version, rewards, reward_terms):
        if index not in self.indices:
            return None
        started = time.perf_counter()
        if row.tactical is None:
            raise RuntimeError("sampled MAPPO lane is missing its public tactical diagnostic")
        game = self.games[self.live[index]]
        record = dict(game_id=game["game_id"], event=row.tactical,
            legal_actions=list(state.legal_actions), learnable=learnable,
            behavior_version=behavior_version if learnable else None, collector_version=behavior_version,
            policy_source="learner" if learnable else "frozen",
            frozen_policy_id=(None if learnable else game["opponent_id"]
                              if row.tactical["actor"] % 2 != game["learner_team"] else game["teammate_id"]),
            rewards=list(rewards), reward_terms=reward_terms, optimizer_uses=0)
        # Keep the exact four private/global post-action rows for replay and input reconstruction.
        # They are never supplied to the actor through this diagnostic object.
        game["steps"].append(dict(row=record, public_rows=row.records.copy(),
                                 critic_rows=row.centralized_records.copy()))
        game["terminal"] = row.terminal
        game["last_ply"] = row.tactical["ply"]
        self.parent_seconds += time.perf_counter() - started
        return record

    def bootstrap(self, values):
        for index, targets in values.items():
            if index in self.live:
                self.games[self.live[index]]["bootstrap_values"] = list(targets)

    def bind(self, samples, *, normalization, objective):
        self.normalization, self.objective = normalization, objective
        self.samples = [s for s in samples if s.training_diagnostic is not None]
        for sample in self.samples:
            row = sample.training_diagnostic
            row.update(action=sample.action, old_log_probability=sample.old_log_prob,
                old_probability=math.exp(sample.old_log_prob), old_value=sample.old_value,
                # Preserve the actual pre-normalization signal, also for
                # archived Actor-only runs with a different Critic clock.
                raw_advantage=row.get("raw_advantage", sample.value_target - sample.old_value),
                normalized_advantage=sample.advantage if sample.learnable else None,
                value_target=sample.value_target, value_player=sample.value_player,
                critic_values=[dict(player=sample.value_player, old_value=sample.old_value,
                                    target=sample.value_target),
                               *[dict(player=v.player, old_value=v.old_value, target=v.target)
                                 for v in sample.additional_values]])
        candidates = [s for s in self.samples if s.learnable]
        rng = random.Random(71683 + self.update * 17 + self.rank)
        selected = {}
        # Per-category selected/missed opportunities first, then a general control sample.
        buckets = []
        for name in TACTIC_NAMES:
            for taken in (True, False):
                bucket = [s for s in candidates if name in s.training_diagnostic["event"]["opportunities"]
                          and (name in s.training_diagnostic["event"]["taken"]) == taken]
                rng.shuffle(bucket)
                buckets.append(bucket[:4])
        for round_index in range(4):
            for bucket in buckets:
                if len(bucket) > round_index and len(selected) < self.settings.probe_samples:
                    selected[id(bucket[round_index])] = bucket[round_index]
        rng.shuffle(candidates)
        for sample in candidates:
            if len(selected) >= self.settings.probe_samples:
                break
            selected[id(sample)] = sample
        self.probes = list(selected.values())

    def commit_step(self, samples):
        """Invoked only by an actual optimizer post-step, including CUDA graphs.

        Uses are not claimed to be unclipped/direct gradients: entropy, clipping
        and gradients shared with other samples can change the observed direction.
        """
        self.optimizer_steps += 1
        for sample in samples:
            if sample.learnable and sample.training_diagnostic is not None:
                sample.training_diagnostic["optimizer_uses"] += 1

    def measure(self, trainer, phase):
        started = time.perf_counter()
        was_training = trainer.policy.training
        trainer.policy.eval()
        try:
            with torch.no_grad(), trainer._autocast():
                for chunk in trainer._learner_chunks(self.probes) if self.probes else ():
                    states = [s.state for s in chunk]
                    # Full forward deliberately avoids mutating the rollout KV cache.
                    features = trainer.policy._encode_full(states, pack_prefixes=True)
                    logs, _, lengths = trainer.policy._all_legal_log_probs(features, states)
                    probs = logs.float().exp().cpu().tolist()
                    offset = 0
                    for sample, length in zip(chunk, lengths, strict=True):
                        values = probs[offset:offset + length]
                        moves = sample.state.legal_actions
                        distribution = dict(zip(moves, values, strict=True))
                        best = max(distribution, key=distribution.get)
                        row = sample.training_diagnostic
                        row[phase] = dict(selected_probability=distribution[sample.action],
                            legal_probabilities=values, argmax_action=best,
                            category_probability={name: sum(distribution[tuple(move)] for move in actions)
                                for name, actions in row["event"]["opportunities"].items()})
                        offset += length
        finally:
            trainer.policy.train(was_training)
            self.probe_seconds += time.perf_counter() - started

    def finish(self):
        rows = [s.training_diagnostic for s in self.samples if s.learnable]
        categories = {}
        for name, title in TACTIC_NAMES.items():
            offered = [r for r in rows if name in r["event"]["opportunities"]]
            chosen = [r for r in offered if name in r["event"]["taken"]]
            missed = [r for r in offered if name not in r["event"]["taken"]]
            probed = [r for r in offered if "before" in r and "after" in r]
            categories[name] = dict(title=title, opportunities=len(offered), selected=len(chosen),
                not_selected=len(missed), selection_fraction=len(chosen) / len(offered) if offered else None,
                raw_positive=sum(r["raw_advantage"] > 0 for r in chosen),
                raw_negative=sum(r["raw_advantage"] < 0 for r in chosen),
                normalized_positive=sum(r["normalized_advantage"] > 0 for r in chosen),
                normalized_negative=sum(r["normalized_advantage"] < 0 for r in chosen),
                selected_raw_advantage_mean=sum(r["raw_advantage"] for r in chosen) / len(chosen) if chosen else None,
                missed_chosen_action_raw_advantage_mean=sum(r["raw_advantage"] for r in missed) / len(missed) if missed else None,
                optimizer_uses=sum(r["optimizer_uses"] for r in chosen),
                never_updated=sum(r["optimizer_uses"] == 0 for r in chosen),
                probe_samples=len(probed),
                probability_before_mean=sum(r["before"]["category_probability"][name] for r in probed) / len(probed) if probed else None,
                probability_after_mean=sum(r["after"]["category_probability"][name] for r in probed) / len(probed) if probed else None)
        games = [g for g in self.games.values() if g["steps"]]
        completed = [g for g in games if g["terminal"]]
        self.report = dict(schema_version=1, report_id=self.report_id, update=self.update, rank=self.rank,
            scope="rotating environment lanes; not all training decisions; probes are stratified samples",
            label_information="actor public information before action; referee outcomes stored separately",
            note="Unchosen actions have no advantage target. Local threat/reversal labels do not prove a strategic mistake.",
            sampled_lanes=self.indices, total_lanes=self.total_lanes, total_decisions=self.total_decisions,
            sampled_decisions=len(self.samples), learner_decisions=len(rows), frozen_decisions=len(self.samples)-len(rows),
            optimizer_steps_applied=self.optimizer_steps, probe_samples=len(self.probes),
            completed_game_statistics=dict(count=len(completed),
                mean_plies=sum(g["last_ply"] for g in completed) / len(completed) if completed else None,
                max_plies=max((g["last_ply"] for g in completed), default=None),
                draws=sum(all(r == 0 for r in g["steps"][-1]["row"]["event"]["referee_outcome"]["terminal_rewards"])
                          for g in completed)),
            players={str(player): dict(decisions=sum(r["event"]["actor"] == player for r in rows),
                raw_advantage_sum=sum(r["raw_advantage"] for r in rows if r["event"]["actor"] == player),
                positive=sum(r["raw_advantage"] > 0 and r["event"]["actor"] == player for r in rows),
                negative=sum(r["raw_advantage"] < 0 and r["event"]["actor"] == player for r in rows))
                     for player in range(4)},
            normalization=self.normalization, objective=self.objective, categories=categories,
            games=[{k: v for k, v in g.items() if k not in ("initial_slot", "steps")}
                   | {"recorded_plies": len(g["steps"]),
                      "result": g["steps"][-1]["row"]["event"]["referee_outcome"]["terminal_rewards"]}
                   for g in games])
        metrics = {"tactical/enabled": 1., "tactical/sampled_decisions": len(self.samples),
            "tactical/sampled_fraction": len(self.samples) / max(1, self.total_decisions),
            "tactical/learner_decisions": len(rows), "tactical/frozen_decisions": len(self.samples)-len(rows),
            "tactical/probe_samples": len(self.probes), "tactical/optimizer_steps_applied": self.optimizer_steps,
            "tactical/replay_segments": len(games),
            "tactical/complete_games": sum(g["starts_at_initial_position"] and g["terminal"] for g in games),
            "timing/tactical_parent_seconds": self.parent_seconds,
            "timing/tactical_probe_seconds": self.probe_seconds,
            "timing/tactical_worker_seconds": sum(r["event"]["diagnostic_seconds"] for r in (s.training_diagnostic for s in self.samples))}
        for name, values in categories.items():
            for key in ("opportunities", "selected", "not_selected", "raw_positive", "raw_negative",
                        "normalized_positive", "normalized_negative", "optimizer_uses", "never_updated"):
                metrics[f"tactical/{name}/{key}"] = values[key]
        self.report["metrics"] = metrics
        return metrics

    def write(self, run_directory, *, environment_plies):
        started = time.perf_counter()
        directory = Path(run_directory) / "tactical_diagnostics"
        updates, details = directory / "updates", directory / "details"
        updates.mkdir(parents=True, exist_ok=True)
        details.mkdir(parents=True, exist_ok=True)
        stem = f"update_{self.update:09d}.rank{self.rank:03d}"
        games = [g for g in self.games.values() if g["steps"]]
        self.report.update(environment_plies=environment_plies,
            action_file=f"details/{stem}.jsonl.gz", replay_file=f"details/{stem}.pt.gz",
            retention=dict(keep_updates=self.settings.keep_updates, max_detail_mib=self.settings.max_detail_mib))
        path = details / f"{stem}.jsonl.gz"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
            for game in games:
                for step in game["steps"]:
                    stream.write(json.dumps(dict(update=self.update, rank=self.rank, **step["row"]), ensure_ascii=False) + "\n")
        os.replace(temporary, path)
        path = details / f"{stem}.pt.gz"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wb", compresslevel=1) as stream:
            torch.save(dict(schema_version=1, mode=self.objective["mode"], report_id=self.report_id,
                update=self.update, objective=self.objective, normalization=self.normalization,
                visibility="referee state plus separate exact public actor histories", games=games), stream)
        os.replace(temporary, path)
        # Retain whole update pairs, under both count and byte caps, per rank.
        archives = sorted(details.glob(f"update_*.rank{self.rank:03d}.pt.gz"))
        sizes = [p.stat().st_size + (p.with_name(p.name.replace(".pt.gz", ".jsonl.gz")).stat().st_size
                 if p.with_name(p.name.replace(".pt.gz", ".jsonl.gz")).exists() else 0) for p in archives]
        total, removed = sum(sizes), 0
        while removed < len(archives) and (len(archives)-removed > self.settings.keep_updates
                                           or total > self.settings.max_detail_mib * 2**20):
            old = archives[removed]
            old.unlink()
            old.with_name(old.name.replace(".pt.gz", ".jsonl.gz")).unlink(missing_ok=True)
            total -= sizes[removed]
            removed += 1
        self.report["retained_detail_bytes"] = total
        self.report["current_details_retained"] = path.exists()
        self.report["metrics"]["timing/tactical_write_seconds"] = time.perf_counter() - started
        payload = json.dumps(self.report, ensure_ascii=False, indent=2)
        _atomic_text(updates / f"{stem}.json", payload)
        _atomic_text(directory / f"latest.rank{self.rank:03d}.json", payload)
        lines = [f"# 训练战术日志 · update {self.update}", "",
            f"抽样环境 {list(self.indices)} / {self.total_lanes}；真实训练步 {len(self.samples)} / {self.total_decisions}。",
            "机会按行动前公开信息判断；下列不是全量成功率，也不是最优战术判定。",
            "原始/归一化优势来自本轮实际个人 GAE；未选择的招法没有优势标签。", "",
            "| 情形 | 机会 | 选中 | 原始正/负优势 | 归一化正/负优势 | 实际优化器使用 | 探针机会概率前→后（样本数） |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for item in self.report["categories"].values():
            probability = (f"{item['probability_before_mean']:.2%} → {item['probability_after_mean']:.2%} ({item['probe_samples']})"
                           if item["probe_samples"] else "未抽到")
            lines.append(f"| {item['title']} | {item['opportunities']} | {item['selected']} | "
                f"{item['raw_positive']}/{item['raw_negative']} | "
                f"{item['normalized_positive']}/{item['normalized_negative']} | {item['optimizer_uses']} | {probability} |")
        lines.extend(["", f"概率前后对照：{len(self.probes)} 步分层抽样；所有抽样棋步均保留优势和价值目标。",
            f"棋谱片段 {len(games)} 段，起始步号及终局状态见 JSON；跨轮截断使用实际 bootstrap，不伪装成终局。",
            f"[逐步明细]({self.report['action_file']}) · [精确棋谱]({self.report['replay_file']})",
            f"明细保留最近 {self.settings.keep_updates} 轮且不超过 {self.settings.max_detail_mib} MiB；汇总长期保留。",
            "优化器使用次数包括可能被裁剪的样本，不等于直接梯度次数；实测前后概率也受其他样本影响。"])
        _atomic_text(directory / f"latest.rank{self.rank:03d}.md", "\n".join(lines) + "\n")
        return self.report["metrics"]
