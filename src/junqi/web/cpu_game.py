"""One CPU-only frozen model/game per subprocess; JSON over private pipes."""
from __future__ import annotations

import os

# Set before importing anything which can transitively import torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import json
from pathlib import Path
import random
import sys
import time
import traceback

import torch

from ..board import PASS_ACTION
from ..training.inference import InferenceEngine
from .replay import ReplayRecorder


def board_geometry(board):
    points = []
    four = board.point_count == 129
    for point in board.points:
        code = point.code
        if four and code >= 120:
            offset = code - 120
            x, y = (offset % 3 - 1) * 2, (1 - offset // 3) * 2
        else:
            arm, offset = divmod(code, 30)
            row, col = offset // 5 + 1, offset % 5 + 1
            x, y = col - 3, row + (2 if four else 0)
            if four:
                for _ in range(arm):
                    x, y = -y, x
            elif arm == 1:
                x, y = -x, -y
        points.append({"code": code, "x": x, "y": y, "kind": point.kind.value})
    return {"points": points, "paths": [{"from": p.start, "to": p.end, "kind": p.kind.value} for p in board.path_records]}


class CpuGame:
    def __init__(self, request):
        threads = request.get("threads", 2)
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        if hasattr(os, "nice"):
            os.nice(10)
        self.human = request.get("seat", 0)
        self.temperature = request.get("temperature", 0.7)
        self.seed = request.get("seed", 20260911)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.engine = InferenceEngine.from_checkpoint(
            request["checkpoint"], mode=request["mode"], device="cpu",
            dead_rules_enabled=request["dead_rules_enabled"], temporal_cache_entries=8,
        )
        for module in (self.engine.policy, self.engine.layout):
            if any(p.device.type != "cpu" or not torch.isfinite(p).all() for p in module.parameters()):
                raise ValueError("模型包含非 CPU 参数或非有限数值")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU worker unexpectedly initialized CUDA")
        self.game, self.history = self.engine.new_game(seed=self.seed)
        self.metadata = {"checkpoint": Path(request["checkpoint"]).name,
                         "update": self.engine.checkpoint_update, "mode": self.engine.mode.value,
                         "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),
                         "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                         "pid": os.getpid(), "threads": torch.get_num_threads(), "seed": self.seed,
                         "dead_rules_enabled": self.engine.policy.config.dead_rules_enabled}
        self.inference_seconds = 0.0
        self.ai_moves = 0
        self.ended = False
        self.review = ReplayRecorder(self.game, self.human, self.state())
        self.opening_frames = [self.state(), *self._advance()]

    def _spectator_reason(self):
        if getattr(self, "ended", False) or self.game.is_terminal or self.game.config.player_count != 4:
            return None
        if not self.game.active_players[self.human]:
            return "eliminated"
        if not any(action != PASS_ACTION for action in self.game.legal_actions(self.human)):
            return "no_legal_moves"
        return None

    def _advance(self):
        # At most one round, including when the human has been eliminated.
        # A spectator can request further rounds without blocking the server.
        started = time.perf_counter()
        frames = []
        for _ in range(self.game.config.player_count):
            if self.ended or self.game.is_terminal:
                break
            if self.game.current_player == self.human:
                if self._spectator_reason() != "no_legal_moves":
                    break
                # A compulsory pass requires no human choice or model inference.
                # Recheck each turn so a temporarily blocked player regains control.
                self.game.step(PASS_ACTION)
                self.history.append_after_step(self.game)
            else:
                self.engine.step(self.game, self.history, temperature=self.temperature)
                self.ai_moves += 1
            self.inference_seconds = time.perf_counter() - started
            # Capture the human's observation now, before a later AI move can
            # change the board or reveal information. The browser paces these
            # snapshots without sleeping in the inference worker.
            frames.append(self._record_frame())
        self.inference_seconds = time.perf_counter() - started
        return frames

    def command(self, request):
        # Also supports small rule-engine fixtures created without a model load.
        if not hasattr(self, "review"):
            self.ended = False
            self.review = ReplayRecorder(self.game, self.human, self.state())
        if self.ended and request["op"] in ("move", "advance"):
            raise ValueError("棋局已经结束，只能查看复盘或开始新棋局")
        if request["op"] == "replay" and request.get("finish") is True:
            # Explicitly ending the game is irreversible for this session. A
            # captured human flag alone must never unlock hidden identities.
            self.ended = True
        frames = []
        if request["op"] == "move":
            if request.get("expected_ply") != self.game.ply_count:
                raise ValueError("棋局已变化，请刷新后再走棋")
            action = request.get("action")
            if type(action) is int and action == 0:
                action = [0, 0]
            if not isinstance(action, list) or len(action) != 2 or any(type(x) is not int for x in action):
                raise ValueError("无效走法")
            if self.game.current_player != self.human or self.game.is_terminal:
                raise ValueError("当前不是你的回合")
            action = tuple(action)
            if action not in self.game.legal_actions(self.human):
                raise ValueError("这一步不符合当前规则")
            self.game.step(action)
            self.history.append_after_step(self.game)
            self.inference_seconds = 0.0
            frames = [self._record_frame(), *self._advance()]
        elif request["op"] == "advance":
            frames = [self.state(), *self._advance()]
        elif request["op"] not in ("state", "replay"):
            raise ValueError("未知操作")
        result = self.state(replay=request["op"] == "replay")
        if request["op"] == "replay" and (self.game.is_terminal or self.ended):
            result["review"] = self.review.export(result=result["result"],
                                                  ended_early=not self.game.is_terminal)
        if frames:
            result["frames"] = frames
        return result

    def _record_frame(self):
        frame = self.state()
        self.review.append(self.game, frame)
        return frame

    def state(self, *, replay=False):
        # Never serialize referee pieces, StepResult.attacker/defender, or the
        # full training history. Observation is the sole visibility boundary.
        observation = self.game.observe(self.human, history_limit=None if replay else 40,
                                        include_legal_masks=False, include_candidate_masks=False)
        spectator_reason = self._spectator_reason()
        ended = getattr(self, "ended", False)
        your_turn = self.game.current_player == self.human and not self.game.is_terminal and not ended and spectator_reason is None
        result = None
        if self.game.result:
            reward = self.game.rewards()[self.human]
            result = {"reason": self.game.result.reason.value,
                      "outcome": "win" if reward > 0 else "loss" if reward < 0 else "draw"}
        return {"model": self.metadata, "human_seat": self.human, "ply": self.game.ply_count,
                "review_supported": True, "ended": ended,
                "current_player": observation.current_player, "your_turn": your_turn,
                "active_players": list(observation.active_players), "result": result,
                "spectator_reason": spectator_reason,
                "no_interaction_plies": self.game.no_interaction_plies,
                "no_capture_plies": self.game.no_interaction_plies,
                "no_capture_draw_plies": self.game.config.no_interaction_draw_plies,
                "passes_remaining": list(observation.passes_remaining),
                "max_passes_per_player": 4,
                "ai_moves": self.ai_moves, "inference_seconds": self.inference_seconds,
                "board": board_geometry(self.game.board_for(self.human)),
                "pieces": [None if p is None else {"owner": p.owner,
                           "name": p.kind.chinese_name if p.kind else "暗棋",
                           "kind": p.kind.value if p.kind else None,
                           "visible": p.identity_visible, "moved": p.has_moved} for p in observation.points],
                "legal_actions": list(self.game.legal_actions(self.human)) if your_turn else [],
                "history": [{"ply": e.ply, "actor": e.actor, "action": e.action,
                             "combat": e.combat.value, "eliminated": e.eliminated_players,
                             "flag_captured_owner": e.flag_captured_owner} for e in observation.history]}


def main():
    game = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with torch.inference_mode():
                if request["op"] == "new":
                    game = CpuGame(request)
                    result = game.state()
                    result["frames"] = game.opening_frames
                    game.opening_frames = []
                elif game is not None:
                    result = game.command(request)
                else:
                    raise ValueError("请先创建棋局")
            response = {"ok": True, "state": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
