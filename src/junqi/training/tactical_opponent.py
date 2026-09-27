"""Public-information hard rules for a frozen, live-champion opponent.

This changes the opponent's behavior distribution, never the learner's actions
or rewards. The only rule inputs are the private Actor observation and its legal
actions; referee identities and centralized Critic states are forbidden.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import torch

from ..board import PASS_ACTION
from ..game import CombatOutcome, JunqiGame
from ..pieces import PIECE_RANKS, PieceType
from .arena import synchronized_error
from .encoding import EXACT_PIECE_CODE_STRIDE, OWN_PIECE_CODES
from .modes import TrainingMode
from .rollout import FrozenPolicyActor


LIVE_CHAMPION_ID = "live_champion_tactical"
RULE_VERSION = 1
RULE_PRIORITY = ("capture_flag", "bomb_known_high", "capture_known_engineer")
_EXACT = {code + EXACT_PIECE_CODE_STRIDE * owner: (owner, kind)
          for kind, code in OWN_PIECE_CODES.items() for owner in range(4)}


def forced_tactical_actions(state):
    """Return the first triggered rule and all its legal actions, or no override.

    Equal-rank/bomb trades also remove an exposed engineer and are permitted.
    Hidden identities, even if obvious to the referee, never trigger a rule.
    """
    if state.centralized:
        raise ValueError("tactical opponent cannot read centralized observations")
    if state.mode is TrainingMode.TWO_PLAYER:
        raise ValueError("tactical champion opponent requires a four-player mode")
    board = state.current_board
    matches = {name: [] for name in RULE_PRIORITY}
    for action in state.legal_actions:
        if action == PASS_ACTION:
            continue
        own = _EXACT.get(int(board[action[0]]))
        enemy = _EXACT.get(int(board[action[1]]))
        if own is None or own[0] != 0 or enemy is None or enemy[0] not in (1, 3):
            continue
        attacker, defender = own[1], enemy[1]
        removes = JunqiGame._combat_outcome_for_kinds(attacker, defender) in (
            CombatOutcome.ATTACKER_WINS, CombatOutcome.BOTH_REMOVED)
        if defender is PieceType.FLAG and removes:
            matches["capture_flag"].append(action)
        if attacker is PieceType.BOMB and PIECE_RANKS.get(defender, 0) >= 7:
            matches["bomb_known_high"].append(action)
        if defender is PieceType.ENGINEER and removes:
            matches["capture_known_engineer"].append(action)
    for name in RULE_PRIORITY:
        if matches[name]:
            return name, tuple(matches[name])
    return None, ()


class TacticalOpponentActor(FrozenPolicyActor):
    """Restrict triggered decisions to rule actions; preserve ordinary sampling.

    Same-priority alternatives retain the base model's masked source/destination
    policy. Returned log-probabilities describe this restricted distribution.
    These actions must remain learnable=False in PPO.
    """
    def __init__(self, policy, *, rule_counts=None, rules=forced_tactical_actions, **kwargs):
        super().__init__(policy, **kwargs)
        self.rule_counts = rule_counts if rule_counts is not None else Counter()
        self.rules = rules

    def sample(self, states, *, count=1, **kwargs):
        selected, names = [], []
        for state in states:
            name, candidates = self.rules(state)
            selected.append(replace(state, legal_actions=candidates) if name else state)
            names.append(name or "model_fallback")
        result = super().sample(selected, count=count, **kwargs)
        # Count only successful calls; internal OOM retries are not extra moves.
        for name in names:
            self.rule_counts[name] += count
        return result


def committed_champion(run_directory, context):
    """Read one committed champion, broadcast it, and use its immutable snapshot.

    best.pt is a repairable mirror. The model-selection commit prevents mixing
    a new champion's metadata with the previous champion's weights.
    """
    entry, error = None, None
    if context.primary:
        try:
            directory = Path(run_directory).resolve() / "model_selection"
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            snapshot, digest = state["best_snapshot"], state["best_sha256"]
            path = (directory / snapshot).resolve()
            if (not path.is_relative_to(directory) or not path.is_file()
                    or not isinstance(digest, str) or len(digest) != 64):
                raise ValueError("live tactical opponent needs a persisted champion snapshot")
            entry = dict(id=LIVE_CHAMPION_ID, file=snapshot, sha256=digest,
                         update=int(state["best_update"]))
        except Exception as exc:
            error = f"live champion: {type(exc).__name__}: {exc}"
    synchronized_error(context, error)
    return context.broadcast_object(entry)


def pack_state_dict(state):
    """Prepare model weights for the existing single-replica loader."""
    groups = {}
    for name, value in state.items():
        groups.setdefault(str(value.dtype), []).append((name, value))
    packs = {}
    for dtype, items in groups.items():
        flat = torch.cat([value.detach().reshape(-1) for _, value in items]).cpu()
        views, offset = {}, 0
        for name, value in items:
            end = offset + value.numel()
            views[name] = (offset, end, tuple(value.shape))
            offset = end
        packs[dtype] = {"flat": flat, "views": views}
    return packs
