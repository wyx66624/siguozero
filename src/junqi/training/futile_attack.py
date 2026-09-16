"""Public battle evidence and a temporary PPO penalty for known losing attacks."""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..board import PASS_ACTION
from ..game import CombatOutcome
from ..pieces import PIECE_RANKS, PieceType


EVIDENCE_ATTRIBUTE = "_training_futile_attack_evidence"


def restore_evidence(game, evidence):
    if evidence is None:
        return
    if (not isinstance(evidence, list) or len(evidence) != game.config.player_count
            or any(not isinstance(table, dict) for table in evidence)):
        raise ValueError("invalid futile attack evidence")
    for table in evidence:
        if any(point not in game.pieces or type(rank) is not int or not 1 <= rank <= 9
               for point, rank in table.items()):
            raise ValueError("invalid futile attack evidence entry")
    setattr(game, EVIDENCE_ATTRIBUTE, [dict(table) for table in evidence])


def step_with_evidence(game, action, *, enabled=False):
    """Track surviving pieces, using only ranks legitimately known by each seat.

    Each entry is the largest known ordinary victim that the surviving piece
    defeated. An attacker of that rank or below is certainly weaker. Bombs can
    trade and engineers can clear possible mines, so those actions are exempt.
    The first encounter cannot use information revealed by its own outcome.
    """
    if not enabled:
        # A disabled/finished schedule needs no stale evidence in future saves.
        if hasattr(game, EVIDENCE_ATTRIBUTE):
            delattr(game, EVIDENCE_ATTRIBUTE)
        return game.step(action), False
    if action == 0 or tuple(action) == PASS_ACTION:
        result = game.step(action)
        if result.eliminated_players:
            for table in getattr(game, EVIDENCE_ATTRIBUTE, ()):
                for point in tuple(table):
                    if game.piece_at(point) is None:
                        del table[point]
        return result, False
    actor = game.current_player
    board = game.board_for(actor)
    start, end = (board.decode(code) for code in action)
    attacker, defender = game.piece_at(start), game.piece_at(end)
    evidence = getattr(game, EVIDENCE_ATTRIBUTE, None)
    if evidence is None:
        evidence = [{} for _ in range(game.config.player_count)]
    futile = False
    attacker_kinds = defender_kinds = ()
    if defender is not None:
        # This is the same exact-identity visibility boundary used by observe().
        # Never inspect a hidden defender's referee type to choose the penalty.
        attacker_kinds = [game._known_piece_kind(viewer, start, attacker)
                          for viewer in range(game.config.player_count)]
        defender_kinds = [game._known_piece_kind(viewer, end, defender)
                          for viewer in range(game.config.player_count)]
        rank = PIECE_RANKS.get(attacker_kinds[actor])
        possible_mine = not defender.has_moved and defender_kinds[actor] in (None, PieceType.MINE)
        exempt_engineer = attacker_kinds[actor] is PieceType.ENGINEER and possible_mine
        futile = (rank is not None and not exempt_engineer
                  and rank <= evidence[actor].get(end, 0))
    result = game.step(action)
    for viewer, table in enumerate(evidence):
        moving_rank, defending_rank = table.pop(start, 0), table.pop(end, 0)
        if result.combat in (CombatOutcome.MOVE, CombatOutcome.ATTACKER_WINS):
            victim_rank = PIECE_RANKS.get(defender_kinds[viewer], 0) if defender_kinds else 0
            rank = max(moving_rank, victim_rank)
        elif result.combat is CombatOutcome.DEFENDER_WINS:
            rank = max(defending_rank, PIECE_RANKS.get(attacker_kinds[viewer], 0))
        else:
            rank = 0
        if rank and game.piece_at(end) is not None:
            table[end] = rank
        if result.eliminated_players:
            for point in tuple(table):
                if game.piece_at(point) is None:
                    del table[point]
    setattr(game, EVIDENCE_ATTRIBUTE, evidence)
    return result, bool(futile and result.combat is CombatOutcome.DEFENDER_WINS)


@dataclass
class FutileAttackPenalty:
    initial: float
    target_environment_plies: int | None
    ema_decay: float = 0.95
    start_environment_plies: int = 0
    rate_ema: float | None = None
    reference_rate: float | None = None
    frequency_scale: float = 1.0
    updates: int = 0

    @property
    def stop_environment_plies(self):
        return (self.target_environment_plies or 0) // 2

    def coefficient(self, environment_plies):
        stop = self.stop_environment_plies
        if self.initial == 0 or environment_plies >= stop or self.start_environment_plies >= stop:
            return 0.0
        progress = min(1.0, max(0.0, (stop - environment_plies) / (stop - self.start_environment_plies)))
        return self.initial * progress * self.frequency_scale

    def observe(self, events, trials):
        if self.initial == 0 or trials == 0:
            return
        if type(events) is not int or type(trials) is not int or not 0 <= events <= trials:
            raise ValueError("invalid futile attack event counts")
        rate = events / trials
        self.rate_ema = (rate if self.rate_ema is None else
                         self.ema_decay * self.rate_ema + (1 - self.ema_decay) * rate)
        if self.reference_rate is None and self.rate_ema > 0:
            self.reference_rate = self.rate_ema
        if self.reference_rate is not None:
            self.frequency_scale = min(self.frequency_scale, self.rate_ema / self.reference_rate)
        self.updates += 1

    def state_dict(self):
        return {"version": 1, "start_environment_plies": self.start_environment_plies,
                "rate_ema": self.rate_ema, "reference_rate": self.reference_rate,
                "frequency_scale": self.frequency_scale, "updates": self.updates}

    def load_state_dict(self, state):
        if state is None:
            return
        if (state.get("version") != 1 or type(state.get("start_environment_plies")) is not int
                or state["start_environment_plies"] < 0 or type(state.get("updates")) is not int
                or state["updates"] < 0):
            raise ValueError("invalid futile attack penalty state")
        for key in ("rate_ema", "reference_rate", "frequency_scale"):
            value = state.get(key)
            if value is None and key != "frequency_scale":
                continue
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid futile attack penalty rate")
        if state.get("reference_rate") == 0:
            raise ValueError("futile attack reference rate must be positive")
        for key in ("start_environment_plies", "rate_ema", "reference_rate", "frequency_scale", "updates"):
            setattr(self, key, state[key])
