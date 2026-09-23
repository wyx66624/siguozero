"""Local tactical patterns from the actor's public information, not an oracle.

Only sampled environment lanes run this diagnostic. No counterfactual is
stepped through the referee and no hidden opponent rank decides a label.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..board import ArmPoint, PASS_ACTION, PointKind
from ..game import CombatOutcome, JunqiGame
from ..pieces import PIECE_RANKS, Piece, PieceType
from .futile_attack import EVIDENCE_ATTRIBUTE


TACTIC_NAMES = {
    "bomb_known_high": "炸已知大子",
    "bomb_inferred_high": "炸公开候选可确认的大子",
    "capture_known_flag": "吃已知军旗",
    "capture_known_weaker": "吃已知较小子",
    "engineer_known_mine": "工兵挖已知雷",
    "known_losing_attack": "攻击公开可确认打不过的子",
    "flag_threat_removal": "吃或兑掉公开可确认的夺旗威胁",
    "engineer_escape": "工兵脱离公开可确认的直接威胁",
    "high_piece_escape": "大子脱离公开可确认的直接威胁",
    "engineer_exposure": "工兵走到公开可确认可被吃兑的位置",
    "quiet_reversal": "同一子无吃子原路返回",
    "quiet_repeat": "无吃子期间重复最近十二次自己的走法",
    "rear_move_streak": "连续三次以上后两排内部无吃子移动",
}


@dataclass(frozen=True)
class PublicPiece:
    owner: int
    kinds: frozenset[PieceType]
    known: PieceType | None
    moved: bool


def _rear(point, player):
    return isinstance(point, ArmPoint) and point.seat == player and point.row >= 5


class PublicThreats:
    """Guaranteed local threats under all publicly possible source identities.

    'Escape' only means no *confirmed* one-move threat on this static board.
    Hidden threats, intervening turns and longer combinations remain unknown.
    """

    def __init__(self, game, player):
        self.game, self.player = game, player
        self.board = game.board_for(player)
        self.pieces = {}
        for point, piece in game.pieces.items():
            kind = game._known_piece_kind(player, point, piece)
            candidates = (frozenset((kind,)) if kind is not None else
                          frozenset(game._inventory.live[player][point]) if game._inventory is not None
                          else game.public_candidates[point])
            self.pieces[self.board.encode(point)] = PublicPiece(piece.owner, candidates, kind, piece.has_moved)

    def attackers(self, pieces):
        occupied = frozenset(pieces)
        result = {}
        for start, piece in pieces.items():
            if self.game.are_allies(self.player, piece.owner) or not self.game.active_players[piece.owner]:
                continue
            if (not piece.kinds or piece.kinds.intersection((PieceType.FLAG, PieceType.MINE))
                    or self.board.point(start).kind is PointKind.HEADQUARTERS):
                continue
            # Only engineer/non-engineer changes movement; intersect both if ambiguous.
            targets = None
            for engineer in {kind is PieceType.ENGINEER for kind in piece.kinds}:
                geometric = self.game._geometric_targets(self.board, start,
                    Piece(piece.owner, PieceType.ENGINEER if engineer else PieceType.BOMB), occupied)
                targets = geometric if targets is None else targets.intersection(geometric)
            for target in targets or ():
                defender = pieces.get(target)
                if defender is None or self.game.are_allies(piece.owner, defender.owner):
                    continue
                if self.board.point(target).kind is PointKind.CAMP:
                    continue
                if all(JunqiGame._combat_outcome_for_kinds(a, d) in (
                        CombatOutcome.ATTACKER_WINS, CombatOutcome.BOTH_REMOVED)
                       for a in piece.kinds for d in defender.kinds):
                    result.setdefault(target, []).append(start)
        return result

    def known_move(self, action):
        """Publicly determined outcome only; ambiguous fights have no safe label."""
        start, end = action
        attacker, defender = self.pieces[start], self.pieces.get(end)
        outcome = CombatOutcome.MOVE
        if defender is not None:
            outcomes = {JunqiGame._combat_outcome_for_kinds(a, d)
                        for a in attacker.kinds for d in defender.kinds}
            if len(outcomes) != 1:
                return None
            outcome = outcomes.pop()
        if outcome not in (CombatOutcome.MOVE, CombatOutcome.ATTACKER_WINS):
            return None  # Trading/dying is not an escape.
        pieces = dict(self.pieces)
        del pieces[start]
        if defender is not None and defender.known is PieceType.FLAG:
            pieces = {code: p for code, p in pieces.items() if p.owner != defender.owner}
        pieces[end] = attacker
        return pieces


def public_tactical_event(game, action):
    player = game.current_player
    public = PublicThreats(game, player)
    board, pieces = public.board, public.pieces
    legal = game.legal_actions()
    opportunities = {}
    def label(name, pair):
        opportunities.setdefault(name, []).append(pair)

    history = []
    for event in reversed(game.public_history):
        if event.was_attack or event.eliminated_players:
            break
        if event.actor == player:
            history.append(event)
            if len(history) == 12:
                break
    rear_streak = 0
    for event in history:
        if not (_rear(event.start, player) and _rear(event.end, player)):
            break
        rear_streak += 1
    evidence = getattr(game, EVIDENCE_ATTRIBUTE, None)
    evidence = evidence[player] if evidence is not None else {}
    threats = public.attackers(pieces)
    flags = [code for code, p in pieces.items() if p.owner == player and p.known is PieceType.FLAG]
    flag_attackers = {start for flag in flags for start in threats.get(flag, ())}
    for pair in legal:
        if pair == PASS_ACTION:
            continue
        source, target = pair
        own, other = pieces[source], pieces.get(target)
        rank = PIECE_RANKS.get(own.known, 0)
        if other is not None:
            target_rank = PIECE_RANKS.get(other.known, 0)
            if own.known is PieceType.BOMB and min(PIECE_RANKS.get(k, 0) for k in other.kinds) >= 7:
                label("bomb_known_high" if other.known is not None else "bomb_inferred_high", pair)
            if other.known is PieceType.FLAG:
                label("capture_known_flag", pair)
            if rank > target_rank > 0:
                label("capture_known_weaker", pair)
            if own.known is PieceType.ENGINEER and other.known is PieceType.MINE:
                label("engineer_known_mine", pair)
            possible_probe = own.known is PieceType.ENGINEER and not other.moved
            loses = all(JunqiGame._combat_outcome_for_kinds(own.known, kind) is CombatOutcome.DEFENDER_WINS
                        for kind in other.kinds)
            if rank and (loses or (not possible_probe and rank <= evidence.get(board.decode(target), 0))):
                label("known_losing_attack", pair)
            if target in flag_attackers and all(JunqiGame._combat_outcome_for_kinds(own.known, kind) in (
                    CombatOutcome.ATTACKER_WINS, CombatOutcome.BOTH_REMOVED) for kind in other.kinds):
                label("flag_threat_removal", pair)
        else:
            if rear_streak >= 2 and _rear(board.decode(source), player) and _rear(board.decode(target), player):
                label("rear_move_streak", pair)
            if history and board.decode(source) == history[0].end and board.decode(target) == history[0].start:
                label("quiet_reversal", pair)
            if any(e.start == board.decode(source) and e.end == board.decode(target) for e in history):
                label("quiet_repeat", pair)
        # Avoid rescanning other moves: only engineers and threatened high ranks need risk probes.
        if own.known is PieceType.ENGINEER or (rank >= 7 and source in threats):
            after = public.known_move(pair)
            if after is not None:
                exposed = target in public.attackers(after)
                if source in threats and not exposed:
                    label("engineer_escape" if own.known is PieceType.ENGINEER else "high_piece_escape", pair)
                if own.known is PieceType.ENGINEER and other is None and exposed:
                    label("engineer_exposure", pair)
    action = tuple(action)
    selected_source = pieces.get(action[0]) if action != PASS_ACTION else None
    selected_target = pieces.get(action[1]) if action != PASS_ACTION else None
    return dict(actor=player, ply=game.ply_count + 1, action=action, legal_count=len(legal),
        opportunities=opportunities, taken=[name for name, moves in opportunities.items() if action in moves],
        not_taken=[name for name, moves in opportunities.items() if action not in moves],
        own_flag_threats=sorted(flag_attackers), threatened_own_pieces=threats,
        no_capture_plies_before=game.no_interaction_plies, prior_rear_move_streak=rear_streak,
        active_players=list(game.active_players), revealed_flags=list(game.revealed_flags),
        attacker_known=selected_source.known.value if selected_source and selected_source.known else None,
        defender_known_before=selected_target.known.value if selected_target and selected_target.known else None,
        defender_candidates_before=sorted(k.value for k in selected_target.kinds) if selected_target else [])


def finish_tactical_event(event, result, game):
    # Referee truth is explicitly separate from pre-action public labels.
    event["referee_outcome"] = dict(combat=result.combat.value,
        attacker=result.attacker.kind.value if result.attacker else None,
        defender=result.defender.kind.value if result.defender else None,
        flag_captured_owner=result.flag_captured_owner, eliminated_players=list(result.eliminated_players),
        terminal=game.is_terminal, terminal_rewards=list(game.rewards()) if game.is_terminal else None,
        next_player=game.current_player, no_capture_plies=game.no_interaction_plies,
        result_reason=game.result.reason.value if game.result else None)
    return event
