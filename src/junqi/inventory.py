"""Persistent, viewer-local inventory constraints. No hidden ranks are inputs.

Each surviving or removed piece is a variable with a public candidate domain.
A perfect matching against the 25 initial slots enforces inventory capacities.
Unknown-type slots represent interchangeable remaining copies. Confirmed
pieces receive persistent viewer-local one-hot identifiers, never hidden IDs.
Mutual-removal pairs remain linked so later evidence can resolve old exchanges.
"""
from __future__ import annotations

from collections import Counter
import os
try:
    from . import _inventory_native as _native_inventory
except ImportError:
    _native_inventory = None
if os.environ.get("SIGUOZERO_NATIVE_INVENTORY", "1") == "0":
    _native_inventory = None
from dataclasses import dataclass, replace
from functools import lru_cache

from .pieces import PIECE_COUNTS, PIECE_RANKS, PIECE_TYPE_ORDER, PieceType


INVENTORY_TYPES = (
    PieceType.FLAG, PieceType.MINE, PieceType.BOMB, PieceType.ENGINEER,
    PieceType.PLATOON_COMMANDER, PieceType.COMPANY_COMMANDER,
    PieceType.BATTALION_COMMANDER, PieceType.REGIMENT_COMMANDER,
    PieceType.BRIGADE_COMMANDER, PieceType.DIVISION_COMMANDER,
    PieceType.ARMY_COMMANDER, PieceType.COMMANDER,
)
INVENTORY_SLOTS = tuple(kind for kind in INVENTORY_TYPES for _ in range(PIECE_COUNTS[kind]))
TYPE_BITS = {kind: 1 << index for index, kind in enumerate(PIECE_TYPE_ORDER)}
ALL_TYPES = frozenset(PIECE_TYPE_ORDER)
INVENTORY_ENCODER_TYPE = "whole_board_inventory_multihot_linear"
# Same public candidate semantics, without direct casualty/pass board features.
INVENTORY_ONLY_ENCODER_TYPE = "whole_board_inventory_candidates_linear"
INVENTORY_MASK_ENCODING = "confirmed_exclusive_slots_v2"
_NATIVE_SLOT_BITS = tuple(TYPE_BITS[kind] for kind in INVENTORY_SLOTS)
_NATIVE_TYPE_BITS = tuple(TYPE_BITS[kind] for kind in PIECE_TYPE_ORDER)

@lru_cache(maxsize=4096)
def _domain_from_bits(bits):
    return frozenset(kind for kind in PIECE_TYPE_ORDER if bits & TYPE_BITS[kind])

SLOTS_BY_TYPE = {kind: tuple(i for i, item in enumerate(INVENTORY_SLOTS) if item is kind)
                 for kind in PIECE_TYPE_ORDER}
REMAINING_SLOT_MASKS = {kind: tuple(sum(1 << slot for slot in slots[:count])
                                  for count in range(len(slots) + 1))
                        for kind, slots in SLOTS_BY_TYPE.items()}


def slot_mask(candidates, casualties, used_slots=0, live_slots=0) -> int:
    """Unknown pieces can use only unassigned, potentially surviving slots.

    A confirmed live or dead piece owns its slot permanently. Anonymous proven
    deaths remove highest unassigned slots; already identified deaths must not
    consume another slot. Without assigned IDs this is the prefix 111->110.
    """
    if not used_slots:
        return sum(REMAINING_SLOT_MASKS[kind][min(PIECE_COUNTS[kind], max(
            0, PIECE_COUNTS[kind] - casualties.get(kind, 0)))] for kind in candidates)
    result = 0
    for kind in candidates:
        type_slots = REMAINING_SLOT_MASKS[kind][-1]
        assigned_deaths = (type_slots & used_slots & ~live_slots).bit_count()
        anonymous_deaths = max(0, casualties.get(kind, 0) - assigned_deaths)
        available = type_slots & ~used_slots
        for _ in range(min(anonymous_deaths, available.bit_count())):
            available ^= 1 << (available.bit_length() - 1)
        result |= available
    return result


@lru_cache(maxsize=4096)
def domain_bits(domain):
    """There are only 2**12 possible immutable candidate domains."""
    return sum(TYPE_BITS[kind] for kind in domain)


@lru_cache(maxsize=8192)
def _cached_slot_mask(domain, casualties, used_slots, live_slots):
    # The immutable counter snapshot keeps even externally restored casualty
    # evidence in the key. No cache state belongs in a game/checkpoint.
    return slot_mask(domain, dict(casualties), used_slots, live_slots)


def supported_domains(masks: tuple[int, ...]) -> tuple[frozenset[PieceType], ...]:
    # Moving a piece usually only permutes the variable order. Cache the
    # multiset so those moves do not rerun the matching/SCC solver.
    canonical = tuple(sorted(masks))
    supported = dict(zip(canonical, _supported_domains(canonical)))
    return tuple(supported[mask] for mask in masks)


@lru_cache(maxsize=8192)
def _supported_domains(masks: tuple[int, ...]) -> tuple[frozenset[PieceType], ...]:
    """Generalized arc consistency for a 25-variable all-different constraint.

    With a perfect matching, an unmatched edge belongs to another perfect
    matching iff it lies on an alternating cycle (the same SCC). This removes
    impossible edges in one graph pass, without enumerating hidden layouts.
    """
    if len(masks) != 25:
        raise ValueError("inventory requires exactly 25 live/removed variables")
    if _native_inventory is not None:
        return tuple(_domain_from_bits(bits) for bits in _native_inventory.supported(masks, _NATIVE_SLOT_BITS))
    edges = [tuple(j for j, kind in enumerate(INVENTORY_SLOTS) if mask & TYPE_BITS[kind])
             for mask in masks]
    right = [-1] * 25

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if right[j] < 0 or augment(right[j], visited):
                right[j] = i
                return True
        return False

    for i in sorted(range(25), key=lambda i: len(edges[i])):
        if not augment(i, set()):
            raise ValueError("public inventory constraints have no feasible assignment")
    graph = [[] for _ in range(50)]
    for i, choices in enumerate(edges):
        for j in choices:
            if right[j] == i:
                graph[25 + j].append(i)
            else:
                graph[i].append(25 + j)
    indices, low, component = [-1] * 50, [0] * 50, [-1] * 50
    stack, active = [], set()
    serial = 0

    def visit(v):
        nonlocal serial
        indices[v] = low[v] = serial
        serial += 1
        stack.append(v)
        active.add(v)
        for w in graph[v]:
            if indices[w] < 0:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in active:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            while True:
                w = stack.pop()
                active.remove(w)
                component[w] = v
                if w == v:
                    break

    for v in range(50):
        if indices[v] < 0:
            visit(v)
    return tuple(frozenset(INVENTORY_SLOTS[j] for j in choices
                           if right[j] == i or component[i] == component[25 + j])
                 for i, choices in enumerate(edges))


@lru_cache(maxsize=8192)
def guaranteed_casualties(live_masks: tuple[int, ...], dead_masks: tuple[int, ...]) -> tuple[int, ...]:
    """Minimum deaths of each type over all retained inventory assignments.

    Reserve m interchangeable copies for live variables and test feasibility.
    Capacities are at most three, so this needs at most two nontrivial matching
    checks per type after arc consistency. Inputs are sorted by the caller.
    """
    if not dead_masks:
        return (0,) * len(PIECE_TYPE_ORDER)
    if _native_inventory is not None:
        return _native_inventory.casualties(live_masks, dead_masks, _NATIVE_SLOT_BITS, _NATIVE_TYPE_BITS)
    masks = live_masks + dead_masks
    edges = [sum(1 << j for j, kind in enumerate(INVENTORY_SLOTS) if mask & TYPE_BITS[kind]) for mask in masks]

    def feasible(reserved):
        choices = edges[:len(live_masks)] + [e & ~reserved for e in edges[len(live_masks):]]
        matched = [-1] * 25

        def augment(i, seen):
            available = choices[i] & ~seen[0]
            while available:
                bit = available & -available
                available -= bit
                if seen[0] & bit:
                    continue
                seen[0] |= bit
                j = bit.bit_length() - 1
                if matched[j] < 0 or augment(matched[j], seen):
                    matched[j] = i
                    return True
            return False

        return all(augment(i, [0]) for i in sorted(range(25), key=lambda i: choices[i].bit_count()))

    counts = []
    for kind in PIECE_TYPE_ORDER:
        capacity = PIECE_COUNTS[kind]
        live_options = sum(bool(mask & TYPE_BITS[kind]) for mask in live_masks)
        maximum = min(capacity, live_options)
        if maximum > 1 and any(mask & TYPE_BITS[kind] for mask in dead_masks):
            copies = [1 << i for i, k in enumerate(INVENTORY_SLOTS) if k is kind]
            while maximum > 1 and not feasible(sum(copies[:maximum])):
                maximum -= 1
        # A supported live edge proves at least one live copy is feasible.
        counts.append(capacity - maximum)
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class MutualRemoval:
    """Current viewer-local knowledge of one past exchange, not a past token.

    Endpoints index append-only, owner-local removed-piece lists. They are
    observation records, never the referee's hidden inventory-instance IDs.
    """

    ply: int
    attacker_owner: int
    attacker_index: int
    defender_owner: int
    defender_index: int
    pairs: frozenset[tuple[PieceType, PieceType]]

    @property
    def classification(self) -> str:
        bomb = [PieceType.BOMB in pair for pair in self.pairs]
        if all(bomb):
            return "bomb"
        return "ambiguous" if any(bomb) else "equal_rank"

    def state_dict(self):
        return {"ply": self.ply, "attacker_owner": self.attacker_owner,
                "attacker_index": self.attacker_index, "defender_owner": self.defender_owner,
                "defender_index": self.defender_index,
                "pairs": tuple(sorted((a.value, b.value) for a, b in self.pairs))}


class InventoryKnowledge:
    """Persistent viewer-local constraints required by all policy encoders."""

    def __init__(self, game):
        count = game.config.player_count
        self.live = [dict() for _ in range(count)]
        self.dead = [[[] for _ in range(count)] for _ in range(count)]
        self.mutual_removals = [[] for _ in range(count)]
        self.known_slots = [dict() for _ in range(count)]
        # Reserve confirmed identifiers even after death, so a later discovery
        # cannot inherit the identity of a different piece.
        self.used_slots = [[0] * count for _ in range(count)]
        # Derived from known_slots; kept current to avoid scanning all known
        # pieces for every point encoded. It is rebuilt on state-dict restore.
        self.live_slots = [[0] * count for _ in range(count)]
        for viewer in range(count):
            for point, piece in game._pieces.items():
                known = game._known_piece_kind(viewer, point, piece)
                self.live[viewer][point] = (frozenset((known,)) if known is not None
                                             else game._piece_candidates[point])
            for owner in range(count):
                if not game._active[owner]:
                    # Elimination itself is public proof of the entire loss,
                    # even when a restored fixture has no casualty tables.
                    self.dead[viewer][owner] = [frozenset((kind,)) for kind in INVENTORY_SLOTS]
                    continue
                # Partial-position fixtures / old public histories cannot reveal
                # absent hidden ranks. Use anonymous unknown removed variables.
                missing = 25 - sum(p.owner == owner for p in game._pieces.values())
                counts = game._known_casualties[viewer][owner]
                exact = [frozenset((kind,)) for kind, number in counts.items() for _ in range(number)]
                if len(exact) > missing:
                    raise ValueError("known casualties exceed publicly missing pieces")
                unknown = ALL_TYPES if game._flag_revealed[owner] else ALL_TYPES - {PieceType.COMMANDER}
                self.dead[viewer][owner] = exact + [unknown] * (missing - len(exact))

    def clone(self):
        result = self.__class__.__new__(self.__class__)
        result.live = [dict(row) for row in self.live]
        result.dead = [[list(row) for row in owners] for owners in self.dead]
        result.mutual_removals = [list(row) for row in self.mutual_removals]
        result.known_slots = [dict(row) for row in self.known_slots]
        result.used_slots = [list(row) for row in self.used_slots]
        result.live_slots = [list(row) for row in self.live_slots]
        return result

    def transition(self, viewer, start, end, attacker_owner, defender_owner,
                   attackers, defenders, combat, *, combat_pairs=(), ply=0):
        live = self.live[viewer]
        known = self.known_slots[viewer]
        attacker_slot = known.pop(start, None)
        defender_slot = known.pop(end, None)
        if attacker_slot is not None:
            self.live_slots[viewer][attacker_owner] &= ~(1 << attacker_slot)
        if defender_slot is not None:
            self.live_slots[viewer][defender_owner] &= ~(1 << defender_slot)
        survivor_slot = defender_slot if combat.value == "defender_wins" else attacker_slot
        if combat.value != "both_removed" and survivor_slot is not None:
            known[end] = survivor_slot
            owner = defender_owner if combat.value == "defender_wins" else attacker_owner
            self.live_slots[viewer][owner] |= 1 << survivor_slot
        live.pop(start)
        if defender_owner is None:
            live[end] = attackers
            return
        live.pop(end)
        attacker_index = len(self.dead[viewer][attacker_owner])
        defender_index = len(self.dead[viewer][defender_owner])
        if combat.value in ("defender_wins", "both_removed"):
            self.dead[viewer][attacker_owner].append(attackers)
        else:
            live[end] = attackers
        if combat.value in ("attacker_wins", "both_removed"):
            self.dead[viewer][defender_owner].append(defenders)
        else:
            live[end] = defenders
        if combat.value == "both_removed":
            if not combat_pairs or ply < 1:
                raise ValueError("mutual removal requires viewer-legal pairs and its ply")
            self.mutual_removals[viewer].append(MutualRemoval(
                ply, attacker_owner, attacker_index, defender_owner, defender_index,
                frozenset(combat_pairs)))

    def eliminate(self, game, owner):
        for viewer, live in enumerate(self.live):
            for point in tuple(live):
                if game._pieces[point].owner == owner:
                    # Elimination proves all 25 are dead, not which earlier
                    # exchange used a bomb. Preserve identities of old records.
                    self.dead[viewer][owner].append(live.pop(point))
                    self.known_slots[viewer].pop(point, None)
            self.live_slots[viewer][owner] = 0

    def _assign_known_slots(self, game):
        """Assign only from each viewer's confirmed domains, in discovery order.

        Simultaneous discoveries use the owner's board order as a public tie
        breaker. Own/visible ally slots agree with the Critic's fixed IDs; a
        hidden opponent's IDs are assigned independently of referee slots.
        """
        pending = {point for viewer, live in enumerate(self.live) for point, domain in live.items()
                   if len(domain) == 1 and point not in self.known_slots[viewer]}
        if not pending:
            return
        points = sorted(pending, key=lambda p: (
            game._pieces[p].owner, game._boards[game._pieces[p].owner].encode(p)))

        def assign(viewer, point, slot=None):
            known = self.known_slots[viewer]
            if point in known or len(self.live[viewer][point]) != 1:
                return
            owner = game._pieces[point].owner
            kind = next(iter(self.live[viewer][point]))
            used = self.used_slots[viewer][owner]
            if slot is None:
                slot = next((i for i in SLOTS_BY_TYPE[kind] if not used & (1 << i)), None)
            if slot is None or INVENTORY_SLOTS[slot] is not kind or used & (1 << slot):
                raise ValueError("confirmed piece slots exceed the public inventory")
            known[point] = slot
            self.used_slots[viewer][owner] |= 1 << slot
            self.live_slots[viewer][owner] |= 1 << slot

        # Initialize owners first, making late Critic initialization and the
        # double-open teammate view use the same already-visible piece IDs.
        for point in points:
            owner = game._pieces[point].owner
            slot = (None if game._critic_inventory_slots is None
                    else game._critic_inventory_slots[point])
            assign(owner, point, slot)
        for viewer in range(game.config.player_count):
            for point in points:
                piece = game._pieces[point]
                slot = (self.known_slots[piece.owner][point]
                        if game._identity_visible_by_mode(viewer, piece) else None)
                assign(viewer, point, slot)

    def mask(self, game, viewer, point):
        domain = self.live[viewer][point]
        if len(domain) == 1:
            return 1 << self.known_slots[viewer][point]
        owner = game._pieces[point].owner
        return _cached_slot_mask(domain, tuple(game._known_casualties[viewer][owner].items()),
                                 self.used_slots[viewer][owner], self.live_slots[viewer][owner])

    def masks(self, game, viewer):
        """Encode one current public view, sharing immutable owner evidence.

        This cache lasts only for this call. Moves, deaths and restored public
        evidence are therefore reflected immediately, without invalidation or
        any persistent checkpoint state. Confirmed physical-copy IDs remain
        point-specific even when several pieces have the same known rank.
        """
        known = self.known_slots[viewer]
        owners = [
            (tuple(casualties.items()), self.used_slots[viewer][owner],
             self.live_slots[viewer][owner])
            for owner, casualties in enumerate(game._known_casualties[viewer])
        ]
        shared = {}
        result = {}
        for point, domain in self.live[viewer].items():
            if len(domain) == 1:
                result[point] = 1 << known[point]
                continue
            owner = game._pieces[point].owner
            key = (owner, domain)
            mask = shared.get(key)
            if mask is None:
                mask = shared[key] = _cached_slot_mask(domain, *owners[owner])
            result[point] = mask
        return result

    def _share_allied_domains(self):
        changed = False
        for a, b in ((0, 2), (1, 3)):
            for point in self.live[a]:
                merged = self.live[a][point] & self.live[b][point]
                changed |= merged != self.live[a][point] or merged != self.live[b][point]
                self.live[a][point] = self.live[b][point] = merged
            # Historical anonymous losses may have different orders. Only
            # exchange records with the same public event can be joined.
            other = {event.ply: (index, event)
                     for index, event in enumerate(self.mutual_removals[b])}
            for index, event in enumerate(self.mutual_removals[a]):
                if event.ply not in other:
                    continue
                peer_index, peer = other[event.ply]
                if (event.attacker_owner, event.defender_owner) != (peer.attacker_owner, peer.defender_owner):
                    raise ValueError("allied mutual-removal histories disagree")
                pairs = event.pairs & peer.pairs
                changed |= pairs != event.pairs or pairs != peer.pairs
                if pairs != event.pairs:
                    self.mutual_removals[a][index] = replace(event, pairs=pairs)
                if pairs != peer.pairs:
                    self.mutual_removals[b][peer_index] = replace(peer, pairs=pairs)
        return changed

    def _propagate_mutual_removals(self, viewer):
        changed = False
        for index, event in enumerate(self.mutual_removals[viewer]):
            attackers = self.dead[viewer][event.attacker_owner][event.attacker_index]
            defenders = self.dead[viewer][event.defender_owner][event.defender_index]
            pairs = frozenset((a, b) for a, b in event.pairs if a in attackers and b in defenders)
            if not pairs:
                raise ValueError("public mutual-removal constraints have no feasible pair")
            a_domain, b_domain = frozenset(a for a, _ in pairs), frozenset(b for _, b in pairs)
            changed |= pairs != event.pairs or a_domain != attackers or b_domain != defenders
            if pairs != event.pairs:
                self.mutual_removals[viewer][index] = replace(event, pairs=pairs)
            self.dead[viewer][event.attacker_owner][event.attacker_index] = a_domain
            self.dead[viewer][event.defender_owner][event.defender_index] = b_domain
        return changed

    def propagate(self, game):
        """Reach a fixed point of inventory matching and historical exchanges."""
        if not game.config.dead_rules_enabled:
            return
        points_by_owner = [[p for p, piece in game._pieces.items() if piece.owner == owner]
                           for owner in range(game.config.player_count)]
        for viewer, live in enumerate(self.live):
            for point in live:
                known = game._known_piece_kind(viewer, point, game._pieces[point])
                live[point] &= (frozenset((known,)) if known is not None
                                else game._piece_candidates[point])
        while True:
            changed = (self._share_allied_domains()
                       if game.config.information_mode.value == "double_open" else False)
            for viewer, live in enumerate(self.live):
                for owner, points in enumerate(points_by_owner):
                    domains = [live[p] for p in points] + self.dead[viewer][owner]
                    supported = supported_domains(tuple(domain_bits(d) for d in domains))
                    changed |= tuple(domains) != supported
                    for point, domain in zip(points, supported):
                        live[point] = domain
                    self.dead[viewer][owner] = list(supported[len(points):])
                changed |= self._propagate_mutual_removals(viewer)
            if not changed:
                break
        for viewer, live in enumerate(self.live):
            for owner, points in enumerate(points_by_owner):
                live_masks = tuple(sorted(domain_bits(live[p]) for p in points))
                dead_masks = tuple(sorted(domain_bits(d) for d in self.dead[viewer][owner]))
                guaranteed = guaranteed_casualties(live_masks, dead_masks)
                counts = game._known_casualties[viewer][owner]
                for kind, number in zip(PIECE_TYPE_ORDER, guaranteed):
                    counts[kind] = max(counts[kind], number)
                # Zero counters are omitted in the referee's serialized contract.
                counts += Counter()
        self._assign_known_slots(game)

    def state_dict(self):
        return {"version": 3, "live": [dict(row) for row in self.live],
                "dead": [[list(row) for row in owners] for owners in self.dead],
                "known_slots": [dict(row) for row in self.known_slots],
                "used_slots": [list(row) for row in self.used_slots],
                "mutual_removals": [[event.state_dict() for event in row] for row in self.mutual_removals]}

    @classmethod
    def from_state_dict(cls, game, state):
        if state.get("version") not in (1, 2, 3):
            raise ValueError("unsupported inventory knowledge version")
        result = cls.__new__(cls)
        result.live = [{p: frozenset(PieceType(k) for k in d) for p, d in row.items()}
                       for row in state["live"]]
        result.dead = [[[frozenset(PieceType(k) for k in d) for d in row] for row in owners]
                       for owners in state["dead"]]
        count = game.config.player_count
        if state["version"] == 3:
            if "known_slots" not in state or "used_slots" not in state:
                raise ValueError("inventory is missing confirmed piece slots")
            result.known_slots = [dict(row) for row in state["known_slots"]]
            result.used_slots = [list(row) for row in state["used_slots"]]
            if len(result.known_slots) != count or len(result.used_slots) != count:
                raise ValueError("confirmed piece slot viewer count mismatch")
        else:
            # Bare old game states can start identifiers from current public
            # knowledge. Old policy histories/checkpoints have a separate
            # incompatible encoding contract and cannot silently resume.
            result.known_slots = [dict() for _ in range(count)]
            result.used_slots = [[0] * count for _ in range(count)]
        # Version 1 never saved pair correlations. Do not reconstruct them from
        # referee identities or invent certainty for an old partial history.
        rows = [[] for _ in range(count)] if state["version"] == 1 else state["mutual_removals"]
        result.mutual_removals = [[MutualRemoval(
            **{field: raw[field] for field in ("ply", "attacker_owner", "attacker_index", "defender_owner", "defender_index")},
            pairs=frozenset((PieceType(a), PieceType(b)) for a, b in raw["pairs"])) for raw in row] for row in rows]
        if len(result.live) != count or len(result.dead) != count or len(result.mutual_removals) != count:
            raise ValueError("inventory viewer count mismatch")
        for viewer, (live, owners, events) in enumerate(zip(result.live, result.dead, result.mutual_removals)):
            if set(live) != set(game._pieces) or len(owners) != count:
                raise ValueError("inventory occupied points/owner count mismatch")
            if any(not d or game._pieces[p].kind not in d for p, d in live.items()):
                raise ValueError("inventory candidates contradict the restored position")
            for owner, dead in enumerate(owners):
                live_count = sum(game._pieces[p].owner == owner for p in live)
                if any(not d for d in dead) or len(dead) + live_count != 25:
                    raise ValueError("inventory requires exactly 25 live/removed variables per owner")
            if state["version"] == 3:
                known, reserved = result.known_slots[viewer], result.used_slots[viewer]
                if (set(known) != {p for p, domain in live.items() if len(domain) == 1}
                        or len(reserved) != count
                        or any(type(bits) is not int or not 0 <= bits < (1 << 25) for bits in reserved)):
                    raise ValueError("invalid confirmed piece slots")
                assigned = set()
                for point, slot in known.items():
                    owner = game._pieces[point].owner
                    if (type(slot) is not int or not 0 <= slot < 25
                            or live[point] != frozenset((INVENTORY_SLOTS[slot],))
                            or (owner, slot) in assigned or not reserved[owner] & (1 << slot)):
                        raise ValueError("confirmed piece slots must be unique and match known types")
                    if (owner == viewer and game._critic_inventory_slots is not None
                            and slot != game._critic_inventory_slots[point]):
                        raise ValueError("own confirmed piece slots disagree with critic slots")
                    assigned.add((owner, slot))
            used, previous_ply = set(), 0
            for event in events:
                if type(event.ply) is not int or not previous_ply < event.ply <= game.ply_count:
                    raise ValueError("invalid mutual-removal ply")
                previous_ply = event.ply
                if event.attacker_owner == event.defender_owner:
                    raise ValueError("invalid mutual-removal owners")
                for owner, index in ((event.attacker_owner, event.attacker_index),
                                     (event.defender_owner, event.defender_index)):
                    if (type(owner) is not int or not 0 <= owner < count or type(index) is not int
                            or not 0 <= index < len(owners[owner]) or (owner, index) in used):
                        raise ValueError("invalid mutual-removal endpoint")
                    used.add((owner, index))
                if not event.pairs or any(
                    a not in PIECE_RANKS and a is not PieceType.BOMB
                    or (PieceType.BOMB not in (a, b) and (a != b or a not in PIECE_RANKS))
                    for a, b in event.pairs
                ):
                    raise ValueError("invalid mutual-removal pairs")
        result.live_slots = [[0] * count for _ in range(count)]
        for viewer, known in enumerate(result.known_slots):
            for point, slot in known.items():
                result.live_slots[viewer][game._pieces[point].owner] |= 1 << slot
        return result
