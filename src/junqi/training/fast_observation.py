"""Allocation-light two-player observations with the original visibility contract."""
from collections import Counter
from functools import lru_cache
import numpy as np

from ..pieces import DEPLOYMENT_PIECE_SEQUENCE
from .encoding import ACTION_PLAYER_PAD, EXACT_PIECE_CODE_STRIDE, NO_CAPTURE_COUNTER_MAX, OWN_PIECE_CODES, action_point_coordinates
from .history_arrays import observation_width

_INVENTORY=Counter(DEPLOYMENT_PIECE_SEQUENCE)

@lru_cache(maxsize=4)
def _coordinates(mode):
    return tuple(action_point_coordinates(i,mode) for i in range(60))

@lru_cache(maxsize=4096)
def _casualties(counts):
    seen=Counter();result=[];counts=dict(counts)
    for kind in DEPLOYMENT_PIECE_SEQUENCE:
        seen[kind]+=1
        result.append(seen[kind]<=counts.get(kind,0))
    return tuple(result)

def _indices(game,points):
    board=game._boards[0]
    lookup=getattr(board,'_encoded_points',None)
    return np.fromiter(map(lookup.__getitem__ if lookup is not None else board.encode,points),dtype=np.intp,count=len(points))

def public_rows(game,mode):
    if game._inventory is None:raise ValueError('actor observations require inventory tracking from game initialization')
    dead=game.config.dead_rules_enabled;width=observation_width(mode,dead)
    rows=np.zeros((2,width+14),dtype=np.int16)
    pieces=tuple(game._pieces.items());points=tuple(p for p,_ in pieces)
    indices=_indices(game,points)
    event=game._public_history[-1] if game._public_history else None
    mask_begin=60+(75 if dead else 0)
    coordinates=_coordinates(mode)
    for viewer,row in enumerate(rows):
        own_indices=indices if viewer==0 else (indices+30)%60
        masks=game._inventory.masks(game,viewer)
        codes=[]
        for point,piece in pieces:
            # Own identity is visible by rule. Opponents retain the referee's
            # exact public-knowledge lookup, including revealed flags.
            kind=piece.kind if piece.owner==viewer else game._known_piece_kind(viewer,point,piece)
            codes.append(2 if kind is None else OWN_PIECE_CODES[kind]+(0 if piece.owner==viewer else 2*EXACT_PIECE_CODE_STRIDE))
        packed=np.fromiter(map(masks.__getitem__,points),dtype=np.int32,count=len(points))
        row[own_indices]=codes
        row[mask_begin+own_indices]=packed&8191
        row[mask_begin+60+own_indices]=packed>>13
        if dead:row[85:110]=_casualties(tuple(game._known_casualties[viewer][1-viewer].items()))
        if event is not None:
            if event.start is None:source=target=0
            else:
                board=game._boards[viewer];lookup=getattr(board,'_encoded_points',None)
                source=lookup[event.start] if lookup is not None else board.encode(event.start)
                target=lookup[event.end] if lookup is not None else board.encode(event.end)
            row[width:width+5]=(*( (0,0,0,0) if (source,target)==(0,0) else (*coordinates[source],*coordinates[target]) ),event.actor^viewer)
        row[width+5:width+10]=(event is not None,min(game.no_interaction_plies,NO_CAPTURE_COUNTER_MAX),
            int(game._active[viewer])+2*int(game._active[1-viewer]),
            int(game._flag_revealed[viewer])+2*int(game._flag_revealed[1-viewer]),
            ACTION_PLAYER_PAD if game.current_player is None else game.current_player^viewer)
        row[width+10:width+12]=(game._passes_remaining[viewer],game._passes_remaining[1-viewer])
    return rows

def centralized_rows(game,mode,public):
    game.enable_inventory_tracking();game.enable_critic_inventory_tracking()
    rows=public_rows(game,mode) if public is None else public.copy()
    width=observation_width(mode,game.config.dead_rules_enabled)
    if rows.shape!=(2,width+14):raise ValueError('centralized observations require all current public rows')
    pieces=tuple(game._pieces.items());points=tuple(p for p,_ in pieces);indices=_indices(game,points)
    owners=np.fromiter((piece.owner for _,piece in pieces),dtype=np.int16,count=len(pieces))
    codes=np.fromiter((OWN_PIECE_CODES[piece.kind] for _,piece in pieces),dtype=np.int16,count=len(pieces))
    masks=np.fromiter((1<<game._critic_inventory_slots[p] for p in points),dtype=np.int32,count=len(points))
    low,high=masks&8191,masks>>13
    remaining=[Counter(),Counter()]
    for _,piece in pieces:remaining[piece.owner][piece.kind]+=1
    dead=game.config.dead_rules_enabled;mask_begin=60+(75 if dead else 0)
    casualties=[_casualties(tuple((_INVENTORY-counts).items())) for counts in remaining] if dead else None
    for viewer,row in enumerate(rows):
        own_indices=indices if viewer==0 else (indices+30)%60
        row[own_indices]=codes+(owners^viewer)*(2*EXACT_PIECE_CODE_STRIDE)
        row[mask_begin+own_indices]=low;row[mask_begin+60+own_indices]=high
        if dead:row[85:110]=casualties[1-viewer]
    return rows
