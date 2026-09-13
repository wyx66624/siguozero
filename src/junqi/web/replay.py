"""Private, compact board changes for post-game replay; no model inference."""
from __future__ import annotations


class ReplayRecorder:
    def __init__(self, game, viewer, initial):
        self.viewer = viewer
        self.initial = {**initial, "pieces": self._pieces(game, initial),
                        "your_turn": False, "legal_actions": [], "history": []}
        self.previous = self.initial["pieces"]
        self.steps = []

    def _pieces(self, game, frame):
        board = game.board_for(self.viewer)
        result = []
        for code, observed in enumerate(frame["pieces"]):
            piece = game.piece_at(board.decode(code))
            result.append(None if piece is None else {
                **observed, "kind": piece.kind.value,
                "name": piece.kind.chinese_name, "visible": True,
            })
        return result

    def append(self, game, frame):
        if frame["ply"] != self.initial["ply"] + len(self.steps) + 1:
            raise ValueError("Replay must record every action exactly once")
        pieces = self._pieces(game, frame)
        changes = [[code, piece] for code, piece in enumerate(pieces)
                   if piece != self.previous[code]]
        fields = ("ply", "current_player", "active_players", "result",
                  "no_interaction_plies", "no_capture_plies", "passes_remaining", "ai_moves")
        self.steps.append({"changes": changes, "event": frame["history"][-1],
                           "state": {key: frame[key] for key in fields}})
        self.previous = pieces

    def export(self, *, result, ended_early):
        return {"format_version": 1, "visibility": "all_pieces", "initial": self.initial,
                "steps": self.steps, "result": result, "ended_early": ended_early}
