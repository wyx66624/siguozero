"""Durable, incremental game-result summaries from completed training updates."""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading


def count(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or value > 2**53 or not math.isfinite(value) or value != int(value):
        return None
    return int(value)


def training_record(record):
    return isinstance(record, dict) and any(key.startswith("rollout/") for key in record)


def result_counts(record, prefix=""):
    values = {name: count(record.get(prefix + name)) for name in ("wins", "draws", "losses")}
    if any(value is None for value in values.values()):
        return None
    return {"games": sum(values.values()), **values}


def rates(values):
    games = values["games"]
    return {**values, "win_rate": values["wins"] / games if games else None,
            "draw_rate": values["draws"] / games if games else None}


class OutcomeStore:
    """The source log remains authoritative; the SQLite index can be rebuilt."""
    def __init__(self, path, *, batch_bytes=4 * 1024 * 1024):
        self.path = Path(path)
        self.batch_bytes = batch_bytes
        self.lock = threading.Lock()

    @staticmethod
    def _fingerprints(stream, offset):
        stream.seek(0)
        head = hashlib.sha256(stream.read(min(256, offset))).hexdigest()
        stream.seek(max(0, offset - 256))
        tail = hashlib.sha256(stream.read(min(256, offset))).hexdigest()
        return head, tail

    def snapshot(self, run, *, algorithm, mode, max_update=None, session_started=None, resumed_update=None):
        run = Path(run).resolve()
        key, source = str(run), run / "metrics.jsonl"
        unit = "games" if algorithm == "ppo" else "branches"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.row_factory = sqlite3.Row
            db.executescript("""
                CREATE TABLE IF NOT EXISTS outcome_sources (
                    run TEXT PRIMARY KEY, cursor TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS outcome_updates (
                    run TEXT NOT NULL, update_number INTEGER NOT NULL,
                    timestamp REAL, unit TEXT NOT NULL, games INTEGER,
                    wins INTEGER, draws INTEGER, losses INTEGER,
                    cumulative_games INTEGER, PRIMARY KEY (run, update_number));
            """)
            row = db.execute("SELECT cursor FROM outcome_sources WHERE run=?", (key,)).fetchone()
            previous = json.loads(row[0]) if row else {}
            catching_up = False
            source_missing = not source.is_file()
            if not source_missing:
                with source.open("rb") as stream, db:
                    stat = os.fstat(stream.fileno())
                    offset = previous.get("offset", 0)
                    identity = [stat.st_dev, stat.st_ino]
                    head, tail = self._fingerprints(stream, offset)
                    reset = (previous.get("identity") != identity or stat.st_size < offset
                             or previous.get("head") != head or previous.get("tail") != tail)
                    if reset:
                        db.execute("DELETE FROM outcome_updates WHERE run=?", (key,))
                        offset = 0
                    top = db.execute("SELECT MAX(update_number) FROM outcome_updates WHERE run=?", (key,)).fetchone()[0] or 0
                    stream.seek(offset)
                    end = offset + self.batch_bytes
                    while stream.tell() < end:
                        start = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith(b"\n"):
                            stream.seek(start)  # Never consume a partially written JSON record.
                            break
                        try:
                            record = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        if not training_record(record) or record.get("mode", mode) != mode:
                            continue
                        update = count(record.get("update"))
                        if not update:
                            continue
                        row_unit = "games" if record.get("algorithm", algorithm) == "ppo" else "branches"
                        values = result_counts(record, "rollout/")
                        expected = count(record.get("rollout/base_games_completed" if row_unit == "games"
                                                    else "rollout/terminal_continuations"))
                        if values is not None and expected is not None and values["games"] != expected:
                            values = None
                        values = values or {name: None for name in ("games", "wins", "draws", "losses")}
                        cumulative = count(record.get("cumulative/base_games" if row_unit == "games"
                                                      else "cumulative/terminal_continuations"))
                        timestamp = record.get("timestamp_unix")
                        if (not isinstance(timestamp, (int, float)) or abs(timestamp) > 2**53
                                or not math.isfinite(timestamp)):
                            timestamp = None
                        payload = (key, update, timestamp, row_unit, values["games"], values["wins"],
                                   values["draws"], values["losses"], cumulative)
                        # A resumed lower update replaces the abandoned suffix. An
                        # exact duplicate of an older result is only a repeated log.
                        if update < top:
                            existing = db.execute("SELECT * FROM outcome_updates WHERE run=? AND update_number=?",
                                                  (key, update)).fetchone()
                            if existing is not None and tuple(existing) == payload:
                                continue
                            db.execute("DELETE FROM outcome_updates WHERE run=? AND update_number>=?", (key, update))
                        top = update
                        db.execute("INSERT OR REPLACE INTO outcome_updates VALUES (?,?,?,?,?,?,?,?,?)",
                                   payload)
                    offset = stream.tell()
                    head, tail = self._fingerprints(stream, offset)
                    cursor = {"identity": identity, "offset": offset, "head": head, "tail": tail}
                    if cursor != previous:
                        db.execute("INSERT OR REPLACE INTO outcome_sources VALUES (?,?)", (key, json.dumps(cursor)))
                    catching_up = offset < stat.st_size
            # Restrict the view immediately on restart, before the first new
            # training row replaces the abandoned suffix in the log index.
            where, parameters = "run=? AND unit=?", [key, unit]
            if max_update is not None:
                where += " AND update_number<=?"
                parameters.append(max_update)
            if session_started is not None and resumed_update is not None:
                where += " AND (update_number<=? OR timestamp>=?)"
                parameters.extend((resumed_update, session_started))
            totals = db.execute("""SELECT COUNT(*) AS updates_seen, COUNT(games) AS recorded_updates,
                MIN(update_number) AS first_update, MAX(update_number) AS last_update,
                COALESCE(SUM(games),0) AS games, COALESCE(SUM(wins),0) AS wins,
                COALESCE(SUM(draws),0) AS draws, COALESCE(SUM(losses),0) AS losses
                FROM outcome_updates WHERE """ + where, parameters).fetchone()
            recent = [dict(row) for row in db.execute("""SELECT update_number AS 'update', timestamp,
                games, wins, draws, losses, cumulative_games FROM outcome_updates
                WHERE """ + where + " ORDER BY update_number DESC LIMIT 12", parameters)]
            values = rates({name: totals[name] for name in ("games", "wins", "draws", "losses")})
            expected = recent[0]["cumulative_games"] if recent else None
            missing = max(0, expected - values["games"]) if expected is not None else None
            return {"available": bool(totals["recorded_updates"]), "unit": unit,
                    "perspective": "root_player" if unit == "branches" else "seat_0" if mode == "two_player" else "team_0_2",
                    "totals": values, "expected_games": expected, "unrecorded_games": missing,
                    "coverage_complete": expected == values["games"] and not catching_up and not source_missing
                        and totals["recorded_updates"] == totals["updates_seen"],
                    "recorded_updates": totals["recorded_updates"], "first_update": totals["first_update"],
                    "last_update": totals["last_update"], "latest": recent[0] if recent else None,
                    "recent": recent, "catching_up": catching_up, "source_missing": source_missing}


def evaluation_outcomes(run, selection, mode):
    """Count only evaluation reports committed in model_selection/state.json."""
    directory = (Path(run) / "model_selection").resolve()
    reports, seen, unavailable = [], set(), 0
    for name in selection.get("rounds", []):
        if not isinstance(name, str):
            unavailable += 1
            continue
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            unavailable += 1
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("evaluation_type") == "historical_only":
                continue  # The fixed panel has its own per-opponent results, never champion W/D/L.
            values = result_counts(record) if isinstance(record, dict) else None
            if (not values or values["games"] != count(record.get("games")) or not values["games"]
                    or record.get("mode", mode) != mode):
                raise ValueError("Incomplete evaluation result")
        except (OSError, ValueError):
            unavailable += 1
            continue
        reports.append({**rates(values), "candidate_update": count(record.get("candidate_update")),
                        "opponent_update": count(record.get("opponent_update")),
                        "score": (values["wins"] + 0.5 * values["draws"]) / values["games"]})
        if record.get("evaluation_type") == "fixed_reference":
            reports[-1].update(evaluation_type="fixed_reference", teammate_results=record.get("teammate_results", {}))
    return {"rounds": len(reports), "unavailable_rounds": unavailable,
            "totals": {name: sum(r[name] for r in reports) for name in ("games", "wins", "draws", "losses")},
            "latest": reports[-1] if reports else None, "recent": list(reversed(reports[-6:]))}
