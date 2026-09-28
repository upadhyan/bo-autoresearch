"""The experiment log: an append-only SQLite `events` table, and the state folded from it."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
EVENT_TYPES = {"run_started", "trial_started", "trial_finished", "trial_failed"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
"""


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(_SCHEMA)
    return con


def append(con: sqlite3.Connection, type: str, actor: str, payload: dict) -> int:
    if type not in EVENT_TYPES:
        raise ValueError(f"unknown event type: {type}")
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    cur = con.execute(
        "INSERT INTO events (type, schema_version, ts, actor, payload) VALUES (?, ?, ?, ?, ?)",
        (type, SCHEMA_VERSION, ts, actor, json.dumps(payload, sort_keys=True)),
    )
    return cur.lastrowid or 0


def read(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute("SELECT seq, type, schema_version, ts, actor, payload FROM events ORDER BY seq")
    return [{"seq": s, "type": t, "schema_version": v, "ts": ts, "actor": a, "payload": json.loads(p)}
            for s, t, v, ts, a, p in rows]


def state(events: list[dict]) -> dict:
    """Fold the events into the run's state. Everything the CLI reports comes from here."""
    # ponytail: refolds the whole log per command; materialise state tables if logs get large
    run: dict = {}
    trials: dict[int, dict] = {}
    for e in events:
        p = {k: v for k, v in e["payload"].items() if k != "rationale"}
        if e["type"] == "run_started":
            run = p
        elif e["type"] == "trial_started":
            trials[p["trial"]] = {**p, "status": "running"}
        elif e["type"] == "trial_finished":
            trials[p["trial"]].update(p, status="finished")
        elif e["type"] == "trial_failed":
            trials[p["trial"]].update(p, status="failed")
    return {"run": run, "trials": [trials[n] for n in sorted(trials)]}
