"""The experiment log: an append-only SQLite `events` table, and the state folded from it."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
EVENT_TYPES = {"run_started", "trial_started", "trial_finished", "trial_failed", "trial_refused",
               "round_started", "round_ended", "noise_estimate", "fidelity_calibration",
               "proxy_accepted_unvalidated", "hypothesis_proposed", "hypothesis_registered",
               "lever_smoke", "lever_committed", "delta_set", "hypothesis_activated",
               "trial_heartbeat", "trial_abandoned", "drift_check", "run_ended", "verdict",
               "hypothesis_rejected", "hypothesis_inconclusive", "hypothesis_escalated", "lever_frozen"}

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


def load(con: sqlite3.Connection) -> dict:
    """The run's state now."""
    return state(read(con))


def state(events: list[dict]) -> dict:
    """Fold the events into the run's state. Everything the CLI reports comes from here."""
    # ponytail: refolds the whole log per command; materialise state tables if logs get large
    run: dict = {}
    trials: dict[int, dict] = {}
    rounds: dict[int, dict] = {}
    hyps: dict[str, dict] = {}
    noise = calib = proxy = delta = ended = escalated = None
    for e in events:
        p = {k: v for k, v in e["payload"].items() if k != "rationale"}
        if e["type"] == "run_started":
            run, delta = p, p.get("delta")
        elif e["type"] == "trial_started":
            trials[p["trial"]] = {**p, "status": "running"}
        elif e["type"] == "trial_heartbeat":
            trials[p["trial"]].update(heartbeat_s=p["elapsed_s"], runner_pid=p.get("runner_pid"))
        elif e["type"] in ("trial_finished", "trial_failed", "trial_abandoned"):
            trials[p["trial"]].update(p, status=e["type"][len("trial_"):])
        elif e["type"] == "round_started":
            rounds[p["round"]] = {**p, "ended": None}
        elif e["type"] == "round_ended":
            rounds[p["round"]]["ended"] = p["trigger"]
        elif e["type"] == "drift_check":
            rounds[p["round"]]["drift"] = p
        elif e["type"] == "delta_set":
            delta = p["delta"]
        elif e["type"] == "run_ended":
            ended = p["reason"]
        elif e["type"] == "hypothesis_activated":
            hyps[p["id"]].update(status="active", activated_round=p["round"])
        elif e["type"] == "noise_estimate" and p["round"] == 0:
            noise = p
        elif e["type"] == "noise_estimate":  # σ re-estimated from a round's replicates
            rounds[p["round"]]["noise"] = p
        elif e["type"] == "fidelity_calibration":
            calib = p
        elif e["type"] == "proxy_accepted_unvalidated":
            proxy = p
        elif e["type"] == "hypothesis_proposed":
            hyps[p["id"]] = {**p, "status": "proposed", "smoke": None, "commit": None,
                             "verdicts": [], "frozen": [], "escalations": []}
        elif e["type"] == "hypothesis_registered":
            hyps[p["id"]]["status"] = "registered"
        elif e["type"] == "lever_smoke":
            hyps[p["id"]]["smoke"] = p
        elif e["type"] == "lever_committed":
            hyps[p["id"]]["commit"] = p["commit"]
        elif e["type"] == "verdict":  # a retain holds only until a check says otherwise
            h = hyps[p["hypothesis"]]
            h["verdicts"].append(p)
            if p["outcome"] == "retained" or h["status"] == "retained":
                h["status"] = "retained" if p["outcome"] == "retained" else "active"
        elif e["type"] == "hypothesis_rejected":
            hyps[p["id"]].update(status="rejected", condition=p["condition"])
        elif e["type"] == "hypothesis_inconclusive":
            hyps[p["id"]].update(status="inconclusive", reason=p["reason"])
        elif e["type"] == "lever_frozen":
            hyps[p["id"]]["frozen"].append(p["lever"])
        elif e["type"] == "hypothesis_escalated":
            hyps[p["id"]]["escalations"].append(p)
            if p["step"] == "rung":
                escalated = {"fidelity": p["fidelity"], "proxy": p["proxy"]}
    r0_complete = rounds.get(0, {}).get("ended") == "calibrated"
    if proxy:
        fidelity = {"fidelity": proxy["fidelity"], "proxy": "unvalidated"}
    elif calib:
        fidelity = {"fidelity": calib["chosen"], "fallback": calib["fallback"],
                    "proxy": "reference" if calib["chosen"] == run["reference_fidelity"] else "validated"}
    else:
        fidelity = {"fidelity": run.get("reference_fidelity"), "proxy": "reference"}
    fidelity = escalated or fidelity  # a stuck hypothesis moved the run up a rung
    return {"run": run, "trials": [trials[n] for n in sorted(trials)], "rounds": rounds,
            "noise": noise, "calibration": calib, "hypotheses": hyps, "r0_complete": r0_complete,
            "fidelity": fidelity, "delta": delta, "run_ended": ended}
