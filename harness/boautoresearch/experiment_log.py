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
               "hypothesis_rejected", "hypothesis_inconclusive", "hypothesis_escalated", "lever_frozen", "record",
               "commit_change", "dependency_added", "equivalence_check", "epoch_started", "narrowed",
               "hypothesis_prioritized", "hypothesis_parked", "hypothesis_unparked", "trial_enqueued",
               "agent_trial_dropped", "checkpoint", "user_pause", "user_resume"}

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
    records: list[dict] = []
    queue: list[dict] = []  # agent-chosen trials, each for one round
    noise = calib = proxy = delta = ended = escalated = paused = None
    epoch = {"epoch": 0, "round": 0}  # the first round whose trials the epoch's evidence counts
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
            for h in hyps.values():  # a BO round the queued hypothesis waited through
                if p["round"] >= 1 and h["status"] == "registered" and h["id"] not in p["hypotheses"]:
                    h["unselected"] += 1
        elif e["type"] == "round_ended":
            rounds[p["round"]]["ended"] = p["trigger"]
        elif e["type"] == "drift_check":
            rounds[p["round"]]["drift"] = p
        elif e["type"] == "delta_set":
            delta = p["delta"]
        elif e["type"] == "run_ended":
            ended = p["reason"]
        elif e["type"] == "trial_enqueued":
            queue.append(p)
        elif e["type"] == "checkpoint":  # checkpoint mode, at a round boundary
            paused = {"by": "checkpoint", "round": p["round"]}
        elif e["type"] == "user_pause":
            paused = {"by": "user"}
        elif e["type"] == "user_resume":
            paused = None
        elif e["type"] == "hypothesis_activated":
            hyps[p["id"]].update(status="active", activated_round=p["round"])
        elif e["type"] == "epoch_started":
            epoch = {"epoch": p["epoch"], "round": p["round"]}
        elif e["type"] == "noise_estimate" and "epoch" in p:  # σ re-estimated in a new epoch
            if noise and p["sigma"] is not None:
                rungs = [{**x, "sigma": p["sigma"], "mean": p["mean"]} if x["fidelity"] == p["fidelity"]
                         else x for x in noise["rungs"]]
                noise = {**noise, "rungs": rungs, "sigma": rungs[0]["sigma"],
                         "replication": any(x["sigma"] > 0 for x in rungs)}
        elif e["type"] == "noise_estimate" and p["round"] == 0:
            noise = p
        elif e["type"] == "noise_estimate":  # σ re-estimated from a round's replicates
            rounds[p["round"]]["noise"] = p
        elif e["type"] == "fidelity_calibration":
            calib, escalated = p, None
        elif e["type"] == "proxy_accepted_unvalidated":
            proxy, escalated = p, None
        elif e["type"] == "hypothesis_proposed":
            hyps[p["id"]] = {**p, "status": "proposed", "smoke": None, "commit": None,
                             "verdicts": [], "frozen": [], "escalations": [], "priority": 0,
                             "unselected": 0}
        elif e["type"] == "hypothesis_registered":  # the queue's tie-break: registration order
            hyps[p["id"]].update(status="registered", registered=e["seq"])
        elif e["type"] == "hypothesis_prioritized":
            hyps[p["id"]]["priority"] = p["priority"]
        elif e["type"] == "hypothesis_parked":  # no verdict: it returns as the same version
            hyps[p["id"]].update(status="parked", parked_from=p["from"], park_reason=p["reason"])
        elif e["type"] == "hypothesis_unparked":  # back to the queue (a proposal to its review)
            h = hyps[p["id"]]
            h.update(status="proposed" if h["parked_from"] == "proposed" else "registered", unselected=0)
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
        elif e["type"] == "narrowed":  # the new range, from the next round on (its checks restart)
            h = hyps[p["id"]]
            h["narrowed_round"] = p["round"]
            levers = h["spec"]["levers"]
            h["spec"] = {**h["spec"], "levers": {**levers, p["lever"]: {**levers[p["lever"]], **p["after"]}}}
        elif e["type"] == "lever_frozen":
            hyps[p["id"]]["frozen"].append(p["lever"])
        elif e["type"] == "record":
            # `round`: the next round to start, the one an expected verdict is for
            records.append({**p, "actor": e["actor"], "seq": e["seq"], "round": max(rounds, default=-1) + 1})
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
            "fidelity": fidelity, "delta": delta, "run_ended": ended, "epoch": epoch,
            "records": records, "paused": paused, "queue": queue}
