"""PROTOTYPE — the agent-side `bo` CLI, faked in-process so a whole session prints in one go.

Two kinds of command, as borrowed from 2608.00316:
  PROBES   read-only, cheap, no budget, never logged as decisions:
           status, summary, show, verdict, trials, untested
  ACTIONS  change state, logged as events, and --rationale is mandatory:
           init, propose, record, register, enqueue, round-run, park, narrow,
           commit-change, add-dependency, stop, wrapup
Every command returns JSON. `round-run` is the deep one: it runs trials until a
harness trigger ends the round, doing replicates, verdict checks, budget refusal
and escalation internally. The agent never runs a single trial by hand.

Everything statistical here is FAKED (random sampler, verdict = crude mean gap).
The question is the interface, not the math.
"""
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "toy_project"))
import bo  # noqa: E402
from train import train_and_eval  # noqa: E402

LOG = []          # stand-in for log.db's events table (in memory: prototype rule)
STATE = {"hyps": {}, "budget_s": 60.0, "spent_s": 0.0, "round": 0, "trials": [],
         "pending_reviews": [], "expected": {}}


class Refused(Exception):
    pass


def _event(kind, actor, **payload):
    LOG.append({"seq": len(LOG) + 1, "type": kind, "actor": actor, **payload})


def action(fn):
    def wrapped(*a, rationale=None, actor="orchestrator", **kw):
        if not rationale:
            raise Refused(f"{fn.__name__}: --rationale is required for actions")
        out = fn(*a, **kw)
        _event(fn.__name__, actor, rationale=rationale, result=out)
        return out
    wrapped.__name__ = fn.__name__
    return wrapped


# ---------------- probes ----------------
def status():
    s = STATE
    return {"round": f"R{s['round']}", "budget": {"spent_s": round(s["spent_s"], 2),
            "remaining_s": round(s["budget_s"] - s["spent_s"], 2)},
            "hypotheses": {h: v["state"] for h, v in s["hyps"].items()},
            "blocking": [f"interplay review missing for {h}" for h in s["pending_reviews"]]
            + [f"expected verdict missing for {h}" for h, v in s["hyps"].items()
               if v["state"] == "active" and h not in s["expected"]]}


def show(h):
    return STATE["hyps"][h]


def untested():
    return [h for h, v in STATE["hyps"].items() if v["state"] in ("proposed", "registered")]


def trials(last=3):
    return STATE["trials"][-last:]


# ---------------- actions ----------------
@action
def propose(spec):
    hid = spec["id"]
    STATE["hyps"][hid] = {**spec, "state": "proposed", "review": None}
    return {"id": hid, "state": "proposed"}


@action
def record(kind, target, payload):
    # In the real plugin the SubagentStop hook schema-checks payload before it lands here.
    if kind == "review":
        STATE["hyps"][target]["review"] = payload
    elif kind == "expected":
        STATE["expected"][target] = payload
    elif kind == "interplay":
        STATE["pending_reviews"].remove(target)
    return {"recorded": kind, "for": target}


@action
def register(h):
    hyp = STATE["hyps"][h]
    if hyp["review"] is None:
        raise Refused(f"register {h}: no reviewer record")
    for name, lv in hyp["levers"].items():
        if not lv["low"] <= lv["baseline"] <= lv["high"]:
            raise Refused(f"register {h}: baseline of {name} outside its range")
    hyp["state"], hyp["commit"] = "registered", f"sha-{h.lower()}"
    return {"id": h, "state": "registered", "commit": hyp["commit"]}


@action
def round_run(max_trials=12):
    blocked = status()["blocking"]
    if blocked:
        raise Refused(f"round-run blocked: {blocked}")
    STATE["round"] += 1
    active = [h for h, v in STATE["hyps"].items() if v["state"] == "active"]
    ended_by = "stall"
    for i in range(max_trials):
        if STATE["spent_s"] + 0.03 > STATE["budget_s"] * 0.95:  # reserve untouchable
            ended_by = "budget"
            break
        levers = {n: random.uniform(lv["low"], lv["high"])
                  for h in active for n, lv in STATE["hyps"][h]["levers"].items()}
        with tempfile.TemporaryDirectory() as d:
            bo._trial = {"levers": levers, "seed": len(STATE["trials"]), "fidelity": {"epochs": 3},
                         "telemetry": {}, "curves": {}, "artifact_dir": d}
            res = bo.run(train_and_eval)
            res["telemetry"] = bo._trial["telemetry"]
            bo._trial = None
        STATE["spent_s"] += res["wall_clock_s"] + 0.03  # fake cost so the budget moves
        STATE["trials"].append({"seq": len(STATE["trials"]) + 1, "levers": levers, **res})
    verdicts = {h: _fake_verdict(h) for h in active}
    for h, v in verdicts.items():
        if v["verdict"] == "rejected":
            STATE["hyps"][h]["state"] = "rejected"
            STATE["pending_reviews"].append(h)  # removal event -> interplay review forced
    STATE["expected"] = {}
    return {"round": f"R{STATE['round']}", "ended_by": ended_by,
            "trials_run": i + 1 if ended_by != "budget" else i, "verdicts": verdicts,
            "incumbent": min(STATE["trials"], key=lambda t: t["objective"])["objective"],
            "next": status()["blocking"]}


@action
def activate(h):
    STATE["hyps"][h]["state"] = "active"
    return {"id": h, "state": "active"}


def _fake_verdict(h):
    name = next(iter(STATE["hyps"][h]["levers"]))
    xs = sorted(STATE["trials"], key=lambda t: t["levers"].get(name, 0))
    half = len(xs) // 2
    gap = abs(sum(t["objective"] for t in xs[:half]) / half
              - sum(t["objective"] for t in xs[half:]) / (len(xs) - half))
    v = "retained" if gap > 0.02 else "rejected"
    return {"record": f"V-R{STATE['round']}-{h}-1", "verdict": v,
            "reason": "no-improvement" if v == "rejected" else None, "effect_gap": round(gap, 4)}


# ---------------- a scripted session ----------------
def say(cmd, fn, *a, **kw):
    print(f"\n$ bo {cmd}")
    try:
        print(json.dumps(fn(*a, **kw), indent=1, default=str))
    except Refused as e:
        print(f"REFUSED: {e}")


if __name__ == "__main__":
    random.seed(0)
    H1 = {"id": "H1.v1", "title": "Warmup stabilises early training",
          "mechanism": "lr schedule", "provenance": "standard",
          "levers": {"H1.warmup_frac": {"type": "float", "low": 0.0, "high": 0.6,
                                        "baseline": 0.0, "predicted": "decrease"}}}
    H2 = {"id": "H2.v1", "title": "Label smoothing reduces overconfidence",
          "mechanism": "loss regulariser", "provenance": "standard",
          "levers": {"H2.smoothing": {"type": "float", "low": 0.0, "high": 0.2,
                                      "baseline": 0.0, "predicted": "decrease"}}}
    say("propose h1.yaml", propose, H1, rationale="loss spikes in first epoch")
    say("propose h2.yaml", propose, H2, rationale="calibration looked poor")
    say("register H1.v1   # before the reviewer ran", register, "H1.v1", rationale="ready")
    for h in ("H1.v1", "H2.v1"):
        say(f"record review {h}   # written by the reviewer subagent", record, "review", h,
            {"directive": "allow", "intent": "fits"}, rationale="reviewer verdict", actor="reviewer")
        say(f"register {h}", register, h, rationale="reviewed, fits")
        say(f"activate {h}", activate, h, rationale="slots free")
    say("round-run   # expected verdicts not yet recorded", round_run, rationale="go")
    for h, exp in (("H1.v1", "retain"), ("H2.v1", "reject")):
        say(f"record expected {h}", record, "expected", h, {"expect": exp},
            rationale="prior belief")
    say("round-run", round_run, rationale="go")
    say("status   # probe: free, not logged", status)
    say("trials --last 2", trials, 2)
    say("record interplay H2.v1   # interplay-review subagent", record, "interplay", "H2.v1",
        {"flagged_pairs": []}, rationale="no untested partner", actor="interplay-reviewer")
    say("status", status)
    print(f"\n--- {len(LOG)} events logged; last: {LOG[-1]['type']} by {LOG[-1]['actor']}")
