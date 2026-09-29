"""The record inlet (#27): subagent output, schema- and quote-checked, logged with its actor.

One verdict-producing run (a harmful lever, rejected no-improvement) serves every test; each test
records under its own agent id.
"""
import json

import pytest

from conftest import bo, events
from test_verdicts import lever, spec, verdict_run


@pytest.fixture(scope="module")
def run(tmp_path_factory, project_python):
    d = tmp_path_factory.mktemp("records")
    harmful = (spec({"x": lever()}), '    term += 2 * lever("H1.x")')
    d1 = ("directives:\n- {id: D1, severity: discouraged, statement: slow schedules, reason: time,"
          " scope: mechanism}\n")
    run_dir, _ = verdict_run(d / "r", project_python, 1, [harmful], extra=d1)
    return run_dir


def record(run_dir, kind, body, agent="a1", actor="round-analyst"):
    f = run_dir.parent / f"{agent}-{kind}.json"
    f.write_text(json.dumps(body))
    return bo(run_dir.parents[1], "record", kind, "--file", str(f), "--agent-id", agent,
              "--actor", actor, "--rationale", "my output")


def recorded(run_dir, agent):
    code, out = bo(run_dir.parents[1], "check", "recorded", agent)
    assert code == 0, out
    return out["recorded"]


def test_check_recorded_turns_true_once_a_valid_record_exists_and_the_record_is_an_event(run):
    assert recorded(run, "exp1") is False
    code, out = record(run, "expected", {"hypothesis": "H1.v1", "verdict": "retain"}, agent="exp1",
                       actor="orchestrator")
    assert code == 1 and "reason" in out["reason"], out  # refused: nothing stored
    assert recorded(run, "exp1") is False
    code, out = record(run, "expected", {"hypothesis": "H1.v1", "verdict": "reject",
                                         "reason": "a harmful lever"}, agent="exp1", actor="orchestrator")
    assert code == 0, out
    assert recorded(run, "exp1") is True and recorded(run, "someone-else") is False
    [e] = [e for e in events(run) if e["type"] == "record" and e["payload"]["agent_id"] == "exp1"]
    assert e["actor"] == "orchestrator" and e["payload"]["kind"] == "expected"
    assert e["payload"]["record"] == {"hypothesis": "H1.v1", "verdict": "reject",
                                      "reason": "a harmful lever"}
    assert e["payload"]["rationale"] == "my output"


def verdict_record(run_dir):
    [e] = [e for e in events(run_dir) if e["type"] == "verdict"][-1:]
    return e["payload"]


def narrative(v, text, quotes, **kw):
    return {"round": 1, "text": text, "cites": [v["id"]], "quotes": quotes, "diagnostics": [],
            "suggestions": [], "generation": False, **kw}


def test_a_narrative_quoting_its_verdict_record_exactly_is_accepted(run):
    v = verdict_record(run)
    vt, dl = v["sqrt_vt"]["upper"], v["delta_stat"]["upper"]
    quotes = [{"record": v["id"], "field": "sqrt_vt.upper", "value": vt},
              {"record": v["id"], "field": "delta_stat.upper", "value": dl},
              {"record": v["id"], "field": "delta", "value": 0.1},
              {"record": v["id"], "field": "best_point.H1.x", "value": v["best_point"]["H1.x"]}]
    text = (f"H1 matters (√V_T upper {vt:.3f}) but only hurts: Δ's upper bound {dl:.4g} "
            f"is below δ = 0.1, so it was rejected ({v['id']}).")
    code, out = record(run, "narrative", narrative(v, text, quotes), agent="an1")
    assert code == 0, out
    assert recorded(run, "an1")


@pytest.mark.parametrize("field", ["sqrt_vt.upper", "delta_stat.lower", "delta"])
def test_a_narrative_misquoting_its_verdict_record_is_refused(run, field):
    v = verdict_record(run)
    actual = v["delta"] if field == "delta" else v[field.split(".")[0]][field.split(".")[1]]
    wrong = actual + 0.01
    code, out = record(run, "narrative", narrative(v, "see the record", [
        {"record": v["id"], "field": field, "value": wrong}]), agent=f"mis-{field}")
    assert code == 1 and field in out["reason"] and repr(wrong) in out["reason"], out
    assert not recorded(run, f"mis-{field}")


def test_a_number_in_the_text_that_no_quote_backs_is_refused(run):
    v = verdict_record(run)
    vt = v["sqrt_vt"]["upper"]
    quotes = [{"record": v["id"], "field": "sqrt_vt.upper", "value": vt}]
    code, out = record(run, "narrative", narrative(v, f"√V_T is at most {vt + 0.05:.3f}", quotes),
                       agent="an2")
    assert code == 1 and f"{vt + 0.05:.3f}" in out["reason"], out
    # a diagnostic quoting δ without backing it is caught too
    code, out = record(run, "narrative", narrative(v, "fine", [], diagnostics=["δ = 0.1 is loose"]),
                       agent="an2")
    assert code == 1 and "0.1" in out["reason"], out
    assert not recorded(run, "an2")


def test_cited_verdict_records_must_exist(run):
    v = verdict_record(run)
    code, out = record(run, "narrative", narrative(v, "fine", [], cites=["V-R9-H1.v1-1"]), agent="an3")
    assert code == 1 and "cites" in out["reason"] and "V-R9-H1.v1-1" in out["reason"], out
    code, out = record(run, "narrative", narrative(v, "fine", [
        {"record": "V-R9-H1.v1-1", "field": "delta", "value": 0.1}]), agent="an3")
    assert code == 1 and "quotes[0].record" in out["reason"], out


def idea(title, mechanism):
    return spec({"y": lever()}, title=title, mechanism=mechanism)


def review(**kw):
    return {"hypothesis": "H1.v1", "directive_verdict": "allow", "intent": "fits", "conflict": "none",
            "conflict_with": [], "rationale": "No directive touches it.", "strict": False, **kw}


def test_an_accepted_proposal_becomes_proposed_hypotheses(run):
    body = {"hypotheses": [idea("Warm up the rate", "Warmup tames early steps."),
                           idea("Clip the gradients", "Clipping bounds each step.")]}
    code, out = record(run, "proposal", body, agent="gen1", actor="hypothesis-generator")
    assert code == 0, out
    new = [e for e in events(run) if e["type"] == "hypothesis_proposed"][-2:]
    assert out["hypotheses"] == [e["payload"]["id"] for e in new]
    assert all(e["actor"] == "hypothesis-generator" for e in new)
    assert [e["payload"]["spec"]["title"] for e in new] == ["Warm up the rate", "Clip the gradients"]
    status = bo(run.parents[1], "status")[1]["hypotheses"]
    assert {h["id"]: h["status"] for h in status if h["id"] in out["hypotheses"]} == \
        dict.fromkeys(out["hypotheses"], "proposed")


def test_review_interplay_and_distill_spec_records_are_accepted(run):
    v = verdict_record(run)
    body = {"hypotheses": [idea("Scale the rate", "Scaling moves the step."),
                           idea("Decay the rate", "Decay shrinks late steps.")]}
    other = record(run, "proposal", body, agent="gen2", actor="hypothesis-generator")[1]["hypotheses"][0]
    assert record(run, "review", review(), agent="rev1", actor="registration-reviewer")[0] == 0
    code, out = record(run, "review", review(hypothesis=other, directive_verdict="deprioritize",
                                             directive="D1", intent="stretch", conflict="rival",
                                             conflict_with=["H1.v1"], strict=True),
                       agent="rev2", actor="registration-reviewer")
    assert code == 0, out
    flags = [{"partner": "H1.v1", "reason": f"H1's harm (Δ upper {v['delta_stat']['upper']}) may "
              "hinge on the rate", "cites": [v["id"]]}]
    body = {"newcomer": other, "flags": flags,
            "quotes": [{"record": v["id"], "field": "delta_stat.upper", "value": v["delta_stat"]["upper"]}]}
    assert record(run, "interplay", body, agent="ip1", actor="interplay-reviewer")[0] == 0
    body = {"content": "# Distill\nKeep nothing: H1 was rejected.", "cites": [v["id"]], "changes": [], "quotes": []}
    assert record(run, "distill_spec", body, agent="ds1")[0] == 0
    kinds = {e["payload"]["agent_id"]: (e["actor"], e["payload"]["kind"])
             for e in events(run) if e["type"] == "record"}
    assert {k: kinds[k] for k in ("rev1", "rev2", "ip1", "ds1")} == {
        "rev1": ("registration-reviewer", "review"), "rev2": ("registration-reviewer", "review"),
        "ip1": ("interplay-reviewer", "interplay"), "ds1": ("round-analyst", "distill_spec")}


BAD = [
    ("proposal", {"hypotheses": [idea("A", "m")]}, "2 mechanisms"),
    ("proposal", {"hypotheses": [idea("A", "m"), {**idea("B", "n"), "provenance": "stolen"}]},
     "hypotheses[1]: provenance"),
    ("review", review(directive_verdict="maybe"), "directive_verdict"),
    ("review", review(directive_verdict="prune"), "directive"),
    ("review", review(intent="meh"), "intent"),
    ("review", review(conflict="rival"), "conflict_with"),
    ("review", review(conflict_with=["H99.v1"]), "H99.v1"),
    ("review", review(strict="yes"), "strict"),
    ("review", review(conflict="rival", conflict_with=["H1.v1"]), "itself"),
    ("review", review(hypothesis=["H1.v1"]), "hypothesis"),
    ("review", review(hypothesis="H99.v1"), "hypothesis"),
    ("review", {**review(), "mood": "good"}, "mood"),
    ("interplay", {"flags": []}, "removed"),
    ("interplay", {"removed": "H1.v1", "flags": [{"partner": "H1.v1", "reason": "x", "cites": []}]},
     "flags[0].partner"),
    ("interplay", {"newcomer": "H1.v1", "flags": []}, "H1.v1 is rejected"),
    ("narrative", {"round": 1, "text": "x", "cites": [], "diagnostics": [], "suggestions": [],
                   "generation": False}, "cites"),
    ("expected", {"hypothesis": "H1.v1", "verdict": "retain", "reason": "x",
                  "quotes": [{"record": ["V"], "field": "delta", "value": 0.1}]}, "quotes[0].record"),
    ("expected", {"hypothesis": "H1.v1", "verdict": "retain", "reason": "x",
                  "quotes": [{"record": "V-R1-H1.v1-1", "field": 3, "value": 0.1}]}, "quotes[0].field"),
    ("distill_spec", {"content": "x", "cites": [], "changes": [],
                      "quotes": [{"record": "V-R1-H1.v1-1", "field": "delta", "value": 0.1}]},
     "not cited"),
    ("narrative", {"round": 7, "text": "x", "cites": [], "diagnostics": [], "suggestions": [],
                   "generation": False}, "round"),
    ("narrative", {"round": 1, "text": "x", "cites": ["V-R1-H1.v1-1"], "diagnostics": [],
                   "suggestions": []}, "generation"),
    ("expected", {"hypothesis": "H1.v1", "verdict": "win", "reason": "x"}, "verdict"),
    ("expected", {"hypothesis": "H1.v1", "verdict": "retain", "reason": "two\nlines"}, "reason"),
    ("distill_spec", {"content": "", "cites": []}, "content"),
]


@pytest.mark.parametrize("kind,body,field", BAD, ids=[f"{k}-{f}" for k, _, f in BAD])
def test_a_payload_failing_its_schema_is_refused_naming_the_field(run, kind, body, field):
    agent = f"bad-{kind}-{BAD.index((kind, body, field))}"
    code, out = record(run, kind, body, agent=agent)
    assert code == 1 and field in out["reason"], out
    assert not recorded(run, agent)


def test_an_unknown_kind_and_an_unreadable_file_are_refused(run):
    code, out = record(run, "memo", {})
    assert code == 1 and "memo" in out["reason"], out
    code, out = bo(run.parents[1], "record", "review", "--file", "/nope.json", "--rationale", "r")
    assert code == 1 and "cannot read" in out["reason"], out
