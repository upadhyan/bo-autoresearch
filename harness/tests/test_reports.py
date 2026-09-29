"""Summaries and exports (#33): SUMMARY.md, rounds/NNN.md, trials.csv and hypotheses.csv, all
regenerated from the experiment log after every state change.

One planted run (δ = 0.1, σ = 0.05) serves most tests: H1 a bowl whose best (x = 0.8) beats the
baseline (0.6) by 0.16, H2 a lever that does nothing, H3 a lever that only hurts, H4 parked with its
code committed, and a discouraged directive that H1's best breaks.
"""
import csv
import json
import re
import shlex
import shutil
from pathlib import Path

import pytest

from conftest import bo, events, register, review, round_run
from test_rounds import BASE, init, make_repo, of_type, toy_env
from test_verdicts import code_hypothesis, lever, spec

DIRECTIVES = """directives:
  - id: keep-x-low
    severity: discouraged
    statement: keep H1's x at or below 0.7
    reason: larger values cost the user memory
    scope: lever values
    predicate: "H1.x <= 0.7"
"""
HYPS = [
    (spec({"x": lever(baseline=0.6, predicted="higher")}, title="The bowl has a better bottom"),
     '    term += 4 * ((lever("H1.x") - 0.8) ** 2 - 0.04)'),
    (spec({"x": lever()}, title="A lever that does nothing"), '    term += 0 * lever("H2.x")'),
    (spec({"x": lever()}, title="A lever that only hurts"), '    term += 2 * lever("H3.x")'),
    (spec({"x": lever(baseline=0.25)}, title="A lever set aside"), '    term += 0 * lever("H4.x")'),
]
GENERATED = ("SUMMARY.md", "rounds", "exports")


def record(repo, kind, body, **kw):
    f = Path(repo).parent / f"{kind}-{len(list(Path(repo).parent.glob(kind + '-*')))}.json"
    f.write_text(json.dumps(body))
    args = [f"--{k.replace('_', '-')}={v}" for k, v in kw.items()]
    return bo(repo, "record", kind, "--file", str(f), "--rationale", "the record", *args)


@pytest.fixture(scope="module")
def planted(tmp_path_factory, project_python):
    """-> (repo, run_dir, round-run outputs, the suggested narrowings' outputs)."""
    d = tmp_path_factory.mktemp("planted")
    repo = make_repo(d)
    run_dir = init(repo, project_python, BASE + "delta: 0.1\nseed: 1\n" + DIRECTIVES)
    env = toy_env(sigma=0.05)
    train = run_dir / "worktree" / "train.py"
    src = train.read_text()  # a failed trial: the plain smoke of broken code
    train.write_text(src.replace("    term = 0.0\n", "    raise RuntimeError('broken')\n"))
    assert bo(repo, "smoke", "--rationale", "does it run", env=env)[1]["trial"]["status"] == "failed"
    train.write_text(src)
    assert round_run(repo, "--rationale", "calibrate", env=env)[0] == 0
    for n, (s, line) in enumerate(HYPS, 1):
        code_hypothesis(repo, run_dir, d, n, s, line, env)
    assert bo(repo, "park", "H4", "--reason", "the user wants it later", "--rationale", "set aside")[0] == 0
    for h, v in (("H1.v1", "retain"), ("H2.v1", "reject")):
        assert record(repo, "expected", {"hypothesis": h, "verdict": v, "reason": "planted"})[0] == 0
    outs = []
    for _ in range(3):
        code, out = round_run(repo, "--rationale", "search", env=env)
        assert code == 0, out
        outs.append(out)
    first = of_type(run_dir, "verdict")[0]["id"]
    story = {"round": 1, "text": f"{first} opened the round; the useless lever went first.", "cites": [first],
             "diagnostics": ["the replicates spread wider than in the calibration round"],
             "suggestions": ["a wider range for the bowl"], "generation": False}
    assert record(repo, "narrative", story, agent_id="ra", actor="round-analyst")[0] == 0
    # R2's suggested narrowing, as the orchestrator would take it up (R2 re-judges the deferred H2 next
    # to the bowl; R3 stalls before H1's next check, so it suggests nothing)
    suggested = [ln for ln in (run_dir / "rounds" / "002.md").read_text().splitlines() if "`narrow " in ln]
    narrowed = [bo(repo, *shlex.split(ln.split("`")[1]), "--rationale", "suggested") for ln in suggested]
    return repo, run_dir, outs, narrowed


def rows(run_dir, name):
    with open(run_dir / "exports" / name) as f:
        return list(csv.DictReader(f))


META = ["seq", "round", "epoch", "commit", "fidelity", "seed", "replicate_of", "chosen_by", "status",
        "objective", "wall_clock_s", "peak_mem"]  # the spec's trials.csv metadata columns


def test_trials_csv_has_a_row_per_trial_of_every_kind_and_the_lever_cell_rule(planted):
    repo, run_dir, *_ = planted
    started = {p["trial"]: p for p in of_type(run_dir, "trial_started")}
    table = rows(run_dir, "trials.csv")
    assert [int(r["seq"]) for r in table] == sorted(started)
    header = list(table[0])
    assert header[:len(META)] == META
    groups = [c.split(":")[0] + ":" for c in header[len(META):]]
    order = ["c:", "t:", "compat:", "L:", "sampled:"]
    assert groups == sorted(groups, key=order.index)
    assert {"smoke", "baseline", "equivalence", "sampler", "confirmation"} <= {r["chosen_by"] for r in table}
    assert "failed" in {r["status"] for r in table}
    assert any(r["replicate_of"] for r in table)
    for r in table:
        t = started[int(r["seq"])]
        assert r["round"] == str(t.get("round", "")) and r["replicate_of"] == str(t.get("replicate_of", ""))
        assert json.loads(r["fidelity"]) == t["fidelity"] and r["commit"] == t["commit"]
    # empty: the lever didn't exist in the trial's code; its baseline: it existed and ran at baseline
    r0 = [r for r in table if r["round"] == "0"]
    assert r0 and all(r["L:H1.x"] == "" and r["L:H4.x"] == "" for r in r0)
    searched = [r for r in table if r["round"] not in ("", "0")]
    assert searched and all(r["L:H4.x"] == "0.25" for r in searched)  # parked: pinned at baseline
    assert all(r["L:H1.x"] != "" for r in searched)
    assert all(r["compat:keep-x-low"] == str(float(r["L:H1.x"]) <= 0.7) for r in searched)


def sections(text):
    """{heading: [lines]} of a markdown file's `## ` sections, in order."""
    out, name = {}, None
    for ln in text.splitlines():
        if ln.startswith("## "):
            name = ln[3:]
            out[name] = []
        elif name and ln.strip():
            out[name].append(ln)
    return out


SECTIONS = ["Where we are", "Working", "Not working", "Still testing", "Set aside", "Up next"]


def test_summary_says_where_the_run_is_and_what_works_in_words_that_match_the_state(planted):
    repo, run_dir, outs, _ = planted
    text = (run_dir / "SUMMARY.md").read_text()
    s = sections(text)
    assert list(s) == SECTIONS
    assert bo(repo, "summary")[1] == {"summary": text}  # the probe: the file is up to date
    status = bo(repo, "status")[1]
    hyps = {h["id"]: h for h in status["hypotheses"]}
    # the planted truth: H1 works, H2 doesn't matter, H3 matters but only hurts, H4 set aside. H2 is
    # rejected `no-improvement`, not `irrelevant`: next to the bowl, BO leaves too much of H1.x's box
    # unexplored to bound H2's range below δ (#21 amendment), so the summary can't say it doesn't matter
    assert {h: x["status"] for h, x in hyps.items()} == {
        "H1.v1": "retained", "H2.v1": "rejected", "H3.v1": "rejected", "H4.v1": "parked"}
    [h1] = s["Working"]
    last = [v for v in of_type(run_dir, "verdict") if v["hypothesis"] == "H1.v1"][-1]["delta_stat"]
    assert h1 == (f"- H1.v1 The bowl has a better bottom: Δ {last['estimate']:.3g} "
                  f"[{last['lower']:.3g}, {last['upper']:.3g}]")
    assert s["Not working"] == ["- H2.v1 A lever that does nothing: doesn't help, and may matter",
                                "- H3.v1 A lever that only hurts: doesn't help, and may matter"]
    assert s["Still testing"] == ["- none"]
    assert s["Set aside"] == ["- H4.v1 A lever set aside: parked: the user wants it later"]
    assert s["Up next"] == ["- 0 queued"]
    incumbent = outs[-1]["incumbent"]
    base = of_type(run_dir, "noise_estimate")[0]["rungs"][0]["mean"]  # R0's baseline replicates
    where = s["Where we are"]
    assert where[0].startswith(f"- Incumbent: loss {incumbent['mean']:.4g} (trial {incumbent['trial']}, "
                               "confirmed) against the baseline's ")
    assert f"against the baseline's {base:.4g}: " in where[0]
    gain, in_delta = where[0].split(": ")[-1].removesuffix("δ").split(" better, ")
    assert float(gain) == pytest.approx(0.16, abs=0.08)  # the bowl's planted gain
    assert float(in_delta) == pytest.approx(float(gain) / 0.1, abs=0.01)
    b = status["budget"]
    assert where[1] == f"- Budget: {b['spent_s']:.1f}s used, {b['remaining_s']:.1f}s of {b['total_s']:.1f}s remaining"
    assert where[2] == f"- Round: R3 ended ({outs[-1]['trigger']})"
    assert len(where) == 3  # no run-end reason while the run goes on


def test_summary_sections_are_capped_at_10_lines_and_up_next_shows_the_first_3_by_priority(repo, run_dir, tmp_path):
    for i in range(1, 17):
        f = tmp_path / f"s{i}.json"
        f.write_text(json.dumps({**spec({"x": lever()}), "title": f"Idea {i:02d}"}))
        assert bo(repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
    ids = [h["id"] for h in bo(repo, "status")[1]["hypotheses"]]  # Idea i is ids[i - 1]
    for i in range(1, 6):
        assert register(repo, ids[i - 1])[0] == 0
    assert bo(repo, "prioritize", ids[4], "3", "--rationale", "most promising")[0] == 0
    assert bo(repo, "prioritize", ids[2], "2", "--rationale", "next")[0] == 0
    for i in range(6, 17):
        assert bo(repo, "park", ids[i - 1], "--reason", f"later {i}", "--rationale", "set aside")[0] == 0
    s = sections((run_dir / "SUMMARY.md").read_text())
    assert s["Up next"] == ["- 5 queued; the first 3 by priority:", f"- {ids[4]} Idea 05 (priority 3)",
                            f"- {ids[2]} Idea 03 (priority 2)", f"- {ids[0]} Idea 01 (priority 0)"]
    assert s["Set aside"] == [f"- {ids[i - 1]} Idea {i:02d}: parked: later {i}" for i in range(6, 15)] + [
        "- … and 2 more: see exports/hypotheses.csv and rounds/"]
    assert s["Where we are"] == ["- Incumbent: none yet (no BO round has ended)",
                                 f"- Budget: 0.0s used, 3600.0s of 3600.0s remaining",
                                 "- Round: none yet (the calibration round is next)"]
    assert bo(repo, "stop", "--rationale", "enough")[0] == 0
    where = sections((run_dir / "SUMMARY.md").read_text())["Where we are"]
    assert where[-1] == "- Run ended: the user stopped it"


def test_hypotheses_csv_has_a_row_per_version_with_its_reject_condition(planted):
    repo, run_dir, *_ = planted
    table = rows(run_dir, "hypotheses.csv")
    assert list(table[0]) == ["id", "title", "lens", "provenance", "mechanism", "state", "reason",
                              "m_u_lower", "m_u_upper", "delta_lower", "delta_upper", "trials_used",
                              "expected_right", "revived_from", "prediction_flag"]
    by = {r["id"]: r for r in table}
    assert list(by) == ["H1.v1", "H2.v1", "H3.v1", "H4.v1"]
    assert [(r["state"], r["reason"]) for r in table] == [
        ("retained", ""), ("rejected", "no-improvement"), ("rejected", "no-improvement"),
        ("parked", "the user wants it later")]
    assert by["H2.v1"]["title"] == "A lever that does nothing" and by["H2.v1"]["lens"] == "optimisation"
    assert by["H2.v1"]["provenance"] == "novel" and by["H2.v1"]["mechanism"] == spec({})["mechanism"]
    for h in ("H1.v1", "H2.v1", "H3.v1"):
        last = [v for v in of_type(run_dir, "verdict") if v["hypothesis"] == h][-1]
        r = by[h]
        assert [float(r[k]) for k in ("m_u_lower", "m_u_upper", "delta_lower", "delta_upper")] == [
            last["m_u"]["lower"], last["m_u"]["upper"], last["delta_stat"]["lower"], last["delta_stat"]["upper"]]
        assert int(r["trials_used"]) == len(last["trials"])
    # H2 does nothing but its range isn't bounded below δ next to the bowl; H3's range is 2
    assert float(by["H2.v1"]["delta_upper"]) < 0.1 <= float(by["H2.v1"]["m_u_upper"])
    assert float(by["H3.v1"]["m_u_lower"]) > 0.1
    assert float(by["H3.v1"]["delta_upper"]) < 0.1 < float(by["H1.v1"]["delta_lower"])
    assert by["H4.v1"]["m_u_lower"] == by["H4.v1"]["trials_used"] == ""  # never tested
    # expected retain / reject before R1, then `undecided` for the rest. H1: retained in R1 (right) and
    # R2 (wrong), no check in R3 (right). H2's R1 reject is deferred behind H3's (two confirmed
    # no-improvement rejects: the smaller UB(Δ) goes first), so R1 reads undecided (wrong), and its R2
    # reject is against `undecided` (wrong). H3: rejected in R1 against `undecided` (wrong)
    assert [r["expected_right"] for r in table] == ["2/3", "0/2", "0/1", ""]
    assert [r["revived_from"] for r in table] == ["", "", "", ""]
    assert by["H1.v1"]["prediction_flag"] == ""  # predicted higher, and its best is higher



ROUND_SECTIONS = ["Trigger", "Verdict records", "Incumbent", "Noise", "Budget", "Discouraged directives",
                  "Suggested narrowings", "Expected versus actual verdicts", "Rivals", "Proposed directives",
                  "Narrative"]


def test_round_summaries_hold_the_mechanical_sections_and_the_analyst_narrative(planted):
    repo, run_dir, outs, narrowed = planted
    assert sorted(p.name for p in (run_dir / "rounds").iterdir()) == ["000.md", "001.md", "002.md", "003.md"]
    assert [p["round"] for p in of_type(run_dir, "round_summary")] == [0, 1, 2, 3]  # once each, at its end
    r1 = sections((run_dir / "rounds" / "001.md").read_text())
    assert list(r1) == ROUND_SECTIONS
    assert r1["Trigger"] == [f"- {outs[0]['trigger']}"]
    assert [ln.split(":")[0] for ln in r1["Verdict records"]] == [f"- {v['id']}" for v in outs[0]["verdicts"]]
    rejects = {p["verdict"]: p["condition"] for p in of_type(run_dir, "hypothesis_rejected")}
    assert all(f"reject ({rejects[v['id']]})" in ln for v, ln in zip(outs[0]["verdicts"], r1["Verdict records"])
               if v["id"] in rejects)
    inc = outs[0]["incumbent"]
    assert r1["Incumbent"][0].startswith(f"- trial {inc['trial']}: loss {inc['mean']:.4g}, the mean over "
                                         f"{inc['replicates'] + 1} trials (confirmed)")
    sigma = [p for p in of_type(run_dir, "noise_estimate") if p.get("round") == 1][0]["sigma"]
    assert r1["Noise"][0].startswith(f"- σ {sigma:.3g} from the round's replicates")
    assert r1["Expected versus actual verdicts"] == [
        "- H1.v1: expected retain, actual retain (right)", "- H2.v1: expected reject, actual undecided (wrong)",
        "- H3.v1: expected undecided, actual reject (wrong)"]
    assert r1["Rivals"] == r1["Proposed directives"] == ["- none"]
    assert r1["Narrative"][0] == f"{outs[0]['verdicts'][0]['id']} opened the round; the useless lever went first."
    assert "- the replicates spread wider than in the calibration round" in r1["Narrative"]
    assert r1["Narrative"][-1].endswith("By round-analyst (ra).")
    assert sections((run_dir / "rounds" / "003.md").read_text())["Narrative"][0].startswith("_No narrative yet")
    # the incumbent (x ≈ 0.8) breaks keep-x-low: the best logged trial that honours it, and the gap
    [line] = r1["Discouraged directives"]
    trial = int(line.split("trial ")[1].split(" ")[0])
    table = {int(r["seq"]): r for r in rows(run_dir, "trials.csv")}
    honours = [r for r in table.values() if r["round"] == "1" and r["status"] == "finished"
               and r["compat:keep-x-low"] == "True"]
    assert table[trial] in honours and table[trial]["replicate_of"] == ""
    gap, lo, hi = (float(x) for x in re.search(r"costs (\S+) \[(\S+), (\S+)\]", line).groups())
    assert lo < gap < hi and gap > 0  # x ≤ 0.7 is off the bowl's bottom: honouring it costs loss
    # R2's best trials sit near x = 0.8, a narrowing `narrow` accepts
    assert narrowed and all(code == 0 for code, _ in narrowed), narrowed
    assert narrowed[0][1]["hypothesis"] == "H1.v1" and narrowed[0][1]["range"]["low"] == 0.6
    r0 = sections((run_dir / "rounds" / "000.md").read_text())
    assert r0["Trigger"] == ["- calibrated"] and r0["Noise"][0].startswith("- σ at {\"epochs\": 4}: ")


def test_two_off_intent_parks_for_one_concern_propose_a_directive(repo, run_dir, tmp_path):
    reasons = ["it rewrites the data pipeline, which the brief keeps out of scope",
               "it swaps the data pipeline, out of scope for the brief",
               "it needs a GPU cluster the user lacks"]
    for i, why in enumerate(reasons):
        f = tmp_path / f"s{i}.json"
        f.write_text(json.dumps({**spec({"x": lever()}), "title": f"Idea {i}"}))
        assert bo(repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
    ids = [h["id"] for h in bo(repo, "status")[1]["hypotheses"]]
    for hid, why in zip(ids, reasons):
        assert review(repo, hid, intent="off-intent", rationale=why)[0] == 0
    assert [h["status"] for h in bo(repo, "status")[1]["hypotheses"]] == ["parked"] * 3
    assert bo(repo, "round-run", "--rationale", "calibrate")[0] == 0
    r0 = sections((run_dir / "rounds" / "000.md").read_text())
    assert r0["Proposed directives"] == [
        f"- {ids[0]}, {ids[1]} were parked off-intent for one concern (\"{reasons[0]}\"): the user may state it "
        "as a directive"]
    aside = sections((run_dir / "SUMMARY.md").read_text())["Set aside"]
    assert aside[2] == f"- {ids[2]} Idea 2: parked: off-intent: {reasons[2]}"



def generated(run_dir):
    return {str(p.relative_to(run_dir)): p.read_bytes() for name in GENERATED for p in
            ([run_dir / name] if (run_dir / name).is_file() else sorted((run_dir / name).rglob("*")))}


def test_deleting_every_generated_file_and_rebuilding_from_the_log_reproduces_them_byte_for_byte(planted):
    repo, run_dir, *_ = planted
    before, log = generated(run_dir), events(run_dir)
    assert {"SUMMARY.md", "exports/trials.csv", "exports/hypotheses.csv", "rounds/003.md"} <= set(before)
    for name in GENERATED:
        p = run_dir / name
        p.unlink() if p.is_file() else shutil.rmtree(p)
    code, out = bo(repo, "rebuild")
    assert code == 0 and out == {"rebuilt": sorted(before)}
    assert generated(run_dir) == before
    assert events(run_dir) == log  # the files are derived: rebuilding logs nothing
