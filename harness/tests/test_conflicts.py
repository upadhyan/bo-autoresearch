"""Conflicts between hypotheses (#32): config-path ownership and merges, masking, rivals."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor

from conftest import bo, review, register, round_run
from test_rounds import BASE, init, make_repo, of_type, toy_env
from test_verdicts import code_hypothesis, lever, repeat, spec, verdict_run


def propose(repo, d, s):
    f = d / f"spec-{len(list(d.glob('spec-*')))}.json"
    f.write_text(json.dumps(s))
    code, out = bo(repo, "propose", "--file", str(f), "--rationale", "idea")
    assert code == 0, out
    return out["hypothesis"]["id"]


def lr(low, high, baseline):
    return {**lever(baseline, "lower", low, high), "path": "optim.lr"}


def test_a_second_hypothesis_on_a_live_config_path_is_refused_until_merged(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE)
    propose(repo, tmp_path, spec({"lr": lr(0.001, 0.01, 0.005)}))
    assert register(repo, "H1")[0] == 0
    propose(repo, tmp_path, spec({"lr": lr(0.0001, 0.001, 0.0005)}))
    code, out = register(repo, "H2")
    assert code == 1 and "optim.lr" in out["reason"] and "H1.v1" in out["reason"]
    assert not [p for p in of_type(run_dir, "hypothesis_registered") if p["id"] == "H2.v1"]

    # the merged hypothesis covers both ranges; a mapping short of H1's range is refused
    short = spec({"lr": lr(0.0001, 0.005, 0.001)},
                 merges={"from": ["H1.v1"], "mapping": {"lr": {"lever": "H1.lr"}}})
    propose(repo, tmp_path, short)
    code, out = register(repo, "H3")
    assert code == 1 and "H1.lr" in out["reason"] and "cover" in out["reason"]

    merged = spec({"lr": lr(0.0001, 0.01, 0.001)},
                  merges={"from": ["H1.v1", "H2.v1"],
                          "mapping": {"lr": [{"lever": "H1.lr"}, {"lever": "H2.lr"}]}})
    propose(repo, tmp_path, merged)
    code, out = register(repo, "H4")
    assert code == 0, out
    assert of_type(run_dir, "hypothesis_merged") == [
        {"id": "H1.v1", "into": "H4.v1", "from": "registered"},
        {"id": "H2.v1", "into": "H4.v1", "from": "proposed"}]
    status = {h["id"]: h["status"] for h in bo(repo, "status")[1]["hypotheses"]}
    assert status == {"H1.v1": "merged", "H2.v1": "merged", "H3.v1": "proposed", "H4.v1": "registered"}


def test_a_parked_hypothesis_returns_only_while_its_config_path_is_free(tmp_path, project_python):
    repo = make_repo(tmp_path)
    init(repo, project_python, BASE)
    propose(repo, tmp_path, spec({"lr": lr(0.001, 0.01, 0.005)}))
    assert register(repo, "H1")[0] == 0
    assert bo(repo, "park", "H1", "--reason", "later", "--rationale", "make room")[0] == 0
    propose(repo, tmp_path, spec({"lr": lr(0.0001, 0.001, 0.0005)}))
    assert register(repo, "H2")[0] == 0  # a parked hypothesis owns nothing
    code, out = bo(repo, "unpark", "H1", "--reason", "back", "--rationale", "try it")
    assert code == 1 and "optim.lr" in out["reason"] and "H2.v1" in out["reason"]


def bool_lever(**lv):
    return {"kind": "bool", "baseline": False, "why_not_graded": "the mechanism is on or off", **lv}


def test_a_conflict_the_review_flags_must_be_resolved_in_the_spec(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE)
    propose(repo, tmp_path, spec({"dropout": bool_lever()}))
    assert register(repo, "H1")[0] == 0
    propose(repo, tmp_path, spec({"wd": lever(0.0, "higher")}))
    code, out = register(repo, "H2", conflict="exclusive", conflict_with=["H1.v1"])
    assert code == 1 and "masked_by" in out["reason"] and "exclusive_with" in out["reason"]
    propose(repo, tmp_path, spec({"bias": lever()}))
    code, out = register(repo, "H3", conflict="shared-lever", conflict_with=["H1.v1"])
    assert code == 1 and "merges" in out["reason"]
    # a rival is no conflict to resolve: both are tested
    assert register(repo, "H3", conflict="rival", conflict_with=["H1.v1"])[0] == 0

    propose(repo, tmp_path, spec({"wd": lever(0.0, "higher")},
                                 masked_by={"H1": "with dropout on, the weight decay is moot"}))
    assert register(repo, "H4", conflict="exclusive", conflict_with=["H1.v1"])[0] == 0
    conflicts = bo(repo, "status")[1]["conflicts"]
    assert conflicts["rivals"] == [["H1.v1", "H3.v1"]]
    assert conflicts["masking"] == [{"masked": "H4.v1", "by": "H1",
                                     "reason": "with dropout on, the weight decay is moot"}]
    assert not of_type(run_dir, "hypothesis_merged")
    # masking declares one precedence: two hypotheses never mask each other
    propose(repo, tmp_path, spec({"a": bool_lever()}, masked_by={"H6": "a yields to b"}))
    assert register(repo, "H5")[0] == 0
    propose(repo, tmp_path, spec({"b": lever()}, masked_by={"H5": "b yields to a"}))
    code, out = register(repo, "H6")
    assert code == 1 and "mask each other" in out["reason"]


def test_exclusive_mechanisms_of_one_slot_merge_into_a_categorical_lever_with_none(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE)
    env = toy_env(sigma=0.05)
    code_hypothesis(repo, run_dir, tmp_path, 1, spec({"bn": bool_lever()}), '    term += 0.1 * lever("H1.bn")', env)
    code_hypothesis(repo, run_dir, tmp_path, 2, spec({"ln": bool_lever()}, exclusive_with=["H1"]),
                    '    term += 0.2 * lever("H2.ln")', env)
    norm = {"kind": "categorical", "options": ["none", "bn", "ln"], "baseline": "none"}
    merged = spec({"norm": norm}, merges={"from": ["H1.v1", "H2.v1"], "mapping": {"norm": [
        {"lever": "H1.bn", "values": [[False, "none"], [True, "bn"]]},
        {"lever": "H2.ln", "values": [[False, "none"], [True, "ln"]]}]}})
    propose(repo, tmp_path, merged)
    code, out = register(repo, "H3", conflict="exclusive", conflict_with=["H1.v1", "H2.v1"])
    assert code == 0, out
    assert [p["id"] for p in of_type(run_dir, "hypothesis_merged")] == ["H1.v1", "H2.v1"]

    ran = {p["trial"]: p["levers"] for p in of_type(run_dir, "trial_started")}
    assert {(lv.get("H1.bn"), lv.get("H2.ln")) for lv in ran.values()} >= {(True, None), (False, True)}
    want = {n: {"H3.norm": "bn" if lv.get("H1.bn") else "ln" if lv.get("H2.ln") else "none"}
            for n, lv in ran.items()}
    code, out = bo(repo, "trials", "--eligible")
    assert code == 0, out
    assert {t["trial"]: t["levers"] for t in out["trials"]} == want


DROP = (spec({"drop": bool_lever()}), '    term += 0.3 * lever("H1.drop")')
WD = (spec({"wd": lever()}, masked_by={"H1": "with dropout on, the weight decay is moot"}),
      '    term += 2 * ((lever("H2.wd") - 0.5) ** 2 - 0.25)')


def test_a_masked_lever_runs_at_its_baseline_while_the_masker_is_on_and_its_gates_count_the_rest(
        tmp_path, project_python):
    shown = 0
    for run_dir, [out] in repeat(tmp_path, project_python, [DROP, WD]):
        repo = run_dir.parents[1]
        started = {p["trial"]: p for p in of_type(run_dir, "trial_started")}
        r1 = [t for t in started.values() if t.get("round") == 1]
        on = [t for t in r1 if t["levers"]["H1.drop"]]
        assert on and len(on) < len(r1)
        for t in r1:  # the effective value runs; the sampled one is logged beside it
            assert set(t["sampled"]) == {"H2.wd"}
            assert t["levers"]["H2.wd"] == (0.0 if t["levers"]["H1.drop"] else t["sampled"]["H2.wd"])
            assert t["masked"] == (["H2.v1"] if t["levers"]["H1.drop"] else [])
            trial_json = json.loads((run_dir / t["artifact_dir"] / "trial.json").read_text())
            assert trial_json["levers"] == t["levers"]  # what lever() reads
        assert any(t["sampled"]["H2.wd"] != 0.0 for t in on)
        rows = {int(r["trial"]): r for r in csv.DictReader((run_dir / "exports" / "trials.csv").open())}
        assert all(float(rows[t["trial"]]["sampled:H2.wd"]) == t["sampled"]["H2.wd"] for t in r1)
        # the GP, Sobol and the warm start take the sampled values
        code, seen = bo(repo, "trials", "--eligible")
        seen = {t["trial"]: t["levers"] for t in seen["trials"]}
        assert all(seen[t["trial"]]["H2.wd"] == t["sampled"]["H2.wd"] for t in r1 if t["trial"] in seen)
        masked_in = [n for v in of_type(run_dir, "verdict") for n in v["trials"] if started[n].get("masked")]
        assert masked_in and set(masked_in) <= {t["trial"] for t in on}  # every verdict GP takes them
        # the masked hypothesis's gates count only the sampler trials where it was effective
        for v in of_type(run_dir, "verdict"):
            fresh = [n for n in v["trials"] if started[n]["kind"] == "sampler"]
            if v["hypothesis"] == "H2.v1":
                fresh = [n for n in fresh if not started[n]["levers"]["H1.drop"]]
            assert v["burn_in"]["fresh"] == v["gates"]["trials"]["sampler"] == len(fresh), v["id"]
        shown += any(v["hypothesis"] == "H2.v1" and v["burn_in"]["fresh"]
                     < sum(started[n]["kind"] == "sampler" for n in v["trials"]) for v in of_type(run_dir, "verdict"))
    assert shown >= 2  # H2 checked with masked sampler trials left out of its count, in most runs


def rival_run(d, python, seed):
    useful = (spec({"x": lever(0.2)}), '    term += 4 * ((lever("H1.x") - 0.7) ** 2 - 0.25)')
    useless = (spec({"x": lever()}), '    term += 0 * lever("H2.x")')
    run_dir, _ = verdict_run(d, python, seed, [useful, useless], rounds=0)
    repo = run_dir.parents[1]
    assert review(repo, "H2", conflict="rival", conflict_with=["H1.v1"])[0] == 0
    for _ in range(2):
        code, out = round_run(repo, "--rationale", "search", env=toy_env(sigma=0.05, cheap_below=2))
        assert code == 0, out
    return run_dir


def test_rivals_are_both_tested_and_each_gets_its_own_verdict(tmp_path, project_python):
    with ThreadPoolExecutor(3) as pool:
        runs = list(pool.map(lambda s: rival_run(tmp_path / f"s{s}", project_python, s), (1, 2, 3)))
    status = []
    for run_dir in runs:
        repo = run_dir.parents[1]
        out = bo(repo, "status")[1]
        assert out["conflicts"]["rivals"] == [["H1.v1", "H2.v1"]]
        [r1] = [p for p in of_type(run_dir, "round_started") if p["round"] == 1]
        assert r1["hypotheses"] == ["H1.v1", "H2.v1"]
        assert {v["hypothesis"] for v in of_type(run_dir, "verdict") if v["round"] == 1} == {"H1.v1", "H2.v1"}
        status.append({h["id"]: h["status"] for h in out["hypotheses"]})
    # independent verdicts: the useful rival is retained, the useless one rejected, in most runs
    assert sum(s == {"H1.v1": "retained", "H2.v1": "rejected"} for s in status) >= 2, status


def test_rivals_join_the_search_together(tmp_path, project_python):
    # ~1.1 s trials and ~110 s left after R0: the dimension cap is 2 (as in test_scheduler)
    repo = make_repo(tmp_path)
    init(repo, project_python, BASE.replace("budget_s: 3600", "budget_s: 115") + "delta: 0.1\n")
    assert bo(repo, "round-run", "--rationale", "calibrate", env=toy_env(sleep=1.0))[0] == 0
    for n, p in ((1, 5), (2, 3), (3, 0)):
        propose(repo, tmp_path, spec({"a": lever()}))
        rival = {"conflict": "rival", "conflict_with": ["H1.v1"]} if n == 3 else {}
        assert register(repo, f"H{n}", **rival)[0] == 0
        assert bo(repo, "prioritize", f"H{n}", str(p), "--rationale", "the ranking")[0] == 0
    s = bo(repo, "status")[1]["schedule"]
    assert s["dimension_cap"] == 2
    assert s["selected"] == ["H1.v1", "H3.v1"]  # H1's rival comes with it, ahead of H2
