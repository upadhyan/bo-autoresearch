"""Acceptance: a whole scripted BOAR run on the toy target, with the agent and reviewer played by the test.

Mirrors design.md "Acceptance" as far as it goes without an LLM: two real levers, a lever with no
effect that gets removed under an accepted review, a cheat the reviewer rejects, warm start carried
into rounds 2 and 3, refusals mid-run, and a report the Stop hook won't release until it is written.
"""

from __future__ import annotations

import json
import re
import statistics

import pytest

from boar.store import TODO_MARKER

SECTIONS = ["Result", "What worked", "What didn't", "Rejected by review", "Caveats", "Experimental log"]
TRIAL_KEYS = {"trial", "round", "commit", "config", "state", "metric", "repeats", "metrics", "duration_s", "error", "queued"}
CHEAT_REASON = "Caching the benchmark output across eval runs is a cheat named in the spec's Known cheats."
TIMER_REASON = "Coarsening the timer changes how cost is measured instead of the work done: a cheat."
REMOVAL_REASON = "noop shows no effect beyond the noise floor across trials that set it both on and off."


def report_sections(report: str) -> dict[str, str]:
    """Map each of the six section names to its body, asserting they appear in order."""
    heads = list(re.finditer(r"^##\s+(?:\d+\.\s*)?(.+?)\s*$", report.replace("’", "'"), re.M))
    found: dict[str, str] = {}
    order = []
    for i, m in enumerate(heads):
        title = m.group(1).lower()
        name = next((s for s in SECTIONS if title.startswith(s.lower())), None)
        if name is None:
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(report)
        found[name] = report[m.end():end]
        order.append(name)
    assert order == SECTIONS, f"report sections {order}, expected {SECTIONS}"
    return found


def summary_section(summary: str, heading: str) -> str:
    m = re.search(rf"^##\s+{re.escape(heading)}.*?$(.*?)(?=^##\s|\Z)", summary, re.M | re.S | re.I)
    assert m, f"summary has no '## {heading}' section:\n{summary}"
    return m.group(1)


def check_summary(d, r: int) -> str:
    path = d.run_dir / "rounds" / str(r) / "summary.md"
    assert path.exists(), f"round {r} wrote no summary.md"
    text = path.read_text()
    for heading in ("Incumbent", "Noise floor", "Warm start", "Lever effects", "Infeasible and failed"):
        summary_section(text, heading)
    assert "drift" in summary_section(text, "Incumbent").lower()
    effects = summary_section(text, "Lever effects")
    assert "fast_a" in effects and "fast_b" in effects
    bad = summary_section(text, "Infeasible and failed")
    for t in d.trials(r):
        if t["state"] != "complete":
            assert f"trials/{t['trial']}" in bad, f"trial {t['trial']} ({t['state']}) not listed in round {r} summary"
    return text


def check_trials(d, guard) -> list[dict]:
    st = d.state()
    trials = d.trials()
    per_round = d.RUN_CONFIG["trials_per_round"]
    assert [t["trial"] for t in trials] == list(range(1, len(trials) + 1))
    for t in trials:
        n = t["trial"]
        assert TRIAL_KEYS <= set(t), f"trial {n} is missing {TRIAL_KEYS - set(t)}"
        assert t["round"] == (n - 1) // per_round + 1
        commit = st["rounds"][str(t["round"])]["commit"]
        assert commit and (commit.startswith(t["commit"]) or t["commit"].startswith(commit))
        cfg = t["config"]
        assert {"fast_a", "fast_b", "noop", "cache_output"} <= set(cfg), "the eval must get every lever ever defined"
        assert cfg["cache_output"] is False, "the rejected cheat's lever stays pinned at its default"
        assert t["state"] in ("complete", "infeasible"), f"trial {n} {t['state']}: {t['error']}"
        assert (t["state"] == "infeasible") == guard(cfg)
        if t["state"] == "complete":
            assert len(t["repeats"]) == d.RUN_CONFIG["repeats"] and t["error"] is None
        else:
            assert 1 <= len(t["repeats"]) <= d.RUN_CONFIG["repeats"]
        assert t["metric"] == pytest.approx(statistics.median(t["repeats"]))
        assert isinstance(t["metrics"], dict) and t["duration_s"] >= 0
        # The incumbent is measured again only on a new commit.
        first = (n - 1) % per_round == 0 and (t["round"] == 1 or commit != st["rounds"][str(t["round"] - 1)]["commit"])
        assert t["queued"] == (("baseline" if t["round"] == 1 else "incumbent") if first else None), n
        tdir = d.run_dir / "trials" / str(n)
        assert json.loads((tdir / "config.json").read_text()) == cfg
        assert (tdir / "repeat-1.stdout").exists() and (tdir / "repeat-1.stderr").exists()
    baseline = trials[0]["config"]
    assert baseline == {"fast_a": False, "fast_b": 1, "noop": False, "cache_output": False}
    return trials


def test_full_scripted_run(driver):
    d = driver
    guard_expr = "config.get('fast_b', 1) >= 7"

    def guard(cfg):
        return cfg.get("fast_b", 1) >= 7

    base_commit = d.git("rev-parse", "main")
    assert d.tag() == "NONE"
    rc, _ = d.hook()
    assert rc == 0

    # S1 is the user's; init freezes the config and moves to the run's branch.
    d.init()
    run_id = d.run_id
    assert d.git("branch", "--show-current") == f"boar/{run_id}"
    cfg = d.config()
    assert {k: cfg[k] for k in d.RUN_CONFIG} == d.RUN_CONFIG and cfg["direction"] == "min"
    assert d.state()["phase"] == "setup"
    assert d.tag() == "S2"
    rc, err = d.hook()
    assert rc == 2 and f"BOAR run {run_id} is not finished" in err

    # S3: three ideas from the user plus a tempting cheat from research.
    ha, hb, hn, hc = d.propose(d.prop_a(), d.prop_b(), d.prop_noop(), d.prop_cheat())
    assert [d.hyp(h)["status"] for h in (ha, hb, hn, hc)] == ["pending"] * 4
    assert d.tag() == "S4"

    # S4: the eval; a guard fails when fast_b is pushed to 7 or more.
    d.write_eval(guard_fail=guard_expr)
    d.eval_check()
    check = json.loads((d.run_dir / "eval_check.json").read_text())
    assert check["ok"] is True and check["trial"]["state"] == "complete"
    assert d.state()["eval"]["status"] == "pending"
    text = d.next()
    assert d.tag_of(text) == "S5"
    assert all(item in text for item in (ha, hb, hn, hc, "eval"))

    # S5: the scripted reviewer pulls its inputs from the harness and rejects the cheat.
    pending = d.ok("review", "pending")
    assert all(item in pending for item in (ha, hb, hn, hc, "eval"))
    assert "Caching the benchmark output" in pending
    d.review(hc, "reject", CHEAT_REASON)
    for h in (ha, hb, hn):
        d.review(h, "accept")
    assert d.state()["phase"] == "setup"
    d.refused("round", "run")
    d.review("eval", "accept", "Dev and holdout are disjoint and typical; the guard covers the spec's ceiling.")
    st = d.state()
    assert st["phase"] == "round" and st["round"] == 1
    assert st["eval"]["status"] == "accepted" and st["eval"]["accepted_hash"] == check["hash"]
    assert [d.hyp(h)["status"] for h in (ha, hb, hn, hc)] == ["active"] * 3 + ["rejected"]
    assert all(d.hyp(h)["activated_round"] == 1 for h in (ha, hb, hn))

    # ---- round 1 -----------------------------------------------------------------------
    text = d.next()
    assert d.tag_of(text) == "R1"
    assert all(h in text for h in (ha, hb, hn)) and "fast_a" in text
    target = d.repo / "target.py"
    target.write_text(target.read_text() + "\n# R1: fast_a, fast_b and noop read from BOAR_CONFIG\n")
    d.round_run()
    st = d.state()
    assert st["rounds"]["1"]["run_status"] == "done"
    r1_commit = st["rounds"]["1"]["commit"]
    assert "# R1:" in d.git("show", f"{r1_commit}:target.py"), "round run must commit the lever work"
    assert d.git("rev-parse", "main") == base_commit, "the user's branch stays untouched"
    assert len(d.trials(1)) == 4
    check_summary(d, 1)
    text = d.next()
    assert d.tag_of(text) == "R3" and all(h in text for h in (ha, hb, hn))
    assert d.hook()[0] == 2
    d.refused("propose", d.write_proposals([d.proposal("Too early", d.lever("too_early"))]))
    for h in (ha, hb, hn):
        d.decide(h, "keep")
    assert d.tag() == "R4"
    d.ok("propose", "--none", "--reason", "Round 1 gives too little evidence to synthesize from.")
    assert d.tag() == "R6"
    d.close()
    st = d.state()
    assert st["round"] == 2 and st["rounds"]["1"]["closed"] is True and st["phase"] == "round"

    # ---- round 2: same search space, so every round-1 trial warm-starts it ----------------
    assert d.tag() == "R1"
    d.round_run()
    st = d.state()
    assert len(d.trials(2)) == 4
    warm2 = st["rounds"]["2"]["warm"]
    assert warm2["copied"] == 4
    assert warm2["excluded"] == {"rule1_state": 0, "rule2_outside_space": 0, "rule3_fixed": 0, "rule4_old": 0}
    summary2 = check_summary(d, 2)
    assert re.search(r"\b4\b", summary_section(summary2, "Warm start"))

    d.decide(ha, "keep", "fast_a=True trials beat the baseline by far more than the noise floor", trials=[1, 2])
    d.decide(hb, "keep")
    ids = [str(t["trial"]) for t in d.trials()]
    d.ok("decide", hn, "remove", "--observable-key", "passes", "--reason", REMOVAL_REASON, "--trials", ",".join(ids[:3]), *ids[3:6], ",".join(ids[6:]))
    st = d.state()
    removal = next(rm for rm in st["removals"] if rm["hypothesis"] == hn)
    assert removal["id"] == f"rm-{hn}-r2" and removal["verdict"] is None
    assert removal["trials"] == [int(i) for i in ids]
    assert d.hyp(hn)["status"] == "active", "a removal waits for review and close"
    assert d.tag() == "R4"
    d.refused("round", "close")

    (hm,) = d.propose(d.proposal("Coarsening the timer resolution cuts cost", d.lever("coarse_timer"), source="synthesis"))
    text = d.next()
    assert d.tag_of(text) == "R5" and removal["id"] in text and hm in text
    d.refused("round", "run")
    d.refused("round", "close")
    pending = d.ok("review", "pending")
    assert removal["id"] in pending and REMOVAL_REASON in pending and hm in pending
    d.review(removal["id"], "accept", "The cited trials set noop both ways; the effect sits inside the noise floor.")
    d.review(hm, "reject", TIMER_REASON)
    assert d.tag() == "R6"
    d.close()
    removed = d.hyp(hn)
    assert removed["status"] == "removed" and removed["removed_round"] == 2
    assert removal["id"] in removed["removed_reason"]
    assert d.hyp(hm)["status"] == "rejected"
    assert d.state()["round"] == 3

    # ---- round 3: trials that switched noop on drop out of the warm start (rule 2) -------
    d.round_run()
    st = d.state()
    t3 = d.trials(3)
    assert len(t3) == 4
    assert all(t["config"]["noop"] is False for t in t3), "a removed hypothesis's lever is pinned at default"
    assert all(t["config"]["coarse_timer"] is False for t in t3), "the eval still gets every lever ever defined"
    earlier = [t for t in d.trials() if t["round"] < 3]
    with_noop = sum(1 for t in earlier if t["config"]["noop"])
    warm3 = st["rounds"]["3"]["warm"]
    assert warm3["excluded"]["rule2_outside_space"] == with_noop
    assert warm3["copied"] == len(earlier) - with_noop > 0
    summary3 = check_summary(d, 3)
    assert re.search(rf"\b{warm3['copied']}\b", summary_section(summary3, "Warm start"))

    for h in (ha, hb):
        d.decide(h, "keep")
    d.refused("decide", hn, "keep", "--observable-key", "passes", "--reason", "it is gone")
    d.refused("finalize")
    d.ok("propose", "--none", "--reason", "Both remaining hypotheses are converging.")
    d.close()
    assert d.state()["phase"] == "finalize"
    assert d.tag() == "FINALIZE"
    assert d.hook()[0] == 2

    # ---- finalize ------------------------------------------------------------------------
    d.ok("finalize", "--foreground")
    trials = check_trials(d, guard)
    assert len(trials) == 12
    holdout = [json.loads(line) for line in (d.run_dir / "holdout.jsonl").read_text().splitlines() if line.strip()]
    assert [h["label"] for h in holdout] == ["baseline", "incumbent"] * 2, "finalize alternates the two configs"
    assert all(h["ok"] and h["guards_ok"] for h in holdout)
    base_med = statistics.median(h["metric"] for h in holdout if h["label"] == "baseline")
    inc_med = statistics.median(h["metric"] for h in holdout if h["label"] == "incumbent")
    assert inc_med < base_med, "the incumbent must beat the baseline on holdout"
    inc_cfg = d.state()["finalize"]["incumbent_config"]
    assert inc_cfg and "noop" not in inc_cfg and "cache_output" not in inc_cfg and not guard(inc_cfg)
    assert inc_cfg.get("fast_a") is True and inc_cfg.get("fast_b", 1) > 1, f"incumbent {inc_cfg} misses a real fix"
    last_inc = next(t for t in trials if t["trial"] == d.state()["rounds"]["3"]["incumbent_trial"])["config"]
    defaults = {"fast_a": False, "fast_b": 1, "noop": False, "cache_output": False, "coarse_timer": False}
    assert inc_cfg == {k: v for k, v in last_inc.items() if v != defaults[k]}, \
        "round 3 kept every hypothesis, so finalize recommends the incumbent its summary named"

    # BO only ever sees dev; holdout is used once, at finalize. The toy eval logs the split it was given.
    for t in trials:
        logs = sorted((d.run_dir / "trials" / str(t["trial"])).glob("repeat-*.stderr"))
        assert logs and all("split=dev" in p.read_text() for p in logs), f"trial {t['trial']} didn't run on dev"
    for h in holdout:
        assert "split=holdout" in (d.run_dir / "holdout" / str(h["i"]) / "repeat-1.stderr").read_text(), h

    report = (d.run_dir / "report.md").read_text()
    assert report.startswith("# ")
    sec = report_sections(report)
    for name in ("What worked", "What didn't", "Caveats"):
        assert TODO_MARKER in sec[name], f"narrative section {name!r} has no todo marker"
    for name in ("Result", "Rejected by review", "Experimental log"):
        assert TODO_MARKER not in sec[name], f"generated section {name!r} should be filled in"
    assert "baseline" in sec["Result"].lower() and "holdout" in sec["Result"].lower()
    assert hc in sec["Rejected by review"] and "Caching the benchmark output" in sec["Rejected by review"]
    assert hm in sec["Rejected by review"]
    assert hn in sec["What didn't"] and "accept" in sec["What didn't"].lower()
    assert "trials.jsonl" in sec["Experimental log"]
    for r in (1, 2, 3):
        assert re.search(rf"^\|\s*{r}\s*\|", sec["Experimental log"], re.M), f"no experimental log row for round {r}"
    for split in ("holdout", "dev"):
        row = next(line for line in sec["Result"].splitlines() if line.startswith(f"| incumbent | {split} |"))
        assert row.endswith("(better) |"), row
    for r in (2, 3):
        w = d.state()["rounds"][str(r)]["warm"]
        row = next(line for line in sec["Experimental log"].splitlines() if re.match(rf"\|\s*{r}\s*\|", line))
        assert f"| {w['copied']} / {sum(w['excluded'].values())} |" in row, row

    kinds = [(rv["item"], rv["kind"], rv["decision"]) for rv in d.reviews()]
    assert (hc, "proposal", "reject") in kinds and ("eval", "eval", "accept") in kinds
    assert (removal["id"], "removal", "accept") in kinds and (hm, "proposal", "reject") in kinds

    # The Stop hook holds until the narrative is written.
    assert d.tag() == "REPORT"
    assert d.hook()[0] == 2
    d.refused("finalize")
    (d.run_dir / "report.md").write_text(report.replace(TODO_MARKER, "Narrative written by the agent."))
    assert d.tag() == "DONE"
    assert d.state()["phase"] == "done"
    assert d.hook()[0] == 0
    assert d.tag() == "DONE"
    d.ok("status")
    d.refused("round", "run")


def test_supersede_pins_the_old_lever_and_activates_the_new_one(driver):
    d = driver.start("round1_run", **driver.SMALL)
    ha, hb, hn = d.hid("fast_a"), d.hid("fast_b"), d.hid("noop")
    for h in (ha, hb, hn):
        d.decide(h, "keep")
    (hw,) = d.propose(
        d.proposal(
            "Batching with a wider range cuts cost",
            d.lever("fast_b_wide", "int", low=1, high=16, default=1),
            source="synthesis",
            supersedes=hb,
        )
    )
    d.review(hw, "accept")
    d.close()
    old, new = d.hyp(hb), d.hyp(hw)
    assert old["status"] == "removed" and hw in (old["removed_reason"] or "")
    assert new["status"] == "active" and new["activated_round"] == 2
    text = d.next()
    assert d.tag_of(text) == "R1" and hw in text and "fast_b_wide" in text

    d.round_run()
    t2 = d.trials(2)
    assert len(t2) == 2
    assert all(t["config"]["fast_b"] == 1 for t in t2), "a superseded lever is pinned at default"
    assert all("fast_b_wide" in t["config"] for t in t2)
    earlier = d.trials(1)
    moved = sum(1 for t in earlier if t["config"]["fast_b"] != 1)
    warm = d.state()["rounds"]["2"]["warm"]
    assert warm["excluded"]["rule2_outside_space"] == moved
    assert warm["copied"] == len(earlier) - moved >= 1


def test_run_finalizes_early_when_no_hypothesis_is_left(driver):
    d = driver.start("round1_run", **driver.SMALL)
    ids = [str(t["trial"]) for t in d.trials()]
    for lever in ("fast_a", "fast_b", "noop"):
        d.ok("decide", d.hid(lever), "remove", "--observable-key", "passes", "--reason", "no effect beyond the noise floor", "--trials", *ids)
    d.ok("propose", "--none", "--reason", "Nothing left to try.")
    for rm in d.state()["removals"]:
        d.review(rm["id"], "accept", "The cited trials show no effect beyond the noise floor.")
    d.close()
    st = d.state()
    assert st["phase"] == "finalize", "with no active hypothesis the run finalizes even though rounds remain"
    assert d.tag() == "FINALIZE"
    d.ok("finalize", "--foreground")
    assert d.state()["finalize"]["incumbent_config"] == {}
    assert d.tag() == "REPORT"


def _wait_until_finished(d, attempts: int = 30) -> str:
    for _ in range(attempts):
        out = d.ok("wait", "--timeout", "10")
        if "still running" not in out.lower():
            assert d.state() and not _worker_alive(d), out
            return out
    pytest.fail("the detached worker never finished")


def _worker_alive(d) -> bool:
    from boar.store import Run

    return Run(root=d.repo, id=d.run_id).running() is not None


def test_detached_round_run_and_finalize_with_wait(driver):
    d = driver.start("round1", rounds=1, trials_per_round=2, repeats=1, holdout_repeats=1)
    d.ok("round", "run")
    assert d.tag() in ("WAIT", "R3")
    _wait_until_finished(d)
    st = d.state()
    assert st["rounds"]["1"]["run_status"] == "done"
    assert len(d.trials(1)) == 2
    assert (d.run_dir / "rounds" / "1" / "worker.log").exists()
    assert d.tag() == "R3"
    for lever in ("fast_a", "fast_b", "noop"):
        d.decide(d.hid(lever), "keep")
    d.ok("propose", "--none", "--reason", "Single-round run.")
    d.close()
    assert d.state()["phase"] == "finalize"
    d.ok("finalize")
    _wait_until_finished(d)
    assert d.state()["finalize"]["status"] == "done"
    assert (d.run_dir / "report.md").exists()
    assert d.tag() == "REPORT"


def test_detached_eval_check_with_wait(driver):
    d = driver.start("proposed")
    d.write_eval(sleep=0.5)
    out = d.ok("eval", "check")
    assert "background" in out
    assert d.tag() in ("WAIT", "S5")
    rc, err = d.hook()
    assert rc == 2
    out = _wait_until_finished(d)
    assert "finished: eval check PASS" in out and "[S5]" in out
    assert d.state()["eval"]["status"] == "pending"
    assert json.loads((d.run_dir / "eval_check.json").read_text())["ok"] is True
    assert (d.run_dir / "eval_check.log").exists()
    assert d.tag() == "S5"


def test_detached_eval_check_failure_is_reported_by_wait(driver):
    d = driver.start("proposed")
    d.write_eval(crash_when="True")
    d.ok("eval", "check")
    out = _wait_until_finished(d)
    assert "finished: eval check FAIL" in out and "exit 3" in out
    assert d.state()["eval"]["status"] == "none"
    assert d.tag() == "S4"


def test_worker_runs_from_the_run_dir_so_a_target_boar_module_cannot_shadow_it(driver):
    d = driver.start("proposed")
    (d.repo / "boar").mkdir()
    (d.repo / "boar" / "__init__.py").write_text("raise SystemExit('shadowed the harness')\n")
    (d.repo / "boar" / "__main__.py").write_text("raise SystemExit('shadowed the harness')\n")
    d.write_eval()
    d.ok("eval", "check")
    out = _wait_until_finished(d)
    assert "finished: eval check PASS" in out, out
    assert ("warning: untracked files that round 1's commit will sweep onto the run branch: boar/__init__.py, "
            "boar/__main__.py;") in out, out
