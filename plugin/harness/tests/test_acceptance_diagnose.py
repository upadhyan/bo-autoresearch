"""Diagnose in the loop: diagnostics, investigations, blocked hypotheses and enablers, driven through the CLI."""

from __future__ import annotations

import json


def test_round_trials_store_the_targets_diagnostics_and_keep_their_diag_dirs(driver):
    d = driver.start("round1_run", **driver.SMALL)
    trials = d.trials(1)
    assert trials and all(t["diag_error"] is None for t in trials)
    for t in trials:
        passes = 2 if t["config"]["fast_a"] else 3
        assert t["diagnostics"] == {"passes": passes, "batch": t["config"]["fast_b"]}
        saved = d.run_dir / "trials" / str(t["trial"]) / "diag-1" / "diag.json"
        assert json.loads(saved.read_text())["passes"] == passes


def test_the_first_decision_binds_the_observable_to_a_measured_diagnostic_key(driver):
    d = driver.start("round1_run", **driver.SMALL)
    ha, hb = d.hid("fast_a"), d.hid("fast_b")
    err = d.refused("decide", ha, "keep", "--reason", "helps")
    assert "--observable-key" in err
    err = d.refused("decide", ha, "keep", "--reason", "helps", "--observable-key", "wall_ms")
    assert "wall_ms" in err and "passes" in err  # names the keys the diagnostics do have
    d.ok("decide", ha, "keep", "--reason", "helps", "--observable-key", "passes")
    assert d.hyp(ha)["observable_key"] == "passes"
    d.ok("decide", ha, "keep", "--reason", "re-decided; the key stays bound")
    assert d.hyp(ha)["observable_key"] == "passes"
    # fix repairs, which may mean instrumentation that never ran: it needs no key.
    d.ok("decide", hb, "fix", "--reason", "the batch lever never applied", "--trials", d.trials(1)[0]["trial"])
    assert d.hyp(hb).get("observable_key") is None


FOUR = {"rounds": 4, "trials_per_round": 2, "repeats": 1, "trial_target_s": 5, "holdout_repeats": 1}


def finish_round_1(d, ha, *rest):
    """Decide keep on every other active hypothesis, propose nothing, and close round 1."""
    for hid in rest:
        d.decide(hid, "keep")
    d.ok("propose", "--none", "--reason", "nothing new")
    d.close()


def new_commit(d):
    """What R1 leaves when it adds the measurements: the round runs on a new commit."""
    (d.repo / "target.py").write_text((d.repo / "target.py").read_text() + "# measurements added\n")


def test_investigate_keeps_the_hypothesis_measures_its_cause_and_runs_the_queued_configs(driver):
    d = driver.start("round1_run", **FOUR)
    ha, hb, hn = d.hid("fast_a"), d.hid("fast_b"), d.hid("noop")
    queue = d.tmp / "queue.json"
    queue.write_text(json.dumps([{"fast_a": False}, {"fast_a": True, "fast_b": 8}]))
    out = d.ok("decide", ha, "investigate", "--observable-key", "passes", "--cause", "the pass count may not drop",
               "--measure", "passes,pass_ms", "--queue", queue)
    assert "investigate" in out
    rec = d.hyp(ha)["decisions"]["1"]
    assert rec["decision"] == "investigate" and rec["cause"] == "the pass count may not drop"
    assert rec["measure"] == ["passes", "pass_ms"] and rec["queue"] == [{"fast_a": False}, {"fast_a": True, "fast_b": 8}]
    assert rec["reason"] == "the pass count may not drop"  # the cause stands in for a missing --reason
    finish_round_1(d, ha, hb, hn)
    assert d.hyp(ha)["status"] == "active"
    text = d.next()
    assert text.startswith("[R1]") and f"measure: {ha}: passes, pass_ms (cause: the pass count may not drop)" in text
    new_commit(d)
    d.round_run()
    round2 = d.trials(2)
    assert len(round2) == 2 + 2
    labels = [t["queued"] for t in round2]
    assert labels[1:3] == [f"investigate {ha} (off-state)", f"investigate {ha} (2/2)"]
    summary = (d.run_dir / "rounds" / "2" / "summary.md").read_text()
    assert f"investigate {ha} (1/2): same as trial {round2[1]['trial']}" in summary  # it queued the off-state
    inc = d.state()["rounds"]["2"]["warm"]["incumbent_config"] or {}
    base = {"fast_a": False, "fast_b": 1, "noop": False, "cache_output": False, **inc}
    assert round2[1]["config"] == {**base, "fast_a": False}
    assert round2[2]["config"] == {**base, "fast_a": True, "fast_b": 8}


def investigate(d, hid, *extra):
    return ("decide", hid, "investigate", "--observable-key", "passes", "--cause", "the pass count may not drop",
            "--measure", "passes", *extra)


def test_an_off_state_equal_to_the_incumbent_is_not_run_and_cites_the_incumbents_trial(driver):
    # A guard fails whenever fast_b moves, so the incumbent keeps it at default: fast_b's off-state is the incumbent.
    d = driver.start("round1_run", guard_fail="config.get('fast_b', 1) != 1", **FOUR)
    hb = d.hid("fast_b")
    d.ok(*investigate(d, hb))
    finish_round_1(d, hb, d.hid("fast_a"), d.hid("noop"))
    new_commit(d)
    d.round_run()
    round2 = d.trials(2)
    [inc] = [t for t in round2 if t["queued"] == "incumbent"]
    assert len(round2) == 2 and inc["config"]["fast_b"] == 1
    summary = (d.run_dir / "rounds" / "2" / "summary.md").read_text()
    assert f"investigate {hb} (off-state): same as trial {inc['trial']}" in summary and "(2 of 2)" in summary
    d.ok("decide", hb, "remove", "--reason", "passes flat, metric flat", "--trials", inc["trial"])
    assert json.dumps(inc) in d.ok("review", "show", f"rm-{hb}-r2")


def test_a_config_two_investigations_queue_runs_once(driver):
    d = driver.start("round1_run", **FOUR)
    ha, hb = d.hid("fast_a"), d.hid("fast_b")
    q = d.tmp / "q.json"
    q.write_text(json.dumps([{"fast_a": False, "fast_b": 2, "noop": True}]))
    d.ok(*investigate(d, ha, "--queue", q))
    d.ok(*investigate(d, hb, "--queue", q))
    finish_round_1(d, ha, d.hid("noop"))
    new_commit(d)
    log = d.round_run()
    round2 = d.trials(2)
    [ran] = [t["trial"] for t in round2 if t["queued"] == f"investigate {ha} (1/1)"]
    assert f"investigate {hb} (1/1)" not in [t["queued"] for t in round2]
    assert f"investigate {hb} (1/1): same as trial {ran}" in (d.run_dir / "rounds" / "2" / "summary.md").read_text()
    assert f"investigate {hb} (1/1): not run, same as trial {ran}" in log


def test_investigate_refusals(driver):
    d = driver.start("round1_run", **FOUR)
    ha, hb = d.hid("fast_a"), d.hid("fast_b")
    q = d.tmp / "q.json"
    assert "--cause" in d.refused("decide", ha, "investigate", "--observable-key", "passes", "--measure", "passes")
    assert "--measure" in d.refused("decide", ha, "investigate", "--observable-key", "passes", "--cause", "c")
    assert "investigate only" in d.refused("decide", ha, "keep", "--reason", "r", "--observable-key", "passes", "--cause", "c")
    q.write_text(json.dumps([{"fast_a": False}, {"fast_a": True}, {"fast_b": 4}]))
    assert "at most 2" in d.refused(*investigate(d, ha, "--queue", q))
    q.write_text(json.dumps([{"fast_a": False}, {"fast_b": 99}, {"cache_output": True}][:2]))
    assert "not a valid value of 'fast_b'" in d.refused(*investigate(d, ha, "--queue", q))
    q.write_text(json.dumps([{"fast_a": False, "cache_output": True}]))
    assert "'cache_output' is not a lever of an active hypothesis" in d.refused(*investigate(d, ha, "--queue", q))
    d.ok(*investigate(d, ha))
    finish_round_1(d, ha, hb, d.hid("noop"))
    d.round_run()
    assert "one investigation" in d.refused(*investigate(d, ha))


def test_no_investigation_in_the_last_two_rounds(driver):
    d = driver.start("round1_run", **{**FOUR, "rounds": 2})
    assert "last two rounds" in d.refused(*investigate(d, d.hid("fast_a")))


def test_investigate_after_a_rejected_removal(driver):
    d = driver.start("round1_run", **FOUR)
    ha = d.hid("fast_a")
    ids = [t["trial"] for t in d.trials(1)]
    d.decide(ha, "remove", "no effect", trials=ids)
    for hid in (d.hid("fast_b"), d.hid("noop")):
        d.decide(hid, "keep")
    d.ok("propose", "--none", "--reason", "nothing new")
    d.review(f"rm-{ha}-r1", "reject", "the trials barely varied the lever")
    d.ok(*investigate(d, ha))
    assert d.hyp(ha)["decisions"]["1"]["decision"] == "investigate"


def to_round2_after_investigating(d, measure="passes,batch"):
    """Round 1: investigate fast_a's hypothesis, keep the rest; round 2 has run."""
    ha = d.hid("fast_a")
    d.ok("decide", ha, "investigate", "--observable-key", "passes", "--cause", "the pass count may not drop",
         "--measure", measure)
    finish_round_1(d, ha, d.hid("fast_b"), d.hid("noop"))
    d.round_run()
    return ha, [t["trial"] for t in d.trials(2)]


def test_resolving_an_investigation_cites_trials_holding_every_measured_key(driver):
    d = driver.start("round1_run", **FOUR)
    ha, ids = to_round2_after_investigating(d, measure="passes,pass_ms")
    assert "cite" in d.refused("decide", ha, "keep", "--reason", "moved weakly")
    err = d.refused("decide", ha, "keep", "--reason", "moved weakly", "--trials", *ids)
    assert "pass_ms" in err and "fix" in err
    # The instrumentation for pass_ms never ran: fix is the way out.
    d.ok("decide", ha, "fix", "--reason", "pass_ms was never written", "--trials", ids[0])


def test_blocked_only_resolves_an_investigation(driver):
    d = driver.start("round1_run", **FOUR)
    ha, ids = to_round2_after_investigating(d)
    hb = d.hid("fast_b")
    assert "investigation" in d.refused("decide", hb, "blocked", "--reason", "x", "--trials", *ids)
    d.ok("decide", ha, "blocked", "--reason", "passes dropped but batch rose", "--trials", *ids)
    assert d.hyp(ha)["decisions"]["2"]["decision"] == "blocked"


def enabler(d, enables, joint, levers=None, **kw):
    lv = levers or [d.lever("fast_b_wide", "int", low=1, high=8, default=1)]
    return {**d.proposal("Widening the batch relieves the bottleneck", *lv, source="synthesis"),
            "enables": enables, "joint_config": joint, **kw}


def to_blocked(d):
    """Round 2 of FOUR: fast_a's hypothesis resolved blocked, the others kept; now in R4."""
    ha, ids = to_round2_after_investigating(d)
    d.ok("decide", ha, "blocked", "--reason", "passes dropped but batch rose", "--trials", *ids)
    for hid in (d.hid("fast_b"), d.hid("noop")):
        d.decide(hid, "keep")
    return ha, ids


def test_an_enables_proposal_names_a_blocked_hypothesis_and_a_joint_config(driver):
    d = driver.start("round1_run", **FOUR)
    ha, _ = to_blocked(d)
    hb = d.hid("fast_b")

    def problems(*props):
        return d.refused("propose", d.write_proposals(list(props)))

    assert "decided blocked this round" in problems(enabler(d, hb, {"fast_b_wide": 8}))
    assert "'joint_config' goes with 'enables'" in problems(enabler(d, None, {"fast_b_wide": 8}))
    assert "needs a 'joint_config'" in problems(enabler(d, ha, None))
    assert "at least one of its own levers away from default" in problems(enabler(d, ha, {"fast_b": 4}))
    assert "'cache_output' is not" in problems(enabler(d, ha, {"fast_b_wide": 8, "cache_output": True}))
    assert "not a valid value of 'fast_b_wide'" in problems(enabler(d, ha, {"fast_b_wide": 80}))
    [he] = d.propose(enabler(d, ha, {"fast_b_wide": 8, "fast_b": 2}))
    assert d.hyp(he)["enables"] == ha and d.hyp(he)["joint_config"] == {"fast_b_wide": 8, "fast_b": 2}
    pending = d.ok("review", "show", he)
    assert "passes dropped but batch rose" in pending  # the blocked hypothesis's reason is in the material


def test_r4_owes_a_supersede_after_a_keep_resolution_and_an_enabler_after_blocked(driver):
    d = driver.start("round1_run", **FOUR)
    ha, ids = to_round2_after_investigating(d)
    d.ok("decide", ha, "keep", "--reason", "passes moved only at fast_b >= 4", "--trials", *ids)
    hb, hn = d.hid("fast_b"), d.hid("noop")
    d.decide(hb, "keep")
    d.decide(hn, "keep")
    d.propose(d.proposal("Something else entirely", d.lever("other_knob"), source="synthesis"))
    text = d.next()
    assert text.startswith("[R4]") and f"supersedes {ha}" in text
    assert f"supersedes {ha}" in d.refused("round", "close")
    [hs] = d.propose(d.proposal("Retuned", d.lever("fast_a2"), source="synthesis", supersedes=ha))
    assert d.next().startswith("[R5]")
    # Re-deciding blocked moves the debt to an enabler.
    d.ok("decide", ha, "blocked", "--reason", "passes dropped but batch rose", "--trials", *ids)
    assert f"enables {ha}" in d.next()
    d.ok("propose", "--none", "--reason", "no enabler worth testing")
    assert d.next().startswith("[R5]")


def test_a_blocked_pair_gets_its_joint_trial(driver):
    d = driver.start("round1_run", **FOUR)
    ha, _ = to_blocked(d)
    [he] = d.propose(enabler(d, ha, {"fast_b_wide": 8}))
    d.review(he, "accept")
    out = d.close()
    assert "round 3 begins" in out and len([h for h in d.hyps() if h["status"] == "active"]) == 4
    d.round_run()
    round3 = d.trials(3)
    assert len(round3) == 2 + 1
    joint = next(t for t in round3 if t["queued"] == f"joint {he}+{ha}")
    inc = d.state()["rounds"]["3"]["warm"]["incumbent_config"] or {}
    assert joint["config"] == {"fast_a": False, "fast_b": 1, "noop": False, "cache_output": False, **inc, "fast_b_wide": 8}
    for hid in (ha, he, d.hid("fast_b"), d.hid("noop")):
        d.decide(hid, "keep", key="batch")
    [hx] = d.propose(d.proposal("One more idea", d.lever("extra_knob"), source="synthesis"))
    d.review(hx, "accept")
    assert "round 4 begins" in d.close()
    assert len([h for h in d.hyps() if h["status"] == "active"]) == 5
    d.round_run()


def test_removing_either_half_of_an_enabled_pair_cites_a_trial_that_moved_both(driver):
    d = driver.start("round1_run", **FOUR)
    ha, _ = to_blocked(d)
    [he] = d.propose(enabler(d, ha, {"fast_b_wide": 8, "fast_a": True}))
    d.review(he, "accept")
    d.close()
    d.round_run()
    round3 = d.trials(3)
    joint = next(t["trial"] for t in round3 if t["queued"] == f"joint {he}+{ha}")
    apart = [t["trial"] for t in d.trials() if not (t["config"]["fast_a"] and t["config"].get("fast_b_wide", 1) != 1)]
    for hid, partner in ((ha, he), (he, ha)):
        err = d.refused("decide", hid, "remove", "--observable-key", "batch", "--reason", "flat", "--trials", *apart)
        assert partner in err and "both" in err
    d.ok("decide", he, "remove", "--observable-key", "batch", "--reason", "flat even jointly", "--trials", joint, *apart)


def test_removal_material_of_an_investigated_hypothesis_holds_the_investigation_and_diag_dirs(driver):
    d = driver.start("round1_run", **FOUR)
    ha, ids = to_round2_after_investigating(d)
    d.ok("decide", ha, "remove", "--reason", "passes moved, nothing else did, metric flat", "--trials", *ids)
    text = d.ok("review", "show", f"rm-{ha}-r2")
    assert "investigated in round 1" in text and "cause: the pass count may not drop" in text
    assert "measured: passes, batch" in text and "observable key: passes" in text
    for n in ids:
        assert str(d.run_dir / "trials" / str(n) / "diag-1") in text


def section(text, heading):
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def test_the_summary_shows_lever_effects_on_each_diagnostic_and_links_diag_dirs(driver):
    d = driver.start("round1_run", **FOUR)
    ha, ids = to_round2_after_investigating(d)
    text = (d.run_dir / "rounds" / "2" / "summary.md").read_text()
    diag = section(text, "## Diagnostics")
    row = next(line for line in diag.splitlines() if line.startswith("| `passes` | `fast_a`"))
    assert "`false`: median 3" in row and "`true`: median 2" in row
    assert any(line.startswith("| `batch` | `fast_b`") and "Spearman" in line for line in diag.splitlines())
    assert f"{ha}: `passes`" in diag  # bound observable keys
    assert f"{ha} (round 1): cause: the pass count may not drop; measures passes, batch" in diag
    trials = section(text, "## All trials this round")
    assert all(f"[diag](../../trials/{n}/)" in trials for n in ids)


def test_the_report_has_a_generated_investigations_table(driver):
    d = driver.start("round1_run", **{**FOUR, "rounds": 3})
    ha, ids = to_round2_after_investigating(d)
    d.ok("decide", ha, "keep", "--reason", "passes moved only at some values", "--trials", *ids)
    for hid in (d.hid("fast_b"), d.hid("noop")):
        d.decide(hid, "keep")
    d.ok("propose", "--none", "--reason", "no retune worth a round")
    d.close()
    d.round_run()
    for h in d.hyps():
        if h["status"] == "active":
            d.decide(h["id"], "keep")
    d.ok("propose", "--none", "--reason", "last round")
    d.close()
    d.ok("finalize", "--foreground")
    report = (d.run_dir / "report.md").read_text()
    table = section(report, "## 7. Investigations")
    row = next(line for line in table.splitlines() if line.startswith(f"| {ha} |"))
    assert "the pass count may not drop" in row and "passes, batch" in row
    assert "keep (round 2)" in row and "passes moved only at some values" in row


def test_diag_prune_runs_only_in_r6_and_spares_what_a_decision_may_need(driver):
    d = driver.start("round1_run", **FOUR)
    round1 = [t["trial"] for t in d.trials(1)]
    assert "R6" in d.refused("diag", "prune")
    d.decide(d.hid("fast_a"), "keep", trials=[round1[0]])
    for hid in (d.hid("fast_b"), d.hid("noop")):
        d.decide(hid, "keep")
    d.ok("propose", "--none", "--reason", "nothing new")
    d.close()
    d.round_run()
    for h in d.hyps():
        if h["status"] == "active":
            d.decide(h["id"], "keep")
    d.ok("propose", "--none", "--reason", "nothing new")
    for t in d.trials():
        (d.run_dir / "trials" / str(t["trial"]) / "diag-1" / "trace.txt").write_text("x" * 100)
    out = d.ok("diag", "prune")
    incumbent = d.state()["rounds"]["2"]["incumbent_trial"]
    protected = {round1[0], incumbent, *(t["trial"] for t in d.trials(2))}
    for t in d.trials():
        diag = d.run_dir / "trials" / str(t["trial"]) / "diag-1"
        assert (diag / "diag.json").exists()
        if t["trial"] in protected:
            assert (diag / "trace.txt").exists(), t["trial"]
    pruned = [t["trial"] for t in d.trials(1)
              if not (d.run_dir / "trials" / str(t["trial"]) / "diag-1" / "trace.txt").exists()]
    assert pruned and set(pruned).isdisjoint(protected) and f"pruned {len(pruned)} trial" in out


def test_next_r3_names_unbound_observables_and_investigations_to_resolve(driver):
    d = driver.start("round1_run", **FOUR)
    text = d.next()
    assert text.startswith("[R3]") and "keep|fix|investigate|remove" in text
    assert "--observable-key" in text and d.hid("fast_a") in text.split("observable key")[1]
    ha, _ = to_round2_after_investigating(d)
    text = d.next()
    assert "keep|fix|investigate|blocked|remove" in text.splitlines()[0]
    assert f"resolve: {ha} (investigated in round 1: the pass count may not drop; measured passes, batch)" in text
