"""Acceptance: every step taken out of order is refused, and refusing changes nothing.

design.md "Control principle": each command refuses to run until its preconditions hold.
"""

from __future__ import annotations

import json
import time


def _removal_id(d, hid: str) -> str:
    return next(rm["id"] for rm in d.state()["removals"] if rm["hypothesis"] == hid)


def test_round_run_refused_until_r5_is_complete(driver):
    d = driver.start("round1_run", **driver.SMALL)
    ha, hb, hn = d.hid("fast_a"), d.hid("fast_b"), d.hid("noop")
    n = len(d.trials())
    d.refused("round", "run")
    d.decide(ha, "keep")
    d.decide(hb, "keep")
    d.ok("decide", hn, "remove", "--observable-key", "passes", "--reason", "no effect beyond the noise floor", "--trials", "1", "2")
    d.refused("round", "run")
    d.ok("propose", "--none", "--reason", "nothing new to try")
    assert d.tag() == "R5"
    d.refused("round", "run")
    d.refused("round", "close")
    assert len(d.trials()) == n

    rm = _removal_id(d, hn)
    d.review(rm, "accept", "The cited trials set noop both ways and show no effect.")
    d.refused("review", "record", rm, "reject", "--reason", "second thoughts")
    d.close()
    assert d.hyp(hn)["status"] == "removed"
    d.round_run()
    assert len(d.trials(2)) == 2


def test_round_run_refused_while_a_proposal_verdict_is_pending(driver):
    d = driver.start("checked", **driver.SMALL)
    ha, hb, hn, hc = (d.hid(x) for x in ("fast_a", "fast_b", "noop", "cache_output"))
    d.review(ha, "accept")
    d.review("eval", "accept")
    assert d.state()["phase"] == "setup"
    d.refused("round", "run")
    assert d.tag() == "S5"
    d.review(hb, "accept")
    d.review(hn, "accept")
    d.review(hc, "reject", "A cheat named in the spec.")
    assert d.state()["phase"] == "round"

    d.round_run()
    for h in (ha, hb, hn):
        d.decide(h, "keep")
    (hx,) = d.propose(d.proposal("Prefetching cuts cost", d.lever("prefetch"), source="synthesis"))
    d.refused("round", "run")
    d.refused("round", "close")
    d.review(hx, "accept")
    d.close()
    assert d.hyp(hx)["status"] == "active" and d.hyp(hx)["activated_round"] == 2


def test_round_run_refused_after_the_accepted_eval_is_edited(driver):
    d = driver.start("round1", **driver.SMALL)
    run = d.run_dir / "eval" / "run"
    original = run.read_text()
    accepted = d.state()["eval"]["accepted_hash"]
    run.write_text(original + "# tweaked after acceptance\n")
    d.refused("round", "run")
    assert d.trials() == []

    rc, out, err = d.boar("eval", "check", "--foreground")
    assert rc == 0
    assert any(w in (out + err).lower() for w in ("changed", "frozen", "hash")), "eval check must warn loudly"
    assert d.state()["eval"]["accepted_hash"] == accepted, "a re-check never re-freezes an accepted eval"
    d.refused("round", "run")

    run.write_text(original)
    pycache = d.run_dir / "eval" / "__pycache__"
    pycache.mkdir()
    (pycache / "helper.cpython-310.pyc").write_bytes(b"\x00bytecode")
    d.round_run()
    assert len(d.trials(1)) == 2


def test_propose_refused_outside_setup_and_r4(driver):
    d = driver.start("round1", **driver.SMALL)
    path = d.write_proposals([d.proposal("Prefetching cuts cost", d.lever("prefetch"), source="synthesis")])
    d.refused("propose", path)
    d.refused("propose", "--none", "--reason", "nothing new")
    d.round_run()
    d.refused("propose", path)
    d.refused("propose", "--none", "--reason", "nothing new")
    count = len(d.hyps())
    for lever in ("fast_a", "fast_b", "noop"):
        d.decide(d.hid(lever), "keep")
    d.ok("propose", path)
    assert len(d.hyps()) == count + 1


def test_propose_refuses_invalid_files_whole(driver):
    d = driver.start("init", **driver.SMALL)
    bad_levers = [
        {"name": "no_default", "type": "int", "low": 1, "high": 4},
        {"name": "no_default_bool", "type": "bool"},
        {"name": "no_range", "type": "int", "default": 1},
        {"name": "no_choices", "type": "categorical", "default": "a"},
        {"name": "odd_type", "type": "complex", "default": 0},
    ]
    for bad in bad_levers:
        d.refused("propose", d.write_proposals([d.proposal("Bad lever", bad)]))
    d.refused("propose", d.write_proposals([d.proposal("One", d.lever("twin")), d.proposal("Two", d.lever("twin"))]))
    d.refused("propose", "--none", "--reason", "setup needs a proposal")
    assert d.hyps() == []

    d.propose(d.prop_a())
    d.refused("propose", d.write_proposals([d.proposal("Same lever again", d.lever("fast_a"))]))
    d.refused("propose", d.write_proposals([d.prop_b(), d.proposal("Bad", {"name": "nd", "type": "bool"})]))
    assert [lv["name"] for h in d.hyps() for lv in h["levers"]] == ["fast_a"], "a refused file adds nothing"


def test_decide_refusals(driver):
    d = driver.start("round1", **driver.SMALL)
    ha, hn, hc = d.hid("fast_a"), d.hid("noop"), d.hid("cache_output")
    d.refused("decide", ha, "keep", "--observable-key", "passes", "--reason", "before the round ran")
    d.round_run()
    d.refused("decide", hc, "keep", "--observable-key", "passes", "--reason", "it was rejected")
    d.refused("decide", "H99", "keep", "--observable-key", "passes", "--reason", "no such hypothesis")
    d.refused("decide", ha, "keep", "--observable-key", "passes", "--reason", "cites a trial from nowhere", "--trials", "999")
    d.refused("decide", ha, "remove", "--observable-key", "passes", "--reason", "a removal must cite trials")
    d.refused("decide", ha, "fix", "--reason", "a fix must cite trials")
    assert d.hyp(ha)["decisions"] == {}
    assert d.state()["removals"] == []

    d.decide(ha, "keep", trials=[1])
    d.decide(d.hid("fast_b"), "keep")
    d.ok("decide", hn, "remove", "--observable-key", "passes", "--reason", "no effect", "--trials", "1,2")
    assert len(d.state()["removals"]) == 1
    d.decide(hn, "keep")
    assert d.state()["removals"] == [], "changing a remove to keep drops its pending removal"
    d.ok("decide", hn, "remove", "--observable-key", "passes", "--reason", "no effect after all", "--trials", "1", "2")
    d.ok("propose", "--none", "--reason", "nothing new")
    d.review(_removal_id(d, hn), "accept", "No effect within the noise floor.")
    d.refused("decide", hn, "keep", "--observable-key", "passes", "--reason", "too late: the removal has a verdict")


def test_finalize_refused_while_rounds_remain(driver):
    d = driver.start("round1", **driver.SMALL)
    d.refused("finalize")
    d.refused("finalize", "--foreground")
    d.round_run()
    d.refused("finalize", "--foreground")
    assert not (d.run_dir / "report.md").exists()
    assert not (d.run_dir / "holdout.jsonl").exists()


def test_round_close_activates_every_accepted_proposal(driver):
    d = driver.start("round1_run", proposals=[driver.prop_a(), driver.prop_cheat()], **driver.SMALL)
    ha = d.hid("fast_a")
    d.refused("round", "close")
    d.ok("decide", ha, "remove", "--observable-key", "passes", "--reason", "no effect", "--trials", "1", "2")
    hb, hn = d.propose(d.prop_b(), d.prop_noop())
    d.review(_removal_id(d, ha), "accept", "No effect within the noise floor.")
    d.review(hb, "accept")
    d.review(hn, "accept")
    d.close()
    st = d.state()
    assert st["round"] == 2
    assert d.hyp(ha)["status"] == "removed"
    assert d.hyp(hb)["status"] == "active" and d.hyp(hn)["status"] == "active"


def test_init_refusals(driver, monkeypatch, tmp_path):
    d = driver
    # Outside the repo: an untracked file in it would be refused for being untracked, not for the spec.
    nodirection = tmp_path / "nodirection.md"
    nodirection.write_text("# Spec\n\n## Metric\nCost, lower is better.\n")
    assert "no direction" in d.refused("init", "--spec", nodirection)
    target = d.repo / "target.py"
    target.write_text(target.read_text() + "# uncommitted\n")
    d.refused("init", "--spec", "spec.md")
    d.git("checkout", "--", "target.py")
    rc, _, _ = d.boar("init", "--spec", "spec.md", "--bogus-key", "3")
    assert rc != 0
    assert not (d.repo / ".boar" / "active").exists()

    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    (plain / "spec.md").write_text((d.repo / "spec.md").read_text())
    monkeypatch.chdir(plain)
    rc, _, err = d.boar("init", "--spec", "spec.md")
    assert rc == 1 and "refused" in err.lower()
    monkeypatch.chdir(d.repo)

    d.init(**d.SMALL)
    first = d.run_id
    d.refused("init", "--spec", "spec.md")
    assert d.run_id == first
    d.ok("abort", "--reason", "the user wants to restart")
    assert d.state()["phase"] == "aborted"
    d.init(**d.SMALL)
    assert d.run_id != first and d.state()["phase"] == "setup"


def test_review_record_refuses_items_that_are_not_pending(driver):
    d = driver.start("checked", **driver.SMALL)
    ha, hb = d.hid("fast_a"), d.hid("fast_b")
    d.review(ha, "accept")
    d.refused("review", "record", ha, "accept", "--reason", "again")
    d.refused("review", "record", ha, "reject", "--reason", "changed my mind")
    d.refused("review", "record", "H99", "accept", "--reason", "no such item")
    d.refused("review", "record", "rm-H1-r1", "accept", "--reason", "no such removal")
    d.refused("review", "record", hb, "accept", "--reason", "")
    d.review("eval", "accept")
    d.refused("review", "record", "eval", "accept", "--reason", "again")
    assert [(r["item"], r["decision"]) for r in d.reviews()] == [(ha, "accept"), ("eval", "accept")]


def test_eval_accept_refused_when_edited_since_the_check(driver):
    d = driver.start("checked", **driver.SMALL)
    run = d.run_dir / "eval" / "run"
    run.write_text(run.read_text() + "# edited after the check\n")
    d.refused("review", "record", "eval", "accept", "--reason", "looks fine")
    assert d.state()["eval"]["status"] != "accepted"
    d.eval_check()
    d.review("eval", "accept")
    ev = d.state()["eval"]
    assert ev["status"] == "accepted" and ev["accepted_hash"] == ev["checked_hash"]


def test_rejected_eval_goes_back_to_s4(driver):
    d = driver.start("checked", **driver.SMALL)
    d.review("eval", "reject", "Holdout overlaps dev.")
    ev = d.state()["eval"]
    assert ev["status"] == "rejected" and ev["rejections"] == 1
    assert d.tag() == "S4"
    d.refused("review", "record", "eval", "accept", "--reason", "not re-checked yet")
    d.eval_check()
    assert d.state()["eval"]["status"] == "pending"
    d.review("eval", "accept")


def test_eval_check_fails_on_contract_and_guard_problems(driver):
    d = driver.start("proposed", **driver.SMALL)
    rc, _, _ = d.boar("eval", "check", "--foreground")
    assert rc == 1, "no eval at all"
    d.refused("review", "record", "eval", "accept", "--reason", "there is nothing to accept")

    d.write_eval(guard_fail="True")
    rc, _, _ = d.boar("eval", "check", "--foreground")
    assert rc == 1, "guards must pass at baseline"
    assert json.loads((d.run_dir / "eval_check.json").read_text())["ok"] is False
    d.refused("review", "record", "eval", "accept", "--reason", "guards fail")

    d.write_eval(crash_when="True")
    rc, _, _ = d.boar("eval", "check", "--foreground")
    assert rc == 1
    assert json.loads((d.run_dir / "eval_check.json").read_text())["trial"]["state"] == "failed"

    run = d.write_eval()
    run.chmod(0o644)
    rc, _, _ = d.boar("eval", "check", "--foreground")
    assert rc == 1, "eval/run must be executable"

    d.write_eval()
    d.eval_check()
    assert d.state()["eval"]["status"] == "pending"


def test_eval_check_refuses_slow_and_hung_evals(driver):
    d = driver.start("proposed", rounds=1, trials_per_round=2, repeats=1, trial_target_s=1, holdout_repeats=1)
    d.write_eval(sleep=2.4)
    rc, _, _ = d.boar("eval", "check", "--foreground")
    assert rc == 1, "a baseline trial over 2 x trial_target_s fails the check"
    assert json.loads((d.run_dir / "eval_check.json").read_text())["ok"] is False

    d.write_eval(sleep=60)
    started = time.monotonic()
    rc, _, _ = d.boar("eval", "check", "--foreground")
    assert rc == 1
    assert time.monotonic() - started < 20, "a hung eval is killed at 3 x trial_target_s"
    trial = json.loads((d.run_dir / "eval_check.json").read_text())["trial"]
    assert trial["state"] == "failed" and "timeout" in (trial["error"] or "").lower()
