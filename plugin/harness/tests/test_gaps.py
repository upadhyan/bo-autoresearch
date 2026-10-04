"""Behaviour no other test pinned down: hang budgets, the final incumbent, summary numbers, median
semantics, the round study's own setup, report cells, and finalize's branch and eval checks.

Engine tests use a run written by hand (as in test_engine_rounds); CLI tests use the conftest driver.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import optuna
import pytest
from optuna.study import StudyDirection

from boar import evalrun, finalize, rounds, stats, store, warmstart
from boar.errors import Refused
from boar.store import Run

RUN_ID = "20261002-130000"
HANG_S = 5.0

EVAL = '''#!{python}
import json, os, sys, time
sys.dont_write_bytecode = True
cfg = json.load(open(os.environ["BOAR_CONFIG"]))
split = os.environ["BOAR_SPLIT"]
count = os.path.join(os.environ["BOAR_RUN_DIR"], "eval-count")
n = int(open(count).read()) if os.path.exists(count) else 0
open(count, "w").write(str(n + 1))
if {hang}:
    time.sleep({hang_s!r})
print(json.dumps({{"metric": {metric}, "guards_ok": True, "metrics": {{}}}}))
'''
COST = '100.0 - (20.0 if cfg["fast_a"] else 0.0) - 3.0 * (cfg["fast_b"] - 1)'

FAST_A = {"name": "fast_a", "type": "bool", "default": False}
FAST_B = {"name": "fast_b", "type": "int", "low": 1, "high": 8, "log": False, "default": 1}
CHEAT = {"name": "cache_output", "type": "bool", "default": False}
DEFAULTS = {"fast_a": False, "fast_b": 1, "cache_output": False}


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def hyp(hid: str, status: str, *levers: dict) -> dict:
    return {
        "id": hid, "round": 0, "status": status, "statement": f"{hid} moves the metric",
        "mechanism": f"mechanism of {hid}", "source": "user", "citations": [], "levers": list(levers),
        "supersedes": None, "verdict": {"decision": "accept", "reason": "fine", "at": store.now_iso()},
        "activated_round": 1 if status == "active" else None, "removed_round": None,
        "removed_reason": None, "decisions": {},
    }


def fresh_round() -> dict:
    return {"commit": None, "run_status": "not_started", "warm": None, "incumbent_trial": None,
            "new_trials": [], "propose_none": None, "closed": False}


def trial(n: int, r: int, config: dict, repeats: list[float], queued: str | None = None) -> dict:
    """A complete trials.jsonl record written by hand; its metric is the median of `repeats`."""
    return {"trial": n, "round": r, "commit": "c", "config": {**DEFAULTS, **config}, "state": "complete",
            "metric": statistics.median(repeats), "repeats": repeats, "metrics": {}, "duration_s": 1.0,
            "error": None, "queued": queued}


def write_eval(run: Run, hang: str = "False", metric: str = COST) -> None:
    exe = run.eval_dir / "run"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(EVAL.format(python=sys.executable, hang=hang, hang_s=HANG_S, metric=metric))
    exe.chmod(0o755)
    h = evalrun.eval_hash(run)
    run.update_state(lambda s: s["eval"].update(status="accepted", checked_hash=h, accepted_hash=h))


def configure(run: Run, **kw) -> None:
    store.write_json(run.config_path, {**run.config(), **kw})


def to_finalize(run: Run) -> None:
    """What closing the last round leaves behind, done by hand."""
    def advance(s: dict) -> None:
        s["rounds"]["1"]["closed"] = True
        s["phase"] = "finalize"

    run.update_state(advance)


@pytest.fixture(autouse=True)
def _reset_stop_flag():
    yield
    evalrun._live["stop"] = False


@pytest.fixture
def run(tmp_path, monkeypatch) -> Run:
    """A run in round 1 of 3 on branch boar/<id>: H1 (fast_a) and H2 (fast_b) active, H3 rejected, eval accepted."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "BOAR Test")
    git(repo, "config", "user.email", "boar-test@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")
    (repo / "target.py").write_text("X = 1\n")
    (repo / ".gitignore").write_text("__pycache__/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "target")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "-b", f"boar/{RUN_ID}")
    (repo / ".git" / "info" / "exclude").write_text("/.boar/\n")
    monkeypatch.chdir(repo)
    r = Run(root=repo, id=RUN_ID)
    store.write_json(r.config_path, {"rounds": 3, "trials_per_round": 2, "repeats": 1, "trial_target_s": 5,
                                     "holdout_repeats": 1, "seed": 0, "direction": "min"})
    r.spec_path.write_text("# Spec\n\n## Goal\nMake the toy cheaper.\n\n## Metric\nDirection: min\n")
    r.save_hypotheses([hyp("H1", "active", FAST_A), hyp("H2", "active", FAST_B), hyp("H3", "rejected", CHEAT)])
    r.save_state({
        "run_id": RUN_ID, "phase": "round", "round": 1, "session_id": None, "branch": f"boar/{RUN_ID}",
        "base_branch": "main", "base_commit": base, "created": store.now_iso(), "next_hypothesis": 4,
        "eval": {"status": "none", "checked_hash": None, "accepted_hash": None, "rejections": 0},
        "setup_failed_batches": 0, "rounds": {"1": fresh_round()}, "removals": [],
        "finalize": {"status": "not_started", "commit": None, "incumbent_config": None}, "abort_reason": None,
    })
    (repo / ".boar" / "active").write_text(RUN_ID + "\n")
    write_eval(r)
    return r


# --- hang budgets --------------------------------------------------------------------------------


def test_a_hung_round_trial_is_killed_at_three_times_the_target(run):
    configure(run, trial_target_s=0.3)
    write_eval(run, hang="True")
    started = time.monotonic()
    rounds.start_round(run, foreground=True)
    trials = run.trials()
    assert [(t["state"], t["error"]) for t in trials] == [("failed", "timeout after 0.9s")] * 2
    assert time.monotonic() - started < HANG_S, "the round waited for the hung eval"
    assert run.state()["rounds"]["1"]["run_status"] == "done"


def test_a_hung_holdout_run_is_killed_at_its_share_of_the_budget(run):
    configure(run, trial_target_s=0.5, repeats=2)
    write_eval(run, hang='split == "holdout"')
    rounds.start_round(run, foreground=True)
    to_finalize(run)
    started = time.monotonic()
    finalize.start_finalize(run, foreground=True)
    holdout = store.read_jsonl(run.holdout_path)
    # One holdout run gets one dev repeat's share: 3 x trial_target_s / repeats.
    assert [(h["ok"], h["error"]) for h in holdout] == [(False, "timeout after 0.75s")] * 2
    assert time.monotonic() - started < HANG_S, "finalize waited for the hung eval"


# --- the final incumbent counts the last round ---------------------------------------------------


def test_the_final_incumbent_counts_the_last_rounds_trials(run):
    for t in (trial(1, 1, {}, [100.0]), trial(2, 1, {"fast_a": True}, [80.0]), trial(3, 2, {"fast_b": 4}, [91.0]),
              trial(4, 3, {"fast_a": True, "fast_b": 8}, [59.0])):
        run.append_trial(t)
    assert finalize.final_incumbent(run) == {"fast_a": True, "fast_b": 8}


# --- summary.md: lever effects, drift, the post-round incumbent ----------------------------------


def test_drift_is_flagged_only_beyond_the_noise_floor():
    before = [trial(1, 1, {"fast_a": True}, [59.0, 59.0])]
    moved = trial(2, 2, {"fast_a": True}, [70.0, 70.0], queued="incumbent")
    steady = trial(2, 2, {"fast_a": True}, [59.1, 59.1], queued="incumbent")
    assert rounds._drift(moved, before, DEFAULTS, 0.2)["flag"] == "yes"
    assert rounds._drift(steady, before, DEFAULTS, 0.2)["flag"] == "no"


def test_drift_far_below_the_baseline_is_judged_against_the_floor_at_that_scale():
    """A baseline at 6.9 s with spread 0.08 must not hide a 23% drift of an incumbent at 0.34 s."""
    before = [trial(1, 1, {"fast_a": True}, [0.3354, 0.3362])]
    moved = trial(2, 2, {"fast_a": True}, [0.4131, 0.4141], queued="incumbent")
    steady = trial(2, 2, {"fast_a": True}, [0.3360, 0.3362], queued="incumbent")
    d = rounds._drift(moved, before, DEFAULTS, 0.0819, 6.895)
    assert d["flag"] == "yes", d
    assert d["why"] == "|Δ| = 0.0778 > noise floor 0.00398869, scaled to this metric from 0.0819 at the baseline's 6.895"
    assert rounds._drift(steady, before, DEFAULTS, 0.0819, 6.895)["flag"] == "no"
    assert rounds._drift(moved, before, DEFAULTS, 0.0819)["flag"] == "no", "unscaled, the baseline's floor hides it"
    assert stats.floor_at(0.2, -100.0, 50.0) == 0.2, "a metric that crosses zero has no scale"


def test_a_numeric_lever_that_raises_the_metric_has_positive_rank_correlation():
    space = {"fast_b": {**FAST_B, "hypothesis": "H2"}}
    trials = [trial(n, 1, {"fast_b": n}, [float(10 * n)]) for n in range(1, 5)]
    assert rounds._lever_effects(trials, space, DEFAULTS) == [["`fast_b`", "H2", "Spearman ρ = 1 (n=4); tried 1 … 4"]]


def test_round_summary_reports_effects_drift_and_incumbent_over_the_warm_set(run):
    for t in (
        trial(1, 1, {}, [100.0, 100.2], "baseline"),
        trial(2, 1, {"fast_a": True}, [80.0, 80.0]),
        trial(3, 1, {"fast_a": True, "fast_b": 7}, [61.0, 61.0]),
        trial(4, 1, {"fast_a": True, "fast_b": 8}, [59.0, 59.0]),
        # Round 2 re-measures round 1's best config and finds it much worse.
        trial(5, 2, {"fast_a": True, "fast_b": 8}, [70.0, 70.0], "incumbent"),
        trial(6, 2, {"fast_b": 2}, [97.0, 97.0]),
        trial(7, 2, {"fast_a": True, "fast_b": 3}, [74.0, 74.0]),
        trial(8, 2, {}, [100.0, 100.0]),
    ):
        run.append_trial(t)
    run.update_state(lambda s: s.update(round=2, rounds={"1": {**fresh_round(), "closed": True}, "2": fresh_round()}))
    summary = rounds.write_summary(run, 2).read_text()

    assert ("**drift: yes** (|Δ| = 11 > noise floor 0.117882, scaled to this metric from 0.2 at the baseline's "
            "100.1)") in summary
    assert "That is 0.1% of the baseline's median 100.05." in summary
    assert "the floor scaled to their metric: 0.0609695 at the incumbent's 61" in summary
    # Pooled with its re-measurement, round 1's best (median 64.5) falls behind a config round 2 never ran.
    assert ('After the round: `{"fast_a": true, "fast_b": 7}`, pooled metric 61 over 2 repeats '
            "(trials 3; best single trial 3).") in summary
    assert "Over the warm-start set plus this round's new trials: 8 complete trials." in summary
    assert "| `fast_a` | H1 | `false`: median 100 (n=3); `true`: median 70 (n=5) |" in summary
    assert re.search(r"\| `fast_b` \| H2 \| Spearman ρ = -0\.\d+ \(n=8\); tried 1 … 8 \|", summary), summary


# --- medians, not means --------------------------------------------------------------------------


def test_a_trial_metric_is_the_median_of_its_repeats(run):
    write_eval(run, metric="[1.0, 1.0, 10.0][n]")
    res = evalrun.run_trial(run, store.full_config(run.hypotheses(), {}), "dev", 3, run.trial_dir(1), 30)
    assert res["repeats"] == [1.0, 1.0, 10.0] and res["metric"] == 1.0


@pytest.mark.parametrize("direction, a, best", [("min", ([1.0, 100.0], [1.0]), 1.0), ("max", ([10.0, -80.0], [10.0]), 10.0)])
def test_the_incumbent_ranks_configs_by_the_median_of_their_pooled_repeats(direction, a, best):
    # Config A pools three repeats whose mean is far worse than config B's, but whose median is better.
    trials = [trial(1, 1, {"fast_a": True}, a[0]), trial(2, 1, {"fast_a": True}, a[1]),
              trial(3, 1, {"fast_b": 2}, [5.0, 5.0])]
    inc = stats.incumbent(trials, DEFAULTS, direction)
    assert inc["config"] == {"fast_a": True} and inc["metric"] == best


# --- the round study's own setup -----------------------------------------------------------------


def test_round_study_follows_the_direction_and_starts_modelling_after_one_rounds_worth(run, monkeypatch):
    configure(run, direction="max", trials_per_round=3)
    made: list[dict] = []
    real = optuna.samplers.TPESampler

    class Recording(real):
        def __init__(self, **kw):
            made.append(kw)
            super().__init__(**kw)

    monkeypatch.setattr(optuna.samplers, "TPESampler", Recording)
    rounds.start_round(run, foreground=True)
    assert made and all(kw["n_startup_trials"] == 3 and kw["multivariate"] is True for kw in made), made
    warmstart.quiet_optuna()
    study = optuna.load_study(study_name="round-1", storage=run.storage_url, sampler=optuna.samplers.RandomSampler())
    assert study.direction == StudyDirection.MAXIMIZE


# --- report cells --------------------------------------------------------------------------------


def test_report_lists_a_rejected_removal_under_rejected_by_review(run):
    run.append_trial(trial(1, 1, {}, [100.0]))
    run.append_trial(trial(2, 1, {"fast_a": True}, [80.0]))
    verdict = {"decision": "reject", "reason": "effect is real", "at": store.now_iso()}
    run.update_state(lambda s: s.update(phase="finalize", removals=[
        {"id": "rm-H1-r1", "hypothesis": "H1", "round": 1, "reason": "no effect", "trials": [1, 2], "verdict": verdict}]))
    run.append_review({"item": "rm-H1-r1", "kind": "removal", "decision": "reject", "reason": "effect is real",
                       "round": 1, "at": store.now_iso()})
    report = finalize.write_report(run).read_text()
    assert "| rm-H1-r1 (remove H1: H1 moves the metric) | removal | 1 | effect is real |" in report


# --- finalize refuses an edited eval or the wrong branch -----------------------------------------


def test_finalize_engine_refuses_an_edited_eval_before_recording_anything(run):
    rounds.start_round(run, foreground=True)
    to_finalize(run)
    (run.eval_dir / "notes.txt").write_text("tweak\n")
    with pytest.raises(Refused, match="frozen"):
        finalize.start_finalize(run, foreground=True)
    assert run.state()["finalize"] == {"status": "not_started", "commit": None, "incumbent_config": None}
    assert not run.holdout_path.exists()


def _at_finalize(driver):
    d = driver.start("round1_run", rounds=1, trials_per_round=2, repeats=1, trial_target_s=5, holdout_repeats=1)
    for lever in ("fast_a", "fast_b", "noop"):
        d.decide(d.hid(lever), "keep")
    d.ok("propose", "--none", "--reason", "Single-round run.")
    d.close()
    assert d.tag() == "FINALIZE"
    return d


UNTOUCHED = {"status": "not_started", "commit": None, "incumbent_config": None}


def test_finalize_refuses_off_the_run_branch_and_leaves_main_alone(driver):
    d = _at_finalize(driver)
    main = d.git("rev-parse", "main")
    target = d.repo / "target.py"
    target.write_text(target.read_text() + "# edited on the user's branch\n")
    d.git("checkout", "-q", "main")
    assert "main is checked out" in d.refused("finalize", "--foreground")
    d.git("checkout", "--", "target.py")
    assert "main is checked out" in d.refused("finalize", "--foreground"), "same commit, still the wrong branch"
    assert d.git("rev-parse", "main") == main
    assert d.state()["finalize"] == UNTOUCHED and not (d.run_dir / "holdout.jsonl").exists()
    d.git("checkout", "-q", f"boar/{d.run_id}")
    d.ok("finalize", "--foreground")
    assert d.git("rev-parse", "main") == main


def test_finalize_refuses_an_eval_edited_after_acceptance(driver):
    d = _at_finalize(driver)
    (d.run_dir / "eval" / "notes.txt").write_text("tweak\n")
    err = d.refused("finalize", "--foreground")
    assert "eval/ has changed since the reviewer accepted it (added: notes.txt)" in err
    assert d.state()["finalize"] == UNTOUCHED and not (d.run_dir / "holdout.jsonl").exists()
