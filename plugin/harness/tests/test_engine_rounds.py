"""rounds, finalize and worker on a throwaway git repo, with run files written by hand in the plan's formats."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
import pytest
from optuna.trial import TrialState

from boar import evalrun, finalize, gitops, rounds, stats, store, warmstart, worker
from boar.errors import Refused
from boar.store import Run

RUN_ID = "20261002-120000"

EVAL = '''#!{python}
import json, os, sys
sys.dont_write_bytecode = True
cfg = json.load(open(os.environ["BOAR_CONFIG"]))
run_dir = os.environ["BOAR_RUN_DIR"]
if {crash}:
    print("boom", file=sys.stderr)
    sys.exit(3)
path = os.path.join(run_dir, "eval-count")
n = int(open(path).read()) if os.path.exists(path) else 0
open(path, "w").write(str(n + 1))
if {hang}:
    import subprocess, time
    child = subprocess.Popen(["sleep", "30"])
    open(os.path.join(run_dir, "eval-child.pid"), "w").write(str(child.pid))
    time.sleep(30)
{extra}
cost = 100.0 - (20.0 if cfg["fast_a"] else 0.0) - 3.0 * (cfg["fast_b"] - 1) + 0.1 * (n % 3)
cost += 0.5 if os.environ["BOAR_SPLIT"] == "holdout" else 0.0
print("some progress output")
print(json.dumps({{"metric": cost, "guards_ok": not ({guard}), "metrics": {{"calls": n}}}}))
'''


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def hyp(hid: str, status: str, *levers: dict, activated: int | None = 1, round_: int = 0) -> dict:
    return {
        "id": hid, "round": round_, "status": status, "statement": f"{hid} moves the metric",
        "mechanism": f"mechanism of {hid}", "source": "user", "citations": [], "levers": list(levers),
        "supersedes": None, "verdict": {"decision": "accept", "reason": "fine", "at": store.now_iso()},
        "activated_round": activated if status in ("active", "removed") else None, "removed_round": None,
        "removed_reason": None, "decisions": {},
    }


FAST_A = {"name": "fast_a", "type": "bool", "default": False}
FAST_B = {"name": "fast_b", "type": "int", "low": 1, "high": 8, "log": False, "default": 1}
CHEAT = {"name": "cache_output", "type": "bool", "default": False}
NOOP = {"name": "noop", "type": "bool", "default": False}


def fresh_round() -> dict:
    return {"commit": None, "run_status": "not_started", "warm": None, "incumbent_trial": None,
            "new_trials": [], "propose_none": None, "closed": False}


def write_eval(run: Run, crash: str = "False", guard: str = "False", hang: str = "False", extra: str = "") -> None:
    """`extra` is a Python statement the eval runs before printing its result (cfg, n and run_dir are in scope)."""
    exe = run.eval_dir / "run"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(EVAL.format(python=sys.executable, crash=crash, guard=guard, hang=hang, extra=extra))
    exe.chmod(0o755)


def accept_eval(run: Run) -> None:
    """What the reviewer's accept leaves: the hash, the file list and the snapshot to restore from."""
    h, manifest = evalrun.eval_hash(run), evalrun.eval_manifest(run)
    shutil.rmtree(run.eval_accepted_dir, ignore_errors=True)
    shutil.copytree(run.eval_dir, run.eval_accepted_dir, symlinks=True)
    run.update_state(lambda s: s["eval"].update(
        status="accepted", checked_hash=h, accepted_hash=h, accepted_manifest=manifest))


@pytest.fixture(autouse=True)
def _reset_stop_flag():
    yield
    evalrun._live["stop"] = False


@pytest.fixture
def run(tmp_path, monkeypatch) -> Run:
    """A run in round 1 on branch boar/<id>, with H1, H2 active, H3 rejected and an accepted eval."""
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
    store.write_json(r.config_path, {"rounds": 3, "trials_per_round": 4, "repeats": 2, "trial_target_s": 5,
                                     "holdout_repeats": 2, "seed": 0, "direction": "min"})
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
    accept_eval(r)
    return r


def load_study(run: Run, r: int) -> optuna.Study:
    warmstart.quiet_optuna()
    return optuna.load_study(study_name=f"round-{r}", storage=run.storage_url)


def close_round(run: Run, r: int, decisions: dict[str, str], new: list[dict] = ()) -> None:
    """What `boar round close` would leave behind, done by hand."""
    hyps = run.hypotheses()
    for h in hyps:
        if h["id"] in decisions:
            h["decisions"][str(r)] = {"decision": decisions[h["id"]], "reason": "by hand", "trials": [], "at": store.now_iso()}
    hyps.extend(new)
    run.save_hypotheses(hyps)

    def advance(s: dict) -> None:
        s["rounds"][str(r)]["closed"] = True
        s["round"] = r + 1
        s["rounds"][str(r + 1)] = fresh_round()

    run.update_state(advance)


def test_round_1_runs_baseline_first_and_records_everything(run):
    (run.root / "target.py").write_text("X = 2  # lever code\n")
    text = rounds.start_round(run, foreground=True)
    assert "round 1 finished" in text
    trials = run.trials()
    assert [t["trial"] for t in trials] == [1, 2, 3, 4] and all(t["round"] == 1 for t in trials)
    first = trials[0]
    assert first["queued"] == "baseline" and first["config"] == {"fast_a": False, "fast_b": 1, "cache_output": False}
    assert all(t["queued"] is None for t in trials[1:])
    assert all(set(t["config"]) == {"fast_a", "fast_b", "cache_output"} and t["config"]["cache_output"] is False
               for t in trials)
    assert all(t["state"] == "complete" and len(t["repeats"]) == 2 for t in trials)
    assert set(first) == {"trial", "round", "commit", "config", "state", "metric", "repeats", "metrics",
                          "duration_s", "error", "queued", "diagnostics", "diag_error"}

    rec = run.state()["rounds"]["1"]
    head = git(run.root, "rev-parse", "HEAD")
    assert rec["commit"] == head == first["commit"]
    assert git(run.root, "log", "-1", "--format=%s") == "boar: round 1"
    assert "target.py" in git(run.root, "show", "--name-only", "--format=", "HEAD")
    assert git(run.root, "branch", "--show-current") == f"boar/{RUN_ID}"
    assert rec["run_status"] == "done" and rec["new_trials"] == [1, 2, 3, 4]
    assert rec["warm"] == {"copied": 0, "excluded": {"rule1_state": 0, "rule2_outside_space": 0, "rule3_fixed": 0},
                           "incumbent_config": None, "queue": [], "same_as": {}}
    defaults = store.lever_defaults(run.hypotheses())
    best = stats.incumbent(trials, defaults, "min")
    assert rec["incumbent_trial"] in best["trials"]

    study = load_study(run, 1)
    assert [t.state for t in study.trials] == [TrialState.COMPLETE] * 4
    assert [t.user_attrs["boar_trial"] for t in study.trials] == [1, 2, 3, 4]
    assert all(t.constraints == {"guards": 0.0} for t in study.trials)
    assert set(study.trials[0].params) == {"fast_a", "fast_b"}  # the rejected hypothesis's lever stays pinned

    out = run.trial_dir(2)
    assert json.loads((out / "config.json").read_text()) == trials[1]["config"]
    assert (out / "repeat-1.stdout").exists() and (out / "repeat-2.stderr").exists()

    summary = run.summary_path(1).read_text()
    for heading in ("# Round 1 summary", "## Incumbent", "## Noise floor", "## Warm start", "## Lever effects",
                    "## Infeasible and failed trials", "## All trials this round"):
        assert heading in summary
    assert "Copied 0 earlier trials" in summary and "drift: n/a" in summary
    assert "`fast_a`" in summary and "`fast_b`" in summary and "`cache_output`" not in summary


def test_infeasible_and_failed_trials_are_recorded_and_told_to_optuna(run):
    store.write_json(run.config_path, {**run.config(), "trials_per_round": 7})
    write_eval(run, crash="cfg['fast_b'] >= 5", guard="cfg['fast_a']")
    accept_eval(run)
    rounds.start_round(run, foreground=True)
    trials = run.trials()

    def expected(config):
        return "failed" if config["fast_b"] >= 5 else "infeasible" if config["fast_a"] else "complete"

    assert [t["state"] for t in trials] == [expected(t["config"]) for t in trials]
    states = {t["state"] for t in trials}
    assert {"failed", "infeasible"} <= states, "the seeded sampler should hit both outcomes"
    for t in trials:
        if t["state"] == "failed":
            assert t["error"] == "exit 3" and t["metric"] is None and t["repeats"] == []
        if t["state"] == "infeasible":
            assert len(t["repeats"]) == 1 and t["metric"] is not None
    by_n = {ft.user_attrs["boar_trial"]: ft for ft in load_study(run, 1).trials}
    for t in trials:
        ft = by_n[t["trial"]]
        if t["state"] == "failed":
            assert ft.state == TrialState.FAIL
        else:
            assert ft.state == TrialState.COMPLETE
            assert ft.constraints == {"guards": 1.0 if t["state"] == "infeasible" else 0.0}
    summary = run.summary_path(1).read_text()
    bad = [t for t in trials if t["state"] != "complete"]
    for t in bad:
        assert f"[trials/{t['trial']}/](../../trials/{t['trial']}/)" in summary
    assert "a guard failed" in summary and "exit 3" in summary


def test_round_2_warm_starts_from_round_1_and_requeues_the_incumbent(run):
    rounds.start_round(run, foreground=True)
    round1 = run.trials()
    # H2 marked fix: round-1 trials that moved fast_b leave the warm-start set. A new hypothesis adds `noop`.
    close_round(run, 1, {"H1": "keep", "H2": "fix"}, new=[hyp("H4", "active", NOOP, activated=2, round_=1)])
    (run.root / "target.py").write_text("X = 2  # R1 repairs H2: a new commit, so the incumbent is measured again\n")
    hyps = run.hypotheses()
    defaults = store.lever_defaults(hyps)
    valid, counts = warmstart.select(round1, hyps, 2)
    moved_b = [t for t in round1 if t["config"]["fast_b"] != 1]
    assert counts == {"rule1_state": 0, "rule2_outside_space": 0, "rule3_fixed": len(moved_b)}
    assert valid, "the baseline trial at least must carry over"

    rounds.start_round(run, foreground=True)
    rec = run.state()["rounds"]["2"]
    inc = stats.incumbent(valid, defaults, "min")
    assert rec["warm"] == {"copied": len(valid), "excluded": counts, "incumbent_config": inc["config"], "queue": [],
                           "same_as": {}}
    round2 = [t for t in run.trials() if t["round"] == 2]
    assert len(round2) == 4 and [t["trial"] for t in round2] == [5, 6, 7, 8]
    assert round2[0]["queued"] == "incumbent"
    assert store.non_default(round2[0]["config"], defaults) == inc["config"]
    assert all("noop" in t["config"] for t in round2)

    study = load_study(run, 2)
    warm = [t for t in study.trials if t.user_attrs.get("warm")]
    assert sorted(t.user_attrs["boar_trial"] for t in warm) == sorted(t["trial"] for t in valid)
    assert all(t.params["noop"] is False for t in warm)  # new lever filled in at default
    assert all(set(t.params) == {"fast_a", "fast_b", "noop"} for t in study.trials)
    summary = run.summary_path(2).read_text()
    assert f"Copied {len(valid)} earlier trials" in summary
    assert f"rule 3 (a lever of a hypothesis marked fix after the trial ran) {len(moved_b)}" in summary
    assert "drift: " in summary and "Re-measured this round (incumbent)" in summary


def test_the_incumbent_is_measured_again_only_on_a_new_commit(run):
    rounds.start_round(run, foreground=True)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    rounds.start_round(run, foreground=True)  # R1 edited nothing: the same commit
    round1, round2 = ([t for t in run.trials() if t["round"] == r] for r in (1, 2))
    assert round2[0]["commit"] == round1[0]["commit"]
    assert len(round2) == 4 and not any(t["queued"] for t in round2), "the slot goes to a sampled trial"
    held = max(stats.incumbent(round1, store.lever_defaults(run.hypotheses()), "min")["trials"])
    rec = run.state()["rounds"]["2"]
    assert rec["warm"]["same_as"] == {"incumbent": held} and rec["drift"] == "n/a"
    summary = run.summary_path(2).read_text()
    assert f"incumbent: same as trial {held}" in summary and "(4 of 4)" in summary

    close_round(run, 2, {"H1": "keep", "H2": "keep"})
    (run.root / "target.py").write_text("X = 2  # R1 changed the code\n")
    rounds.start_round(run, foreground=True)
    round3 = [t for t in run.trials() if t["round"] == 3]
    assert round3[0]["queued"] == "incumbent" and round3[0]["commit"] != round2[0]["commit"]
    assert run.state()["rounds"]["3"]["warm"]["same_as"] == {}
    assert "Re-measured this round (incumbent)" in run.summary_path(3).read_text()


def test_a_config_that_failed_on_the_rounds_commit_runs_again(run):
    write_eval(run, crash="not cfg['fast_a'] and cfg['fast_b'] == 1")  # the baseline fails
    accept_eval(run)
    rounds.start_round(run, foreground=True)
    base = run.trials()[0]
    assert base["queued"] == "baseline" and base["state"] == "failed"
    close_round(run, 1, {"H1": "investigate", "H2": "keep"})
    hyps = run.hypotheses()
    hyps[0]["decisions"]["1"].update(cause="c", measure=[], queue=[{"fast_a": False, "fast_b": 1}])
    run.save_hypotheses(hyps)
    rounds.start_round(run, foreground=True)  # the same commit
    reran = [t for t in run.trials() if t["round"] == 2 and t["queued"] and t["config"] == base["config"]]
    assert len(reran) == 1 and reran[0]["state"] == "failed" and reran[0]["commit"] == base["commit"]
    assert base["trial"] not in run.state()["rounds"]["2"]["warm"]["same_as"].values()


def _interrupt_after(monkeypatch, n_ok: int):
    """Make the (n_ok+1)-th eval trial raise Interrupted; returns a function that restores run_trial."""
    real = evalrun.run_trial
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > n_ok:
            raise evalrun.Interrupted()
        return real(*args, **kwargs)

    monkeypatch.setattr(evalrun, "run_trial", flaky)
    return lambda: monkeypatch.setattr(evalrun, "run_trial", real)


def test_interrupted_round_resumes_to_n_trials(run, monkeypatch):
    restore = _interrupt_after(monkeypatch, 2)
    with pytest.raises(evalrun.Interrupted):
        rounds.start_round(run, foreground=True)
    assert len(run.trials()) == 2 and run.state()["rounds"]["1"]["run_status"] == "running"
    restore()
    # A hard kill would leave the in-flight optuna trial RUNNING.
    study = load_study(run, 1)
    study.ask()
    rounds.start_round(run, foreground=True)
    trials = run.trials()
    assert [t["trial"] for t in trials] == [1, 2, 3, 4]
    assert sum(t["queued"] == "baseline" for t in trials) == 1
    study = load_study(run, 1)
    assert not study.get_trials(states=(TrialState.RUNNING,))
    assert run.state()["rounds"]["1"]["run_status"] == "done"


def test_interrupted_queued_trial_is_queued_again(run, monkeypatch):
    restore = _interrupt_after(monkeypatch, 0)
    with pytest.raises(evalrun.Interrupted):
        rounds.start_round(run, foreground=True)
    assert run.trials() == []
    restore()
    rounds.start_round(run, foreground=True)
    trials = run.trials()
    assert trials[0]["queued"] == "baseline" and trials[0]["config"]["fast_b"] == 1
    assert len(trials) == 4


def test_a_round_started_by_older_code_resumes_with_every_queued_config(run, monkeypatch):
    rounds.start_round(run, foreground=True)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    restore = _interrupt_after(monkeypatch, 1)
    with pytest.raises(evalrun.Interrupted):
        rounds.start_round(run, foreground=True)
    restore()
    run.update_state(lambda s: s["rounds"]["2"]["warm"].pop("same_as"))  # older code recorded no skips
    rounds.start_round(run, foreground=True)
    round2 = [t for t in run.trials() if t["round"] == 2]
    assert len(round2) == 4 and [t["queued"] for t in round2].count("incumbent") == 1
    assert run.state()["rounds"]["2"]["run_status"] == "done"


def test_resume_refuses_when_the_code_moved(run, monkeypatch):
    restore = _interrupt_after(monkeypatch, 1)
    with pytest.raises(evalrun.Interrupted):
        rounds.start_round(run, foreground=True)
    restore()
    commit = run.state()["rounds"]["1"]["commit"]

    (run.root / "target.py").write_text("X = 3\n")
    with pytest.raises(Refused, match="working tree has changed"):
        rounds.start_round(run, foreground=True)
    git(run.root, "commit", "-qam", "sneaky change")
    with pytest.raises(Refused, match="HEAD is now"):
        rounds.start_round(run, foreground=True)
    git(run.root, "reset", "-q", "--hard", commit)
    (run.root / "stray.pyc").write_bytes(b"x")  # untracked bytecode doesn't count as a code change
    rounds.start_round(run, foreground=True)
    assert len(run.trials()) == 4


def test_round_refuses_when_the_eval_changed(run):
    (run.eval_dir / "notes.txt").write_text("tweak\n")
    with pytest.raises(Refused, match="frozen"):
        rounds.start_round(run, foreground=True)
    assert run.state()["rounds"]["1"]["commit"] is None


def test_finalize_alternates_holdout_runs_and_writes_the_report(run, monkeypatch):
    rounds.start_round(run, foreground=True)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    write_eval(run, extra="os.environ['BOAR_SPLIT'] == 'holdout' and "
                          "open(os.path.join(run_dir, 'holdout-repeats'), 'a').write(os.environ['BOAR_REPEAT'] + ' ')")
    accept_eval(run)
    restore = _interrupt_after(monkeypatch, 3)
    with pytest.raises(evalrun.Interrupted):
        finalize.start_finalize(run, foreground=True)
    assert [h["i"] for h in store.read_jsonl(run.holdout_path)] == [0, 1, 2]
    restore()
    text = finalize.start_finalize(run, foreground=True)
    assert "write the sections" in text

    fin = run.state()["finalize"]
    defaults = store.lever_defaults(run.hypotheses())
    expected = stats.incumbent(warmstart.select(run.trials(), run.hypotheses(), 99)[0], defaults, "min")["config"]
    assert fin["status"] == "done" and fin["incumbent_config"] == expected
    assert fin["commit"] == git(run.root, "rev-parse", "HEAD")
    holdout = store.read_jsonl(run.holdout_path)
    assert [h["i"] for h in holdout] == [0, 1, 2, 3]
    assert [h["label"] for h in holdout] == ["baseline", "incumbent", "baseline", "incumbent"]
    assert run.path("holdout-repeats").read_text().split() == ["1", "1", "2", "2"]  # one BOAR_REPEAT per pair
    assert all(h["ok"] and h["guards_ok"] for h in holdout)
    assert holdout[1]["config"] == store.full_config(run.hypotheses(), expected)
    assert (run.path("holdout", "3") / "repeat-1.stdout").exists()

    report = run.report_path.read_text()
    assert report.startswith("# BOAR report: Make the toy cheaper.")
    for heading in ("## 1. Result", "## 2. What worked", "## 3. What didn't", "## 4. Rejected by review",
                    "## 5. Caveats", "## 6. Experimental log"):
        assert heading in report
    assert report.splitlines().count(store.TODO_MARKER) == 3
    assert report.count(store.TODO_MARKER) == 3
    assert f"boar/{RUN_ID}" in report and "trials.jsonl" in report
    assert "| incumbent | holdout |" in report and "| baseline | dev |" in report

    with pytest.raises(Refused, match="already finalized"):
        finalize.start_finalize(run, foreground=True)


def test_report_lists_removals_and_rejections(run):
    rounds.start_round(run, foreground=True)
    hyps = run.hypotheses()
    h2 = store.hypothesis(hyps, "H2")
    h2.update(status="removed", removed_round=1, removed_reason="removal rm-H2-r1 accepted: no effect")
    run.save_hypotheses(hyps)
    verdict = {"decision": "accept", "reason": "spread of values, within noise", "at": store.now_iso()}
    run.update_state(lambda s: s.update(phase="finalize", removals=[
        {"id": "rm-H2-r1", "hypothesis": "H2", "round": 1, "reason": "no effect", "trials": [2, 3], "verdict": verdict}]))
    run.append_review({"item": "H3", "kind": "proposal", "decision": "reject", "reason": "a known cheat", "round": 0,
                       "at": store.now_iso()})
    finalize.start_finalize(run, foreground=True)
    report = run.report_path.read_text()
    assert "| H2 | H2 moves the metric | 1 | removal rm-H2-r1 accepted: no effect | 2, 3 | accept: spread of values" in report
    assert "| H3: H3 moves the metric | proposal | 0 | a known cheat |" in report


def _wait_for(pred, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_detached_worker_runs_the_round_and_wait_reports_it(run):
    text = rounds.start_round(run, foreground=False)
    assert "running in the background" in text
    rec = store.read_json(run.running_path)
    assert rec["command"] == "round run" and rec["round"] == 1 and isinstance(rec["pid"], int)
    assert rec["log"].endswith("rounds/1/worker.log")
    out = worker.wait(run, 60)
    assert out.startswith("finished: round 1"), out + "\n" + Path(rec["log"]).read_text()
    assert not run.running_path.exists()
    assert len(run.trials()) == 4 and run.state()["rounds"]["1"]["run_status"] == "done"
    assert "worker started" in Path(rec["log"]).read_text()


def test_wait_times_out_while_the_worker_runs_then_sigterm_stops_it(run):
    write_eval(run, hang="True")
    accept_eval(run)
    rounds.start_round(run, foreground=False)
    pid = store.read_json(run.running_path)["pid"]
    assert _wait_for(lambda: run.path("eval-child.pid").exists()), "the eval never started"
    assert worker.wait(run, 0.2).startswith("still running: round 1, 0/4 trials done")
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    assert _wait_for(lambda: not store.pid_alive(pid), 15), "the worker ignored SIGTERM"
    assert _wait_for(lambda: not store.pid_alive(int(run.path("eval-child.pid").read_text())), 5)
    assert not run.running_path.exists()
    assert run.trials() == []
    assert run.state()["rounds"]["1"]["run_status"] == "running"
    log = run.round_dir(1) / "worker.log"
    assert "stopped by SIGTERM" in log.read_text()
    assert "stopped before finishing" in worker.wait(run, 1)
    study = load_study(run, 1)
    assert [t.state for t in study.trials] == [TrialState.FAIL]


def test_wait_points_a_died_inline_run_at_its_shell_not_at_a_worker_log(run):
    """A `--foreground` run prints to its own shell; a worker.log from an earlier detached attempt is stale."""
    run.round_dir(1).mkdir(parents=True, exist_ok=True)
    (run.round_dir(1) / "worker.log").write_text("stale line from an earlier worker\n")
    with worker.claim(run, worker.ROUND, 1):
        rec = store.read_json(run.running_path)
    assert rec["log"] is None
    dead = subprocess.Popen(["true"])
    dead.wait()
    store.write_json(run.running_path, {**rec, "pid": dead.pid, "identity": None})
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="running"))
    out = worker.wait(run, 1)
    assert "stopped before finishing" in out
    assert "stale line" not in out and "Last log lines" not in out
    assert f"It ran inline (--foreground, pid {dead.pid})" in out and "resume with `boar round run`" in out


def test_wait_after_an_inline_run_exited_does_not_point_at_a_missing_worker_log(run):
    """SIGTERM stops a `--foreground` round cleanly: the slot is gone and no worker.log was ever written."""
    with worker.claim(run, worker.ROUND, 1):
        pass
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="running"))
    assert not (run.round_dir(1) / "worker.log").exists() and not run.running_path.exists()
    out = worker.wait(run, 1)
    assert "stopped before finishing" in out and "Last log lines" not in out and "worker.log" not in out
    assert "It ran inline (--foreground), so its output went to the shell that started it" in out


# --- regressions ----------------------------------------------------------------------------


def test_inline_round_holds_the_worker_slot(run, monkeypatch):
    real, seen = evalrun.run_trial, []

    def watching(*args, **kwargs):
        seen.append(run.running())
        with pytest.raises(Refused, match="still running"):
            store.require_live_run()
        return real(*args, **kwargs)

    monkeypatch.setattr(evalrun, "run_trial", watching)
    rounds.start_round(run, foreground=True)
    assert len(seen) == 4 and all(rec and rec["pid"] == os.getpid() and rec["command"] == "round run" for rec in seen)
    assert not run.running_path.exists()


def test_round_refuses_while_another_process_holds_the_slot(run):
    with worker.claim(run, worker.ROUND, 1):
        with pytest.raises(Refused, match="still running in the background"):
            rounds.start_round(run, foreground=True)
        with pytest.raises(Refused, match="still running in the background"):
            finalize.start_finalize(run, foreground=True)
    assert run.state()["rounds"]["1"]["commit"] is None and run.trials() == []


def _after_recorded(monkeypatch, k: int, act):
    """Run `act()` right after the k-th trial is recorded, between two trials; returns an undo of the hook."""
    real = Run.append_trial
    calls = {"n": 0}

    def hooked(self, record):
        real(self, record)
        calls["n"] += 1
        if calls["n"] == k:
            act()

    monkeypatch.setattr(Run, "append_trial", hooked)
    return lambda: monkeypatch.setattr(Run, "append_trial", real)


def test_a_code_change_between_trials_stops_the_round(run, monkeypatch):
    restore = _after_recorded(monkeypatch, 2, lambda: (run.root / "target.py").write_text("X = 1000\n"))
    with pytest.raises(Refused, match=r"round 1 started at commit \w+ but the working tree has changed \(target.py\); "
                                      r"it changed between trials"):
        rounds.start_round(run, foreground=True)
    assert [t["trial"] for t in run.trials()] == [1, 2]
    restore()
    git(run.root, "checkout", "--", "target.py")
    rounds.start_round(run, foreground=True)
    commit = run.state()["rounds"]["1"]["commit"]
    assert len(run.trials()) == 4 and all(t["commit"] == commit for t in run.trials())


def test_an_untracked_file_left_between_trials_stops_the_round(run, monkeypatch):
    _after_recorded(monkeypatch, 1, lambda: (run.root / "notes.txt").write_text("x\n"))
    with pytest.raises(Refused, match=r"working tree has changed \(notes.txt\)"):
        rounds.start_round(run, foreground=True)


def test_a_trial_that_rewrites_a_committed_file_is_recorded_failed_and_put_back(run):
    """The eval's output was swept into the round commit; each trial rewrites it (it used to wedge the round)."""
    (run.root / "results.txt").write_text("from eval check\n")
    write_eval(run, extra="open('results.txt', 'a').write('trial %d\\n' % n)")
    accept_eval(run)
    rounds.start_round(run, foreground=True)
    trials = run.trials()
    assert len(trials) == 4 and {t["state"] for t in trials} == {"failed"}
    assert all(t["error"] == "the trial wrote into the repo: results.txt" for t in trials)
    assert (run.root / "results.txt").read_text() == "from eval check\n", "put back from the round commit"
    assert "trial" in (run.trial_dir(1) / "repo-writes" / "results.txt").read_text(), "what it wrote is kept"
    assert gitops.status(run.root) == {}
    assert [t.state for t in load_study(run, 1).trials] == [TrialState.FAIL] * 4
    assert run.state()["rounds"]["1"]["run_status"] == "done"
    assert "the trial wrote into the repo: results.txt" in run.summary_path(1).read_text()


def test_new_files_a_lever_writes_are_moved_out_so_later_rounds_and_finalize_run(run):
    write_eval(run, extra="cfg['fast_a'] and (os.makedirs('logs', exist_ok=True), open('logs/%d.prof' % n, 'w').write('p'))")
    accept_eval(run)
    rounds.start_round(run, foreground=True)
    wrote = [t for t in run.trials() if t["config"]["fast_a"]]
    assert wrote and all(t["state"] == "failed" and "wrote into the repo: logs/" in t["error"] for t in wrote)
    assert all(t["state"] == "complete" for t in run.trials() if not t["config"]["fast_a"])
    assert not (run.root / "logs").exists() and gitops.status(run.root) == {}
    assert list((run.trial_dir(wrote[0]["trial"]) / "repo-writes" / "logs").glob("*.prof"))
    commit = run.state()["rounds"]["1"]["commit"]
    assert "logs/" not in git(run.root, "show", "--stat", "--format=", commit)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    finalize.start_finalize(run, foreground=True)
    assert run.state()["finalize"]["status"] == "done"


def test_a_trial_that_writes_into_eval_is_recorded_failed_and_eval_is_restored(run):
    write_eval(run, extra="cfg['fast_a'] and open(os.path.join(run_dir, 'eval', 'cache.json'), 'w').write('{}')")
    (run.eval_dir / "data.txt").write_text("frozen\n")
    accept_eval(run)
    store.write_json(run.config_path, {**run.config(), "trials_per_round": 6})
    rounds.start_round(run, foreground=True)
    trials = run.trials()
    wrote = [t for t in trials if t["config"]["fast_a"]]
    assert len(trials) == 6 and wrote, "the round finishes instead of stopping at the first such trial"
    assert all(t["state"] == "failed" and t["error"] == "the trial wrote into eval/: added: cache.json" for t in wrote)
    assert evalrun.eval_hash(run) == run.state()["eval"]["accepted_hash"]
    assert (run.trial_dir(wrote[0]["trial"]) / "eval-writes" / "cache.json").read_text() == "{}"


def test_eval_writes_that_cannot_be_put_back_stop_the_round_after_recording(run, monkeypatch):
    shutil.rmtree(run.eval_accepted_dir)
    real = evalrun.run_trial

    def writes(r, *args, **kwargs):
        res = real(r, *args, **kwargs)
        (r.eval_dir / "cache.json").write_text("{}")
        return res

    monkeypatch.setattr(evalrun, "run_trial", writes)
    with pytest.raises(Refused, match=r"trial 1 was recorded as failed, but eval/ could not be put back .*boar eval restore"):
        rounds.start_round(run, foreground=True)
    assert [t["state"] for t in run.trials()] == ["failed"]
    with pytest.raises(Refused, match=r"added: cache.json.*boar eval restore"):
        rounds.start_round(run, foreground=True)


def test_a_moved_head_during_a_trial_is_refused_without_recording(run, monkeypatch):
    real = evalrun.run_trial

    def commits(r, *args, **kwargs):
        res = real(r, *args, **kwargs)
        (r.root / "target.py").write_text("X = 3\n")
        git(r.root, "commit", "-qam", "sneaky")
        return res

    monkeypatch.setattr(evalrun, "run_trial", commits)
    with pytest.raises(Refused, match=r"HEAD is now \w+; the trial that just ran can't be tied to one commit"):
        rounds.start_round(run, foreground=True)
    assert run.trials() == [] and [t.state for t in load_study(run, 1).trials] == [TrialState.FAIL]


def test_frozen_eval_refusal_names_the_diff(run):
    (run.eval_dir / "run").write_text((run.eval_dir / "run").read_text() + "# tweak\n")
    with pytest.raises(Refused, match=r"\(changed: run\).*frozen.*`boar eval restore`"):
        rounds.check_eval_frozen(run)


def test_finalize_measures_the_last_rounds_commit(run):
    rounds.start_round(run, foreground=True)
    commit = run.state()["rounds"]["1"]["commit"]
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    (run.root / "target.py").write_text("X = 50  # edited after the last round\n")
    with pytest.raises(Refused, match=r"the last round \(round 1\) ran at commit .* finalize must measure the code"):
        finalize.start_finalize(run, foreground=True)
    assert run.state()["finalize"]["commit"] is None and not run.holdout_path.exists()
    git(run.root, "checkout", "--", "target.py")
    finalize.start_finalize(run, foreground=True)
    assert run.state()["finalize"]["commit"] == commit == git(run.root, "rev-parse", "HEAD")
    assert "boar: finalize" not in git(run.root, "log", "--format=%s")


def test_finalize_stops_when_the_code_changes_between_holdout_runs(run, monkeypatch):
    rounds.start_round(run, foreground=True)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    real = store.append_jsonl

    def hooked(path, obj):
        real(path, obj)
        if path == run.holdout_path:
            (run.root / "target.py").write_text("X = 9\n")

    monkeypatch.setattr(store, "append_jsonl", hooked)
    with pytest.raises(Refused, match="finalize started at commit .* changed between trials"):
        finalize.start_finalize(run, foreground=True)
    assert [h["i"] for h in store.read_jsonl(run.holdout_path)] == [0]


def test_a_holdout_run_that_writes_into_the_repo_keeps_its_measurement_and_is_put_back(run):
    """eval check runs dev only, so a holdout-only write shows up first at finalize, which commits nothing after."""
    rounds.start_round(run, foreground=True)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    write_eval(run, extra="os.environ['BOAR_SPLIT'] == 'holdout' and open('holdout.log', 'w').write('x')")
    accept_eval(run)
    finalize.start_finalize(run, foreground=True)
    runs = store.read_jsonl(run.holdout_path)
    assert len(runs) == 4 and all(h["ok"] and h["guards_ok"] and h["error"] is None for h in runs)
    assert all(h["wrote"] == "the trial wrote into the repo: holdout.log" for h in runs)
    assert gitops.status(run.root) == {} and (run.path("holdout", "0", "repo-writes", "holdout.log")).exists()
    report = run.report_path.read_text()
    row = next(line for line in report.splitlines() if line.startswith("| incumbent | holdout |"))
    assert "| 2 |" in row and "better" in row, row
    assert "Holdout runs that wrote into the repo or eval/" in report
    assert "- run 0 (baseline): the trial wrote into the repo: holdout.log" in report
    assert "Warning" not in report


def test_nested_repo_with_uncommitted_edits_is_refused_before_the_round_starts(run):
    lib = run.root / "vendor" / "lib"
    lib.mkdir(parents=True)
    git(lib, "init", "-q", "-b", "main")
    git(lib, "config", "user.name", "t")
    git(lib, "config", "user.email", "t@example.invalid")
    (lib / "lib.py").write_text("FAST = False\n")
    git(lib, "add", "-A")
    git(lib, "commit", "-qm", "lib")
    (lib / "lib.py").write_text("FAST = True  # the lever\n")
    with pytest.raises(Refused, match=r"could not be committed \(vendor/lib\); a nested git repo"):
        rounds.start_round(run, foreground=True)
    assert run.state()["rounds"]["1"]["commit"] is None and run.state()["rounds"]["1"]["run_status"] == "not_started"
    git(lib, "commit", "-qam", "lever")
    rounds.start_round(run, foreground=True)
    assert len(run.trials()) == 4


def test_resumed_round_does_not_replay_the_random_draws(run, monkeypatch):
    store.write_json(run.config_path, {**run.config(), "trials_per_round": 6})
    restore = _interrupt_after(monkeypatch, 3)
    with pytest.raises(evalrun.Interrupted):
        rounds.start_round(run, foreground=True)
    restore()
    rounds.start_round(run, foreground=True)
    sampled = [json.dumps(t["config"], sort_keys=True) for t in run.trials() if not t["queued"]]
    assert len(sampled) == 5 and len(set(sampled)) == 5, sampled


@pytest.mark.parametrize("seed", [-5, 2**32 - 1])
def test_any_integer_seed_runs(run, seed):
    store.write_json(run.config_path, {**run.config(), "seed": seed, "trials_per_round": 2})
    rounds.start_round(run, foreground=True)
    assert len(run.trials()) == 2


def _hand_trial(n: int, r: int, config: dict, repeats: list[float], queued=None, state="complete") -> dict:
    return {"trial": n, "round": r, "commit": "c", "config": {"fast_a": False, "fast_b": 1, "cache_output": False, **config},
            "state": state, "metric": stats.median(repeats), "repeats": repeats, "metrics": {}, "duration_s": 1.0,
            "error": None, "queued": queued}


def test_drift_is_judged_against_the_floor_from_earlier_rounds(run):
    for t in (_hand_trial(1, 1, {}, [100.0, 100.2], "baseline"), _hand_trial(2, 1, {"fast_a": True}, [130.0, 130.0]),
              _hand_trial(3, 2, {}, [110.0, 110.2], "incumbent")):
        run.append_trial(t)
    run.update_state(lambda s: s["rounds"].update({"2": fresh_round()}))
    facts = rounds._facts(run, 2)
    assert facts["drift"]["flag"] == "yes", facts["drift"]
    assert facts["noise_floor"] == pytest.approx(0.2)


def test_an_incumbent_that_breaks_a_guard_on_re_measurement_is_flagged_and_dethroned(run):
    rounds.start_round(run, foreground=True)
    defaults = store.lever_defaults(run.hypotheses())
    inc = stats.incumbent(run.trials(), defaults, "min")["config"]
    assert inc.get("fast_a") is True, inc
    assert run.state()["rounds"]["1"]["drift"] == "n/a"
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    write_eval(run, guard="cfg['fast_a']")
    (run.root / "target.py").write_text("X = 2  # a new commit, so the incumbent is measured again\n")
    accept_eval(run)
    rounds.start_round(run, foreground=True)
    rec = run.state()["rounds"]["2"]
    queued = next(t for t in run.trials() if t["round"] == 2 and t["queued"])
    assert queued["state"] == "infeasible" and rec["drift"] == "yes"
    summary = run.summary_path(2).read_text()
    assert "**drift: yes** (the re-measurement was infeasible)" in summary
    assert store.non_default(queued["config"], defaults) == inc
    after = stats.incumbent(warmstart.select(run.trials(), run.hypotheses(), 3)[0], defaults, "min")
    assert after is not None and after["config"] != inc, "a config that broke a guard is no longer the incumbent"
    assert f"After the round: {rounds.config_text(inc)}" not in summary


def test_an_incumbent_that_fails_on_re_measurement_is_dethroned(run):
    rounds.start_round(run, foreground=True)
    defaults = store.lever_defaults(run.hypotheses())
    inc = stats.incumbent(run.trials(), defaults, "min")["config"]
    assert inc.get("fast_a") is True, inc
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    write_eval(run, crash="cfg['fast_a']")
    (run.root / "target.py").write_text("X = 2  # a new commit, so the incumbent is measured again\n")
    accept_eval(run)
    rounds.start_round(run, foreground=True)
    queued = next(t for t in run.trials() if t["round"] == 2 and t["queued"])
    assert queued["state"] == "failed" and store.non_default(queued["config"], defaults) == inc
    assert f"After the round: {rounds.config_text(inc)}" not in run.summary_path(2).read_text()
    assert finalize.final_incumbent(run) != inc


def test_report_warns_when_the_recommended_config_fails_on_holdout(run):
    rounds.start_round(run, foreground=True)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    write_eval(run, crash="cfg['fast_a'] and os.environ['BOAR_SPLIT'] == 'holdout'")
    accept_eval(run)
    finalize.start_finalize(run, foreground=True)
    assert run.state()["finalize"]["incumbent_config"].get("fast_a") is True
    report = run.report_path.read_text()
    assert "**Warning: the recommended config failed in 2/2 holdout runs.** Check the failed runs above" in report
    row = next(line for line in report.splitlines() if line.startswith("| incumbent | holdout |"))
    assert "failed in 2/2 holdout runs" in row


def test_report_flags_holdout_guard_failures_of_the_recommended_config(run):
    rounds.start_round(run, foreground=True)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    write_eval(run, guard="cfg['fast_a'] and os.environ['BOAR_SPLIT'] == 'holdout'")
    accept_eval(run)
    finalize.start_finalize(run, foreground=True)
    assert run.state()["finalize"]["incumbent_config"].get("fast_a") is True
    report = run.report_path.read_text()
    assert "**Warning: the recommended config broke a guard in 2/2 holdout runs.**" in report
    assert report.index("Holdout runs that failed or broke a guard:") < report.index("check the failed runs above")
    row = next(line for line in report.splitlines() if line.startswith("| incumbent | holdout |"))
    assert "broke a guard in 2/2 holdout runs" in row and "better" not in row


def test_report_log_shows_the_pooled_incumbent_metric(run):
    rounds.start_round(run, foreground=True)
    rec = run.state()["rounds"]["1"]
    inc = stats.incumbent(run.trials(), store.lever_defaults(run.hypotheses()), "min")
    assert rec["incumbent_metric"] == inc["metric"] and rec["incumbent_n"] == len(inc["repeats"])
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    finalize.start_finalize(run, foreground=True)
    assert f"(pooled {rounds.fmt(inc['metric'])}, n={len(inc['repeats'])})" in run.report_path.read_text()


def test_report_does_not_call_the_baseline_against_itself_better_or_worse(run):
    write_eval(run, guard="cfg['fast_a'] or cfg['fast_b'] > 1")
    accept_eval(run)
    rounds.start_round(run, foreground=True)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    finalize.start_finalize(run, foreground=True)
    assert run.state()["finalize"]["incumbent_config"] == {}
    rows = [line for line in run.report_path.read_text().splitlines() if line.startswith("| incumbent |")]
    assert len(rows) == 2 and all("(incumbent is the baseline)" in r for r in rows)
    assert not any("worse" in r or "better" in r for r in rows)


def test_report_says_nothing_was_compared_when_no_trial_passed(run):
    write_eval(run, crash="True")
    accept_eval(run)
    rounds.start_round(run, foreground=True)
    close_round(run, 1, {"H1": "keep", "H2": "keep"})
    run.update_state(lambda s: s.update(phase="finalize"))
    finalize.start_finalize(run, foreground=True)
    report = run.report_path.read_text()
    assert "No config's latest dev trial completed feasibly" in report and "No tested config beat it" not in report


def test_report_names_the_better_trials_an_accepted_removal_left_out(run):
    """Rule 2 drops every trial that set the removed lever; the report must not say nothing beat the baseline."""
    hyps = run.hypotheses()
    for h in hyps:
        if h["id"] == "H1":
            h["decisions"]["1"] = {"decision": "keep", "reason": "helps", "trials": [], "at": store.now_iso()}
        if h["id"] == "H2":
            h.update(status="removed", removed_round=1, removed_reason="removal rm-H2-r1 accepted: flat")
            h["decisions"]["1"] = {"decision": "remove", "reason": "flat", "trials": [2, 3], "at": store.now_iso()}
    hyps.append(hyp("H4", "active", NOOP))
    run.save_hypotheses(hyps)
    verdict = {"decision": "accept", "reason": "flat across the range", "at": store.now_iso()}
    run.update_state(lambda s: (
        s.update(phase="finalize", removals=[{"id": "rm-H2-r1", "hypothesis": "H2", "round": 1, "reason": "flat",
                                               "trials": [2, 3], "verdict": verdict}]),
        s["rounds"]["1"].update(run_status="done", closed=True),
        s["finalize"].update(status="done", incumbent_config={}),
    ))
    points = [({}, 100.0), ({"fast_a": True, "fast_b": 4, "noop": True}, 70.0),
              ({"fast_a": True, "fast_b": 6, "noop": True}, 65.0), ({"fast_a": True}, 120.0)]
    for n, (cfg, metric) in enumerate(points, 1):
        run.append_trial({"trial": n, "round": 1, "commit": "abc", "state": "complete", "metric": metric,
                          "config": {"fast_a": False, "fast_b": 1, "cache_output": False, "noop": False, **cfg},
                          "repeats": [metric, metric], "metrics": {}, "duration_s": 1.0, "error": None, "queued": None})
    report = finalize.write_report(run).read_text()
    assert "No tested config beat it on dev" not in report
    assert ("Recommended config: the baseline. No config in the final search space beat it on dev. Trials 2–3 did, "
            "but they set levers of hypotheses removed, superseded or marked fix (`fast_b` of H2) away from default"
            ) in report, report
    assert "`fast_a`+`noop`" not in report, "trials 2 and 3 set them together"
    assert "H1 keep, H2 remove (accepted)" in report


def test_change_within_the_noise_floor_is_no_effect():
    assert finalize._change(100.1, 100.0, "min", 0.5) == "+0.1% (within noise)"
    assert finalize._change(90.0, 100.0, "min", 0.5) == "-10.0% (better)"
    assert finalize._change(90.0, 100.0, "min", None) == "-10.0% (better)"


def test_drift_is_not_flagged_when_nothing_was_measured_before():
    defaults = {"fast_a": False, "fast_b": 1}
    crashed = {"trial": 1, "round": 1, "config": defaults, "state": "failed", "error": "exit 3", "metric": None, "repeats": []}
    d = rounds._drift(crashed, [], defaults, None)
    assert d["flag"] == "n/a" and d["why"] == "no earlier repeats of this config; this trial was failed (exit 3)"
    earlier = [{"trial": 0, "round": 0, "config": defaults, "state": "complete", "metric": 100.0, "repeats": [100.0]}]
    assert rounds._drift(crashed, earlier, defaults, None)["flag"] == "yes"
