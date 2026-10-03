from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
import types

import pytest

import boar as boar_pkg
from boar import cli, control, store
from boar import evalrun as real_evalrun

SPEC = "# Spec\n\n## Metric\nwall time in s\nDirection: min\n"


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k, v in {
        "HOME": str(tmp_path),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "XDG_STATE_HOME": str(tmp_path / "state"),  # the session registry
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)  # the hook falls back to it
    (tmp_path / "spec.md").write_text(SPEC)
    return tmp_path


@pytest.fixture
def repo(env, monkeypatch):
    root = env / "target"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "app.py").write_text("print('hi')\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def boar(capsys):
    def call(*args):
        code = cli.main([str(a) for a in args])
        out, err = capsys.readouterr()
        return code, out, err

    return call


def fake_module(monkeypatch, name, **funcs):
    mod = types.ModuleType(f"boar.{name}")
    for k, v in funcs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, f"boar.{name}", mod)
    monkeypatch.setattr(boar_pkg, name, mod, raising=False)
    return mod


@pytest.fixture
def evalfake(monkeypatch):
    ctl = {"ok": True, "hash": "h1"}

    def eval_check(run):
        res = {"ok": ctl["ok"], "hash": ctl["hash"], "problems": [] if ctl["ok"] else ["last line is not JSON"],
               "trial": {"state": "complete", "metric": 1.0, "repeats": [1.0], "duration_s": 1.0}, "limit_s": 1200,
               "at": store.now_iso()}
        store.write_json(run.eval_check_path, res)
        return res

    fake_module(
        monkeypatch, "evalrun", eval_check=eval_check, eval_hash=lambda run: ctl.get("current", ctl["hash"]),
        eval_manifest=lambda run: {"run": ctl.get("current", ctl["hash"])}, eval_diff=real_evalrun.eval_diff,
        eval_unchanged=lambda run, manifest: real_evalrun.eval_diff(manifest, {"run": ctl.get("current", ctl["hash"])}),
        diff_text=real_evalrun.diff_text,
    )
    return ctl


def props(env, *names, supersedes=None, file="props.json"):
    data = [
        {"statement": f"{n} helps", "mechanism": "m", "source": "research", "citations": [],
         "levers": [{"name": n, "type": "int", "low": 1, "high": 8, "default": 1}], "supersedes": supersedes}
        for n in names
    ]
    path = env / file
    path.write_text(json.dumps(data))
    return path


def init(boar, env, *extra):
    code, _, err = boar("init", "--spec", env / "spec.md", *extra)
    assert code == 0, err
    return store.active_run()


def to_round1(boar, env, *extra):
    run = init(boar, env, *extra)
    boar("propose", props(env, "a", "b"))
    boar("eval", "check", "--foreground")
    boar("review", "record", "H1", "accept", "--reason", "ok")
    boar("review", "record", "H2", "accept", "--reason", "ok")
    boar("review", "record", "eval", "accept", "--reason", "ok")
    assert run.state()["phase"] == "round"
    return run


def add_trials(run, r, n):
    start = len(run.trials()) + 1
    for t in range(start, start + n):
        run.append_trial({"trial": t, "round": r, "commit": "c", "config": {"a": 1, "b": 1}, "state": "complete",
                          "metric": 1.0, "repeats": [1.0], "metrics": {}, "duration_s": 1.0, "error": None, "queued": None})
    return list(range(start, start + n))


def live_record(**rec):
    """A running.json record naming this (live) test process, as worker.claim writes it."""
    return {"pid": os.getpid(), "identity": store.process_identity(os.getpid()), **rec}


REPORT = "# report\n" + "".join(f"{h}\n| table |\n" for h in control.REPORT_SECTIONS)


def first(boar):
    return boar("next")[1].splitlines()[0]


def hook(boar, monkeypatch, stdin=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin if isinstance(stdin, str) else json.dumps(stdin)))
    return boar("next", "--hook")


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / ".boar").rglob("*"))
        if p.is_file() and p.name != ".lock"
    }


# --- tag order ----------------------------------------------------------------------------------


def test_none_without_a_run(boar, repo, monkeypatch):
    assert first(boar) == "[NONE] No active BOAR run. Start one with /boar <request>."
    assert hook(boar, monkeypatch, {"session_id": "s"}) == (0, "", "")


def test_setup_tags_in_order(boar, env, repo, evalfake):
    run = init(boar, env)
    out = boar("next")[1]
    assert out.startswith(f"[S2] Research the problem into {run.research_path}")
    assert "Every user idea" in out and "S4 can go in parallel" in out
    boar("propose", props(env, "a", "b"))
    assert first(boar).startswith(f"[S4] Build the eval in {run.eval_dir}/")
    evalfake["ok"] = False
    boar("eval", "check", "--foreground")
    out = boar("next")[1]
    assert out.startswith("[S4] Build the eval") and "last line is not JSON" in out
    evalfake["ok"] = True
    boar("eval", "check", "--foreground")
    assert first(boar) == "[S5] Send the boar-reviewer agent these items: H1, H2, eval."
    boar("review", "record", "H1", "accept", "--reason", "ok")
    boar("review", "record", "H2", "accept", "--reason", "ok")
    for attempt in (1, 2):
        boar("review", "record", "eval", "reject", "--reason", f"dev overlaps holdout {attempt}")
        assert first(boar) == (
            f"[S4] Eval rejected (failed attempt {attempt}/3): dev overlaps holdout {attempt}; "
            "rebuild it and re-run `boar eval check`."
        )
        boar("eval", "check", "--foreground")
        assert first(boar) == "[S5] Send the boar-reviewer agent these items: eval."
    boar("review", "record", "eval", "reject", "--reason", "metric is wrong")
    out = boar("next")[1]
    assert out.startswith("[ASK_USER] The eval was rejected 3 times") and "metric is wrong" in out
    boar("eval", "check", "--foreground")
    assert first(boar).startswith("[S5]")


def test_s3_after_a_failed_batch(boar, env, repo):
    init(boar, env)
    boar("propose", props(env, "a"))
    boar("review", "record", "H1", "reject", "--reason", "caches eval output")
    out = boar("next")[1]
    assert out.startswith("[S3] No hypothesis accepted (failed attempt 1/3)") and "H1: caches eval output" in out


def test_round_tags_in_order(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    out = boar("next")[1]
    assert out.startswith("[R1] Implement levers for: H1, H2; repair: none; then `boar round run`.")
    assert "new: H1 (a)" in out and "BOAR_CONFIG" in out and "default must reproduce baseline" in out

    run.update_state(lambda s: s["rounds"]["1"].update(run_status="running", commit="c"))
    assert first(boar) == "[R1] Round 1 was interrupted; resume with `boar round run`."
    # An inline (--foreground) round writes no worker.log: its output went to the shell that started it.
    out = boar("next")[1]
    assert "it ran inline (--foreground), so its output went to the shell that started it" in out
    assert "worker log" not in out
    run.round_dir(1).mkdir(parents=True, exist_ok=True)
    (run.round_dir(1) / "worker.log").write_text("x\n")
    assert f"worker log: {run.round_dir(1) / 'worker.log'}" in boar("next")[1]
    (run.round_dir(1) / "worker.log").unlink()

    add_trials(run, 1, 2)
    store.write_json(run.running_path, live_record(command="round run", round=1, log="w.log"))
    assert first(boar) == "[WAIT] boar round run is running (round 1, 2/6 trials done). Run `boar wait`."
    run.running_path.unlink()

    ids = add_trials(run, 1, 4)
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="done"))
    out = boar("next")[1]
    assert out.startswith("[R3] Decide on: H1, H2") and str(run.summary_path(1)) in out
    boar("decide", "H1", "fix", "--reason", "crashes at 8", "--trials", ids[0])
    assert first(boar).startswith("[R3] Decide on: H2")
    boar("decide", "H2", "keep", "--reason", "helps")
    out = boar("next")[1]
    assert out.startswith("[R4] Propose new hypotheses") and "last round" not in out
    boar("propose", props(env, "c"))
    assert first(boar) == "[R5] Send the boar-reviewer agent: H3."
    boar("review", "record", "H3", "accept", "--reason", "ok")
    out = boar("next")[1]
    assert out.startswith("[R6] Run `boar round close`.") and "activates H3" in out
    boar("round", "close")
    out = boar("next")[1]
    assert out.startswith("[R1] Implement levers for: H3; repair: H1; then `boar round run`.")
    assert "repair: H1: crashes at 8" in out


def test_r5_lists_pending_removals(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    add_trials(run, 1, 3)
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="done"))
    boar("decide", "H1", "remove", "--reason", "flat", "--trials", "1", "2", "3")
    boar("decide", "H2", "keep", "--reason", "x")
    boar("propose", "--none", "--reason", "nothing new")
    assert first(boar) == "[R5] Send the boar-reviewer agent: rm-H1-r1."
    boar("review", "record", "rm-H1-r1", "accept", "--reason", "flat across the range")
    out = boar("next")[1]
    assert out.startswith("[R6]") and "removes H1: removal rm-H1-r1 accepted: flat" in out


def test_finalize_report_and_done(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    run.update_state(lambda s: s.update(phase="finalize"))
    assert first(boar) == "[FINALIZE] Run `boar finalize`."
    run.update_state(lambda s: s["finalize"].update(status="running"))
    assert first(boar) == "[FINALIZE] Finalize was interrupted; resume with `boar finalize`."
    store.write_json(run.running_path, live_record(command="finalize"))
    store.append_jsonl(run.holdout_path, {"i": 0})
    assert first(boar) == "[WAIT] boar finalize is running (finalize, holdout 1/12 runs done). Run `boar wait`."
    run.running_path.unlink()
    run.update_state(lambda s: s["finalize"].update(status="done"))
    run.report_path.write_text(f"{REPORT}{store.TODO_MARKER}\n{store.TODO_MARKER}\n")
    out = boar("next")[1]
    assert out.startswith(f"[REPORT] Write the narrative sections marked {store.TODO_MARKER} in {run.report_path}")
    assert "2 marker(s) left" in out
    assert run.state()["phase"] == "finalize"
    run.report_path.write_text(f"{REPORT}Both fixes.\n")
    assert first(boar) == f"[DONE] Run {run.id} is finished. Report: {run.report_path}"
    assert run.state()["phase"] == "done"
    assert first(boar).startswith("[DONE]")


# --- the Stop hook ----------------------------------------------------------------------------------


def test_hook_blocks_while_work_remains(boar, env, repo, monkeypatch):
    run = init(boar, env)
    code, out, err = hook(boar, monkeypatch, {"session_id": "any", "hook_event_name": "Stop"})
    assert code == 2 and out == ""
    assert err.startswith(f"BOAR run {run.id} is not finished. [S2] Research")


def test_hook_without_stdin_still_works(boar, env, repo):
    init(boar, env)
    assert boar("next", "--hook")[0] == 2


@pytest.mark.parametrize(
    "recorded, incoming, blocks",
    [
        ("sess-A", {"session_id": "sess-A"}, True),
        ("sess-A", {"session_id": "sess-B"}, False),
        ("sess-A", {"session_id": ""}, True),
        ("sess-A", {"session_id": "${CLAUDE_SESSION_ID}"}, True),
        ("sess-A", {}, True),
        ("sess-A", "not json at all", True),
        ("sess-A", "[1, 2]", True),
        (None, {"session_id": "sess-B"}, True),
        ("$CLAUDE_SESSION_ID", {"session_id": "sess-B"}, True),
    ],
)
def test_hook_session_filtering(boar, env, repo, monkeypatch, recorded, incoming, blocks):
    extra = ("--session", recorded) if recorded else ()
    init(boar, env, *extra)
    code, out, err = hook(boar, monkeypatch, incoming)
    assert code == (2 if blocks else 0) and out == ""
    assert bool(err) == blocks


def test_hook_releases_on_ask_user_done_and_aborted(boar, env, repo, monkeypatch):
    run = init(boar, env)
    run.update_state(lambda s: s["eval"].update(status="rejected", rejections=3))
    assert hook(boar, monkeypatch, {}) == (0, "", "")
    run.update_state(lambda s: s.update(phase="done"))
    assert hook(boar, monkeypatch, {}) == (0, "", "")
    run.update_state(lambda s: s.update(phase="aborted"))
    assert hook(boar, monkeypatch, {}) == (0, "", "")


def test_hook_marks_a_complete_report_done(boar, env, repo, monkeypatch):
    run = init(boar, env)
    run.update_state(lambda s: (s.update(phase="finalize"), s["finalize"].update(status="done")))
    run.report_path.write_text(REPORT)
    assert hook(boar, monkeypatch, {}) == (0, "", "")
    assert run.state()["phase"] == "done"


def test_hook_never_crashes(boar, env, repo, monkeypatch):
    run = init(boar, env)
    run.state_path.write_text("{corrupt")
    assert hook(boar, monkeypatch, {}) == (0, "", "")
    monkeypatch.setattr(cli.nextstep, "compute", lambda run: 1 / 0)
    run.state_path.write_text(json.dumps({"session_id": None}))
    assert hook(boar, monkeypatch, {}) == (0, "", "")


def test_hook_finds_the_run_from_the_hook_cwd(boar, env, repo, monkeypatch):
    init(boar, env)
    elsewhere = env / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert hook(boar, monkeypatch, {"cwd": str(repo)})[0] == 2


def test_hook_subprocess_does_not_hang_on_an_open_stdin(boar, env, repo):
    run = init(boar, env, "--session", "sess-A")
    cmd = [sys.executable, "-m", "boar", "next", "--hook"]
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert p.wait(timeout=30) == 2
        assert p.stderr.read().decode().startswith(f"BOAR run {run.id} is not finished. [S2]")
    finally:
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()
    assert time.monotonic() - t0 < 30
    other = subprocess.run(cmd, cwd=repo, input=json.dumps({"session_id": "sess-B"}), capture_output=True, text=True)
    assert (other.returncode, other.stdout, other.stderr) == (0, "", "")
    same = subprocess.run(cmd, cwd=repo, input=json.dumps({"session_id": "sess-A"}), capture_output=True, text=True)
    assert same.returncode == 2


# --- session binding and read-only commands ---------------------------------------------------------


def test_next_session_binds_the_run(boar, env, repo):
    run = init(boar, env, "--session", "old")
    boar("next", "--session", "${CLAUDE_SESSION_ID}")
    assert run.state()["session_id"] == "old"
    boar("next", "--session", "new")
    assert run.state()["session_id"] == "new"


def test_status_next_and_review_are_read_only(boar, env, repo, evalfake, monkeypatch):
    run = to_round1(boar, env)
    add_trials(run, 1, 2)
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="done"))
    fake_module(
        monkeypatch, "warmstart",
        incumbent=lambda trials, hyps, r, direction: {"config": {"a": 4}, "metric": 0.5, "repeats": [0.5, 0.6], "trials": [2]},
    )
    before = snapshot(repo)
    code, out, _ = boar("status")
    assert code == 0
    assert f"run {run.id}  phase round  round 1/12" in out
    assert "eval: accepted" in out and "H1  active" in out
    assert "trials: 2 (complete 2, infeasible 0, failed 0)" in out
    assert 'incumbent: {"a": 4} metric 0.5 over 2 repeats (trials 2)' in out
    assert "worker: none" in out and "next: [R3] Decide on: H1, H2" in out
    for args in (("next",), ("review", "pending"), ("review", "show", "H1"), ("review", "show", "eval")):
        assert boar(*args)[0] == 0
    assert snapshot(repo) == before


def test_status_survives_a_missing_engine(boar, env, repo, evalfake, monkeypatch):
    run = to_round1(boar, env)
    add_trials(run, 1, 1)
    fake_module(monkeypatch, "warmstart")
    code, out, _ = boar("status")
    assert code == 0 and "incumbent: unavailable" in out


def test_hook_holds_a_report_that_was_deleted(boar, env, repo, monkeypatch):
    run = init(boar, env)
    run.update_state(lambda s: (s.update(phase="finalize"), s["finalize"].update(status="done")))
    code, _, err = hook(boar, monkeypatch, {})
    assert code == 2 and "[REPORT] report.md is missing; run `boar finalize --report`" in err
    run.report_path.write_text("")
    assert "[REPORT] report.md has lost generated headings: ## 1. Result;" in hook(boar, monkeypatch, {})[2]
    assert run.state()["phase"] == "finalize"


def test_next_session_says_when_it_takes_the_hook_from_another_session(boar, env, repo):
    run = init(boar, env, "--session", "sess-A")
    out = boar("next", "--session", "sess-A")[1]
    assert out.startswith("[S2]") and "note:" not in out
    out = boar("next", "--session", "sess-B")[1]
    assert out.startswith("[S2]")
    assert out.splitlines()[-1] == "note: the Stop hook now holds session sess-B to this run (was sess-A)"
    assert run.state()["session_id"] == "sess-B"
    assert "note:" not in boar("next")[1]


def test_hook_falls_back_to_the_project_dir_when_the_session_cwd_is_elsewhere(boar, env, repo, monkeypatch):
    init(boar, env)
    elsewhere = env / "vendor-clone"
    elsewhere.mkdir()
    git(elsewhere, "init", "-q")
    monkeypatch.chdir(elsewhere)
    assert hook(boar, monkeypatch, {"cwd": str(elsewhere)})[0] == 0
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
    assert hook(boar, monkeypatch, {"cwd": str(elsewhere)})[0] == 2
    monkeypatch.delenv("CLAUDE_PROJECT_DIR")
    monkeypatch.chdir(repo)
    assert hook(boar, monkeypatch, {"cwd": str(elsewhere)})[0] == 2


def test_rebuilding_after_the_third_eval_rejection_goes_back_to_s4(boar, env, repo, evalfake, monkeypatch):
    run = init(boar, env)
    boar("propose", props(env, "a"))
    for attempt in range(3):
        boar("eval", "check", "--foreground")
        boar("review", "record", "eval", "reject", "--reason", f"overlap {attempt}")
    assert first(boar).startswith("[ASK_USER] The eval was rejected 3 times")
    assert hook(boar, monkeypatch, {})[0] == 0
    evalfake["ok"] = False
    boar("eval", "check", "--foreground")
    out = boar("next")[1]
    assert out.startswith("[S4] Eval rejected (failed attempt 3/3): overlap 2")
    assert "last check failed:" in out and "last line is not JSON" in out
    assert hook(boar, monkeypatch, {})[0] == 2
    evalfake["ok"] = True
    boar("eval", "check", "--foreground")
    assert first(boar).startswith("[S5]")
    boar("review", "record", "eval", "reject", "--reason", "still overlaps")
    assert first(boar).startswith("[ASK_USER] The eval was rejected 4 times")
    assert run.state()["eval"]["rejections"] == 4


def test_an_eval_edited_after_its_check_goes_back_to_s4_not_to_review(boar, env, repo, evalfake):
    init(boar, env)
    boar("propose", props(env, "a"))
    boar("eval", "check", "--foreground")
    evalfake["current"] = "edited"
    out = boar("next")[1]
    assert out.startswith("[S5] Send the boar-reviewer agent these items: H1.")
    assert "eval/ changed since its passing `boar eval check`" in out
    boar("review", "record", "H1", "accept", "--reason", "ok")
    assert first(boar) == "[S4] eval/ changed since its passing `boar eval check`; re-run the check before the eval goes to review."
    evalfake["hash"] = "edited"
    boar("eval", "check", "--foreground")
    assert first(boar) == "[S5] Send the boar-reviewer agent these items: eval."


def test_a_frozen_eval_violation_shows_at_r1_and_finalize(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    evalfake["current"] = "edited"
    out = first(boar)
    assert out == (
        "[R1] eval/ has changed since the reviewer accepted it (changed: run); the eval is frozen: run "
        "`boar eval restore` to put the accepted version back; then `boar round run`."
    )
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="running", commit="c"))
    assert first(boar).startswith("[R1] eval/ has changed since the reviewer accepted it (changed: run)")
    run.update_state(lambda s: s.update(phase="finalize"))
    assert first(boar).startswith("[FINALIZE] eval/ has changed since the reviewer accepted it (changed: run)")
    evalfake["current"] = "h1"
    assert first(boar) == "[FINALIZE] Run `boar finalize`."


# --- round 2 fixes ------------------------------------------------------------------------------------


def test_r6_does_not_suggest_fix_in_the_last_round(boar, env, repo, evalfake):
    run = to_round1(boar, env, "--rounds", "1")
    add_trials(run, 1, 3)
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="done"))
    boar("decide", "H1", "remove", "--reason", "flat", "--trials", "1", "2", "3")
    boar("decide", "H2", "keep", "--reason", "x")
    boar("propose", "--none", "--reason", "last round")
    boar("review", "record", "rm-H1-r1", "reject", "--reason", "trial 3 crashed from a bug; calls for fix")
    out = boar("next")[1]
    assert out.startswith("[R6] Run `boar round close`.")
    assert "rm-H1-r1 was rejected (trial 3 crashed from a bug; calls for fix); H1 stays active" in out
    assert "This is the last round: no R1 follows" in out and "`boar decide H1 keep`" in out
    assert "boar decide H1 fix" not in out


def test_r6_does_not_suggest_fix_for_a_hypothesis_superseded_this_round(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    add_trials(run, 1, 3)
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="done"))
    boar("decide", "H1", "remove", "--reason", "flat", "--trials", "1", "2", "3")
    boar("decide", "H2", "keep", "--reason", "x")
    boar("propose", props(env, "c", supersedes="H1"))
    boar("review", "record", "rm-H1-r1", "reject", "--reason", "a bug")
    boar("review", "record", "H3", "accept", "--reason", "ok")
    out = boar("next")[1]
    assert "removes H1: superseded by H3" in out
    assert "rm-H1-r1 was rejected (a bug), but H1 goes anyway: superseded by H3." in out
    assert "stays active" not in out and "boar decide H1 fix" not in out


def registry(env, sid):
    return env / "state" / "boar" / "sessions" / sid


def test_hook_finds_the_sessions_run_through_the_registry(boar, env, repo, monkeypatch):
    """Claude started in a parent dir: neither the hook cwd, the project dir nor the process cwd is in the repo."""
    run = init(boar, env, "--session", "sess-A")
    assert registry(env, "sess-A").read_text().strip() == str(repo)
    scratch = env / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(scratch)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(env))
    code, _, err = hook(boar, monkeypatch, {"session_id": "sess-A", "cwd": str(scratch)})
    assert code == 2 and err.startswith(f"BOAR run {run.id} is not finished. [S2]")
    assert hook(boar, monkeypatch, {"session_id": "sess-B", "cwd": str(scratch)})[0] == 0
    assert hook(boar, monkeypatch, {"cwd": str(scratch)})[0] == 0

    # In another target repo whose run belongs to another session, the hook still finds this session's run.
    other = env / "other"
    other.mkdir()
    git(other, "init", "-q", "-b", "main")
    (other / "x.py").write_text("x\n")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "init")
    monkeypatch.chdir(other)
    assert boar("init", "--spec", env / "spec.md", "--session", "sess-T")[0] == 0
    theirs = store.active_run()
    code, _, err = hook(boar, monkeypatch, {"session_id": "sess-A", "cwd": str(other)})
    assert code == 2 and err.startswith(f"BOAR run {run.id} is not finished.")
    code, _, err = hook(boar, monkeypatch, {"session_id": "sess-T", "cwd": str(other)})
    assert code == 2 and err.startswith(f"BOAR run {theirs.id} is not finished.")

    # A stale entry releases: the run there was aborted, or the entry points nowhere.
    monkeypatch.chdir(repo)
    boar("abort", "--reason", "done with it")
    monkeypatch.chdir(scratch)
    assert hook(boar, monkeypatch, {"session_id": "sess-A", "cwd": str(scratch)})[0] == 0
    # ... and stays stale when a new unbound run starts there: the entry leads only to a run bound to this session.
    monkeypatch.chdir(repo)
    assert boar("init", "--spec", env / "spec.md")[0] == 0
    assert store.active_run().state()["session_id"] is None
    monkeypatch.chdir(scratch)
    assert hook(boar, monkeypatch, {"session_id": "sess-A", "cwd": str(scratch)})[0] == 0
    registry(env, "sess-A").write_text(str(env / "gone") + "\n")
    assert hook(boar, monkeypatch, {"session_id": "sess-A", "cwd": str(scratch)})[0] == 0


def test_next_session_registers_the_repo_and_unsafe_ids_are_not_written(boar, env, repo):
    init(boar, env)
    boar("next", "--session", "sess-N")
    assert registry(env, "sess-N").read_text().strip() == str(repo)
    boar("next", "--session", "../escape")
    assert not (env / "state" / "boar" / "escape").exists()
    assert sorted(p.name for p in (env / "state" / "boar" / "sessions").iterdir()) == ["sess-N"]


def test_wait_points_at_a_log_only_for_a_detached_worker(boar, env, repo):
    run = init(boar, env)
    inline = subprocess.Popen(["sleep", "30"])
    detached = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        for proc, shown in ((inline, "it runs inline (--foreground, pid"), (detached, "log: w.log")):
            store.write_json(run.running_path, {"pid": proc.pid, "identity": store.process_identity(proc.pid),
                                                "command": "eval check", "round": None, "log": "w.log"})
            out = boar("next")[1]
            assert out.startswith("[WAIT] boar eval check is running") and shown in out
            assert ("log:" in out) == (proc is detached)
    finally:
        for proc in (inline, detached):
            proc.kill()
            proc.wait()


def test_an_unreadable_eval_pending_review_holds_the_hook(boar, env, repo, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root reads anything")
    run = init(boar, env, "--repeats", "1")
    run.eval_dir.mkdir()
    (run.eval_dir / "run").write_text("#!/bin/sh\necho '{\"metric\": 1.0, \"guards_ok\": true}'\n")
    (run.eval_dir / "run").chmod(0o755)
    (run.eval_dir / "dev.json").write_text("[1]\n")
    boar("propose", props(env, "a"))
    assert boar("eval", "check", "--foreground")[0] == 0
    (run.eval_dir / "dev.json").chmod(0)
    try:
        out = boar("next")[1]
        assert out.startswith("[S5] Send the boar-reviewer agent these items: H1.")
        assert "eval/ could not be read (unreadable: dev.json: Permission denied)" in out
        boar("review", "record", "H1", "accept", "--reason", "ok")
        assert first(boar).startswith("[S4] eval/ could not be read (unreadable: dev.json: Permission denied)")
        assert hook(boar, monkeypatch, {})[0] == 2
    finally:
        (run.eval_dir / "dev.json").chmod(0o644)
    assert first(boar) == "[S5] Send the boar-reviewer agent these items: eval."


def test_r3_does_not_offer_fix_in_the_last_round(boar, env, repo, evalfake):
    run = to_round1(boar, env, "--rounds", "1")
    add_trials(run, 1, 3)
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="done"))
    out = boar("next")[1]
    assert out.startswith("[R3] Decide on: H1, H2 — `boar decide <id> keep|remove --reason")
    assert "R1 repairs it" not in out and "fix: not in the last round (1/1): no R1 follows" in out
