"""Acceptance: the Stop hook (`boar next --hook`) blocks exactly while there is work to do.

Exit 2 with the next step on stderr keeps Claude Code working; exit 0 lets it stop. Only the
session that owns the run is held, and nothing on stdin may make the hook crash.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from boar.store import process_identity

REPO = Path(__file__).resolve().parents[2]


def test_no_run_lets_the_agent_stop(boar, tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    rc, _, err = boar("next", "--hook", stdin='{"session_id": "s1"}')
    assert rc == 0 and err == ""
    rc, out, _ = boar("next")
    assert rc == 0 and out.startswith("[NONE]")


def test_no_run_in_a_git_repo_lets_the_agent_stop(boar, target_repo):
    rc, _, _ = boar("next", "--hook", stdin="{}")
    assert rc == 0


def test_mid_run_blocks_with_the_next_step_on_stderr(driver):
    d = driver.start("init")
    rc, err = d.hook()
    assert rc == 2
    assert f"BOAR run {d.run_id} is not finished" in err
    first_line = d.next().strip().splitlines()[0]
    assert first_line in err


def test_hook_holds_only_the_session_that_owns_the_run(driver):
    d = driver.start("init", session="session-A")
    assert d.state()["session_id"] == "session-A"
    assert d.hook(json.dumps({"session_id": "session-B", "hook_event_name": "Stop"}))[0] == 0
    assert d.hook(json.dumps({"session_id": "session-A", "hook_event_name": "Stop"}))[0] == 2
    assert d.hook("{}")[0] == 2, "hook input without a session id blocks"
    assert d.hook("")[0] == 2, "no hook input blocks"

    d.ok("next", "--session", "session-C")
    assert d.state()["session_id"] == "session-C"
    assert d.hook(json.dumps({"session_id": "session-A"}))[0] == 0
    assert d.hook(json.dumps({"session_id": "session-C"}))[0] == 2


def test_unsubstituted_session_variable_counts_as_no_session(driver):
    d = driver.start("init", session="${CLAUDE_SESSION_ID}")
    assert d.hook(json.dumps({"session_id": "whatever"}))[0] == 2


def test_run_without_a_session_blocks_every_session(driver):
    d = driver.start("init")
    assert d.hook(json.dumps({"session_id": "any"}))[0] == 2


def test_ask_user_after_three_eval_rejections_releases_the_hook(driver):
    d = driver.start("checked", **driver.SMALL)
    for attempt in range(1, 4):
        if attempt > 1:
            d.eval_check()
        assert d.hook()[0] == 2
        d.review("eval", "reject", f"Attempt {attempt}: holdout overlaps dev.")
    assert d.state()["eval"]["rejections"] == 3
    assert d.tag() == "ASK_USER"
    assert d.hook()[0] == 0
    d.eval_check()
    assert d.tag() != "ASK_USER", "a re-checked eval goes back to review"
    assert d.hook()[0] == 2


def test_ask_user_after_three_rejected_batches_releases_the_hook(driver):
    d = driver.start("checked", proposals=[driver.prop_a()], **driver.SMALL)
    d.review("eval", "accept")
    batches = [None, driver.prop_b(), driver.prop_noop()]
    for i, extra in enumerate(batches, 1):
        hid = d.hid("fast_a") if extra is None else d.propose(extra)[0]
        d.review(hid, "reject", "Out of scope for the spec.")
        assert d.state()["setup_failed_batches"] == i
        if i < 3:
            assert d.tag() == "S3"
            assert d.hook()[0] == 2
    assert d.tag() == "ASK_USER"
    assert d.hook()[0] == 0


def test_aborted_run_releases_the_hook(driver):
    d = driver.start("round1", **driver.SMALL)
    assert d.hook()[0] == 2
    d.ok("abort", "--reason", "the user asked to stop")
    st = d.state()
    assert st["phase"] == "aborted" and st["abort_reason"] == "the user asked to stop"
    assert d.tag() == "ABORTED"
    assert d.hook()[0] == 0
    d.refused("round", "run")


def test_malformed_hook_input_is_sane(driver, boar):
    for stdin in ("not json{", "[1, 2]", '"a string"', '{"session_id": 123}', '{"session_id": null}', "\x00\xff"):
        rc, out, err = boar("next", "--hook", stdin=stdin)
        assert rc == 0 and "Traceback" not in err, (stdin, err)
    d = driver.start("init")
    rc, err = d.hook("not json{")
    assert rc == 2, "unparseable hook input is ignored, not a reason to let go"
    for stdin in ("[1, 2]", '"a string"', '{"session_id": 123}', '{"session_id": null}', "\x00\xff"):
        rc, out, err = d.boar("next", "--hook", stdin=stdin)
        assert rc in (0, 2) and "Traceback" not in err, (stdin, err)


def test_live_worker_means_wait_and_blocks_other_steps(driver):
    d = driver.start("round1", **driver.SMALL)
    sleeper = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        (d.run_dir / "running.json").write_text(
            json.dumps({"pid": sleeper.pid, "identity": process_identity(sleeper.pid), "command": "round run",
                        "round": 1, "started": "2026-10-02T00:00:00+00:00",
                        "log": str(d.run_dir / "rounds" / "1" / "worker.log")})
        )
        text = d.next()
        assert d.tag_of(text) == "WAIT" and "boar wait" in text
        assert d.hook()[0] == 2
        d.refused("round", "run")
        d.ok("abort", "--reason", "stop the run and its worker")
        assert sleeper.wait(timeout=10) is not None, "abort must stop the background worker"
        assert d.hook()[0] == 0
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait()


def test_hook_exit_codes_through_a_real_process(driver):
    d = driver.start("init", session="S1")

    def hook(stdin: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "boar", "next", "--hook"],
            input=stdin, cwd=d.repo, capture_output=True, text=True, timeout=60,
        )

    p = hook(json.dumps({"session_id": "S1", "hook_event_name": "Stop"}))
    assert p.returncode == 2 and f"BOAR run {d.run_id} is not finished" in p.stderr
    assert hook(json.dumps({"session_id": "S2"})).returncode == 0


def _stop_hook_commands() -> list[str]:
    data = json.loads((REPO / "plugin" / "hooks" / "hooks.json").read_text())
    return [h["command"] for entry in data["hooks"]["Stop"] for h in entry["hooks"] if h.get("type") == "command"]


def test_plugin_stop_hook_runs_boar_next_hook():
    assert any("boar next --hook" in c for c in _stop_hook_commands())


def test_plugin_stop_hook_command_blocks_mid_run_and_tolerates_missing_boar(driver):
    bindir = Path(sys.executable).parent
    if not (bindir / "boar").exists():
        pytest.skip("no boar entry point next to the interpreter")
    d = driver.start("init")
    (command,) = [c for c in _stop_hook_commands() if "boar next --hook" in c]
    shell = shutil.which("sh") or "/bin/sh"
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    p = subprocess.run([shell, "-c", command], input="{}", cwd=d.repo, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and "not finished" in p.stderr
    bare = {**os.environ, "PATH": "/nonexistent"}
    p = subprocess.run([shell, "-c", command], input="{}", cwd=d.repo, env=bare, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0
