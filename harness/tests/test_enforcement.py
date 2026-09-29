"""Enforcement checks (`check write|bash|read|stop|recorded`) and the hook adapters that call them.

Each check answers {"allow": true} or {"allow": false, "reason": ...}; every block is logged as
`hook_blocked`. The adapters are driven as Claude Code runs them: hook JSON on stdin, exit code out.
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import bo, events, git, write_run_yaml

HOOK = Path(__file__).resolve().parents[2] / "hooks" / "hook.py"


def check(cwd, *args):
    code, out = bo(cwd, "check", *args)
    assert code == 0, out
    return out


def resolved(path):
    """How a check names a path: absolute, its directories resolved (not the file's own symlink)."""
    return str(Path(path).parent.resolve() / Path(path).name)


def blocks(run_dir):
    return [e["payload"] for e in events(run_dir) if e["type"] == "hook_blocked"]


def assert_blocked(run_dir, out, check_name, agent, subject, words):
    """Blocked with a reason naming `words`, and logged as hook_blocked with the check, agent and reason."""
    assert out["allow"] is False, out
    assert words in out["reason"], out["reason"]
    assert blocks(run_dir)[-1] == {"check": check_name, "agent": agent, "subject": subject,
                                   "reason": out["reason"]}


# --- check write ------------------------------------------------------------------------------

@pytest.fixture
def guarded(init_run):
    """A run whose registry protects data/ too."""
    return init_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\nprotected_paths: [data/]\n")


WRITE_CASES = [
    # (path relative to the run dir or to the repo, allowed, words in the reason)
    ("worktree/train.py", True, None),
    ("worktree/new_module.py", True, None),
    ("worktree/runner.py", False, "protected path"),
    ("worktree/levers.json", False, "protected path"),
    ("worktree/data/train.csv", False, "protected path"),
    ("log.db", False, "harness"),
    ("run.yaml", False, "harness"),
    ("venv/bin/python", False, "harness"),
    ("artifacts/trial-1/note.txt", False, "harness"),
    ("SUMMARY.md", False, "harness"),
    ("REPO/train.py", False, "user's checkout"),
    ("REPO/run.yaml", True, None),  # the draft of the next run.yaml, before its init
]


@pytest.mark.parametrize("rel,allowed,words", WRITE_CASES)
def test_check_write_guards_protected_paths_the_harness_files_and_the_users_checkout(guarded, rel, allowed, words):
    repo = guarded.parents[1]
    path = str(repo / rel[len("REPO/"):]) if rel.startswith("REPO/") else str(guarded / rel)
    out = check(repo, "write", "--agent", "lever-coder", "--", path)
    if allowed:
        assert out == {"allow": True}
        assert blocks(guarded) == []
    else:
        assert_blocked(guarded, out, "write", "lever-coder", resolved(path), words)


def test_check_write_resolves_relative_paths_against_the_callers_directory(guarded):
    out = check(guarded / "worktree", "write", "--agent", "orchestrator", "--", "runner.py")
    assert_blocked(guarded, out, "write", "orchestrator", resolved(guarded / "worktree" / "runner.py"),
                   "protected path")
    assert check(guarded / "worktree", "write", "--agent", "orchestrator", "--", "train.py") == {"allow": True}


def test_checks_allow_everything_outside_the_repository_and_when_there_is_no_run(guarded, repo, tmp_path):
    assert check(repo, "write", "--agent", "lever-coder", "--", str(tmp_path / "spec.json")) == {"allow": True}
    bare = tmp_path / "other"
    bare.mkdir()
    git(bare, "init", "-q")
    assert check(bare, "write", "--agent", "lever-coder", "--", "anything.py") == {"allow": True}
    assert check(bare, "stop") == {"allow": True}


# --- check bash -------------------------------------------------------------------------------

RECORD_HEREDOC = "boautoresearch record review --file - <<'EOF'\n{\"x\": \"cat > runner.py\"}\nEOF"

BASH_CASES = [
    # (agent, cwd: RUN | WT | REPO, command, allowed, words in the reason)
    ("boautoresearch:round-analyst", "REPO", "boautoresearch status", True, None),
    ("boautoresearch:round-analyst", "REPO", "cd WT && boautoresearch verdict H1", True, None),
    ("boautoresearch:registration-reviewer", "REPO", RECORD_HEREDOC, True, None),
    ("boautoresearch:round-analyst", "REPO", "cat train.py", False, "read-only"),
    ("boautoresearch:interplay-reviewer", "REPO", "boautoresearch round-run --rationale go", False, "read-only"),
    ("boautoresearch:round-analyst", "REPO", "boautoresearch status > out.json", False, "read-only"),
    ("boautoresearch:round-analyst", "REPO", "boautoresearch status $(rm -rf WT)", False, "read-only"),
    ("orchestrator", "REPO", "pip install numpy", False, "add-dependency"),
    ("boautoresearch:lever-coder", "WT", "python -m pip install -q numpy", False, "add-dependency"),
    ("boautoresearch:lever-coder", "WT", "uv pip install --python RUN/venv/bin/python numpy", False, "add-dependency"),
    ("boautoresearch:lever-coder", "WT", "conda install numpy", False, "add-dependency"),
    ("boautoresearch:lever-coder", "WT", "pip list", True, None),
    ("boautoresearch:lever-coder", "WT", "echo 'x=1' > runner.py", False, "protected path"),
    ("boautoresearch:lever-coder", "REPO", "echo x >> WT/levers.json", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "sed -i '' s/a/b/ runner.py", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "rm -f levers.json", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "mv train.py runner.py", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "cp /tmp/r.py runner.py", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "echo x | tee -a runner.py", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "echo x > RUN/log.db", False, "harness"),
    ("boautoresearch:lever-coder", "WT", "sed -i '' s/a/b/ train.py && cp train.py /tmp/t.py 2>&1", True, None),
    ("boautoresearch:lever-coder", "WT", "python train.py > /dev/null 2>&1", True, None),
    ("boautoresearch:lever-coder", "WT", "git commit -am 'lever'", False, "through the harness"),
    ("orchestrator", "REPO", "git -C WT reset --hard HEAD~1", False, "through the harness"),
    ("boautoresearch:lever-coder", "WT", "git stash", False, "through the harness"),
    ("boautoresearch:lever-coder", "REPO", "cd WT; git checkout -- train.py", False, "through the harness"),
    ("boautoresearch:lever-coder", "WT", "git diff HEAD && git log -3 && git show HEAD", True, None),
    ("orchestrator", "REPO", "sqlite3 RUN/log.db 'select * from events'", False, "probes"),
    ("orchestrator", "RUN", "cat artifacts/trial-1/note.txt", False, "probes"),
    ("orchestrator", "REPO", "cat .bo-research/*/log.db", False, "probes"),
    ("orchestrator", "REPO", "boautoresearch trials", True, None),
    ("orchestrator", "REPO", "cat RUN/SUMMARY.md", True, None),
    ("boautoresearch:round-analyst", "RUN", "boautoresearch trials", True, None),
    # forms that once slipped past the parser
    ("boautoresearch:lever-coder", "WT", "echo '<<X'\nrm levers.json\nX", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "cat <<X\nhi\nX\nrm levers.json\nX", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "echo x >& runner.py", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "env -i rm levers.json", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "bash -c 'rm levers.json'", False, "protected path"),
    ("boautoresearch:lever-coder", "WT", "eval git commit -m x", False, "through the harness"),
    ("orchestrator", "REPO", "boautoresearch record review --file r.json --agent-id rr --actor registration-reviewer",
     False, "another agent's name"),
]


@pytest.mark.parametrize("agent,where,command,allowed,words", BASH_CASES)
def test_check_bash_enforces_the_read_only_allowlist_installs_protected_writes_git_and_raw_reads(
        guarded, agent, where, command, allowed, words):
    repo = guarded.parents[1]
    places = {"RUN": str(guarded), "WT": str(guarded / "worktree"), "REPO": str(repo)}
    command = re.sub(r"\b(RUN|WT)\b", lambda m: places[m.group(1)], command)
    out = check(places[where], "bash", "--agent", agent, "--", command)
    if allowed:
        assert out == {"allow": True}, out
        assert blocks(guarded) == []
    else:
        assert_blocked(guarded, out, "bash", agent, command, words)


def test_a_subagents_record_comes_back_carrying_its_agent_id_and_role(guarded):
    repo = guarded.parents[1]
    out = check(repo, "bash", "--agent", "boautoresearch:registration-reviewer", "--agent-id", "a7", "--",
                RECORD_HEREDOC)
    assert out == {"allow": True, "updated_command": RECORD_HEREDOC.replace(
        "boautoresearch record", "boautoresearch record --agent-id a7 --actor registration-reviewer", 1)}
    # the orchestrator (no agent id) records as itself; a subagent can't name another's id
    assert check(repo, "bash", "--agent", "orchestrator", "--", "boautoresearch record expected --file e.json") == {
        "allow": True}
    out = check(repo, "bash", "--agent", "boautoresearch:hypothesis-generator", "--agent-id", "g1", "--",
                "boautoresearch record proposal --file p.json --agent-id rr")
    assert_blocked(guarded, out, "bash", "boautoresearch:hypothesis-generator",
                   "boautoresearch record proposal --file p.json --agent-id rr", "the hook adds")


def test_the_record_rewrite_leaves_the_heredoc_body_alone(guarded):
    cmd = "boautoresearch record narrative --file - <<'EOF'\n{\"text\": \"ran boautoresearch record expected\"}\nEOF"
    out = check(guarded.parents[1], "bash", "--agent", "boautoresearch:round-analyst", "--agent-id", "an1", "--", cmd)
    assert out["updated_command"] == cmd.replace("record narrative", "record --agent-id an1 --actor round-analyst narrative")


def test_the_rewritten_record_logs_the_subagents_record_from_a_heredoc(guarded, tmp_path):
    repo = guarded.parents[1]
    body = {"hypothesis": "H9.v1", "verdict": "undecided", "reason": "no view yet"}
    cmd = f"{sys.executable} -m boautoresearch record expected --file - --rationale mine <<'EOF'\n{json.dumps(body)}\nEOF"
    out = check(repo, "bash", "--agent", "boautoresearch:round-analyst", "--agent-id", "an1", "--", cmd)
    assert out["allow"] is True
    p = subprocess.run(out["updated_command"], shell=True, cwd=repo, capture_output=True, text=True)
    assert p.returncode == 1 and "H9.v1" in json.loads(p.stdout)["reason"]  # validated like a file's body


# --- check read -------------------------------------------------------------------------------

READ_CASES = [
    ("orchestrator", "log.db", False),
    ("orchestrator", "log.db-wal", False),
    ("orchestrator", "artifacts/trial-1/note.txt", False),
    ("orchestrator", ".", False),  # Grep or Glob over the run dir reads its artifacts
    ("orchestrator", "SUMMARY.md", True),
    ("orchestrator", "worktree/train.py", True),
    ("boautoresearch:round-analyst", "artifacts/trial-1/note.txt", True),
    ("boautoresearch:round-analyst", "log.db", True),
]


@pytest.mark.parametrize("agent,rel,allowed", READ_CASES)
def test_check_read_keeps_the_orchestrator_off_the_raw_log_and_artifacts(guarded, agent, rel, allowed):
    out = check(guarded, "read", "--agent", agent, "--", rel)
    if allowed:
        assert out == {"allow": True}
    else:
        assert_blocked(guarded, out, "read", agent, resolved(guarded / rel).rstrip("/."), "probes")


# --- check stop, and the worktree freeze while a round runs --------------------------------------

def slow_r0(repo, run_dir):
    """R0 running in the background (one slow trial); -> the process, once its round has started."""
    env = {**os.environ, "TOY_SLEEP_PER_EPOCH": "30"}
    p = subprocess.Popen([sys.executable, "-m", "boautoresearch", "round-run", "--rationale", "calibrate"],
                         cwd=repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    while not any(e["type"] == "trial_started" for e in events(run_dir)):
        assert time.monotonic() < deadline and p.poll() is None
        time.sleep(0.1)
    return p


def stop_blocked(run_dir, cwd, words):
    out = check(cwd, "stop")
    assert_blocked(run_dir, out, "stop", "orchestrator", "stop", words)


def test_check_stop_blocks_mid_run_and_allows_a_pause_a_background_round_and_a_finished_wrap_up(init_run):
    run_dir = init_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\ngo: true\n")
    repo = run_dir.parents[1]
    stop_blocked(run_dir, repo, "round-run (the calibration round)")  # headless: no interview to wait on

    p = slow_r0(repo, run_dir)
    try:
        assert check(repo, "stop") == {"allow": True}  # it wakes the orchestrator when it returns
        out = check(repo, "write", "--agent", "lever-coder", "--", str(run_dir / "worktree" / "train.py"))
        assert_blocked(run_dir, out, "write", "lever-coder", resolved(run_dir / "worktree" / "train.py"),
                       "a round is running")
    finally:
        p.kill()
        p.wait()
    stop_blocked(run_dir, repo, "round-run")  # the killed round is no background round
    assert check(repo, "write", "--agent", "lever-coder", "--", str(run_dir / "worktree" / "train.py")) == {
        "allow": True}

    assert bo(repo, "checkpoint", "--rationale", "the user pauses")[0] == 0
    assert check(repo, "stop") == {"allow": True}
    assert bo(repo, "checkpoint", "--resume", "--rationale", "go on")[0] == 0
    stop_blocked(run_dir, repo, "round-run")

    assert bo(repo, "stop", "--rationale", "the user stops")[0] == 0
    stop_blocked(run_dir, repo, "wrap-up")
    n = len(blocks(run_dir))
    con = sqlite3.connect(run_dir / "log.db")  # wrap-up is #37's; its last event ends enforcement
    con.execute("INSERT INTO events (type, schema_version, ts, actor, payload) "
                "VALUES ('wrapup_finished', 1, '2026-01-01T00:00:00+00:00', 'harness', '{}')")
    con.commit()
    con.close()
    assert check(repo, "stop") == {"allow": True}
    assert check(repo, "write", "--agent", "lever-coder", "--", str(repo / "train.py")) == {"allow": True}
    assert len(blocks(run_dir)) == n


def test_check_stop_allows_the_pre_loop_interviews_of_an_interactive_run(guarded):
    assert check(guarded.parents[1], "stop") == {"allow": True}


# --- check recorded ---------------------------------------------------------------------------

def test_check_recorded_holds_a_recording_subagent_until_its_record_exists(guarded, tmp_path):
    repo = guarded.parents[1]
    out = check(repo, "recorded", "g1", "--agent", "boautoresearch:hypothesis-generator")
    assert out["recorded"] is False
    assert_blocked(guarded, out, "recorded", "boautoresearch:hypothesis-generator", "g1", "record")
    # the lever coder commits through the harness and records nothing: never held
    assert check(repo, "recorded", "c1", "--agent", "boautoresearch:lever-coder") == {"allow": True, "recorded": False}
    spec = {"title": "t", "rationale": "r", "mechanism": "m", "provenance": "novel", "lens": "l",
            "directives": [], "fidelity_sensitive": False,
            "levers": {"a": {"kind": "float", "low": 0, "high": 1, "baseline": 0.5, "predicted": "higher"}}}
    body = {"hypotheses": [spec, {**spec, "mechanism": "another", "levers": {"b": spec["levers"]["a"]}}]}
    f = tmp_path / "p.json"
    f.write_text(json.dumps(body))
    code, out = bo(repo, "record", "proposal", "--file", str(f), "--agent-id", "g1", "--actor",
                   "hypothesis-generator", "--rationale", "my lens")
    assert code == 0, out
    assert check(repo, "recorded", "g1", "--agent", "boautoresearch:hypothesis-generator") == {
        "allow": True, "recorded": True}


# --- the hook adapters ------------------------------------------------------------------------

def hook(cwd, event, **fields):
    """Run the adapter as Claude Code does (system python3, hook JSON on stdin)."""
    body = {"session_id": "s", "cwd": str(cwd), "hook_event_name": event, **fields}
    p = subprocess.run(["python3", str(HOOK)], input=json.dumps(body), capture_output=True, text=True,
                       timeout=60)
    return p.returncode, p.stdout, p.stderr


def test_the_adapter_maps_allow_to_exit_0_and_block_to_exit_2_with_the_reason(init_run):
    run_dir = init_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\ngo: true\n")
    repo, wt = run_dir.parents[1], run_dir / "worktree"
    code, out, err = hook(wt, "PreToolUse", tool_name="Write", tool_input={"file_path": str(wt / "train.py"), "content": ""})
    assert (code, out, err) == (0, "", "")
    code, _, err = hook(wt, "PreToolUse", tool_name="Edit", agent_type="boautoresearch:lever-coder", agent_id="c1",
                        tool_input={"file_path": "runner.py", "old_string": "a", "new_string": "b"})
    assert code == 2 and "protected path" in err
    assert blocks(run_dir)[-1]["agent"] == "boautoresearch:lever-coder"
    code, _, err = hook(repo, "PreToolUse", tool_name="Read", tool_input={"file_path": str(run_dir / "log.db")})
    assert code == 2 and "probes" in err
    code, _, err = hook(repo, "PreToolUse", tool_name="Grep", tool_input={"pattern": "loss", "path": str(run_dir)})
    assert code == 2 and "probes" in err
    code, _, err = hook(repo, "PreToolUse", tool_name="Bash", tool_input={"command": "pip install torch"})
    assert code == 2 and "add-dependency" in err
    code, _, err = hook(repo, "Stop", stop_hook_active=False)
    assert code == 2 and "round-run (the calibration round)" in err
    code, _, err = hook(repo, "SubagentStop", agent_type="boautoresearch:round-analyst", agent_id="an1")
    assert code == 2 and "record" in err
    code, _, err = hook(repo, "PreToolUse", tool_name="SubagentHandback", agent_type="boautoresearch:round-analyst",
                        agent_id="an1", tool_input={"message": "done"})
    assert code == 2 and "record" in err
    assert hook(repo, "SubagentStop", agent_type="Explore", agent_id="e1")[0] == 0
    assert hook(repo, "SubagentStop", agent_id="x1")[0] == 2  # no agent type: held, not taken for the orchestrator
    code, _, err = hook(repo, "PreToolUse", tool_name="Glob", tool_input={"pattern": ".bo-research/*/log.db"})
    assert code == 2 and "probes" in err
    code, _, err = hook(repo, "PreToolUse", tool_name="Grep", tool_input={"pattern": "x", "glob": ".bo-research/**"})
    assert code == 2 and "probes" in err
    assert hook(repo, "PreToolUse", tool_name="Glob", tool_input={"pattern": "**/*.py"})[0] == 0
    assert [b["check"] for b in blocks(run_dir)] == ["write", "read", "read", "bash", "stop", "recorded", "recorded",
                                                     "recorded", "read", "read"]


def test_the_adapter_rewrites_a_subagents_record_and_injects_status_and_next(guarded):
    repo = guarded.parents[1]
    cmd = "boautoresearch record review --file r.json"
    code, out, _ = hook(repo, "PreToolUse", tool_name="Bash", agent_type="boautoresearch:registration-reviewer",
                        agent_id="rr1", tool_input={"command": cmd, "description": "record"})
    assert code == 0
    assert json.loads(out) == {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": {
        "command": "boautoresearch record --agent-id rr1 --actor registration-reviewer review --file r.json",
        "description": "record"}}}
    code, out, _ = hook(repo, "SessionStart", source="startup")
    assert code == 0 and '"next"' in out and "round-run (the calibration round)" in out
    code, out, _ = hook(repo, "PostToolUse", tool_name="Bash", tool_input={"command": "boautoresearch status"},
                        tool_response={})
    assert code == 0
    context = json.loads(out)["hookSpecificOutput"]
    assert context["hookEventName"] == "PostToolUse" and "round-run (the calibration round)" in context["additionalContext"]
    assert hook(repo, "PostToolUse", tool_name="Bash", tool_input={"command": "ls"}, tool_response={}) == (0, "", "")


def fake_run(tmp_path, check_script):
    """A repository with a run whose venv's `boautoresearch` is `check_script`."""
    repo = tmp_path / "fake"
    repo.mkdir()
    git(repo, "init", "-q")
    run_dir = repo / ".bo-research" / "20260101-000000"
    (run_dir / "venv" / "bin").mkdir(parents=True)
    sqlite3.connect(run_dir / "log.db").executescript(
        "CREATE TABLE events (seq INTEGER PRIMARY KEY, type TEXT, schema_version INTEGER, ts TEXT, actor TEXT, payload TEXT);")
    exe = run_dir / "venv" / "bin" / "boautoresearch"
    exe.write_text(check_script)
    exe.chmod(0o755)
    return repo, run_dir


def test_the_adapter_fails_closed_when_the_check_crashes(tmp_path):
    repo, run_dir = fake_run(tmp_path, "#!/bin/sh\necho 'Traceback: boom' >&2\nexit 1\n")
    code, _, err = hook(repo, "PreToolUse", tool_name="Write", tool_input={"file_path": "x.py", "content": ""})
    assert code == 2 and "boom" in err
    code, _, err = hook(repo, "Stop")
    assert code == 2 and "boom" in err
    (run_dir / "venv" / "bin" / "boautoresearch").write_text("#!/bin/sh\necho not json\n")
    assert hook(repo, "Stop")[0] == 2
    (run_dir / "venv" / "bin" / "boautoresearch").write_text("#!/bin/sh\necho '{}'\n")  # JSON, but no answer
    assert hook(repo, "Stop")[0] == 2
    (run_dir / "venv" / "bin" / "boautoresearch").unlink()  # the run venv is gone mid-run
    assert hook(repo, "Stop")[0] == 2


def test_the_adapter_is_inert_without_an_active_run(tmp_path, repo):
    write = {"tool_name": "Write", "tool_input": {"file_path": "x.py", "content": ""}}
    assert hook(repo, "PreToolUse", **write) == (0, "", "")  # a repository with no run
    assert hook(tmp_path, "Stop") == (0, "", "")  # not a repository
    fake, run_dir = fake_run(tmp_path, "#!/bin/sh\nexit 1\n")
    con = sqlite3.connect(run_dir / "log.db")
    con.execute("INSERT INTO events (type, schema_version, ts, actor, payload) "
                "VALUES ('wrapup_finished', 1, '2026-01-01T00:00:00+00:00', 'harness', '{}')")
    con.commit()
    assert hook(fake, "PreToolUse", **write) == (0, "", "")  # wrapped up: the check isn't even asked
    assert hook(fake, "Stop") == (0, "", "")
