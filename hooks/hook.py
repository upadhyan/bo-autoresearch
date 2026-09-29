"""Claude Code hook adapter: hook JSON on stdin -> the run venv's `boautoresearch check …` -> exit code.

Standard library only, no decisions of its own. Inert (exit 0 at once) without an active run: no
`.bo-research/<run>/log.db` under the repository, or its wrap-up has finished. During a run it fails
closed: a check that crashes, times out or answers nonsense blocks (exit 2, the error on stderr).
"""
from __future__ import annotations  # the system python3 may be 3.9

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

TIMEOUT_S = 20  # hooks.json gives each hook 30 s


def active_run(cwd: str) -> Path | None:
    p = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                       cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        return None  # not a repository: nothing to enforce
    runs = sorted((Path(p.stdout.strip()).parent / ".bo-research").glob("*/log.db"))
    if not runs:
        return None
    con = sqlite3.connect(runs[-1])  # a read-only open can't join a WAL log with no -shm file yet
    try:
        done = con.execute("SELECT 1 FROM events WHERE type = 'wrapup_finished' LIMIT 1").fetchone()
    finally:
        con.close()
    return None if done else runs[-1].parent


def command(hook: dict) -> list[str] | None:
    """The harness call for this hook event, or None when there's nothing to ask."""
    event, tool, ti = hook.get("hook_event_name"), hook.get("tool_name"), hook.get("tool_input") or {}
    agent, agent_id = hook.get("agent_type") or "orchestrator", hook.get("agent_id")
    if event == "SessionStart":
        return ["status"]
    if event == "PostToolUse":
        return ["next"] if "boautoresearch" in ti.get("command", "") else None
    if event == "Stop":
        return ["check", "stop"]
    if event == "SubagentStop" or tool == "SubagentHandback":  # no agent_type: held like a recording role
        return ["check", "recorded", *(["--agent", hook["agent_type"]] if hook.get("agent_type") else []),
                "--", agent_id or ""]
    if tool in ("Edit", "Write", "NotebookEdit"):
        return ["check", "write", "--agent", agent, "--", ti.get("file_path") or ti.get("notebook_path") or ""]
    if tool == "Bash":
        return ["check", "bash", "--agent", agent, *(["--agent-id", agent_id] if agent_id else []),
                "--", ti.get("command", "")]
    if tool in ("Read", "Grep", "Glob"):  # a Glob pattern or Grep glob is part of what is read
        where = ti.get("file_path") or os.path.join(ti.get("path") or ".", ti.get("pattern" if tool == "Glob" else "glob") or "")
        return ["check", "read", "--agent", agent, "--", where]
    return None


def main() -> int:
    try:
        hook = json.load(sys.stdin)
        cwd = hook.get("cwd") or "."
        if (run_dir := active_run(cwd)) is None or (args := command(hook)) is None:
            return 0
        p = subprocess.run([str(run_dir / "venv" / "bin" / "boautoresearch"), *args], cwd=cwd,
                           capture_output=True, text=True, timeout=TIMEOUT_S)
        if p.returncode:  # a refusal (JSON) or a crash (a traceback)
            raise RuntimeError(p.stdout.strip() or p.stderr.strip())
        out = json.loads(p.stdout)
        if args[0] == "status":  # SessionStart: stdout becomes context
            print(f"boautoresearch run status (the harness's `next` list is what to do):\n{p.stdout}")
        elif args[0] == "next":
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                                     "additionalContext": f"boautoresearch next: {json.dumps(out['next'])}"}}))
        elif out["allow"] is not True:
            print(out["reason"], file=sys.stderr)
            return 2
        elif "updated_command" in out:  # the permission flow still applies to the rewritten call
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": {
                **hook["tool_input"], "command": out["updated_command"]}}}))
        return 0
    except Exception as e:  # fail closed: during a run, a check that can't answer (or answers nonsense) blocks
        print(f"boautoresearch hook failed, so it blocks: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
