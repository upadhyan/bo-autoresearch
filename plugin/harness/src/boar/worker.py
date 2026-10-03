"""Detached workers for `round run`, `finalize` and `eval check`, and `boar wait`.

A round (or one long baseline trial) outlives the agent's shell tool, so the work runs in its own
session and the agent polls with `boar wait`. running.json is the worker slot: it names the one process doing a
run's work, the detached worker or the CLI running it inline. A record whose process is gone means the worker was
killed, and re-running the same command resumes it.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from boar import evalrun, store
from boar.errors import Refused
from boar.store import Run

ROUND, FINALIZE, EVAL = "round run", "finalize", "eval check"
_SUBCOMMAND = {ROUND: "round", FINALIZE: "finalize", EVAL: "eval"}
_COMMAND = {"round": ROUND, ROUND: ROUND, "finalize": FINALIZE, "eval": EVAL, EVAL: EVAL}

# Popen handles of spawned workers, kept so they are not reaped (and warned about) by GC.
_children: list[subprocess.Popen] = []

# Tells a detached worker which CLI process spawned it: that pid holds the slot until the worker takes it over.
_PARENT_ENV = "BOAR_WORKER_PARENT"


def log_path(run: Run, command: str, r: int | None) -> Path:
    if command == ROUND:
        return run.round_dir(r) / "worker.log"
    return run.path("eval_check.log" if command == EVAL else "finalize.log")


def _record(pid: int, command: str, r: int | None, log: Path | None, started: str | None = None) -> dict:
    """running.json; `log` is None while a CLI holds the slot, since an inline run prints to its own shell."""
    return {
        "pid": pid, "identity": store.process_identity(pid), "command": command, "round": r,
        "started": started or store.now_iso(), "log": str(log) if log else None,
    }


def _busy(rec: dict) -> Refused:
    return Refused(
        f"`boar {rec.get('command')}` is still running in the background (pid {rec['pid']}); "
        "wait for it with `boar wait`"
    )


def kill_orphans(run: Run) -> list[int]:
    """SIGKILL the eval process group a dead worker left running; returns the pgids killed.

    Evals run in their own session, so killing a worker without cleanup (SIGKILL, OOM) leaves its eval
    measuring alongside whatever runs next. Call only while no live worker owns the slot.
    """
    rec = store.read_json(run.eval_group_path)
    if not rec:
        return []
    pgid, killed = rec.get("pgid"), []
    # A live leader must be the recorded eval (not a reused pid). A dead leader's pid can't be reused while its
    # group still has members, so signalling the group then reaches only what the eval left behind.
    if isinstance(pgid, int) and pgid > 1 and pgid != os.getpgrp() and (
        not store.pid_alive(pgid) or store.same_process(pgid, rec.get("identity"))
    ):
        try:
            os.killpg(pgid, signal.SIGKILL)
            killed.append(pgid)
        except (ProcessLookupError, PermissionError):
            pass
    with contextlib.suppress(FileNotFoundError):
        run.eval_group_path.unlink()
    return killed


@dataclass
class Claim:
    """This process's hold on the worker slot. `spawn` hands it to a detached worker."""

    run: Run
    command: str
    r: int | None
    rec: dict
    spawned: bool = False

    def spawn(self) -> dict:
        """Start `python -m boar _worker …` in its own session; running.json names it before this returns."""
        run, command, r = self.run, self.command, self.r
        log = log_path(run, command, r)
        log.parent.mkdir(parents=True, exist_ok=True)
        # sys.executable is the interpreter boar was imported by, so `-m boar` finds the same package.
        # cwd is the run dir, not the repo root, so a target repo's own `boar` module can't shadow it.
        # The environment is passed on (the worker drops _PARENT_ENV): the evals the worker starts inherit it.
        with open(log, "ab") as out:
            proc = subprocess.Popen(
                [sys.executable, "-m", "boar", "_worker", _SUBCOMMAND[command]],
                cwd=run.dir,
                env={**os.environ, _PARENT_ENV: str(os.getpid())},
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _children.append(proc)
        rec = _record(proc.pid, command, r, log, self.rec["started"])
        with run.locked():
            # The worker may already have taken the slot over (or even finished); only replace our own claim.
            if store.read_json(run.running_path) == self.rec:
                store.write_json(run.running_path, rec)
        self.spawned = True
        return rec


@contextlib.contextmanager
def claim(run: Run, command: str, r: int | None) -> Iterator[Claim]:
    """Take the worker slot for `command`, or refuse if a live process holds it.

    The check and the write are atomic, so two concurrent `round run`s can't both start. Until `spawn`, the
    slot names this process, and SIGTERM stops the eval it runs inline. On exit the record is removed unless
    it was handed to a worker (or is no longer ours).
    """
    with run.launch_locked(), run.locked():
        live = run.running()
        if live is not None:
            raise _busy(live)
        rec = _record(os.getpid(), command, r, None)
        store.write_json(run.running_path, rec)
    c = Claim(run, command, r, rec)
    old = None
    with contextlib.suppress(ValueError):  # not the main thread: no handler, as before
        old = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        kill_orphans(run)
        yield c
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
        if not c.spawned:
            with run.locked():
                if store.read_json(run.running_path) == rec:
                    with contextlib.suppress(FileNotFoundError):
                        run.running_path.unlink()


def spawn(run: Run, command: str, r: int | None) -> dict:
    """Claim the slot and start the detached worker for `command`."""
    with claim(run, command, r) as c:
        return c.spawn()


def _on_sigterm(signum, frame) -> None:
    evalrun.request_stop()


def worker_main(command: str) -> int:
    """Body of the hidden `boar _worker round|finalize|eval`; cwd is inside the run dir."""
    command = _COMMAND.get(command, command)
    if command not in _SUBCOMMAND:
        print(f"unknown worker command {command!r}", file=sys.stderr)
        return 2
    run = store.require_run()
    signal.signal(signal.SIGTERM, _on_sigterm)
    pid = os.getpid()
    parent = os.environ.pop(_PARENT_ENV, "")
    parent_pid = int(parent) if parent.isdigit() else None
    r = run.state()["round"] if command == ROUND else None
    with run.locked():
        prev = run.running()
        if prev is not None and prev["pid"] not in (pid, parent_pid):
            print(f"[{store.now_iso()}] another worker (pid {prev['pid']}) holds this run; exiting", flush=True)
            return 1
        prev = prev or {}
        store.write_json(run.running_path, _record(
            pid, command, r, Path(prev.get("log") or log_path(run, command, r)), prev.get("started")
        ))
    print(f"[{store.now_iso()}] boar {command} worker started (pid {pid})", flush=True)
    try:
        killed = kill_orphans(run)
        if killed:
            print(f"[{store.now_iso()}] killed eval process groups a dead worker left running: {killed}", flush=True)
        if command == ROUND:
            from boar import rounds

            rounds.run_round_worker(run)
        elif command == EVAL:
            from boar import control

            print(control.eval_check_body(run)[0], flush=True)
        else:
            from boar import finalize

            finalize.run_finalize_worker(run)
        return 0
    except evalrun.Interrupted:
        print(f"[{store.now_iso()}] stopped by SIGTERM; `boar {command}` resumes it", flush=True)
        return 143
    except Refused as e:
        print(f"[{store.now_iso()}] refused: {e}", flush=True)
        return 1
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        return 1
    finally:
        with run.locked():
            cur = store.read_json(run.running_path)
            if cur and cur.get("pid") == pid:
                with contextlib.suppress(FileNotFoundError):
                    run.running_path.unlink()


def _log_tail(path: Path, n: int = 15) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join("    " + line for line in lines[-n:])


def progress(run: Run, command: str, r: int | None) -> str:
    cfg = run.config()
    if command == EVAL:
        return f"eval check, one baseline trial of {cfg['repeats']} repeats on dev"
    if command == ROUND:
        k = sum(1 for t in run.trials() if t["round"] == r)
        return f"round {r}, {k}/{cfg['trials_per_round']} trials done"
    k = len(store.read_jsonl(run.holdout_path))
    return f"finalize, {k}/{2 * cfg['holdout_repeats']} holdout runs done"


def wait(run: Run, timeout_s: float) -> str:
    """Block until no worker is alive or `timeout_s` passes; say what happened in one line (plus a log tail on failure)."""
    deadline = time.monotonic() + timeout_s
    rec = run.running()
    # The worker this wait is about: the live one, else a record left by a worker that was killed.
    seen = rec or store.read_json(run.running_path) or {}
    while rec is not None and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        rec = run.running()
    if rec is not None:
        return f"still running: {progress(run, rec['command'], rec.get('round'))}; call `boar wait` again"
    state = run.state()
    phase, r = state["phase"], state["round"]
    if phase == store.ABORTED:
        return f"nothing is running: the run was aborted ({state.get('abort_reason')})"
    if seen.get("command") == EVAL or (not seen and phase == store.SETUP):
        return _eval_finished(run, seen)
    fin = state.get("finalize") or {}
    if phase in (store.FINALIZE, store.DONE) or fin.get("status") not in (None, "not_started"):
        if fin.get("status") == "done":
            return _with_next(run, f"finished: finalize (report: {run.report_path})")
        if fin.get("status") == "running":
            return _died(run, FINALIZE, None, seen)
        return "nothing is running"
    rr = state.get("rounds", {}).get(str(r)) or {}
    status = rr.get("run_status")
    if status == "done":
        return _with_next(run, f"finished: round {r} (summary: {run.summary_path(r)})")
    if status == "running" and phase == store.ROUND:
        return _died(run, ROUND, r, seen)
    return "nothing is running"


def _with_next(run: Run, text: str) -> str:
    try:
        from boar import nextstep

        return f"{text}, next: {nextstep.next_line(run)}"
    except Exception:
        return text


def _eval_finished(run: Run, seen: dict) -> str:
    started = seen.get("started")
    check = store.read_json(run.eval_check_path)
    if not check or (started and str(check.get("at", "")) < started):
        if started:
            return _died(run, EVAL, None, seen)
        return "nothing is running"
    trial = check.get("trial") or {}
    verdict = "PASS" if check.get("ok") else "FAIL"
    text = (
        f"finished: eval check {verdict} (baseline {trial.get('state')}, metric {trial.get('metric')}, "
        f"{trial.get('duration_s')} s for all repeats, limit {check.get('limit_s')} s)"
    )
    problems = check.get("problems") or []
    if problems:
        text += "; problems: " + "; ".join(problems)
    warnings = check.get("warnings") or []
    if warnings:
        text += "; warning: " + "; ".join(warnings)
    return _with_next(run, text)


def _died(run: Run, command: str, r: int | None, seen: dict) -> str:
    """`seen` is the slot's last record ({} once a clean exit removed it)."""
    killed = kill_orphans(run)
    orphans = f" Killed the eval it left running (process group {', '.join(map(str, killed))})." if killed else ""
    again = f"{'re-run' if command == EVAL else 'resume with'} `boar {command}`"
    log = log_path(run, command, r)
    if (seen and not seen.get("log")) or not log.exists():  # only a detached worker writes a log
        pid = f", pid {seen['pid']}" if seen.get("pid") else ""
        return (
            f"`boar {command}` stopped before finishing ({progress(run, command, r)}).{orphans} It ran inline "
            f"(--foreground{pid}), so its output went to the shell that started it. Fix the "
            f"cause if that output shows one, then {again}."
        )
    return (
        f"`boar {command}` stopped before finishing ({progress(run, command, r)}).{orphans} Last log lines "
        f"({log}):\n{_log_tail(log)}\nFix the cause if the log shows one, then {again}."
    )
