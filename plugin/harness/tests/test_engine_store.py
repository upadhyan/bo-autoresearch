"""store and worker: the jsonl files, the study storage URL, and the worker slot (running.json, orphaned evals)."""

from __future__ import annotations

import os
import signal
import subprocess
import time

import optuna
import pytest

from boar import store, warmstart, worker
from boar.errors import Refused
from boar.store import Run


@pytest.fixture
def run(tmp_path, monkeypatch) -> Run:
    r = Run(root=tmp_path / "repo", id="run1")
    r.dir.mkdir(parents=True)
    r.save_state({"phase": "round", "round": 1})
    (r.root / ".boar" / "active").write_text("run1\n")
    monkeypatch.chdir(r.root)
    return r


@pytest.fixture
def sleeper():
    procs = []

    def start() -> subprocess.Popen:
        p = subprocess.Popen(["sleep", "30"], start_new_session=True)
        procs.append(p)
        return p

    yield start
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()


def _gone(p: subprocess.Popen, timeout: float = 5.0) -> bool:
    try:
        p.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


# --- jsonl --------------------------------------------------------------------------------


def test_torn_last_line_is_skipped_on_read_and_cut_on_the_next_append(tmp_path):
    path = tmp_path / "trials.jsonl"
    store.append_jsonl(path, {"trial": 1})
    with open(path, "a") as f:
        f.write('{"trial": 2, "state": "compl')  # a writer died mid-append
    assert store.read_jsonl(path) == [{"trial": 1}]
    store.append_jsonl(path, {"trial": 2})
    assert path.read_text() == '{"trial": 1}\n{"trial": 2}\n'
    assert store.read_jsonl(path) == [{"trial": 1}, {"trial": 2}]


def test_complete_last_line_without_newline_is_kept(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}')
    assert store.read_jsonl(path) == [{"a": 1}, {"a": 2}]
    store.append_jsonl(path, {"a": 3})
    assert store.read_jsonl(path) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_a_bad_line_in_the_middle_still_raises(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{"a": \n{"a": 3}\n')
    with pytest.raises(ValueError):
        store.read_jsonl(path)


# --- study storage --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["proj?v1", "pct%41dir", "sp ace#x"])
def test_storage_url_keeps_the_study_in_the_run_dir(tmp_path, name):
    r = Run(root=tmp_path / name, id="run1")
    r.dir.mkdir(parents=True)
    warmstart.quiet_optuna()
    optuna.create_study(study_name="round-1", storage=r.storage_url)
    assert r.study_db.exists()
    assert optuna.load_study(study_name="round-1", storage=r.storage_url).study_name == "round-1"


# --- the worker slot ----------------------------------------------------------------------


def test_an_unreaped_child_counts_as_dead():
    """A zombie still answers kill(0); pid_alive must see through it with /proc or, without it (macOS), ps."""
    child = subprocess.Popen(["true"])
    deadline = time.monotonic() + 5
    while store.pid_alive(child.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not store.pid_alive(child.pid)
    child.wait()


def test_running_needs_the_recorded_process_not_just_a_live_pid(run, sleeper):
    other = sleeper()
    store.write_json(run.running_path, {"pid": other.pid, "command": "round run", "round": 1})
    assert run.running() is None, "a live pid that isn't boar (a reused pid) is not the worker"
    store.write_json(run.running_path, {"pid": other.pid, "identity": {"start": "1", "boot_id": "x"},
                                        "command": "round run", "round": 1})
    assert run.running() is None, "a pid whose start time doesn't match is not the worker"
    rec = {"pid": other.pid, "identity": store.process_identity(other.pid), "command": "round run", "round": 1}
    store.write_json(run.running_path, rec)
    assert run.running() == rec
    other.kill()
    other.wait()
    assert run.running() is None


def test_claim_registers_this_process_and_refuses_a_second_claim(run):
    with worker.claim(run, worker.ROUND, 1) as c:
        rec = store.read_json(run.running_path)
        assert rec["pid"] == os.getpid() and rec["identity"] == store.process_identity(os.getpid())
        assert rec == c.rec and run.running() == rec
        with pytest.raises(Refused, match="still running in the background"), worker.claim(run, worker.ROUND, 1):
            pass
        with pytest.raises(Refused, match="still running in the background"):
            store.require_live_run()
        assert store.read_json(run.running_path) == rec, "a refused claim leaves the holder's record alone"
    assert not run.running_path.exists()


def test_claim_installs_a_sigterm_handler_only_while_held(run):
    before = signal.getsignal(signal.SIGTERM)
    with worker.claim(run, worker.EVAL, None):
        assert signal.getsignal(signal.SIGTERM) is worker._on_sigterm
    assert signal.getsignal(signal.SIGTERM) is before


def test_claim_exit_leaves_a_record_that_is_no_longer_its_own(run, sleeper):
    other = sleeper()
    with worker.claim(run, worker.ROUND, 1):
        theirs = {"pid": other.pid, "identity": store.process_identity(other.pid), "command": "round run"}
        store.write_json(run.running_path, theirs)
    assert store.read_json(run.running_path) == theirs


def test_worker_refuses_to_start_while_another_worker_holds_the_run(run, sleeper, capsys):
    other = sleeper()
    theirs = {"pid": other.pid, "identity": store.process_identity(other.pid), "command": "round run", "round": 1}
    store.write_json(run.running_path, theirs)
    assert worker.worker_main("round") == 1
    assert "another worker" in capsys.readouterr().out
    assert store.read_json(run.running_path) == theirs


def test_kill_orphans_kills_a_dead_workers_eval_group(run, sleeper):
    orphan = sleeper()
    store.write_json(run.eval_group_path, {"pgid": orphan.pid, "identity": store.process_identity(orphan.pid)})
    assert worker.kill_orphans(run) == [orphan.pid]
    assert _gone(orphan)
    assert not run.eval_group_path.exists()
    assert worker.kill_orphans(run) == []


def test_kill_orphans_spares_a_reused_pid(run, sleeper):
    other = sleeper()
    store.write_json(run.eval_group_path, {"pgid": other.pid, "identity": {"start": "1", "boot_id": "x"}})
    assert worker.kill_orphans(run) == []
    assert other.poll() is None
    assert not run.eval_group_path.exists()


def test_claim_kills_an_orphaned_eval_before_new_work(run, sleeper):
    orphan = sleeper()
    store.write_json(run.eval_group_path, {"pgid": orphan.pid, "identity": store.process_identity(orphan.pid)})
    with worker.claim(run, worker.ROUND, 1):
        assert _gone(orphan)


def test_wait_died_path_kills_the_orphaned_eval(run, sleeper):
    orphan = sleeper()
    run.update_state(lambda s: s.update(rounds={"1": {"run_status": "running"}}))
    store.write_json(run.config_path, {"trials_per_round": 4})
    store.write_json(run.eval_group_path, {"pgid": orphan.pid, "identity": store.process_identity(orphan.pid)})
    out = worker.wait(run, 0)
    assert "stopped before finishing" in out and f"process group {orphan.pid}" in out
    assert _gone(orphan)


def test_spawned_worker_takes_over_the_slot(run):
    started = time.monotonic()
    with worker.claim(run, worker.EVAL, None) as c:
        rec = c.spawn()
    assert rec["pid"] != os.getpid() and rec["identity"] == store.process_identity(rec["pid"])
    # The worker exits at once (no config to run an eval check with); the slot never names the CLI again.
    cur = store.read_json(run.running_path)
    assert cur is None or cur["pid"] == rec["pid"]
    deadline = started + 20
    while store.pid_alive(rec["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not run.running_path.exists()
    assert "worker started" in (run.path("eval_check.log")).read_text()
