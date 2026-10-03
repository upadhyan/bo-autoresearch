"""evalrun: the eval contract, repeats, hang kill, the eval hash and `eval check`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from boar import evalrun, store
from boar.store import Run

OK_LINE = '{"metric": 2.5, "guards_ok": true, "metrics": {"rss": 10}}'


@pytest.fixture(autouse=True)
def _reset_stop_flag():
    yield
    evalrun._live["stop"] = False
    evalrun._live["proc"] = None


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def run(tmp_path, monkeypatch) -> Run:
    """A run directory in a one-commit git repo (`eval check` compares the repo before and after), no hypotheses."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / ".git" / "info" / "exclude").write_text("/.boar/\n")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "--allow-empty", "-m", "empty")
    r = Run(root=tmp_path, id="run1")
    store.write_json(r.config_path, {"rounds": 2, "trials_per_round": 3, "repeats": 3, "trial_target_s": 5,
                                     "holdout_repeats": 2, "max_active": 6, "seed": 0, "direction": "min"})
    store.write_json(r.hypotheses_path, [])
    return r


def write_eval(run: Run, body: str, mode: int = 0o755) -> Path:
    exe = run.eval_dir / "run"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(mode)
    return exe


# A counter in the run dir lets an eval behave differently on each repeat.
COUNT = 'n=$(cat "$BOAR_RUN_DIR/count" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$BOAR_RUN_DIR/count"\n'


def trial(run: Run, config=None, repeats=3, budget=5.0, split="dev"):
    return evalrun.run_trial(run, config or {"x": 1}, split, repeats, run.path("trials", "1"), budget)


def test_complete_trial_records_repeats_files_and_environment(run):
    write_eval(run, COUNT + 'pwd > "$BOAR_RUN_DIR/cwd"\ncp "$BOAR_CONFIG" "$BOAR_RUN_DIR/seen.json"\n'
               'echo "progress: not the result"\necho "warming up" >&2\n'
               'echo "{\\"metric\\": $n, \\"guards_ok\\": true, \\"metrics\\": {\\"rss\\": $((n*10)), \\"split\\": \\"$BOAR_SPLIT\\"}}"\n')
    res = trial(run, {"x": 4, "flag": True}, split="holdout")
    assert res["state"] == "complete" and res["error"] is None
    assert res["repeats"] == [1.0, 2.0, 3.0] and res["metric"] == 2.0
    assert res["metrics"] == {"rss": 20, "split": "holdout"}
    assert Path(run.path("cwd").read_text().strip()) == run.root
    assert json.loads(run.path("seen.json").read_text()) == {"x": 4, "flag": True}
    out = run.path("trials", "1")
    assert json.loads((out / "config.json").read_text()) == {"x": 4, "flag": True}
    assert (out / "repeat-3.stdout").read_text().startswith("progress")
    assert (out / "repeat-1.stderr").read_text() == "warming up\n"


def test_infeasible_stops_at_first_guard_failure(run):
    write_eval(run, COUNT + 'g=true; [ $n -ge 2 ] && g=false\necho "{\\"metric\\": $n, \\"guards_ok\\": $g}"\n')
    res = trial(run)
    assert res["state"] == "infeasible" and res["error"] is None
    assert res["repeats"] == [1.0, 2.0] and res["metric"] == 1.5
    assert not run.path("trials", "1", "repeat-3.stdout").exists()


@pytest.mark.parametrize(
    "line, error",
    [
        ("not json at all", "unparseable output"),
        ('["a list"]', "unparseable output"),
        ('{"metric": NaN, "guards_ok": true}', "metric not a finite number"),
        ('{"metric": Infinity, "guards_ok": true}', "metric not a finite number"),
        ('{"metric": "fast", "guards_ok": true}', "metric not a finite number"),
        ('{"metric": true, "guards_ok": true}', "metric not a finite number"),
        ('{"metric": 1.0}', "guards_ok missing"),
        ('{"metric": 1.0, "guards_ok": "yes"}', "guards_ok missing"),
        ('{"metric": 1.0, "guards_ok": true, "metrics": [1]}', "metrics is not a JSON object"),
    ],
)
def test_contract_violations_fail_the_trial_at_once(run, line, error):
    write_eval(run, COUNT + f"echo '{line}'\n")
    res = trial(run)
    assert res["state"] == "failed" and res["error"].startswith(error)
    assert run.path("count").read_text().strip() == "1"


def test_empty_stdout_fails(run):
    write_eval(run, "true\n")
    assert trial(run)["error"] == "unparseable output: stdout is empty"


def test_non_zero_exit_fails_even_with_a_valid_line(run):
    write_eval(run, COUNT + f"echo '{OK_LINE}'\n[ $n -ge 2 ] && exit 3\nexit 0\n")
    res = trial(run)
    assert res["state"] == "failed" and res["error"] == "exit 3"
    assert res["repeats"] == [2.5] and res["metric"] == 2.5


def test_missing_or_non_executable_eval_fails(run):
    assert "not found" in trial(run)["error"]
    write_eval(run, f"echo '{OK_LINE}'\n", mode=0o644)
    res = trial(run)
    assert res["state"] == "failed" and "not executable" in res["error"]


def test_hang_is_killed_with_its_whole_process_group(run):
    write_eval(run, 'sleep 30 &\necho $! > "$BOAR_RUN_DIR/child.pid"\nwait\n')
    start = time.monotonic()
    res = trial(run, repeats=3, budget=0.5)
    assert time.monotonic() - start < 5
    assert res["state"] == "failed" and res["error"] == "timeout after 0.5s"
    child = int(run.path("child.pid").read_text())
    time.sleep(0.1)
    assert not store.pid_alive(child)


def test_repeats_share_one_budget(run):
    write_eval(run, f"sleep 0.3\necho '{OK_LINE}'\n")
    res = trial(run, repeats=3, budget=0.75)
    assert res["state"] == "failed" and res["error"].startswith("timeout")
    assert len(res["repeats"]) == 2


def test_request_stop_kills_the_running_eval_and_interrupts(run):
    write_eval(run, 'sleep 30 &\necho $! > "$BOAR_RUN_DIR/child.pid"\nwait\n')
    caught = {}

    def go():
        try:
            trial(run, budget=20)
        except BaseException as e:  # noqa: BLE001 - the test inspects what was raised
            caught["e"] = e

    t = threading.Thread(target=go)
    t.start()
    deadline = time.monotonic() + 5
    while not run.path("child.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    evalrun.request_stop()
    t.join(5)
    assert isinstance(caught.get("e"), evalrun.Interrupted)
    time.sleep(0.1)
    assert not store.pid_alive(int(run.path("child.pid").read_text()))
    with pytest.raises(evalrun.Interrupted):
        trial(run)


needs_proc = pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="the concurrent-eval check reads /proc")


@pytest.fixture
def fast_poll(monkeypatch):
    monkeypatch.setattr(evalrun, "CONCURRENT_POLL_S", 0.1)


def bystander(cwd: Path, **env: str) -> subprocess.Popen:
    """A long-running process: a hand-run eval when it has BOAR_CONFIG and runs in the repo."""
    base = {k: v for k, v in os.environ.items() if k != "BOAR_CONFIG"}
    return subprocess.Popen(["sleep", "30"], cwd=cwd, env={**base, **env})


@needs_proc
def test_a_hand_run_eval_alongside_the_trial_stops_it_unrecorded(run, fast_poll):
    write_eval(run, f"sleep 5\necho '{OK_LINE}'\n")
    other = bystander(run.root, BOAR_CONFIG="hand.json")
    try:
        start = time.monotonic()
        with pytest.raises(evalrun.ConcurrentEval, match=f"pid {other.pid}"):
            trial(run, budget=30)
        assert time.monotonic() - start < 3
    finally:
        other.kill()
        other.wait()
    assert not run.eval_group_path.exists()


@needs_proc
@pytest.mark.parametrize("where, env", [("elsewhere", {"BOAR_CONFIG": "x.json"}), ("repo", {})])
def test_other_processes_do_not_stop_the_trial(run, tmp_path_factory, fast_poll, where, env):
    write_eval(run, f"sleep 0.5\necho '{OK_LINE}'\n")
    other = bystander(run.root if where == "repo" else tmp_path_factory.mktemp("elsewhere"), **env)
    try:
        assert trial(run, repeats=1)["state"] == "complete"
    finally:
        other.kill()
        other.wait()


@needs_proc
def test_the_trials_own_children_do_not_stop_it(run, fast_poll):
    # A child in a session of its own: outside the eval's process group, but still its descendant.
    write_eval(run, f"{sys.executable} -c 'import os, time; os.setsid(); time.sleep(0.6)' &\nwait\necho '{OK_LINE}'\n")
    assert trial(run, repeats=1)["state"] == "complete"


def test_eval_hash_ignores_bytecode_but_tracks_edits(run):
    assert evalrun.eval_hash(run) is None
    write_eval(run, f"echo '{OK_LINE}'\n")
    (run.eval_dir / "lib").mkdir()
    (run.eval_dir / "lib" / "helper.py").write_text("X = 1\n")
    h0 = evalrun.eval_hash(run)
    (run.eval_dir / "lib" / "__pycache__").mkdir()
    (run.eval_dir / "lib" / "__pycache__" / "helper.cpython-314.pyc").write_bytes(b"\x00junk")
    (run.eval_dir / "stray.pyc").write_bytes(b"\x01")
    assert evalrun.eval_hash(run) == h0
    (run.eval_dir / "lib" / "helper.py").write_text("X = 2\n")
    h1 = evalrun.eval_hash(run)
    assert h1 != h0
    (run.eval_dir / "lib" / "helper.py").rename(run.eval_dir / "lib" / "other.py")
    assert evalrun.eval_hash(run) not in (h0, h1)


def test_eval_check_passes_and_records_hash(run):
    write_eval(run, f"echo '{OK_LINE}'\n")
    res = evalrun.eval_check(run)
    assert res["ok"] and res["problems"] == [] and res["hash"] == evalrun.eval_hash(run)
    assert res["limit_s"] == 10 and res["trial"]["state"] == "complete" and len(res["trial"]["repeats"]) == 3
    assert store.read_json(run.eval_check_path)["ok"] is True
    assert store.read_json(run.eval_check_path)["manifest"] == evalrun.eval_manifest(run)
    assert (run.path("eval_check") / "repeat-1.stdout").exists()


def test_eval_check_passes_baseline_config_of_all_levers(run):
    store.write_json(run.hypotheses_path, [
        {"id": "H1", "status": "pending", "levers": [{"name": "w", "type": "int", "low": 1, "high": 4, "default": 1}]},
        {"id": "H2", "status": "rejected", "levers": [{"name": "f", "type": "bool", "default": False}]},
    ])
    write_eval(run, f'cp "$BOAR_CONFIG" "$BOAR_RUN_DIR/seen.json"\necho \'{OK_LINE}\'\n')
    assert evalrun.eval_check(run)["ok"]
    assert json.loads(run.path("seen.json").read_text()) == {"w": 1, "f": False}


def test_eval_check_reports_guard_failure_slowness_and_self_modification(run):
    write_eval(run, "echo '{\"metric\": 1, \"guards_ok\": false}'\n")
    res = evalrun.eval_check(run)
    assert not res["ok"] and any("guards fail at baseline" in p for p in res["problems"])

    cfg = run.config()
    store.write_json(run.config_path, {**cfg, "trial_target_s": 0.5})
    write_eval(run, f"sleep 0.4\necho '{OK_LINE}'\n")
    res = evalrun.eval_check(run)
    assert res["trial"]["state"] == "complete"
    assert not res["ok"] and any("more than 2 × trial_target_s" in p for p in res["problems"])

    store.write_json(run.config_path, cfg)
    write_eval(run, f'date >> "$0.log"\necho \'{OK_LINE}\'\n')
    res = evalrun.eval_check(run)
    assert not res["ok"] and "eval/ changed during the check: added: run.log; the eval must not write inside eval/" in res["problems"]


def test_eval_check_reports_contract_problem(run):
    res = evalrun.eval_check(run)
    assert not res["ok"] and res["hash"] is None and "no eval directory" in res["problems"][0]
    write_eval(run, "echo nope\n")
    res = evalrun.eval_check(run)
    assert not res["ok"] and res["problems"][0].startswith("contract: unparseable output")


def test_bytecode_written_by_a_python_eval_does_not_change_the_hash(run, monkeypatch):
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    (run.eval_dir).mkdir(parents=True)
    (run.eval_dir / "helper.py").write_text("VALUE = 3.0\n")
    exe = run.eval_dir / "run"
    exe.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
        "import helper\nprint(json.dumps({'metric': helper.VALUE, 'guards_ok': True}))\n"
    )
    exe.chmod(0o755)
    before = evalrun.eval_hash(run)
    assert evalrun.eval_check(run)["ok"]
    assert (run.eval_dir / "__pycache__").is_dir()
    assert evalrun.eval_hash(run) == before


def test_huge_integer_metric_fails_the_trial_instead_of_raising(run):
    huge = "1" + "0" * 400
    assert evalrun.parse_result(f'{{"metric": {huge}, "guards_ok": true}}') == (None, "metric not a finite number")
    _, err = evalrun.parse_result(f'{{"metric": 1, "guards_ok": true, "metrics": {{"big": {huge}, "x": 2}}}}')
    assert err is None
    write_eval(run, f"echo '{{\"metric\": 1, \"guards_ok\": true, \"metrics\": {{\"big\": {huge}}}}}'\n")
    res = trial(run, repeats=2)
    assert res["state"] == "complete" and res["metrics"]["big"] == int(huge)


@pytest.mark.parametrize("stdout", [
    "working 99%\r" + OK_LINE + "\n",
    "\r 10/10\r   \r" + OK_LINE + "\n",
    "progress\n" + OK_LINE + "\r\n",
])
def test_a_progress_line_overwritten_by_the_result_still_parses(stdout):
    parsed, err = evalrun.parse_result(stdout)
    assert err is None and parsed["metric"] == 2.5


def test_result_line_may_hold_raw_line_separators_inside_strings():
    line = '{"metric": 1, "guards_ok": true, "metrics": {"note": "a b\u0085c"}}'
    parsed, err = evalrun.parse_result("progress\n" + line + "\r\n")
    assert err is None and parsed["metrics"] == {"note": "a b\u0085c"}


def test_eval_group_is_recorded_while_the_eval_runs(run):
    write_eval(run, f'cat "$BOAR_RUN_DIR/eval_group.json" > "$BOAR_RUN_DIR/seen-group"\necho \'{OK_LINE}\'\n')
    trial(run, repeats=1)
    seen = json.loads(run.path("seen-group").read_text())
    assert isinstance(seen["pgid"], int) and seen["identity"]
    assert not run.eval_group_path.exists(), "cleared once the eval is done"


def test_symlinked_directory_in_eval_is_part_of_the_hash(run, tmp_path):
    write_eval(run, f"echo '{OK_LINE}'\n")
    for d in ("dev_a", "dev_b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "w1.txt").write_text(d)
    (run.eval_dir / "dev").symlink_to(tmp_path / "dev_a")
    h0 = evalrun.eval_hash(run)
    assert evalrun.eval_manifest(run)["dev"] == f"symlink:{tmp_path / 'dev_a'}"
    (run.eval_dir / "dev").unlink()
    (run.eval_dir / "dev").symlink_to(tmp_path / "dev_b")
    assert evalrun.eval_hash(run) != h0, "repointing the link changes the hash"
    (run.eval_dir / "dev").unlink()
    assert evalrun.eval_hash(run) != h0, "removing the link changes the hash"


def test_eval_manifest_and_diff_name_each_changed_file(run):
    assert evalrun.eval_manifest(run) is None
    write_eval(run, f"echo '{OK_LINE}'\n")
    (run.eval_dir / "data").mkdir()
    (run.eval_dir / "data" / "x.json").write_text("{}")
    (run.eval_dir / "gone.txt").write_text("bye")
    (run.eval_dir / "__pycache__").mkdir()
    (run.eval_dir / "__pycache__" / "m.pyc").write_bytes(b"\0")
    old, h0 = evalrun.eval_manifest(run), evalrun.eval_hash(run)
    assert set(old) == {"run", "data/x.json", "gone.txt"}
    assert evalrun.eval_diff(old, evalrun.eval_manifest(run)) == []
    (run.eval_dir / "data" / "x.json").write_text('{"tampered": 1}')
    (run.eval_dir / "gone.txt").unlink()
    (run.eval_dir / ".DS_Store").write_bytes(b"\1")
    new = evalrun.eval_manifest(run)
    assert evalrun.eval_hash(run) != h0
    assert evalrun.eval_diff(old, new) == ["added: .DS_Store", "changed: data/x.json", "removed: gone.txt"]
    assert evalrun.eval_diff(old, None) == ["removed: data/x.json", "removed: gone.txt", "removed: run"]
    assert evalrun.diff_text([f"added: f{i}" for i in range(12)], limit=2) == "added: f0; added: f1; … and 10 more"


def test_evals_do_not_inherit_the_harness_virtualenv(monkeypatch):
    """Under `uv run boar` the harness venv is active; the eval must see the user's interpreter, not ours."""
    ours = os.path.join(sys.prefix, "bin")
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    monkeypatch.setenv("PATH", os.pathsep.join([ours, "/usr/local/bin", "/usr/bin"]))
    env = evalrun.eval_environ()
    assert "VIRTUAL_ENV" not in env and env["PATH"] == os.pathsep.join(["/usr/local/bin", "/usr/bin"])
    monkeypatch.setenv("VIRTUAL_ENV", "/home/u/project/.venv")
    monkeypatch.setenv("PATH", os.pathsep.join(["/home/u/project/.venv/bin", "/usr/bin"]))
    env = evalrun.eval_environ()
    assert env["VIRTUAL_ENV"] == "/home/u/project/.venv", "the user's own venv stays"
    assert env["PATH"] == os.pathsep.join(["/home/u/project/.venv/bin", "/usr/bin"])


def test_eval_check_fails_and_puts_back_what_the_eval_wrote_into_the_repo(run):
    (run.root / "tracked.csv").write_text("committed\n")
    git(run.root, "add", "tracked.csv")
    git(run.root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "csv")
    (run.root / "draft.md").write_text("the agent's own untracked file\n")
    write_eval(run, f'echo row >> tracked.csv\nmkdir -p out && date > out/log.txt\necho \'{OK_LINE}\'\n')
    res = evalrun.eval_check(run)
    assert not res["ok"]
    assert any(p.startswith("the eval wrote into the repo: out/log.txt, tracked.csv; write runtime outputs to a temp dir")
               for p in res["problems"]), res["problems"]
    assert (run.root / "tracked.csv").read_text() == "committed\n" and not (run.root / "out").exists()
    [keep] = run.path("eval_check-repo-writes").iterdir()
    assert any(f"put back: out/log.txt, tracked.csv; new files moved to {keep})" in p for p in res["problems"])
    assert (keep / "out" / "log.txt").exists() and "row" in (keep / "tracked.csv").read_text()
    assert (run.root / "draft.md").exists(), "a file that was there before the check is not the eval's"


def _commit_out_dir(run: Run) -> None:
    (run.root / "out").mkdir()
    (run.root / "out" / "a.txt").write_text("a\n")
    (run.root / "out" / "b.txt").write_text("b\n")
    git(run.root, "add", "out")
    git(run.root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "out")


def test_eval_check_puts_back_a_tracked_directory_the_eval_replaced_with_a_file(run):
    """rmtree(out) then a file `out`: the tracked out/a.txt, out/b.txt come back, not just get named."""
    _commit_out_dir(run)
    write_eval(run, f"rm -rf out && echo results > out\necho '{OK_LINE}'\n")
    res = evalrun.eval_check(run)
    assert not res["ok"]
    [keep] = run.path("eval_check-repo-writes").iterdir()
    assert any(p.startswith("the eval wrote into the repo: out, out/a.txt, out/b.txt;")
               and "put back: out, out/a.txt, out/b.txt;" in p and "still changed" not in p
               for p in res["problems"]), res["problems"]
    assert (run.root / "out" / "a.txt").read_text() == "a\n" and (run.root / "out" / "b.txt").read_text() == "b\n"
    assert (keep / "out").read_text() == "results\n"
    assert "out/a.txt" not in git(run.root, "status", "--porcelain")


def test_eval_check_names_what_it_could_not_put_back(run):
    """It never claims "put back" for a path still changed; that path is named with how to restore it."""
    _commit_out_dir(run)
    (run.root / "out" / "b.txt").write_text("edited\n")  # already uncommitted before the check: left alone by design
    write_eval(run, f"rm -rf out && echo results > out\necho '{OK_LINE}'\n")
    res = evalrun.eval_check(run)
    [problem] = [p for p in res["problems"] if p.startswith("the eval wrote into the repo")]
    assert "put back: out, out/a.txt;" in problem and "still changed: out/b.txt; restore them" in problem, problem
    assert (run.root / "out" / "a.txt").read_text() == "a\n"


def test_a_second_eval_check_keeps_what_the_first_moved_aside(run):
    write_eval(run, f"date > results.txt\necho '{OK_LINE}'\n")
    assert not evalrun.eval_check(run)["ok"]
    write_eval(run, f"echo '{OK_LINE}'\n")
    assert evalrun.eval_check(run)["ok"]
    [keep] = run.path("eval_check-repo-writes").iterdir()
    assert (keep / "results.txt").exists(), "moved, never deleted"


def test_eval_check_warns_about_tracked_changes_round_1_would_commit(run):
    _commit_out_dir(run)
    (run.root / "out" / "a.txt").unlink()
    write_eval(run, f"echo '{OK_LINE}'\n")
    res = evalrun.eval_check(run)
    assert res["ok"], res["problems"]
    assert any(w.startswith("tracked files with uncommitted changes (edited or deleted) that round 1's commit will "
                            "record on the run branch: out/a.txt;") for w in res["warnings"]), res["warnings"]


def test_eval_check_ignores_metadata_only_touches_of_an_uncommitted_file(run, tmp_path):
    """A same-mode chmod or a hard link made and removed elsewhere changes only ctime: nothing was written."""
    (run.root / "bench.sh").write_text("#!/bin/sh\n")
    (run.root / "bench.sh").chmod(0o755)
    link = tmp_path / "sandbox-link"
    write_eval(run, f"chmod 755 bench.sh && ln bench.sh {link} && rm {link}\necho '{OK_LINE}'\n")
    res = evalrun.eval_check(run)
    assert res["ok"], res["problems"]


def test_eval_check_fails_when_the_eval_rewrites_a_file_that_was_already_uncommitted(run):
    """A manual run before the check leaves results.txt untracked; round 1's commit would sweep it in."""
    (run.root / "results.txt").write_text("from a manual run\n")
    (run.root / "draft.md").write_text("the agent's own untracked file\n")
    write_eval(run, f'echo row >> results.txt\necho \'{OK_LINE}\'\n')
    res = evalrun.eval_check(run)
    assert not res["ok"]
    assert any(p.startswith("the eval wrote into the repo: results.txt; write runtime outputs to a temp dir")
               and "already uncommitted" in p for p in res["problems"]), res["problems"]
    assert "row" in (run.root / "results.txt").read_text(), "its earlier content is unknown, so it is left alone"
    assert (run.root / "draft.md").read_text() == "the agent's own untracked file\n"


def test_eval_check_fails_when_the_eval_rewrites_an_uncommitted_file_with_the_same_bytes(run):
    """A deterministic output left by a manual run: the bytes match, but round 1 would still commit it."""
    (run.root / "results.txt").write_text("same\n")
    os.utime(run.root / "results.txt", ns=(1, 1))
    write_eval(run, f"printf 'same\\n' > results.txt\necho '{OK_LINE}'\n")
    res = evalrun.eval_check(run)
    assert not res["ok"]
    assert any(p.startswith("the eval wrote into the repo: results.txt;") and "already uncommitted" in p
               for p in res["problems"]), res["problems"]


def test_eval_check_warns_about_untracked_files_round_1_would_commit(run):
    """A cache primed before the check isn't written during it; only a warning can name it."""
    (run.root / "cache").mkdir()
    (run.root / "cache" / "k1.txt").write_text("cached\n")
    write_eval(run, f"echo '{OK_LINE}'\n")
    res = evalrun.eval_check(run)
    assert res["ok"], res["problems"]
    assert len(res["warnings"]) == 1 and res["warnings"][0].startswith(
        "untracked files that round 1's commit will sweep onto the run branch: cache/k1.txt;"), res["warnings"]
    assert store.read_json(run.eval_check_path)["warnings"] == res["warnings"]


def test_eval_check_fails_on_links_that_leave_eval(run, tmp_path):
    write_eval(run, f"echo '{OK_LINE}'\n")
    (tmp_path / "workloads").mkdir()
    (run.eval_dir / "inside.txt").write_text("x")
    (run.eval_dir / "dev").symlink_to(tmp_path / "workloads")
    (run.eval_dir / "alias").symlink_to("inside.txt")
    res = evalrun.eval_check(run)
    assert not res["ok"]
    assert any(p.startswith(f"eval/ links outside itself: dev -> {tmp_path / 'workloads'};") for p in res["problems"])
    assert not any("alias" in p for p in res["problems"]), "a link that stays inside eval/ is fine"


def test_eval_unchanged_rehashes_only_entries_whose_stat_changed(run, monkeypatch):
    write_eval(run, f"echo '{OK_LINE}'\n")
    (run.eval_dir / "data").mkdir()
    (run.eval_dir / "data" / "big.bin").write_bytes(b"x" * 1000)
    manifest = evalrun.eval_manifest(run)
    assert evalrun.eval_unchanged(run, manifest) == []
    assert set(evalrun.eval_fingerprint(run)) == {"run", "data/big.bin"}
    hashed = []
    real = evalrun._digest
    monkeypatch.setattr(evalrun, "_digest", lambda p: hashed.append(p.name) or real(p))
    assert evalrun.eval_unchanged(run, manifest) == [] and hashed == [], "nothing changed, nothing read"
    (run.eval_dir / "data" / "big.bin").write_bytes(b"y" * 1000)
    assert evalrun.eval_unchanged(run, manifest) == ["changed: data/big.bin"] and hashed == ["big.bin"]
    (run.eval_dir / "data" / "big.bin").write_bytes(b"x" * 1000)
    assert evalrun.eval_unchanged(run, manifest) == []


def test_eval_unchanged_reports_unreadable_entries_instead_of_raising(run):
    if os.geteuid() == 0:
        pytest.skip("root reads anything")
    write_eval(run, f"echo '{OK_LINE}'\n")
    (run.eval_dir / "secret.txt").write_text("s")
    manifest = evalrun.eval_manifest(run)
    (run.eval_dir / "secret.txt").write_text("t")
    (run.eval_dir / "secret.txt").chmod(0)
    try:
        assert evalrun.eval_unchanged(run, manifest) == ["unreadable: secret.txt: Permission denied"]
    finally:
        (run.eval_dir / "secret.txt").chmod(0o644)


def test_restore_eval_puts_back_the_snapshot_and_keeps_what_was_written(run, tmp_path):
    import shutil

    write_eval(run, f"echo '{OK_LINE}'\n")
    (run.eval_dir / "data").mkdir()
    (run.eval_dir / "data" / "w.txt").write_text("frozen")
    manifest, h = evalrun.eval_manifest(run), evalrun.eval_hash(run)
    shutil.copytree(run.eval_dir, run.eval_accepted_dir, symlinks=True)
    (run.eval_dir / "data" / "w.txt").write_text("tampered")
    (run.eval_dir / "cache.json").write_text("{}")
    (run.eval_dir / "run").unlink()
    keep = tmp_path / "kept"
    assert evalrun.restore_eval(run, manifest, keep) is None
    assert evalrun.eval_hash(run) == h
    assert (keep / "data" / "w.txt").read_text() == "tampered" and (keep / "cache.json").exists()
    assert os.access(run.eval_dir / "run", os.X_OK)
