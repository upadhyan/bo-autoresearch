"""The eval contract: run `eval/run` once per repeat, parse its last stdout line, judge the trial.

Each invocation runs in its own session so that a hang, or a SIGTERM to the worker, can kill
the eval and everything it spawned without touching the harness.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from boar import gitops, stats, store
from boar.errors import Refused
from boar.store import Run

COMPLETE, INFEASIBLE, FAILED = "complete", "infeasible", "failed"

# Only the tail of stdout is parsed; an eval may print a lot before its result line.
_TAIL_BYTES = 64 * 1024

# How often a running trial looks for another eval of the same repo running alongside it.
CONCURRENT_POLL_S = 2.0

# A repeat's BOAR_DIAG_DIR larger than this keeps only its diag.json.
DIAG_CAP_BYTES = 20 * 1024 * 1024


class Interrupted(Exception):
    """The worker was told to stop mid-trial. The trial must not be recorded."""


class ConcurrentEval(Refused):
    """Another process ran this repo's eval during a trial. The trial is stopped and must not be recorded."""


_live: dict[str, Any] = {"proc": None, "stop": False}


def request_stop() -> None:
    """Kill the running eval and refuse to start another. Safe to call from a signal handler."""
    _live["stop"] = True
    proc = _live["proc"]
    if proc is not None:
        _kill_group(proc)


def _kill_group(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


def _skipped(name: str, is_dir: bool) -> bool:
    """Bytecode caches: running the eval writes them, so they are not part of it."""
    return name == "__pycache__" if is_dir else name.endswith(".pyc")


def eval_entries(root: Path, errors: dict[str, str] | None = None) -> list[tuple[str, Path]]:
    """(relative posix path, path) of every file and symlink under `root`, sorted, bytecode skipped.

    A symlinked directory is an entry itself (hashed by its target), not walked into. A directory
    that can't be listed goes into `errors` (relpath -> error) if given; otherwise it is skipped.
    """

    def unlisted(e: OSError) -> None:
        if errors is not None:
            rel = Path(e.filename).relative_to(root).as_posix() if e.filename else "."
            errors[rel] = e.strerror or str(e)

    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=unlisted):
        links = [d for d in dirnames if Path(dirpath, d).is_symlink()]
        dirnames[:] = [d for d in dirnames if d not in links and not _skipped(d, True)]
        found.extend(Path(dirpath, name) for name in [*links, *filenames] if not _skipped(name, False))
    return sorted(((p.relative_to(root).as_posix(), p) for p in found), key=lambda e: e[0])


def _entry_bytes(path: Path) -> bytes:
    return b"symlink:" + os.readlink(path).encode() if path.is_symlink() else path.read_bytes()


def _digest(path: Path) -> str:
    """An entry's eval_manifest value: "symlink:<target>" for a link, else the sha256 of its bytes."""
    if path.is_symlink():
        return f"symlink:{os.readlink(path)}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp(path: Path) -> tuple[str, int, int] | None:
    """`_digest` of a repo path with its mtime and mode, or None when it is gone or unreadable.

    The mtime catches a rewrite with the same bytes: that file still lands in round 1's commit. Not the
    ctime: a chmod to the same mode or a hard link made and removed elsewhere changes it, writing nothing.
    """
    try:
        st = path.lstat()
        return _digest(path), st.st_mtime_ns, st.st_mode
    except OSError:
        return None


def eval_hash(run: Run) -> str | None:
    """sha256 over (relative path, bytes) of eval/; a symlink contributes its target, not what it points at."""
    root = run.eval_dir
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for rel, path in eval_entries(root):
        for chunk in (rel.encode(), _entry_bytes(path)):
            h.update(len(chunk).to_bytes(8, "big"))
            h.update(chunk)
    return h.hexdigest()


def eval_manifest(run: Run) -> dict[str, str] | None:
    """relpath -> sha256 of each entry eval_hash covers ("symlink:<target>" for a symlink); None if eval/ is missing.

    Kept at acceptance so a later change to eval/ can be named file by file.
    """
    root = run.eval_dir
    if not root.is_dir():
        return None
    return {rel: _digest(path) for rel, path in eval_entries(root)}


def _stat_key(path: Path) -> list[int]:
    st = path.lstat()
    # ctime too: a tool that edits a file and then resets its mtime can't reset ctime.
    return [st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def eval_fingerprint(run: Run) -> dict[str, list[int]]:
    """relpath -> [size, mtime_ns, ctime_ns] of each eval_hash entry, from lstat alone (nothing is read)."""
    root = run.eval_dir
    if not root.is_dir():
        return {}
    out: dict[str, list[int]] = {}
    for rel, path in eval_entries(root):
        with contextlib.suppress(OSError):
            out[rel] = _stat_key(path)
    return out


def _current(run: Run) -> tuple[dict[str, str] | None, dict[str, str]]:
    """eval_manifest of eval/ now, and relpath -> error for entries that can't be read.

    Digests are cached in run.eval_fingerprint_path by fingerprint, so an entry is re-hashed only
    when its stat changed: `boar next` and the per-trial checks stay cheap with large data in eval/.
    """
    root = run.eval_dir
    if not root.is_dir():
        return None, {}
    errors: dict[str, str] = {}
    try:
        cache = store.read_json(run.eval_fingerprint_path, {})
    except (OSError, ValueError):
        cache = {}
    cache = cache if isinstance(cache, dict) else {}
    manifest: dict[str, str] = {}
    fresh: dict[str, list] = {}
    for rel, path in eval_entries(root, errors):
        try:
            key = _stat_key(path)
            hit = cache.get(rel)
            digest = hit[3] if isinstance(hit, list) and len(hit) == 4 and hit[:3] == key else _digest(path)
        except OSError as e:
            errors[rel] = e.strerror or str(e)
            continue
        manifest[rel] = digest
        fresh[rel] = [*key, digest]
    if fresh != cache:
        with contextlib.suppress(OSError):
            store.write_json(run.eval_fingerprint_path, fresh)
    return manifest, errors


def eval_unchanged(run: Run, accepted_manifest: dict[str, str]) -> list[str]:
    """eval_diff lines from `accepted_manifest` to eval/ now, plus "unreadable: <path>: <error>"; [] means unchanged.

    Only entries whose fingerprint changed since the last call are re-hashed (see `_current`).
    """
    manifest, errors = _current(run)
    if manifest is None:
        return eval_diff(accepted_manifest, None)
    readable = {p: d for p, d in accepted_manifest.items() if p not in errors}
    return eval_diff(readable, manifest) + [f"unreadable: {p}: {e}" for p, e in sorted(errors.items())]


def restore_eval(run: Run, accepted_manifest: dict[str, str], keep: Path) -> str | None:
    """Put back every eval/ entry that differs from the accepted one, from the snapshot taken at acceptance.

    What was there instead is moved under `keep`, never deleted. Returns why eval/ still differs, or None.
    """
    snap = run.eval_accepted_dir
    if not snap.is_dir():
        return f"no copy of the accepted eval at {snap}"
    manifest, errors = _current(run)
    manifest = manifest or {}
    differ = {p for p in {*accepted_manifest, *manifest} if accepted_manifest.get(p) != manifest.get(p)}
    # An unlistable eval/ itself (".") can't be put back entry by entry; the check below reports it.
    paths = sorted(differ | {p for p in errors if p != "."})
    try:
        for rel in paths:
            here = run.eval_dir / rel
            if os.path.lexists(here):
                (keep / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(here), str(keep / rel))
        for rel in paths:
            if rel in accepted_manifest:
                (run.eval_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snap / rel, run.eval_dir / rel, follow_symlinks=False)
    except OSError as e:
        return f"restoring eval/ failed: {e}"
    left = eval_unchanged(run, accepted_manifest)
    return f"eval/ still differs from the accepted version ({diff_text(left)})" if left else None


def links_outside(run: Run) -> list[str]:
    """Each symlink in eval/ (or eval/ itself) that resolves outside eval/, as "<link> -> <target>".

    The hash covers a link's target path, not what it points at, so eval/ must be self-contained.
    """
    root = run.eval_dir
    if root.is_symlink():
        return [f"eval -> {os.readlink(root)}"]
    if not root.is_dir():
        return []
    inside = Path(os.path.realpath(root))
    # realpath, not resolve: it doesn't raise on a link loop on any supported Python.
    return [
        f"{rel} -> {os.readlink(path)}" for rel, path in eval_entries(root)
        if path.is_symlink() and not Path(os.path.realpath(path)).is_relative_to(inside)
    ]


def eval_diff(old: dict[str, str], new: dict[str, str] | None) -> list[str]:
    """Sorted "changed: <path>", "added: <path>", "removed: <path>" lines from manifest `old` to `new`."""
    new = new or {}
    lines = [f"changed: {p}" for p in old if p in new and new[p] != old[p]]
    lines += [f"added: {p}" for p in new if p not in old]
    lines += [f"removed: {p}" for p in old if p not in new]
    return sorted(lines)


def diff_text(lines: list[str], limit: int = 10) -> str:
    """An eval_diff on one line, cut after `limit` entries."""
    shown = "; ".join(lines[:limit])
    return shown + (f"; … and {len(lines) - limit} more" if len(lines) > limit else "")


def _clip(text: str, n: int = 120) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _finite(x: Any) -> float | None:
    """`x` as a finite float, or None. JSON ints are unbounded; one too large for a float is not finite."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def parse_result(stdout: str) -> tuple[dict | None, str | None]:
    """The last non-empty stdout line as {"metric", "guards_ok", "metrics"}, or (None, why it breaks the contract)."""
    # Only "\n" ends a line: JSON allows U+2028 and friends raw inside strings, and splitlines breaks on them.
    lines = [line for line in stdout.split("\n") if line.strip()]
    if not lines:
        return None, "unparseable output: stdout is empty"
    # A raw "\r" can't be inside JSON; before the result it is a progress line being overwritten.
    last = [seg for seg in lines[-1].split("\r") if seg.strip()][-1].strip()
    try:
        obj = json.loads(last)
    except ValueError:
        return None, f"unparseable output: {_clip(last)}"
    if not isinstance(obj, dict):
        return None, f"unparseable output: last line is not a JSON object: {_clip(last)}"
    metric = _finite(obj.get("metric"))
    if metric is None:
        return None, "metric not a finite number"
    if not isinstance(obj.get("guards_ok"), bool):
        return None, "guards_ok missing or not true/false"
    metrics = obj.get("metrics")
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, dict):
        return None, "metrics is not a JSON object"
    return {"metric": metric, "guards_ok": obj["guards_ok"], "metrics": metrics}, None


def _read_tail(path: Path) -> str:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - _TAIL_BYTES))
        return f.read().decode(errors="replace")


def _exe_problem(exe: Path) -> str | None:
    if not exe.exists():
        return f"eval/run not found (expected {exe})"
    if not exe.is_file():
        return "eval/run is not a file"
    if not os.access(exe, os.X_OK):
        return "eval/run is not executable (chmod +x it)"
    return None


def other_evals(root: Path, own: int) -> list[str]:
    """This user's processes that run an eval of the repo at `root` but are not `own` or its descendants.

    A run of the eval is a process with BOAR_CONFIG in its environment and its working directory in the
    repo, as a hand-run `BOAR_CONFIG=… <run dir>/eval/run` from the repo root has. Needs /proc (Linux);
    elsewhere it finds nothing.
    """
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return []
    root, uid = root.resolve(), os.getuid()
    parents: dict[int, int] = {}
    candidates: list[int] = []
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if entry.stat().st_uid != uid:
                continue
            parents[pid] = int((entry / "stat").read_text().rsplit(") ", 1)[1].split()[1])
            if not any(v.startswith(b"BOAR_CONFIG=") for v in (entry / "environ").read_bytes().split(b"\0")):
                continue
            cwd = Path(os.readlink(entry / "cwd"))
        except (OSError, IndexError, ValueError):
            continue
        if cwd == root or root in cwd.parents:
            candidates.append(pid)

    def descends_from_own(pid: int) -> bool:
        seen: set[int] = set()
        while pid > 1 and pid not in seen:
            if pid == own:
                return True
            seen.add(pid)
            pid = parents.get(pid, 0)
        return False

    return [f"pid {p}: {store.process_cmdline(p).strip()[:120]}" for p in candidates if not descends_from_own(p)]


def _wait(proc: subprocess.Popen, timeout_s: float, root: Path) -> int:
    """`proc.wait(timeout_s)`, raising ConcurrentEval as soon as another run of this repo's eval shows up."""
    deadline = time.monotonic() + timeout_s
    while True:
        left = deadline - time.monotonic()
        try:
            return proc.wait(timeout=max(0.0, min(CONCURRENT_POLL_S, left)))
        except subprocess.TimeoutExpired:
            if left <= CONCURRENT_POLL_S:
                raise
        others = other_evals(root, proc.pid)
        if others:
            raise ConcurrentEval(
                f"another run of this repo's eval was going during the trial ({'; '.join(others)}); it competes "
                "for the machine and can trip the trial's guards, so the trial was stopped and not recorded. Let "
                "that run finish or stop it, and run nothing else while a worker runs"
            )


def _invoke(
    exe: Path, cwd: Path, env: dict, out_dir: Path, k: int, timeout_s: float, group_path: Path
) -> tuple[dict | None, str | None]:
    """One repeat. Returns (parsed result, None), or (None, error); error "timeout" means the budget ran out.

    The eval's process group is recorded in `group_path` while it runs: if this process dies without
    cleaning up, the next worker (or `boar wait`, or `boar abort`) kills what it left behind.
    """
    stdout_path = out_dir / f"repeat-{k}.stdout"
    with open(stdout_path, "wb") as out, open(out_dir / f"repeat-{k}.stderr", "wb") as err:
        try:
            proc = subprocess.Popen(
                [str(exe)], cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True
            )
        except OSError as e:
            return None, f"can't execute eval/run: {e.strerror or e}"
        _live["proc"] = proc
        store.write_json(group_path, {"pgid": proc.pid, "identity": store.process_identity(proc.pid)})
        try:
            if _live["stop"]:
                _kill_group(proc)
            try:
                rc = _wait(proc, timeout_s, cwd)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.wait()
                return None, "timeout"
            except BaseException:
                _kill_group(proc)
                proc.wait()
                raise
        finally:
            _live["proc"] = None
            # Anything the eval left running in its session would disturb the next measurement.
            _kill_group(proc)
            with contextlib.suppress(FileNotFoundError):
                group_path.unlink()
    if _live["stop"]:
        raise Interrupted()
    if rc != 0:
        return None, f"exit {rc}" if rc > 0 else f"killed by signal {-rc}"
    return parse_result(_read_tail(stdout_path))


def _median_metrics(per_repeat: list[dict]) -> dict:
    """Per key: the median of the finite numeric values, else the last value reported."""
    out: dict[str, Any] = {}
    for key in dict.fromkeys(k for m in per_repeat for k in m):
        vals = [m[key] for m in per_repeat if key in m]
        nums = [f for f in map(_finite, vals) if f is not None]
        out[key] = stats.median(nums) if nums else vals[-1]
    return out


def eval_environ() -> dict[str, str]:
    """The environment an eval inherits: the worker's, minus the harness's own virtualenv.

    Under `uv run boar` that venv is active (VIRTUAL_ENV, its bin/ first on PATH), so a
    `#!/usr/bin/env python3` eval would run the harness's interpreter, not the one the agent's manual
    runs use. Another venv the user activated is left as it is.
    """
    env = dict(os.environ)
    venv = env.get("VIRTUAL_ENV")
    try:
        ours = bool(venv) and Path(venv).resolve() == Path(sys.prefix).resolve()
    except OSError:
        ours = False
    if not ours:
        return env
    env.pop("VIRTUAL_ENV", None)
    env.pop("VIRTUAL_ENV_PROMPT", None)
    bindir = (Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")).resolve()

    def ours_bin(entry: str) -> bool:
        try:
            return bool(entry) and Path(entry).resolve() == bindir
        except OSError:
            return False

    if "PATH" in env:
        env["PATH"] = os.pathsep.join(e for e in env["PATH"].split(os.pathsep) if not ours_bin(e))
    return env


def _tree_bytes(path: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(dirpath, name)).st_size
    return total


def _keep_diag(tmp: Path, dest: Path, k: int) -> tuple[dict | None, str | None]:
    """Move repeat k's BOAR_DIAG_DIR `tmp` to `dest`; returns (diag.json's numbers or None, a problem or None).

    An empty directory leaves nothing behind. Over DIAG_CAP_BYTES only diag.json is kept. Diagnostics never
    fail a trial, so every problem is returned, never raised.
    """
    problems = []
    try:
        if not any(tmp.iterdir()):
            return None, None
        size = _tree_bytes(tmp)
        if size > DIAG_CAP_BYTES:
            dest.mkdir(parents=True)
            if (tmp / "diag.json").is_file() and not (tmp / "diag.json").is_symlink():
                shutil.copyfile(tmp / "diag.json", dest / "diag.json")
            problems.append(
                f"repeat {k}: the diag dir held {size / 2**20:.1f} MB, over the {DIAG_CAP_BYTES // 2**20} MB cap; "
                "kept only diag.json"
            )
        else:
            shutil.move(str(tmp), str(dest))
    except OSError as e:
        problems.append(f"repeat {k}: the diag dir could not be saved ({e.strerror or e})")
    values, why = None, None
    diag = dest / "diag.json"
    if diag.is_file():
        try:
            obj = json.loads(diag.read_text())
        except (OSError, ValueError) as e:
            why = f"repeat {k}: diag.json is not valid JSON ({_clip(str(e), 80)})"
        else:
            if not isinstance(obj, dict) or not all(_finite(v) is not None for v in obj.values()):
                why = f"repeat {k}: diag.json must be a flat JSON object of finite numbers"
            else:
                values = obj
    return values, "; ".join(problems + ([why] if why else [])) or None


def run_trial(
    run: Run, config: dict, split: str, repeats: int, out_dir: Path, budget_s: float, first_repeat: int = 1
) -> dict:
    """Evaluate one config `repeats` times within a shared wall-clock budget.

    Repeat k (from 1) runs with BOAR_REPEAT = first_repeat + k - 1, the index the eval derives its seeds from.

    Returns {"state", "metric", "repeats", "metrics", "duration_s", "error"}. A guard failure stops
    the repeats (infeasible); a crash, non-zero exit, hang or contract violation stops them (failed).
    Raises Interrupted if `request_stop` was called.
    """
    if _live["stop"]:
        raise Interrupted()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    config_path = out_dir / "config.json"
    store.write_json(config_path, config)
    exe = run.eval_dir / "run"
    env = {
        **eval_environ(),
        "BOAR_CONFIG": str(config_path.resolve()),
        "BOAR_SPLIT": split,
        "BOAR_RUN_DIR": str(run.dir.resolve()),
    }
    values: list[float] = []
    per_repeat: list[dict] = []
    diags: list[dict] = []
    diag_errors: list[str] = []
    state, error = COMPLETE, _exe_problem(exe)
    start = time.monotonic()
    if error:
        state = FAILED
    else:
        for k in range(1, repeats + 1):
            remaining = budget_s - (time.monotonic() - start)
            if remaining <= 0:
                outcome, problem = None, "timeout"
            else:
                tmp = Path(tempfile.mkdtemp(prefix="boar-diag-"))
                try:
                    repeat_env = {"BOAR_REPEAT": str(first_repeat + k - 1), "BOAR_DIAG_DIR": str(tmp)}
                    outcome, problem = _invoke(
                        exe, run.root, {**env, **repeat_env}, out_dir, k, remaining, run.eval_group_path
                    )
                    diag_values, diag_problem = _keep_diag(tmp, out_dir / f"diag-{k}", k)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                if diag_values is not None:
                    diags.append(diag_values)
                if diag_problem:
                    diag_errors.append(diag_problem)
            if problem:
                state, error = FAILED, f"timeout after {budget_s:g}s" if problem == "timeout" else problem
                break
            values.append(outcome["metric"])
            per_repeat.append(outcome["metrics"])
            if not outcome["guards_ok"]:
                state = INFEASIBLE
                break
    return {
        "state": state,
        "metric": stats.median(values),
        "repeats": values,
        "metrics": _median_metrics(per_repeat),
        "duration_s": round(time.monotonic() - start, 3),
        "error": error,
        "diagnostics": _median_metrics(diags),
        "diag_error": "; ".join(diag_errors) or None,
    }


def _names(paths: list[str], limit: int = 10) -> str:
    return ", ".join(paths[:limit]) + (" …" if len(paths) > limit else "")


def _fresh(base: Path) -> Path:
    """A new, not yet existing directory under `base`, named by the time: one per check, none overwritten."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    n = 0
    while (path := base / (stamp if not n else f"{stamp}-{n}")).exists():
        n += 1
    return path


def eval_check(run: Run) -> dict:
    """One baseline trial on dev, judged against the contract, the guards, the time limit and what it wrote.

    Writes eval_check.json: {"ok", "hash", "manifest", "problems", "warnings", "trial", "limit_s", "at"}; the manifest
    lets `eval_unchanged` tell cheaply whether eval/ still is the checked version. What the eval wrote
    into the repo is put back (new files moved under eval_check-repo-writes/<time>/), so the round commit
    never sweeps eval outputs into the run's code.
    """
    cfg = run.config()
    target = cfg["trial_target_s"]
    limit = 2 * target
    out_dir = run.path("eval_check")
    before = eval_hash(run)
    before_manifest = eval_manifest(run)
    links = links_outside(run)
    head, repo_before, dirs = gitops.head_commit(run.root), gitops.status(run.root), gitops.untracked_dirs(run.root)
    # Already-uncommitted paths keep their status when rewritten, so their content and times are compared instead.
    stamps = {p: _stamp(run.root / p) for p in repo_before}
    trial = run_trial(run, store.lever_defaults(run.hypotheses()), "dev", cfg["repeats"], out_dir, 3 * target)
    wrote = gitops.changed(repo_before, gitops.status(run.root))
    rewrote = sorted(p for p, c in stamps.items() if p not in wrote and _stamp(run.root / p) != c)
    # A primed cache or a manual run's output is written before the check, so only a warning can name it.
    untracked = sorted(p for p, c in repo_before.items() if c == "??" and p not in wrote and p not in rewrote)
    tracked = sorted(p for p, c in repo_before.items() if c != "??" and p not in wrote and p not in rewrote)
    problems: list[str] = []
    if before is None:
        problems.append(f"no eval directory at {run.eval_dir}")
    elif trial["error"]:
        problems.append(f"contract: {trial['error']} (stdout/stderr of each repeat in {out_dir})")
    elif trial["state"] == INFEASIBLE:
        problems.append(f"guards fail at baseline: a repeat reported guards_ok false (output in {out_dir})")
    if trial["duration_s"] > limit:
        problems.append(
            f"the baseline trial took {trial['duration_s']:.1f}s over {cfg['repeats']} repeats, more than "
            f"2 × trial_target_s = {limit}s; make the eval smaller"
        )
    if before_manifest is not None:
        changed = eval_diff(before_manifest, eval_manifest(run))
        if changed:
            problems.append(f"eval/ changed during the check: {diff_text(changed)}; the eval must not write inside eval/")
    if links:
        problems.append(
            f"eval/ links outside itself: {'; '.join(links)}; copy what it points at into eval/ (the frozen eval "
            "must be self-contained)"
        )
    if wrote:
        # Outside out_dir: the next check's run_trial clears that, and these files were moved, not deleted.
        keep = _fresh(run.path("eval_check-repo-writes"))
        left = gitops.undo_writes(run.root, head, repo_before, keep, dirs)
        back = [p for p in wrote if p not in left and p not in repo_before]
        stuck = [p for p in wrote if p not in back]
        how = [f"put back: {_names(back)}; new files moved to {keep}"] if back else []
        if stuck:
            how.append(
                f"still changed: {_names(stuck)}; restore them (e.g. `git checkout -- <path>`, or `git stash -u`) "
                "before round 1 commits the worktree"
            )
        problems.append(
            f"the eval wrote into the repo: {_names(wrote)}; write runtime outputs to a temp dir instead "
            f"({'; '.join(how)})"
        )
    if rewrote:
        problems.append(
            f"the eval wrote into the repo: {_names(rewrote)}; write "
            "runtime outputs to a temp dir instead (these were already uncommitted before the check, so they are "
            "left as they are; delete them, or commit them if they are yours, before round 1 commits the worktree)"
        )
    warnings = [
        f"untracked files that round 1's commit will sweep onto the run branch: {_names(untracked)}; delete any "
        "that an eval run wrote (a manual run's output, a cache), since a trial that rewrites a committed output fails"
    ] if untracked else []
    if tracked:
        warnings.append(
            f"tracked files with uncommitted changes (edited or deleted) that round 1's commit will record on the run "
            f"branch: {_names(tracked)}; restore any that an eval run changed (`git checkout -- <path>`)"
        )
    reps = trial["repeats"]
    if len(reps) > 1 and stats.spread(reps) == 0:
        warnings.append(
            f"the baseline's {len(reps)} repeats returned the same metric bit for bit, so the noise floor stays "
            "unknown; unless the workload is truly deterministic, derive its seeds from BOAR_SPLIT and BOAR_REPEAT"
        )
    result = {
        "ok": not problems, "hash": before, "manifest": before_manifest, "problems": problems, "warnings": warnings,
        "trial": trial, "limit_s": limit, "at": store.now_iso(),
    }
    store.write_json(run.eval_check_path, result)
    return result
