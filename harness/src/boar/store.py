"""Run directory layout and the files the harness owns.

Everything a command reads or writes about a run goes through here, so the
on-disk layout lives in one place:

    .boar/
      active                # id of the current run; read by the Stop hook
      <run-id>/
        config.json         # defaults plus overrides, frozen at init
        state.json          # phase, round records, eval status, removals (harness-only)
        spec.md
        research.md
        hypotheses.json     # written only by the harness
        reviews.jsonl
        eval/               # frozen once accepted
        eval_check.json     # last `boar eval check` result
        study.db            # Optuna, one study per round
        trials.jsonl
        trials/<n>/         # config and stdout/stderr for each repeat
        rounds/<r>/summary.md
        holdout.jsonl       # finalize's holdout runs
        report.md
        running.json        # the worker slot: names the live worker (or the CLI running one inline)
        eval_group.json     # the eval process group a worker is running, so a dead worker's eval can be killed
        eval.accepted/      # copy of eval/ as the reviewer accepted it
        eval.fingerprint.json  # eval/ digests by (size, mtime, ctime): only changed files are re-hashed
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import os
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from boar.errors import Refused

BOAR_DIR = ".boar"
TODO_MARKER = "<!-- boar:todo -->"

# Hypothesis statuses. A hypothesis is `pending` until reviewed; an accepted
# proposal stays `pending` (verdict accept) until activation.
PENDING, ACTIVE, REJECTED, REMOVED, WITHDRAWN = "pending", "active", "rejected", "removed", "withdrawn"

# Run phases.
SETUP, ROUND, FINALIZE, DONE, ABORTED = "setup", "round", "finalize", "done", "aborted"


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --- file primitives -------------------------------------------------------


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(path: Path, obj: Any) -> None:
    """Write atomically: a crash leaves either the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _parses(raw: bytes) -> bool:
    try:
        json.loads(raw)
    except ValueError:
        return False
    return True


def append_jsonl(path: Path, obj: Any) -> None:
    """Append one record as one line. A torn last line (a writer died mid-append) is cut off first:
    that record was never acknowledged, so the step that wrote it re-runs on resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(obj, sort_keys=False) + "\n").encode()
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            data = os.pread(fd, size, 0)
            cut = data.rfind(b"\n") + 1
            if _parses(data[cut:]):
                line = b"\n" + line
            else:
                os.ftruncate(fd, cut)
        while line:
            line = line[os.write(fd, line):]
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: Path) -> list:
    """Every record; an unparseable last line with no newline is a torn append and is skipped."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return []
    lines = data.split(b"\n")
    tail = lines.pop()
    out = [json.loads(line) for line in lines if line.strip()]
    if tail.strip() and _parses(tail):
        out.append(json.loads(tail))
    return out


# --- locating the run --------------------------------------------------------


def find_repo_root(start: Path | None = None) -> Path:
    """The target repo root: the nearest ancestor holding `.boar/`, else the git top level, else cwd."""
    start = (start or Path.cwd()).resolve()
    for d in (start, *start.parents):
        if (d / BOAR_DIR).is_dir():
            return d
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=start, capture_output=True, text=True, check=True
        ).stdout.strip()
        if top:
            return Path(top)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return start


def _sqlalchemy_decodes_paths() -> bool:
    from sqlalchemy.engine import make_url

    return make_url("sqlite:///%25").database == "%"


@dataclass(frozen=True)
class Run:
    """Handle on one run directory. Holds paths only; data is read fresh on each call."""

    root: Path  # target repo root; the eval runs from here
    id: str

    @property
    def dir(self) -> Path:
        return self.root / BOAR_DIR / self.id

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    # Named paths, so the layout above is the only place file names appear.
    @property
    def config_path(self) -> Path:
        return self.path("config.json")

    @property
    def state_path(self) -> Path:
        return self.path("state.json")

    @property
    def spec_path(self) -> Path:
        return self.path("spec.md")

    @property
    def research_path(self) -> Path:
        return self.path("research.md")

    @property
    def hypotheses_path(self) -> Path:
        return self.path("hypotheses.json")

    @property
    def reviews_path(self) -> Path:
        return self.path("reviews.jsonl")

    @property
    def eval_dir(self) -> Path:
        return self.path("eval")

    @property
    def eval_check_path(self) -> Path:
        return self.path("eval_check.json")

    @property
    def study_db(self) -> Path:
        return self.path("study.db")

    @property
    def storage_url(self) -> str:
        """Quoted, because SQLAlchemy reads '?' in the path as a query string.

        SQLAlchemy 2.1 decodes '%xx' in the path; 2.0 (the newest on Python 3.10)
        does not, so there the quoted path goes to SQLite as a URI, which does.
        """
        path = urllib.parse.quote(str(self.study_db))
        if _sqlalchemy_decodes_paths():
            return "sqlite:///" + path
        return "sqlite:///file:" + path + "?uri=true"

    @property
    def trials_path(self) -> Path:
        return self.path("trials.jsonl")

    def trial_dir(self, n: int) -> Path:
        return self.path("trials", str(n))

    def round_dir(self, r: int) -> Path:
        return self.path("rounds", str(r))

    def summary_path(self, r: int) -> Path:
        return self.round_dir(r) / "summary.md"

    @property
    def holdout_path(self) -> Path:
        return self.path("holdout.jsonl")

    @property
    def report_path(self) -> Path:
        return self.path("report.md")

    @property
    def running_path(self) -> Path:
        return self.path("running.json")

    @property
    def eval_group_path(self) -> Path:
        return self.path("eval_group.json")

    @property
    def eval_accepted_dir(self) -> Path:
        return self.path("eval.accepted")

    @property
    def eval_fingerprint_path(self) -> Path:
        return self.path("eval.fingerprint.json")

    @property
    def lock_path(self) -> Path:
        return self.path(".lock")

    @property
    def launch_lock_path(self) -> Path:
        return self.path(".launch.lock")

    # --- data ---------------------------------------------------------------

    def config(self) -> dict:
        return read_json(self.config_path)

    def state(self) -> dict:
        return read_json(self.state_path)

    def save_state(self, state: dict) -> None:
        write_json(self.state_path, state)

    def hypotheses(self) -> list[dict]:
        return read_json(self.hypotheses_path, [])

    def save_hypotheses(self, hyps: list[dict]) -> None:
        write_json(self.hypotheses_path, hyps)

    def trials(self) -> list[dict]:
        return read_jsonl(self.trials_path)

    def append_trial(self, record: dict) -> None:
        append_jsonl(self.trials_path, record)

    def reviews(self) -> list[dict]:
        return read_jsonl(self.reviews_path)

    def append_review(self, record: dict) -> None:
        append_jsonl(self.reviews_path, record)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Serialize read-modify-write of the run's files across processes. Not reentrant."""
        with _flock(self.dir, self.lock_path):
            yield

    @contextlib.contextmanager
    def launch_locked(self) -> Iterator[None]:
        """Serialize claiming the worker slot. A separate file from `locked`, so it may wrap code that takes that one."""
        with _flock(self.dir, self.launch_lock_path):
            yield

    def update_state(self, fn: Callable[[dict], Any]) -> Any:
        """Apply `fn` to the state under the lock and save it; returns what `fn` returns."""
        with self.locked():
            state = self.state()
            result = fn(state)
            self.save_state(state)
            return result

    # --- background worker --------------------------------------------------

    def running(self) -> dict | None:
        """The live worker's record, or None if none is running.

        A record is stale (the worker was killed, maybe by a reboot) unless its pid is alive and is still
        the process that wrote it; a reused pid must not keep the run waiting.
        """
        rec = read_json(self.running_path)
        if not isinstance(rec, dict) or not pid_alive(rec.get("pid")):
            return None
        if not same_process(rec["pid"], rec.get("identity")):
            return None
        return rec


@contextlib.contextmanager
def _flock(d: Path, path: Path) -> Iterator[None]:
    d.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def process_identity(pid: int) -> dict | None:
    """What tells this process apart from a later one given the same pid: its start time and the boot.

    From /proc where there is one, else `ps`; None if neither answers.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            start = f.read().rsplit(") ", 1)[1].split()[19]
        with open("/proc/sys/kernel/random/boot_id") as f:
            return {"start": start, "boot_id": f.read().strip()}
    except (OSError, IndexError):
        pass
    try:
        out = _ps(pid, "lstart=").strip()
    except OSError:
        return None
    return {"start": out, "boot_id": None} if out else None


def _ps(pid: int, field: str) -> str:
    return subprocess.run(["ps", "-o", field, "-p", str(pid)], capture_output=True, text=True, check=False).stdout


def process_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        pass
    try:
        return _ps(pid, "command=")
    except OSError:
        return ""


def same_process(pid: int, identity: Any) -> bool:
    """Whether `pid` is still the process `identity` was recorded from. Without a usable identity, a
    record only counts if the process looks like boar."""
    if isinstance(identity, dict):
        now = process_identity(pid)
        if now is not None:
            return now == identity
    return "boar" in process_cmdline(pid)


def pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie child still answers kill(0); treat it as dead.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ", 1)[1][:1] != "Z"
    except (OSError, IndexError):
        return True


def active_run(start: Path | None = None) -> Run | None:
    """The run named by `.boar/active`, or None. Finished and aborted runs are still returned."""
    root = find_repo_root(start)
    marker = root / BOAR_DIR / "active"
    try:
        run_id = marker.read_text().strip()
    except FileNotFoundError:
        return None
    if not run_id or not (root / BOAR_DIR / run_id / "state.json").exists():
        return None
    return Run(root=root, id=run_id)


def require_run(start: Path | None = None) -> Run:
    run = active_run(start)
    if run is None:
        raise Refused("no active BOAR run here; start one with `boar init --spec .boar/spec-draft.md`")
    return run


def require_live_run(start: Path | None = None) -> Run:
    """An active run that is neither finished nor aborted, with no background worker mid-flight."""
    run = require_run(start)
    phase = run.state()["phase"]
    if phase in (DONE, ABORTED):
        raise Refused(f"run {run.id} is {phase}; start a new one with `boar init`")
    rec = run.running()
    if rec is not None:
        raise Refused(
            f"`boar {rec.get('command')}` is still running in the background (pid {rec['pid']}); "
            "wait for it with `boar wait`"
        )
    return run


# --- hypotheses and levers -----------------------------------------------------


def hypothesis(hyps: list[dict], hid: str) -> dict:
    for h in hyps:
        if h["id"] == hid:
            return h
    raise Refused(f"no hypothesis {hid!r} in this run")


def all_levers(hyps: list[dict]) -> dict[str, dict]:
    """Every lever ever defined in the run, by name, with `hypothesis` set to its owner's id."""
    out: dict[str, dict] = {}
    for h in hyps:
        for lever in h["levers"]:
            out[lever["name"]] = {**lever, "hypothesis": h["id"]}
    return out


def lever_defaults(hyps: list[dict]) -> dict[str, Any]:
    return {name: lever["default"] for name, lever in all_levers(hyps).items()}


def active_hypotheses(hyps: list[dict]) -> list[dict]:
    return [h for h in hyps if h["status"] == ACTIVE]


def search_space(hyps: list[dict]) -> dict[str, dict]:
    """The levers of all active hypotheses, by name. Every other lever stays pinned at its default."""
    return {
        name: lever for name, lever in all_levers(hyps).items() if hypothesis(hyps, lever["hypothesis"])["status"] == ACTIVE
    }


def full_config(hyps: list[dict], values: dict[str, Any]) -> dict[str, Any]:
    """The config handed to the eval: every lever ever defined, `values` over the defaults."""
    config = lever_defaults(hyps)
    unknown = set(values) - set(config)
    if unknown:
        raise ValueError(f"unknown levers {sorted(unknown)}")
    config.update(values)
    return config


def non_default(config: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """The levers a config sets away from default. A lever missing from the config is at default."""
    return {k: v for k, v in config.items() if k in defaults and v != defaults[k]}


def config_key(config: dict[str, Any], defaults: dict[str, Any]) -> str:
    """Canonical identity of a config: its non-default levers. Trials with equal keys pool their repeats."""
    return json.dumps(non_default(config, defaults), sort_keys=True)
