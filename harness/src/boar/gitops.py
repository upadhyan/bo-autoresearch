"""Git operations on the target repo. Every call runs in the repo root."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from pathlib import Path

from boar.errors import Refused
from boar.store import BOAR_DIR


def git(root: Path, *args: str, check: bool = True, strip: bool = True) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except FileNotFoundError:
        raise Refused("git is not installed") from None
    if check and proc.returncode != 0:
        raise Refused(f"`git {' '.join(args)}` failed: {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip() if strip else proc.stdout


def is_repo(root: Path) -> bool:
    return git(root, "rev-parse", "--is-inside-work-tree", check=False) == "true"


def current_branch(root: Path) -> str:
    """The checked-out branch name, or "" when HEAD is detached."""
    return git(root, "branch", "--show-current", check=False)


def head_commit(root: Path) -> str:
    return git(root, "rev-parse", "HEAD")


def has_commits(root: Path) -> bool:
    return bool(git(root, "rev-parse", "--verify", "--quiet", "HEAD", check=False))


def tracked_changes(root: Path) -> str:
    """Uncommitted changes to tracked files (staged or not)."""
    return git(root, "status", "--porcelain", "--untracked-files=no")


def worktree_changes(root: Path) -> str:
    """Any uncommitted change, untracked files included; `.boar/` is excluded via info/exclude."""
    return git(root, "status", "--porcelain")


def exclude_boar_dir(root: Path) -> None:
    """Keep `.boar/` out of git without touching the user's tracked .gitignore."""
    exclude = Path(git(root, "rev-parse", "--git-path", "info/exclude"))
    if not exclude.is_absolute():
        exclude = root / exclude
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text() if exclude.exists() else ""
    entry = f"/{BOAR_DIR}/"
    if entry not in existing.splitlines():
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        exclude.write_text(f"{existing}{prefix}{entry}\n")


def create_branch(root: Path, branch: str) -> None:
    git(root, "checkout", "-b", branch)


# Running Python code (the eval, a lever check) writes bytecode; it is not part of the run's code.
_BYTECODE = (":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.pyc")


def commit_worktree(root: Path, message: str) -> str:
    """Commit every change in the working tree except bytecode; return HEAD's sha (HEAD itself if nothing changed)."""
    if worktree_changes(root):
        git(root, "add", "-A", "--", ".", *_BYTECODE)
        if git(root, "diff", "--cached", "--name-only"):
            git(root, "commit", "--no-verify", "-m", message)
    return head_commit(root)


def _bytecode(path: str) -> bool:
    return "/__pycache__/" in f"/{path}" or path.endswith(".pyc")


def status(root: Path) -> dict[str, str]:
    """path -> porcelain XY code of every uncommitted path, each untracked file listed on its own.

    Bytecode and `.boar/` don't count. Paths are relative to the repo top, whatever the cwd.
    """
    fields = git(root, "status", "--porcelain", "-z", "--untracked-files=all", strip=False).split("\0")
    out: dict[str, str] = {}
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        out[path] = code
        if code[0] in "RC" and i < len(fields):
            # A rename or copy: its source path is the next field.
            out[fields[i]] = code
            i += 1
    return {p: c for p, c in out.items() if not _bytecode(p) and not p.startswith(f"{BOAR_DIR}/")}


def changed(start: dict[str, str], now: dict[str, str]) -> list[str]:
    """Paths whose status differs between two `status` snapshots, sorted."""
    return sorted(p for p in {*start, *now} if start.get(p) != now.get(p))


def untracked_dirs(root: Path) -> list[str]:
    """Untracked directories, each named by its top, empty ones included (`status` lists files only).

    Taken before a trial, it tells the directories the trial made from the ones already there.
    """
    out = git(root, "ls-files", "--others", "--directory", "--exclude-standard", "-z", "--full-name", "--", ":/",
              strip=False)
    return sorted(p.rstrip("/") for p in out.split("\0") if p.endswith("/"))


def _aside(keep: Path, path: str, merge: bool = False) -> Path:
    """A free place for `path` under `keep`, its parent made; `keep/dup-<n>/path` when `keep/path` is taken.

    Taken: something is there already, or a file sits where a parent directory goes (a file the trial
    left where a tracked directory was, moved there first). With `merge`, an existing directory is fine.
    """
    for n in range(100):
        dest = (keep / f"dup-{n}" if n else keep) / path
        if (dest.is_symlink() or dest.exists()) and not (merge and dest.is_dir() and not dest.is_symlink()):
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return dest
    raise OSError(f"no free place under {keep} for {path}")


def undo_writes(root: Path, commit: str, start: dict[str, str], keep: Path, dirs: list[str]) -> list[str]:
    """Put back what was written into the repo since the `start` snapshot; returns what is still changed.

    New untracked files are moved under `keep` first; then a tracked path gets its content from `commit` again
    (what was written there, a directory too, is copied under `keep` first). Nothing is deleted, except the
    directories left empty by the moves that the trial made: not in or under `dirs` (`untracked_dirs` at start).
    A path already uncommitted at `start` is left alone: its earlier content is unknown.
    """
    now = status(root)
    paths = [p for p in changed(start, now) if p not in start]
    moved = [p for p in paths if now.get(p) == "??"]
    for path in moved:
        with contextlib.suppress(OSError):  # reported below as still changed
            shutil.move(str(root / path), str(_aside(keep, path)))
    for path in paths:
        if path in moved:
            continue
        src = root / path
        # Only what is there now is copied aside; a deleted tracked file (one under a directory the trial
        # replaced with a file or a link, too) has nothing to keep and is always checked out again.
        if src.is_symlink() or src.exists():
            try:
                if src.is_symlink() or src.is_file():
                    shutil.copy2(src, _aside(keep, path), follow_symlinks=False)
                elif src.is_dir():  # a directory where a tracked file was: checkout removes it with what is left in it
                    shutil.copytree(src, _aside(keep, path, merge=True), symlinks=True, dirs_exist_ok=True)
            except OSError:
                continue  # its content could not be kept: leave it rather than lose it
        git(root, "checkout", commit, "--", f":(literal){path}", check=False)
    kept = [root / d for d in dirs]
    for path in moved:
        parent = (root / path).parent
        while parent != root and not any(parent == k or k in parent.parents for k in kept):
            try:
                parent.rmdir()  # only an empty one; a file there now (a restored tracked path) stops the walk too
            except OSError:
                break
            parent = parent.parent
    return changed(start, status(root))
