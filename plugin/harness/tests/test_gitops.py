"""gitops: commits carry the code, never bytecode; a trial's writes into the repo are found and put back."""

from __future__ import annotations

import subprocess

import pytest

from boar import gitops


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def test_commit_worktree_skips_bytecode(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    head = gitops.head_commit(tmp_path)

    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-310.pyc").write_bytes(b"\0")
    (tmp_path / "sub" / "__pycache__").mkdir(parents=True)
    (tmp_path / "sub" / "__pycache__" / "b.pyc").write_bytes(b"\0")
    (tmp_path / "top.pyc").write_bytes(b"\0")
    assert gitops.commit_worktree(tmp_path, "boar: round 1") == head, "bytecode alone makes no commit"

    (tmp_path / "a.py").write_text("x = 2\n")
    (tmp_path / "sub" / "c.py").write_text("y = 1\n")
    sha = gitops.commit_worktree(tmp_path, "boar: round 1")
    assert sha != head
    assert sorted(_git(tmp_path, "show", "--name-only", "--format=", sha).split()) == ["a.py", "sub/c.py"]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.txt").write_text("b\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_status_lists_untracked_files_one_by_one_without_bytecode_or_boar(repo):
    (repo / "out" / "deep").mkdir(parents=True)
    (repo / "out" / "deep" / "x.log").write_text("x")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "a.pyc").write_bytes(b"\0")
    (repo / ".boar" / "r").mkdir(parents=True)
    (repo / ".boar" / "r" / "state.json").write_text("{}")
    (repo / "a.py").write_text("x = 2\n")
    (repo / "has space.txt").write_text("s")
    assert gitops.status(repo) == {"a.py": " M", "has space.txt": "??", "out/deep/x.log": "??"}


def test_undo_writes_restores_tracked_files_and_moves_new_ones(repo, tmp_path_factory):
    head = gitops.head_commit(repo)
    (repo / "mine.txt").write_text("there before")
    start, dirs = gitops.status(repo), gitops.untracked_dirs(repo)
    (repo / "a.py").write_text("x = 99\n")
    (repo / "b.txt").unlink()
    (repo / "out").mkdir()
    (repo / "out" / "r.csv").write_text("1,2\n")
    (repo / "mine.txt").write_text("changed by the trial too")
    keep = tmp_path_factory.mktemp("keep")
    assert gitops.changed(start, gitops.status(repo)) == ["a.py", "b.txt", "out/r.csv"]
    assert gitops.undo_writes(repo, head, start, keep, dirs) == []
    assert (repo / "a.py").read_text() == "x = 1\n" and (repo / "b.txt").read_text() == "b\n"
    assert not (repo / "out").exists() and (keep / "out" / "r.csv").read_text() == "1,2\n"
    assert (keep / "a.py").read_text() == "x = 99\n", "what the trial wrote is kept"
    assert (repo / "mine.txt").read_text() == "changed by the trial too", "a path dirty at start is left alone"


def test_undo_writes_keeps_files_under_a_directory_that_replaced_a_tracked_file(repo, tmp_path_factory):
    head = gitops.head_commit(repo)
    (repo / "logs").mkdir()  # an empty directory of the user's, there before the trial
    dirs = gitops.untracked_dirs(repo)
    assert dirs == ["logs"]
    (repo / "b.txt").unlink()
    (repo / "b.txt").mkdir()
    (repo / "b.txt" / "trial.txt").write_text("trial output")
    (repo / "b.txt" / "__pycache__").mkdir()
    (repo / "b.txt" / "__pycache__" / "m.pyc").write_bytes(b"\0")
    (repo / "logs" / "debug.txt").write_text("d")
    (repo / "new" / "deep").mkdir(parents=True)
    (repo / "new" / "deep" / "x").write_text("x")
    keep = tmp_path_factory.mktemp("keep")
    assert gitops.undo_writes(repo, head, {}, keep, dirs) == []
    assert (repo / "b.txt").read_text() == "b\n"
    assert (keep / "b.txt" / "trial.txt").read_text() == "trial output", "moved before checkout removed the dir"
    assert (keep / "b.txt" / "__pycache__" / "m.pyc").exists(), "what status can't see is copied too"
    assert (repo / "logs").is_dir() and not any((repo / "logs").iterdir()), "a directory there before stays"
    assert not (repo / "new").exists(), "directories the trial made go"
    assert (keep / "logs" / "debug.txt").exists() and (keep / "new" / "deep" / "x").exists()


@pytest.mark.parametrize("replacement", ["file", "dangling-link"])
def test_undo_writes_restores_a_tracked_directory_the_trial_replaced(repo, tmp_path_factory, replacement):
    """Moving the new `d` aside put a file at keep/d; that must not stop d/f and d/g from coming back."""
    (repo / "d").mkdir()
    (repo / "d" / "f").write_text("f\n")
    (repo / "d" / "g").write_text("g\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "dir")
    head, dirs = gitops.head_commit(repo), gitops.untracked_dirs(repo)
    for name in ("f", "g"):
        (repo / "d" / name).unlink()
    (repo / "d").rmdir()
    if replacement == "file":
        (repo / "d").write_text("results\n")
    else:
        (repo / "d").symlink_to(repo / "no-such-cache")
    keep = tmp_path_factory.mktemp("keep")
    assert gitops.undo_writes(repo, head, {}, keep, dirs) == []
    assert gitops.status(repo) == {}
    assert (repo / "d" / "f").read_text() == "f\n" and (repo / "d" / "g").read_text() == "g\n"
    if replacement == "file":
        assert (keep / "d").read_text() == "results\n", "what the trial wrote is kept"
    else:
        assert (keep / "d").is_symlink()


def test_undo_writes_uses_a_free_name_when_the_keep_path_is_taken(repo, tmp_path_factory):
    """A keep dir reused across runs may hold a file where this run's path goes; nothing is skipped or lost."""
    head, dirs = gitops.head_commit(repo), gitops.untracked_dirs(repo)
    keep = tmp_path_factory.mktemp("keep")
    (keep / "a.py").write_text("earlier\n")
    (keep / "out").write_text("earlier file\n")
    (repo / "a.py").write_text("x = 99\n")
    (repo / "out").mkdir()
    (repo / "out" / "r.csv").write_text("1,2\n")
    assert gitops.undo_writes(repo, head, {}, keep, dirs) == []
    assert (repo / "a.py").read_text() == "x = 1\n" and not (repo / "out").exists()
    assert (keep / "a.py").read_text() == "earlier\n" and (keep / "dup-1" / "a.py").read_text() == "x = 99\n"
    assert (keep / "dup-1" / "out" / "r.csv").read_text() == "1,2\n"
