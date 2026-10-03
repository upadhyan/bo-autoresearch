"""The plugin's `bin/boar` launcher: it runs the harness venv's entry point, or says what is missing.

The launcher is what Claude's Bash finds on PATH once the plugin is installed. These tests run copies of
it against stub venvs, so they need no network and never call uv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[2]
LAUNCHER = PLUGIN / "bin" / "boar"


def _plugin_copy(tmp_path: Path) -> Path:
    """A plugin root holding a copy of the launcher and no harness venv yet."""
    root = tmp_path / "plugin"
    (root / "bin").mkdir(parents=True)
    shutil.copy2(LAUNCHER, root / "bin" / "boar")
    return root


def test_launcher_is_executable():
    assert os.access(LAUNCHER, os.X_OK)


def test_launcher_runs_the_venv_entry_point_from_the_callers_directory(tmp_path):
    root = _plugin_copy(tmp_path)
    venv_bin = root / "harness" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    stub = venv_bin / "boar"
    stub.write_text("#!/bin/sh\necho \"${VIRTUAL_ENV-unset}\"\npwd -P\nprintf '%s\\n' \"$@\"\n")
    stub.chmod(0o755)
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    args = ["review", "record", "H1", "accept", "--reason", "two words"]
    p = subprocess.run(
        [str(root / "bin" / "boar"), *args], cwd=repo, env=env, capture_output=True, text=True, timeout=60
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == ["unset", str(repo.resolve()), *args]


@pytest.mark.parametrize("venv", ["missing", "interpreter gone"])
def test_launcher_without_uv_says_how_to_install_it(tmp_path, venv):
    root = _plugin_copy(tmp_path)
    if venv == "interpreter gone":
        venv_bin = root / "harness" / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(tmp_path / "uninstalled-python")
        (venv_bin / "boar").write_text("#!/bin/sh\nexit 0\n")
        (venv_bin / "boar").chmod(0o755)
    tools = tmp_path / "tools"  # PATH holds only what the launcher needs besides uv
    tools.mkdir()
    (tools / "dirname").symlink_to(shutil.which("dirname"))
    p = subprocess.run(
        [str(root / "bin" / "boar"), "next"], cwd=tmp_path, env={"PATH": str(tools)},
        capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 127
    assert "needs uv" in p.stderr and "https://astral.sh/uv/install.sh" in p.stderr
    assert p.stdout == ""
