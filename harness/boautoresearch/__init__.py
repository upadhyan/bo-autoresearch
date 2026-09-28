"""Runner-side interface of the harness: the only names lever code and runner scripts import.

Inside a trial the harness points BOAUTORESEARCH_TRIAL at the trial's resolved file.
Outside the harness (plain `python runner.py`) lever() reads the committed levers.json
in the working directory, so the code still runs on its own.
"""
import json
import os
import tempfile
from pathlib import Path

__all__ = ["lever", "seed", "fidelity", "log", "artifact_dir", "run"]

_TRIAL_FILE = os.environ.get("BOAUTORESEARCH_TRIAL")
_trial = json.loads(Path(_TRIAL_FILE).read_text()) if _TRIAL_FILE else None
_telemetry: dict = {}
_curves: dict = {}
_local_dir = None


def lever(name):
    """Effective value of a lever for this trial. The ONLY way lever code reads config."""
    levers = _trial["levers"] if _trial else json.loads(Path("levers.json").read_text())
    if name not in levers:
        raise KeyError(f"{name} is not a declared lever")
    return levers[name]


def seed():
    """The harness-assigned seed. 0 outside the harness."""
    return _trial["seed"] if _trial else 0


def fidelity():
    """This round's rung, e.g. {"epochs": 3}. Empty outside the harness: use the code's own defaults."""
    return dict(_trial["fidelity"]) if _trial else {}


def log(name, value, step=None):
    """Telemetry only. Scalars land on the trial; stepped values go to curves.json."""
    if _trial is None:
        return
    if step is None:
        _telemetry[name] = value
    else:
        _curves.setdefault(name, []).append([step, value])


def artifact_dir():
    """The trial's artifact folder, created on demand. A temp folder outside the harness."""
    global _local_dir
    if _trial:
        d = Path(_trial["artifact_dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d
    if _local_dir is None:  # never write into the user's checkout
        _local_dir = Path(tempfile.mkdtemp(prefix="boautoresearch-"))
    return _local_dir


def run(fn):
    """Runner entry point. fn() returns the objective, or (objective, {constraint: value})."""
    out = fn()
    objective, constraints = out if isinstance(out, tuple) else (out, {})
    result = {"objective": float(objective),
              "constraints": {k: float(v) for k, v in constraints.items()}}
    print(json.dumps(result))
    if _trial:
        d = artifact_dir()
        if _curves:
            (d / "curves.json").write_text(json.dumps(_curves))
        (d / "result.json").write_text(json.dumps({**result, "telemetry": _telemetry}))
