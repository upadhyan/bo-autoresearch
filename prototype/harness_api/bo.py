"""PROTOTYPE — throwaway. Runner-side surface of the harness.

This is everything lever code and runner scripts may import. Six names:
    lever, seed, fidelity, log, artifact_dir, run
Everything else (sampling, seeds, replicates, budget, verdicts) is the harness's
business and is invisible from here.
"""
import json
import time
from pathlib import Path

# Set by the harness for the duration of one trial. None = running outside the
# harness (plain `python train.py`), in which case lever() returns the committed
# default: the baseline during research, the incumbent value on the distilled branch.
_trial = None
_LEVER_DEFAULTS = json.loads(Path("levers.json").read_text()) if Path("levers.json").exists() else {}


def lever(name):
    """Value of a declared lever for this trial. The ONLY way lever code reads config."""
    if _trial is None:
        return _LEVER_DEFAULTS[name]
    if name not in _trial["levers"]:
        raise KeyError(f"{name} is not a declared lever in this round")
    return _trial["levers"][name]  # the *effective* value (masking already applied)


def seed():
    """Harness-owned seed. Replicates get a fresh one; the runner never picks its own."""
    return _trial["seed"] if _trial else 0


def fidelity():
    """The round's fidelity rung, e.g. {"epochs": 3}. Reference fidelity outside the harness."""
    return _trial["fidelity"] if _trial else {"epochs": 20}


def log(name, value, step=None):
    """Telemetry. Scalars (step=None) land on the trial record; stepped values go to
    a curve in the artifact dir. Never feeds the sampler; there is no pruning."""
    if _trial is None:
        return
    if step is None:
        _trial["telemetry"][name] = value
    else:
        _trial["curves"].setdefault(name, []).append((step, value))


def artifact_dir():
    """Per-trial directory for checkpoints, plots, anything. Kept forever."""
    d = Path(_trial["artifact_dir"]) if _trial else Path("artifacts/local")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(fn):
    """Runner entry point. fn() returns the objective, or (objective, {constraint: value}).
    The harness measures wall-clock, heartbeats and catches exceptions (-> trial_failed)."""
    t0 = time.monotonic()
    out = fn()
    objective, constraints = out if isinstance(out, tuple) else (out, {})
    result = {"objective": float(objective), "constraints": constraints,
              "wall_clock_s": time.monotonic() - t0}
    if _trial is None:
        print(json.dumps(result))
    return result
