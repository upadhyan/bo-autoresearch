"""Toy trainer with a planted analytic loss: scale * (1 + 1/epochs), plus planted truth from env.

TOY_SIGMA: gaussian noise drawn from the harness seed. TOY_SLEEP_PER_EPOCH: seconds per epoch.
Below TOY_CHEAP_BELOW epochs (a cheap rung): TOY_SCRAMBLE reverses the ranking, and
TOY_CHEAP_REPLICATE_SIGMA adds noise to baseline replicates only (so σ there is planted, ranks intact).
"""
import json
import os
import random
import sys
import time
from pathlib import Path

from boautoresearch import artifact_dir, fidelity, lever, log, seed


def _env(name):
    return float(os.environ.get(f"TOY_{name}", 0))


def train_and_eval():
    if os.environ.get("TOY_RAISE"):
        raise RuntimeError("toy trainer diverged")
    epochs = fidelity().get("epochs", 10)
    time.sleep(_env("SLEEP_PER_EPOCH") * epochs)
    scale, cheap = lever("H1.scale"), epochs < _env("CHEAP_BELOW")
    if cheap and _env("SCRAMBLE"):
        scale = 10 - scale
    loss = scale * (1 + 1 / epochs)
    rng = random.Random(seed())
    loss += _env("SIGMA") * rng.gauss(0, 1)
    trial_file = os.environ.get("BOAUTORESEARCH_TRIAL")
    if cheap and trial_file and json.loads(Path(trial_file).read_text())["kind"] == "baseline":
        loss += _env("CHEAP_REPLICATE_SIGMA") * rng.gauss(0, 1)
    for step in range(epochs):
        log("train_loss", loss + 1 / (step + 1), step=step)
    log("python", sys.executable)
    log("seed", seed())
    (artifact_dir() / "note.txt").write_text(f"loss={loss}\n")
    return loss, {"runtime_s": 0.0}
