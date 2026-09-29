"""Toy for BO rounds, with no levers yet: loss = 1 + noise. Tests add lever code (a planted optimum).

TOY_SIGMA: gaussian noise drawn from the harness seed. TOY_SLEEP: seconds per trial.
TOY_SLEEP_FROM: sleep only from this trial number on. Below TOY_CHEAP_BELOW epochs
(a cheap rung) with TOY_SCRAMBLE, the lever code's term is negated (the proxy ranks backwards).
"""
import json
import os
import random
import time
from pathlib import Path

from boautoresearch import fidelity, seed


def _env(name):
    return float(os.environ.get(f"TOY_{name}", 0))


def train_and_eval():
    n = json.loads(Path(os.environ["BOAUTORESEARCH_TRIAL"]).read_text())["trial"]
    if n >= _env("SLEEP_FROM"):
        time.sleep(_env("SLEEP"))
    flip = -1 if fidelity().get("epochs", 10) < _env("CHEAP_BELOW") and _env("SCRAMBLE") else 1
    term = 0.0
    loss = 1.0 + flip * term + _env("SIGMA") * random.Random(seed()).gauss(0, 1)
    return loss
