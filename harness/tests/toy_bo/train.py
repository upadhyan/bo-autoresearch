"""Toy for BO rounds, with no levers yet: loss = 1 + noise. Tests add lever code (a planted optimum).

TOY_SIGMA: gaussian noise drawn from the harness seed. TOY_REF_SIGMA: more noise, at fidelities of at
least TOY_CHEAP_BELOW epochs only. TOY_SLEEP: seconds per trial, from trial number TOY_SLEEP_FROM on,
and only for trials of kind TOY_SLEEP_KIND when that is set. TOY_SLEEP_PER_EPOCH: seconds per epoch.
Below TOY_CHEAP_BELOW epochs (a cheap rung) with TOY_SCRAMBLE, the lever code's term is negated
(the proxy ranks backwards).
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
    trial = json.loads(Path(os.environ["BOAUTORESEARCH_TRIAL"]).read_text())
    if trial["trial"] >= _env("SLEEP_FROM") and os.environ.get("TOY_SLEEP_KIND", trial["kind"]) == trial["kind"]:
        time.sleep(_env("SLEEP"))
    epochs = fidelity().get("epochs", 10)
    time.sleep(_env("SLEEP_PER_EPOCH") * epochs)
    cheap = epochs < _env("CHEAP_BELOW")
    flip = -1 if cheap and _env("SCRAMBLE") else 1
    term = 0.0
    rng = random.Random(seed())
    loss = 1.0 + flip * term + _env("SIGMA") * rng.gauss(0, 1)
    if not cheap:
        loss += _env("REF_SIGMA") * rng.gauss(0, 1)
    return loss
