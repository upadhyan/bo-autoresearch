"""Toy for wrap-up: loss = 1 + term + noise. It imports nothing from the harness itself (the seed comes
from the trial file the runner is handed), so a distilled branch can be checked for having no harness
dependency at all. Tests plant their terms in `term`. TOY_SIGMA: gaussian noise drawn from the harness seed.
TOY_SLEEP: seconds per trial.
"""
import json
import os
import random
import time
from pathlib import Path


def train_and_eval():
    trial = json.loads(Path(os.environ["BOAUTORESEARCH_TRIAL"]).read_text())
    time.sleep(float(os.environ.get("TOY_SLEEP", 0)))
    term = 0.0
    rng = random.Random(trial["seed"])
    return 1.0 + term + float(os.environ.get("TOY_SIGMA", 0)) * rng.gauss(0, 1)
