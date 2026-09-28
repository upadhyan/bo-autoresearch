"""Toy trainer with a planted analytic loss: scale * (1 + 1/epochs), plus planted truth from env.

TOY_SIGMA: gaussian noise drawn from the harness seed. TOY_SLEEP_PER_EPOCH: seconds per epoch.
TOY_SCRAMBLE_BELOW: below this many epochs the ranking is reversed (a broken cheap rung).
"""
import os
import random
import sys
import time

from boautoresearch import artifact_dir, fidelity, lever, log, seed


def train_and_eval():
    if os.environ.get("TOY_RAISE"):
        raise RuntimeError("toy trainer diverged")
    epochs = fidelity().get("epochs", 10)
    time.sleep(float(os.environ.get("TOY_SLEEP_PER_EPOCH", 0)) * epochs)
    scale = lever("H1.scale")
    if epochs < float(os.environ.get("TOY_SCRAMBLE_BELOW", 0)):
        scale = 10 - scale
    loss = scale * (1 + 1 / epochs)
    loss += float(os.environ.get("TOY_SIGMA", 0)) * random.Random(seed()).gauss(0, 1)
    for step in range(epochs):
        log("train_loss", loss + 1 / (step + 1), step=step)
    log("python", sys.executable)
    log("seed", seed())
    (artifact_dir() / "note.txt").write_text(f"loss={loss}\n")
    return loss, {"runtime_s": 0.0}
