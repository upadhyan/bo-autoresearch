"""Toy trainer with a planted analytic loss: scale * (1 + 1/epochs)."""
import os
import sys

from boautoresearch import artifact_dir, fidelity, lever, log, seed


def train_and_eval():
    if os.environ.get("TOY_RAISE"):
        raise RuntimeError("toy trainer diverged")
    epochs = fidelity().get("epochs", 10)
    loss = lever("H1.scale") * (1 + 1 / epochs)
    for step in range(epochs):
        log("train_loss", loss + 1 / (step + 1), step=step)
    log("python", sys.executable)
    log("seed", seed())
    (artifact_dir() / "note.txt").write_text(f"loss={loss}\n")
    return loss, {"runtime_s": 0.0}
