"""PROTOTYPE — a toy "user project" after the agent added lever code for two hypotheses.

Objective (minimise): a noisy bowl. Planted truth:
  H1 warmup really helps (optimum warmup_frac ~0.3), and helps more with a higher lr.
  H2 label smoothing does nothing.
At baseline (warmup_frac=0, smoothing=0) the code behaves exactly as before the levers.
"""
import random

from bo import artifact_dir, fidelity, lever, log, seed


def train_and_eval():
    rng = random.Random(seed())
    epochs = fidelity()["epochs"]
    lr = 0.1  # user's existing hyperparameter, untouched

    # --- H1.v1: "Warmup stabilises early training" (lever code added by the agent)
    warmup = lever("H1.warmup_frac")          # baseline 0.0 == old code path
    if warmup > 0:
        lr = lr * (1 + 2 * warmup)            # toy stand-in for a warmup schedule

    # --- H2.v1: "Label smoothing reduces overconfidence"
    smoothing = lever("H2.smoothing")         # baseline 0.0; planted as useless

    loss = 1.0 - 0.02 * epochs + (warmup - 0.3) ** 2 * (1 + lr) + 0 * smoothing
    loss += rng.gauss(0, 0.01)
    for step in range(epochs):
        log("train_loss", loss + 0.5 / (step + 1), step=step)
    log("grad_norm", 1.3)
    (artifact_dir() / "note.txt").write_text(f"loss={loss:.4f}\n")
    return loss, {"runtime_s": 0.01 * epochs}
