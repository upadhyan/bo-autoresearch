"""Toy with no levers yet: loss = 3 * (1 + 1/epochs). Tests add lever code to it in the worktree.

TOY_SLEEP_PER_EPOCH: seconds per epoch.
"""
import os
import time

from boautoresearch import fidelity


def train_and_eval():
    epochs = fidelity().get("epochs", 10)
    time.sleep(float(os.environ.get("TOY_SLEEP_PER_EPOCH", 0)) * epochs)
    loss = 3.0 * (1 + 1 / epochs)
    return loss
