"""A small training loop over a synthetic task: a toy trainer.

Its settings live in CONFIG. A research change sets them in `train_and_eval` (where the comment says),
before training starts; CONFIG itself holds the values the code runs with today. The validation loss
comes from objective.py (protected). runner.py passes in the epochs and the seed.
"""
import time

from objective import evaluate

SLEEP_PER_EPOCH = 0.03

CONFIG = {
    "lr_log2": 0.0,  # log2 multiplier on the base learning rate
    "warmup_frac": 0.0,  # share of training spent ramping the learning rate up
    "cosine_tail": 0.0,  # share of training spent in a long cosine-decay tail
    "grad_clip": 0.0,  # gradient-clipping strength (0: no clipping)
    "weight_decay": 0.0,
    "label_smoothing": 0.0,  # label-smoothing epsilon
    "smoothing_prior": 0.0,  # how far smoothing mixes toward the class prior instead of uniform
    "smoothing_ramp": 0.0,  # share of training over which the epsilon ramps in
    "ema": 0.0,  # strength of the exponential moving average of the weights
    "distill_weight": 0.0,  # weight of a distillation loss (0: plain one-hot targets)
    "mixed_precision": False,
    "pretrained_init": False,
    "attention_block": False,
}


def train_and_eval(epochs=8, seed=0):
    cfg = dict(CONFIG)
    # research changes to cfg go here

    for _ in range(epochs):
        time.sleep(SLEEP_PER_EPOCH)  # one epoch of "training"
    return evaluate(cfg, epochs, seed)
