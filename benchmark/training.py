"""The shared hard scenario of #47-#49: a training run tuned for dev error, with 20 levers, interactions, a memory
limit that crashes, a noisy two-lever guard and a removal. bench.simulate runs it as a target:

    bench.simulate(variant, seed, cfg, training.TARGETS["training"])

docs/experiments/47-ax-gp.md describes it and why each part is there.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

import bench
from boar import evalrun, stats, store

# --- the target: a model trained for a fixed wall-clock time, scored on dev error (%) ---------------------------


def _float(name: str, low: float, high: float, default: float, log: bool = False) -> dict:
    return {"name": name, "type": "float", "low": low, "high": high, "log": log, "default": default}


def _int(name: str, low: int, high: int, default: int, log: bool = False) -> dict:
    return {"name": name, "type": "int", "low": low, "high": high, "log": log, "default": default}


def _bool(name: str) -> dict:
    return {"name": name, "type": "bool", "default": False}


def _choice(name: str, *choices: str) -> dict:
    return {"name": name, "type": "categorical", "choices": list(choices), "default": choices[0]}


PROPOSALS = {
    "H1": bench._proposal("A larger learning rate, scaled with the batch, converges further in the time budget",
                          _float("lr_scale", 0.1, 10.0, 1.0, log=True), _float("warmup_frac", 0.0, 0.3, 0.0)),
    "H2": bench._proposal("Bigger batches use the GPU better", _int("batch", 16, 1024, 64, log=True)),
    "H3": bench._proposal("Pinned memory and more loader workers feed the GPU faster",
                          _bool("pin_memory"), _int("num_workers", 0, 16, 0)),
    "H4": bench._proposal("An auxiliary reconstruction loss regularises the encoder",
                          _float("aux_weight", 0.0, 1.0, 0.0)),
    "H5": bench._proposal("Wider hidden layers fit better", _int("width", 64, 1024, 128, log=True)),
    "H6": bench._proposal("Gradient checkpointing frees activation memory", _bool("checkpointing")),
    "H7": bench._proposal("Stronger augmentation generalises better", _float("aug_strength", 0.0, 1.0, 0.0)),
    "H8": bench._proposal("Label smoothing calibrates the classifier", _float("label_smoothing", 0.0, 0.3, 0.0)),
    "H9": bench._proposal("Dropout regularises wide layers", _float("dropout", 0.0, 0.5, 0.0)),
    "H10": bench._proposal("A cosine schedule with a floor anneals better",
                           _choice("schedule", "constant", "step", "cosine"), _float("lr_floor", 0.0, 0.5, 0.0)),
    "H11": bench._proposal("Mixed precision runs more steps in the budget",
                           _choice("precision", "fp32", "bf16", "fp16")),
    "H12": bench._proposal("An average of the weights smooths the final model",
                           _bool("ema"), _int("ema_halflife", 10, 10000, 1000, log=True)),
    "H13": bench._proposal("A larger init scale breaks symmetry sooner", _float("init_scale", 1.0, 10.0, 1.0)),
    "H14": bench._proposal("Weight decay curbs overfitting", _float("weight_decay", 1e-6, 0.1, 1e-6, log=True)),
    "H15": bench._proposal("cuDNN autotuning and less logging save step time",
                           _bool("cudnn_benchmark"), _int("log_every", 1, 1000, 1, log=True)),
}
DEFAULTS = {lever["name"]: lever["default"] for p in PROPOSALS.values() for lever in p["levers"]}

MEMORY = 393216  # activations fit if batch × width × bytes per value (× 0.35 with checkpointing) stays within this
BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}
FLOOR = 0.73  # the guard: the regression head's R² stays at or above this
NOISE = 0.05  # relative sd of the noise every config shares at a repeat, and of each config's own noise
GUARD_NOISE = 0.01  # the same for the measured R², in R² units

# What the code does at a commit: whether aux_weight changes how lr behaves (#48), and how far a commit has moved
# the guard (#49).
WORLD = {"aux_interacts": True, "r2_shift": 0.0}


def _scenario(round7: dict) -> list[dict]:
    """The rounds, as bench.SCENARIO; only round 7 differs between targets. 19 levers are active from round 5."""
    return [
        {"add": ["H1", "H2", "H3", "H7", "H10", "H13", "H14"]},
        {"add": ["H4", "H5", "H8"]},  # the guard can break; fp32 only, so the memory limit bounds batch × width
        {"add": ["H6", "H11"]},  # fp16 or checkpointing puts the limit past the best batch and width
        {"add": ["H9"], "investigate": {"H4": [{"aux_weight": 0.1}]}},  # H4's off-state and a probe, then removed
        {"remove": ["H4"], "add": ["H12", "H15"]},
        {},  # R1 changed nothing: the same commit
        round7,
        {},
        {},
    ]


def _dynamics(lr, warmup, batch, lr_best):
    """Error from the learning rate, its warmup and the batch, for floats or numpy arrays: lr is best at `lr_best`
    (scaled with √batch), warmup's best grows with lr, and past 3× its best lr the run diverges and recovers late."""
    return (1.5 * np.log(lr / lr_best) ** 2 + 20 * (warmup - 0.05 - 0.04 * np.log(lr)) ** 2
            + 2.0 * (lr > 3 * lr_best) + 0.4 * np.log(batch / 192) ** 2)


def _capacity(width, dropout):
    """Error from width and dropout, for floats or numpy arrays: a wider model fits better but runs fewer steps in
    the budget, and the best dropout grows with width."""
    return (2.0 * (128 / width) ** 0.8 + 0.5 * np.log(width / 128)
            + 4 * (dropout - 0.05 - 0.06 * np.log(width / 128)) ** 2)


def _regularisation(aug, smoothing):
    return 1.5 * (aug - 1) ** 2 + 8 * (smoothing - 0.25) ** 2


def _lr_best(c: dict, world: dict) -> float:
    return math.sqrt(c["batch"] / 64) * (1 + 1.5 * c["aux_weight"] if world["aux_interacts"] else 1)


def crashes(config: dict) -> bool:
    """Whether the config runs out of memory: a crash at every commit."""
    c = {**DEFAULTS, **config}
    return c["batch"] * c["width"] * BYTES[c["precision"]] * (0.35 if c["checkpointing"] else 1) > MEMORY


def reg_r2(config: dict, world: dict) -> float:
    """The guard metric without noise: augmentation and smoothing together, or much aux_weight, cost R²."""
    c = {**DEFAULTS, **config}
    return (0.80 - 0.06 * c["aug_strength"] ** 2 - 0.4 * c["aug_strength"] * c["label_smoothing"]
            - 0.1 * c["aux_weight"] + world["r2_shift"])


def true_metric(config: dict, world: dict) -> tuple[float, bool]:
    """Noise-free dev error, and whether the noise-free guard holds. Levers not defined yet are at default."""
    c = {**DEFAULTS, **config}
    error = (
        3.0
        + _dynamics(c["lr_scale"], c["warmup_frac"], c["batch"], _lr_best(c, world))
        + _capacity(c["width"], c["dropout"])
        + _regularisation(c["aug_strength"], c["label_smoothing"])
        + {"constant": 0.5, "step": 0.3, "cosine": 0.15 + 4 * (c["lr_floor"] - 0.25) ** 2}[c["schedule"]]
        + {"fp32": 0.25, "bf16": 0.05, "fp16": 0.0}[c["precision"]] + 0.12 * c["checkpointing"]  # fewer steps
        + (-0.25 + 0.2 * math.log(c["ema_halflife"] / 300) ** 2 if c["ema"] else 0.0)
        + 0.6 * ((c["init_scale"] - 4) / 3) ** 2
        + 0.05 * (math.log10(c["weight_decay"]) + 3.5) ** 2
        + 0.6 * c["aux_weight"]
    )  # pin_memory, num_workers, cudnn_benchmark and log_every change nothing
    return float(error), reg_r2(c, world) >= FLOOR


def best_config(world: dict, space=None) -> dict:
    """The feasible config with the lowest true metric at `world` over the levers in `space` (all if None), the
    others at default. Batch, width, precision and checkpointing are enumerated, with lr, warmup and dropout at their
    best for each; augmentation runs along a fine grid with smoothing at its best for each; every other lever acts
    alone. aux_weight's best is 0: it only adds error, and lr can follow how it moves lr's best."""
    names = set(DEFAULTS if space is None else space)

    def values(name: str, grid) -> np.ndarray:
        return np.asarray(grid if name in names else [DEFAULTS[name]])

    batch, width = values("batch", range(16, 1025)), values("width", range(64, 1025))
    lr_best = np.sqrt(batch / 64)
    lr = lr_best if "lr_scale" in names else np.full(batch.shape, DEFAULTS["lr_scale"])
    warmup = np.clip(0.05 + 0.04 * np.log(lr), 0, 0.3) if "warmup_frac" in names else 0.0
    dropout = 0.05 + 0.06 * np.log(width / 128) if "dropout" in names else 0.0
    dynamics, capacity = _dynamics(lr, warmup, batch, lr_best), _capacity(width, dropout)
    cells = []
    for precision in values("precision", PROPOSALS["H11"]["levers"][0]["choices"]):
        for ckpt in values("checkpointing", [False, True]):
            memory = batch[:, None] * width[None, :] * BYTES[precision] * (0.35 if ckpt else 1)
            error = dynamics[:, None] + capacity[None, :] + {"fp32": 0.25, "bf16": 0.05, "fp16": 0.0}[precision]
            error = np.where(memory > MEMORY, np.inf, error + 0.12 * ckpt)
            i, j = np.unravel_index(np.argmin(error), error.shape)
            cells.append((error[i, j], {"batch": int(batch[i]), "width": int(width[j]), "precision": str(precision),
                                        "checkpointing": bool(ckpt), "lr_scale": float(lr[i]),
                                        "warmup_frac": float(np.broadcast_to(warmup, lr.shape)[i]),
                                        "dropout": float(np.broadcast_to(dropout, width.shape)[j])}))
    best = min(cells, key=lambda cell: cell[0])[1]
    # Smoothing as high as the guard lets it go, up to its own best of 0.25, a hair inside the floor.
    aug = values("aug_strength", np.linspace(0, 1, 100001))
    room = 0.80 + world["r2_shift"] - FLOOR - 0.06 * aug**2
    with np.errstate(divide="ignore"):
        smoothing = np.clip(np.minimum(0.25, room / (0.4 * aug)) - 1e-9, 0, 0.3) if "label_smoothing" in names else 0.0
    smoothing = np.broadcast_to(smoothing, aug.shape)
    i = np.argmin(np.where(room - 0.4 * aug * smoothing >= 0, _regularisation(aug, smoothing), np.inf))
    best |= {"aug_strength": float(aug[i]), "label_smoothing": float(smoothing[i])}
    best |= {"schedule": "cosine", "lr_floor": 0.25, "ema": True, "ema_halflife": 300, "init_scale": 4.0,
             "weight_decay": 10**-3.5, "aux_weight": 0.0}
    return {name: value for name, value in {**DEFAULTS, **best}.items() if name in names}


def measure(config: dict, world: dict, commit: str, seed: int, repeats: int) -> dict:
    """One trial as bench.measure records it, with the guard measured on each repeat: R² carries noise paired like
    the metric's, and a repeat below the floor breaks the guard and stops the trial. `margin` is the lowest measured
    R² minus the floor, below 0 exactly when the guard broke; None for a crash."""
    c = {**DEFAULTS, **config}
    if crashes(c):
        return {"state": evalrun.FAILED, "metric": None, "repeats": [], "runs": 1, "margin": None}
    true, r2, key = true_metric(c, world)[0], reg_r2(c, world), store.config_key(c, DEFAULTS)
    values, margins = [], []
    for k in range(1, repeats + 1):
        values.append(true * (1 + NOISE * (bench._normal(seed, "dev", k) + bench._normal(seed, commit, key, "dev", k))))
        margins.append(r2 - FLOOR + GUARD_NOISE * (bench._normal(seed, "guard", k)
                                                   + bench._normal(seed, commit, key, "guard", k)))
        if margins[-1] < 0:
            break
    state = evalrun.COMPLETE if margins[-1] >= 0 else evalrun.INFEASIBLE
    return {"state": state, "metric": stats.median(values), "repeats": values, "runs": len(values),
            "margin": min(margins)}


def _target(world: dict, round7: dict) -> SimpleNamespace:
    return SimpleNamespace(PROPOSALS=PROPOSALS, DEFAULTS=DEFAULTS, WORLD={**WORLD, **world}, SCENARIO=_scenario(round7),
                           measure=measure, true_metric=true_metric, best_config=best_config)


# Round 7 is a new commit on each; only `training` moves the guard there.
TARGETS = {
    "training": _target({}, {"world": {"r2_shift": 0.06}}),  # a fix to the regression head loosens the guard
    "training-fixed-guard": _target({}, {"world": {}}),
    "training-additive": _target({"aux_interacts": False}, {"world": {}}),  # aux_weight only adds error
}
