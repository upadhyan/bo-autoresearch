"""Run config: defaults, `init` overrides, and the metric direction from the spec."""

from __future__ import annotations

import re

from boar.errors import Refused

DEFAULTS: dict[str, int] = {
    "rounds": 12,
    "trials_per_round": 6,
    "repeats": 3,
    "trial_target_s": 600,
    "holdout_repeats": 6,
    "max_active": 6,
    "seed": 0,
}

# Keys that must be at least 1. trials_per_round needs 2: one trial each round re-measures the incumbent.
_POSITIVE = {"rounds", "trials_per_round", "repeats", "trial_target_s", "holdout_repeats", "max_active"}
_MINIMUM = {"trials_per_round": 2}
# Optuna takes seeds in [0, 2**32); the engine reduces seed + round modulo 2**32, so keep the given one in range.
_SEED_LIMIT = 2**32


def parse_overrides(tokens: list[str]) -> dict[str, int]:
    """Parse `init` overrides given as `--key value`, `--key=value` or `key=value`.

    Dashes in keys are read as underscores, so `--trials-per-round 4` works.
    """
    out: dict[str, int] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            body = tok[2:]
            if "=" in body:
                key, raw = body.split("=", 1)
            else:
                if i + 1 >= len(tokens):
                    raise Refused(f"config override {tok} has no value")
                key, raw = body, tokens[i + 1]
                i += 1
        elif "=" in tok:
            key, raw = tok.split("=", 1)
        else:
            raise Refused(f"can't parse config override {tok!r}; use --key value")
        key = key.replace("-", "_")
        if key not in DEFAULTS:
            raise Refused(f"unknown config key {key!r}; known keys: {', '.join(DEFAULTS)}")
        try:
            value = int(raw)
        except ValueError:
            raise Refused(f"config key {key} needs an integer, got {raw!r}") from None
        if key in _POSITIVE and value < _MINIMUM.get(key, 1):
            why = "; one trial each round re-measures the incumbent" if key in _MINIMUM else ""
            raise Refused(f"config key {key} must be at least {_MINIMUM.get(key, 1)}, got {value}{why}")
        if key == "seed" and not 0 <= value < _SEED_LIMIT:
            raise Refused(f"config key seed must be between 0 and {_SEED_LIMIT - 1}, got {value}")
        out[key] = value
        i += 1
    return out


def build_config(overrides: dict[str, int], direction: str) -> dict:
    """The frozen run config: defaults, then overrides, plus the metric direction."""
    config: dict = dict(DEFAULTS)
    config.update(overrides)
    config["direction"] = direction
    return config


# A line such as "Direction: min", "- **Direction**: maximize" or "direction = max".
_DIRECTION_LINE = re.compile(
    r"^[\s>*_-]*direction[*_]*\s*[:=]\s*[*_`]*\s*(minimi[sz]e|maximi[sz]e|min|max)\b",
    re.IGNORECASE | re.MULTILINE,
)


def direction_from_spec(spec_text: str) -> str:
    """Return "min" or "max" from the spec's `Direction:` line (the SKILL puts it under Metric)."""
    found = {("min" if m.group(1).lower().startswith("min") else "max") for m in _DIRECTION_LINE.finditer(spec_text)}
    if not found:
        raise Refused("spec has no direction; add a line 'Direction: min' or 'Direction: max' under Metric")
    if len(found) > 1:
        raise Refused("spec gives both directions; keep one 'Direction:' line under Metric")
    return found.pop()
