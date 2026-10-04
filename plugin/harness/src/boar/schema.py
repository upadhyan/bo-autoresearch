"""Validation of proposal files (design "Schemas").

Every error is collected and reported at once, so the agent fixes a file in one
pass instead of one refusal per mistake.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

SOURCES = ("user", "research", "synthesis")
LEVER_TYPES = ("bool", "int", "float", "categorical")
LEVER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")

_PROPOSAL_REQUIRED = ("statement", "mechanism", "source", "levers")
_PROPOSAL_KEYS = {*_PROPOSAL_REQUIRED, "citations", "supersedes", "enables", "joint_config"}
_LEVER_KEYS = {
    "bool": {"name", "type", "default"},
    "int": {"name", "type", "low", "high", "log", "default"},
    "float": {"name", "type", "low", "high", "log", "default"},
    "categorical": {"name", "type", "choices", "default"},
}


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return (_is_int(x) or isinstance(x, float)) and math.isfinite(x)


def _is_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, bool, int)) or (isinstance(x, float) and math.isfinite(x))


def _same(a: Any, b: Any) -> bool:
    """Equal and of the same JSON type, so `true` never matches `1` and `1` never matches `1.0`."""
    return type(a) is type(b) and a == b


def validate(
    data: Any, existing_levers: set[str], active_ids: set[str], allow_supersedes: bool,
    blocked_ids: set[str] = frozenset(), space: dict[str, dict] | None = None,
) -> list[str]:
    """Every problem with a proposal file, as messages naming the proposal index and lever.

    `blocked_ids` are the hypotheses an `enables` may name; `space` holds the active levers a `joint_config` may set.
    """
    if not isinstance(data, list):
        return ["the proposal file must hold a JSON list of proposals"]
    if not data:
        return ["the proposal file is empty; in R4 use `boar propose --none --reason …` instead"]
    errors: list[str] = []
    seen: dict[str, str] = {}
    for i, p in enumerate(data):
        where = f"proposal[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{where}: must be a JSON object")
            continue
        for key in _PROPOSAL_REQUIRED:
            if key not in p:
                errors.append(f"{where}: missing {key!r}")
        for key in sorted(set(p) - _PROPOSAL_KEYS):
            errors.append(f"{where}: unknown key {key!r} (allowed: {', '.join(sorted(_PROPOSAL_KEYS))})")
        for key in ("statement", "mechanism"):
            if key in p and not (isinstance(p[key], str) and p[key].strip()):
                errors.append(f"{where}: {key!r} must be a non-empty string")
        if "source" in p and p["source"] not in SOURCES:
            errors.append(f"{where}: 'source' must be one of {', '.join(SOURCES)}, got {p['source']!r}")
        citations = p.get("citations", [])
        if not (isinstance(citations, list) and all(isinstance(c, str) for c in citations)):
            errors.append(f"{where}: 'citations' must be a list of strings")
        sup = p.get("supersedes")
        if sup is not None:
            if not allow_supersedes:
                errors.append(f"{where}: 'supersedes' must be null in setup (nothing is active yet)")
            elif not isinstance(sup, str):
                errors.append(f"{where}: 'supersedes' must be null or a single hypothesis id string, got {sup!r}")
            elif sup not in active_ids:
                listed = ", ".join(sorted(active_ids)) or "none"
                errors.append(f"{where}: 'supersedes' must name an active hypothesis (active: {listed}), got {sup!r}")
        enables, joint = p.get("enables"), p.get("joint_config")
        if enables is not None:
            if not allow_supersedes:
                errors.append(f"{where}: 'enables' must be null in setup (nothing is blocked yet)")
            elif not isinstance(enables, str) or enables not in blocked_ids:
                listed = ", ".join(sorted(blocked_ids)) or "none"
                errors.append(f"{where}: 'enables' must name a hypothesis decided blocked this round (blocked: {listed}), got {enables!r}")
            if joint is None:
                errors.append(f"{where}: an 'enables' proposal needs a 'joint_config': the partial config of its joint trial")
        elif joint is not None:
            errors.append(f"{where}: 'joint_config' goes with 'enables'; leave it null otherwise")
        if "levers" not in p:
            continue
        levers = p["levers"]
        if not isinstance(levers, list) or not levers:
            errors.append(f"{where}: 'levers' must be a non-empty list")
            continue
        found: list[str] = []
        for j, lever in enumerate(levers):
            found.extend(_lever_errors(f"{where}.levers[{j}]", lever, existing_levers, seen))
        errors.extend(found)
        if enables is not None and joint is not None and not found:
            errors.extend(_joint_errors(where, joint, levers, space or {}))
    return errors


def _joint_errors(where: str, joint: Any, levers: list[dict], space: dict[str, dict]) -> list[str]:
    from boar import warmstart

    if not isinstance(joint, dict) or not joint:
        return [f"{where}: 'joint_config' must be a non-empty object of lever values"]
    own = {lever["name"]: lever for lever in levers}
    errors = []
    for name, value in joint.items():
        lever = own.get(name) or space.get(name)
        if lever is None:
            errors.append(f"{where}: 'joint_config': {name!r} is not one of its levers or a lever of an active hypothesis")
        elif not warmstart.allowed(lever, value):
            errors.append(f"{where}: 'joint_config': {json.dumps(value)} is not a valid value of {name!r}")
    if not any(n in own and not _same(v, own[n]["default"]) for n, v in joint.items()):
        errors.append(f"{where}: 'joint_config' must set at least one of its own levers away from default")
    return errors


def _lever_errors(where: str, lever: Any, existing: set[str], seen: dict[str, str]) -> list[str]:
    if not isinstance(lever, dict):
        return [f"{where}: must be a JSON object"]
    name = lever.get("name")
    if isinstance(name, str):
        where = f"{where} {name!r}"
    errors: list[str] = []
    if not isinstance(name, str) or not LEVER_NAME.match(name):
        errors.append(f"{where}: 'name' must match {LEVER_NAME.pattern}, got {name!r}")
    elif name in existing:
        errors.append(f"{where}: lever name is already used in this run; lever names are unique across the run")
    elif name in seen:
        errors.append(f"{where}: lever name is also used by {seen[name]}")
    else:
        seen[name] = where.rsplit(" ", 1)[0]
    kind = lever.get("type")
    if kind not in LEVER_TYPES:
        errors.append(f"{where}: 'type' must be one of {', '.join(LEVER_TYPES)}, got {kind!r}")
        return errors
    for key in sorted(set(lever) - _LEVER_KEYS[kind]):
        errors.append(f"{where}: unknown key {key!r} for a {kind} lever (allowed: {', '.join(sorted(_LEVER_KEYS[kind]))})")
    if "default" not in lever:
        errors.append(f"{where}: missing 'default' (it must reproduce the baseline behaviour)")
    default = lever.get("default")
    if kind == "bool":
        if "default" in lever and not isinstance(default, bool):
            errors.append(f"{where}: default must be true or false, got {default!r}")
    elif kind in ("int", "float"):
        errors.extend(_range_errors(where, lever, kind))
    else:
        choices = lever.get("choices")
        if not isinstance(choices, list) or not choices:
            errors.append(f"{where}: 'choices' must be a non-empty list")
        elif not all(_is_scalar(c) for c in choices):
            errors.append(f"{where}: every choice must be a string, number, boolean or null")
        else:
            if any(choices[k] == c for i, c in enumerate(choices) for k in range(i)):
                errors.append(f"{where}: 'choices' must be distinct (note true equals 1 and 1 equals 1.0)")
            if "default" in lever and not any(_same(c, default) for c in choices):
                errors.append(f"{where}: default {default!r} is not one of the choices")
    return errors


def _range_errors(where: str, lever: dict, kind: str) -> list[str]:
    ok = _is_int if kind == "int" else _is_number
    noun = "an integer" if kind == "int" else "a finite number"
    errors = []
    for key in ("low", "high"):
        if key not in lever:
            errors.append(f"{where}: missing {key!r}")
        elif not ok(lever[key]):
            errors.append(f"{where}: {key!r} must be {noun}, got {lever[key]!r}")
    log = lever.get("log", False)
    if not isinstance(log, bool):
        errors.append(f"{where}: 'log' must be true or false, got {log!r}")
    default = lever.get("default")
    if "default" in lever and not ok(default):
        errors.append(f"{where}: default must be {noun}, got {default!r}")
    if errors:
        return errors
    low, high = lever["low"], lever["high"]
    if not low < high:
        errors.append(f"{where}: needs low < high, got low={low!r} high={high!r}")
    elif "default" in lever and not low <= default <= high:
        errors.append(f"{where}: default {default!r} is outside [{low!r}, {high!r}]")
    if log is True and not low > 0:
        errors.append(f"{where}: a log-scale lever needs low > 0, got low={low!r}")
    return errors


def normalize_lever(lever: dict) -> dict:
    """The stored form: keys in a fixed order, `log` explicit for numeric levers, float bounds as floats."""
    kind = lever["type"]
    out: dict[str, Any] = {"name": lever["name"], "type": kind}
    if kind == "int":
        out.update(low=lever["low"], high=lever["high"], log=lever.get("log", False))
    elif kind == "float":
        out.update(low=float(lever["low"]), high=float(lever["high"]), log=lever.get("log", False))
    elif kind == "categorical":
        out["choices"] = list(lever["choices"])
    out["default"] = float(lever["default"]) if kind == "float" else lever["default"]
    return out
