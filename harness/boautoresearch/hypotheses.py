"""Hypothesis specs (pre-registration), random in-range points, and the static lever-code check.

Pure functions; the CLI turns their ValueError into a refusal.
"""
import ast
import math
from collections import Counter

SPEC_KEYS = {"title", "rationale", "mechanism", "provenance", "source", "lens", "directives",
             "fidelity_sensitive", "fidelity_reason", "levers"}
LEVER_KEYS = {"float": {"low", "high", "log", "predicted"}, "int": {"low", "high", "log", "predicted"},
              "categorical": {"options"}, "bool": {"why_not_graded"}}
GRADED = ("float", "int")


def _text(d: dict, key: str, where: str = "", limit: int = 0) -> None:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{where}{key} is required and must be non-empty text")
    if limit and len(v) > limit:
        raise ValueError(f"{where}{key} must be {limit} characters or fewer, not {len(v)}")


def _number(v, integer: bool) -> bool:
    if isinstance(v, bool):
        return False
    return isinstance(v, int) if integer else isinstance(v, (int, float)) and math.isfinite(v)


def validate(spec) -> None:
    """Raise ValueError naming the first failing field of a hypothesis spec."""
    if not isinstance(spec, dict):
        raise ValueError("the spec must be a mapping")
    if unknown := set(spec) - SPEC_KEYS:
        raise ValueError(f"unknown spec fields: {sorted(unknown)}")
    _text(spec, "title", limit=80)
    for key in ("rationale", "mechanism", "lens"):
        _text(spec, key)
    if spec.get("provenance") not in ("novel", "adapted", "standard"):
        raise ValueError("provenance must be novel, adapted or standard")
    if spec["provenance"] == "adapted":
        _text(spec, "source")
    d = spec.get("directives")
    if not isinstance(d, list) or not all(isinstance(i, str) and i for i in d):
        raise ValueError("directives must list the directive ids the hypothesis touches ([] for none)")
    if not isinstance(spec.get("fidelity_sensitive"), bool):
        raise ValueError("fidelity_sensitive must be true or false")
    if spec["fidelity_sensitive"]:
        _text(spec, "fidelity_reason")
    levers = spec.get("levers")
    if not isinstance(levers, dict) or not levers:
        raise ValueError("levers must map at least one lever name to its declaration")
    for name, lv in levers.items():
        _validate_lever(name, lv)


def _validate_lever(name: str, lv) -> None:
    where = f"levers.{name}."
    if not name.isidentifier():
        raise ValueError(f"levers.{name}: a lever name must be an identifier (the harness prefixes H<n>.)")
    if not isinstance(lv, dict) or lv.get("kind") not in LEVER_KEYS:
        raise ValueError(f"{where}kind must be float, int, categorical or bool")
    kind = lv["kind"]
    if unknown := set(lv) - LEVER_KEYS[kind] - {"kind", "baseline", "path"}:
        raise ValueError(f"{where}: fields {sorted(unknown)} don't apply to a {kind} lever")
    if "path" in lv and not isinstance(lv["path"], str):
        raise ValueError(f"{where}path must be a config path")
    if "baseline" not in lv:
        raise ValueError(f"{where}baseline is required: the value at which the code runs unchanged")
    b = lv["baseline"]
    if kind in GRADED:
        integer = kind == "int"
        for key in ("low", "high"):
            if not _number(lv.get(key), integer):
                raise ValueError(f"{where}{key} must be {'an integer' if integer else 'a number'}")
        if lv["low"] >= lv["high"]:
            raise ValueError(f"{where}low must be below high")
        if not isinstance(lv.get("log", False), bool) or (lv.get("log") and lv["low"] <= 0):
            raise ValueError(f"{where}log must be true or false, and a log range needs low > 0")
        if not _number(b, integer) or not lv["low"] <= b <= lv["high"]:
            raise ValueError(f"{where}baseline {b!r} must be {kind} inside [{lv['low']}, {lv['high']}]")
        if lv.get("predicted") not in ("higher", "lower"):
            raise ValueError(f"{where}predicted must be higher or lower: the side of the baseline "
                             "where the improving values lie")
    elif kind == "categorical":
        opts = lv.get("options")
        if (not isinstance(opts, list) or len(opts) < 2
                or not all(isinstance(o, (str, int, float, bool)) for o in opts)
                or len(set(opts)) < len(opts)):  # set: 1, 1.0 and True are one value
            raise ValueError(f"{where}options must list at least 2 distinct scalar values")
        if not any(type(o) is type(b) and o == b for o in opts):
            raise ValueError(f"{where}baseline {b!r} must be one of the options")
    else:
        _text(lv, "why_not_graded", where)
        if not isinstance(b, bool):
            raise ValueError(f"{where}baseline must be true or false")


def random_point(levers: dict, rng) -> dict:
    """One uniformly random in-range value per lever (log-uniform for log ranges).

    A categorical or bool lever never draws its baseline, so the smoke runs its other branch.
    """
    out = {}
    for name, lv in levers.items():
        if lv["kind"] in GRADED:
            lo, hi = lv["low"], lv["high"]
            x = (math.exp(rng.uniform(math.log(lo), math.log(hi))) if lv.get("log")
                 else rng.uniform(lo, hi))
            out[name] = min(max(round(x), lo), hi) if lv["kind"] == "int" else x
        else:
            opts = lv["options"] if lv["kind"] == "categorical" else [False, True]
            out[name] = rng.choice([o for o in opts if o != lv["baseline"]])
    return out


# ponytail: a denylist of the usual config routes (env, argv, parsers, the trial/levers files);
# a determined author can still slip past it. Tighten with an allowlist if it gets gamed.
CONFIG_NAMES = {"environ", "getenv", "environb", "getenvb", "argv", "BOAUTORESEARCH_TRIAL"}
CONFIG_MODULES = {"argparse", "optparse", "getopt", "configparser", "click", "typer", "fire",
                  "hydra", "omegaconf", "dotenv", "tomllib", "toml", "tomli"}
CONFIG_STRINGS = ("levers.json", "BOAUTORESEARCH_TRIAL", "trial.json")


def _config_reads(tree: ast.AST) -> Counter:
    sites: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in CONFIG_NAMES:
            sites[node.id] += 1
        elif isinstance(node, ast.Attribute) and node.attr in CONFIG_NAMES:
            sites[node.attr] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            sites.update(f"import {m}" for m in mods if m.split(".")[0] in CONFIG_MODULES)
            if isinstance(node, ast.ImportFrom):  # `from os import getenv as g` hides the name
                sites.update(a.name for a in node.names if a.name in CONFIG_NAMES)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            sites.update(f"{s!r}" for s in CONFIG_STRINGS if s in node.value)
    return sites


def _lever_reads(tree: ast.AST) -> list:
    """The argument of every lever(...) call: a str literal, or None for anything else."""
    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)) == "lever":
                a = node.args[0] if len(node.args) == 1 and not node.keywords else None
                reads.append(a.value if isinstance(a, ast.Constant) and isinstance(a.value, str)
                             else None)
    return reads


def check_lever_code(changed: dict, levers: list, declared: set) -> None:
    """Refuse lever code that reads config other than through lever().

    changed maps each changed .py path to (source before, source after; None if deleted).
    levers are the hypothesis's lever names, each of which must be read with lever("<name>");
    declared are all lever names the code may read.
    """
    read: set = set()
    for path, (before, after) in sorted(changed.items()):
        if after is None:
            continue
        try:
            new, old = ast.parse(after, path), ast.parse(before or "", path)
        except SyntaxError as e:
            raise ValueError(f"{path} does not parse: {e}")
        if added := _config_reads(new) - _config_reads(old):
            raise ValueError(f"{path} reads config other than through lever(): {sorted(added)}")
        for name in _lever_reads(new):
            if name is None:
                raise ValueError(f"{path}: lever() must be called with a literal lever name")
            if name not in declared:
                raise ValueError(f"{path} reads lever {name!r}, which no hypothesis declares")
            read.add(name)
    if missing := [n for n in levers if n not in read]:
        raise ValueError(f"the changed code never reads {missing} through lever(\"<name>\")")
