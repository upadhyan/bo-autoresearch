"""The directive registry: validation, predicates (the allowed region), compat flags, forbidden
patterns, similarity and the protected-path manifest.

Pure functions (the manifest reads files); the CLI turns their ValueError into a refusal.
"""
import ast
import fnmatch
import functools
import hashlib
import itertools
import math
import operator
import re
from pathlib import Path
from typing import Any

from .hypotheses import GRADED, options

BRIEF_KEYS = ("purpose", "contribution", "complexity", "provenance")
DIRECTIVE_KEYS = {"id", "severity", "statement", "reason", "scope", "predicate", "forbidden_patterns"}
REGISTRY_KEYS = {"brief", "directives", "protected_paths"}


def _text(d: dict, key: str, where: str) -> None:
    if not isinstance(d.get(key), str) or not d[key].strip():
        raise ValueError(f"{where}{key} is required and must be non-empty text")


def validate(reg: dict) -> dict:
    """{brief, directives, protected_paths} as run.yaml (or a revision) states them, checked."""
    brief = reg.get("brief")
    if brief is not None:
        if not isinstance(brief, dict) or set(brief) != set(BRIEF_KEYS):
            raise ValueError(f"brief must state exactly {', '.join(BRIEF_KEYS)}")
        for k in BRIEF_KEYS:
            _text(brief, k, "brief.")
    ds = reg.get("directives") or []
    if not isinstance(ds, list):
        raise ValueError("directives must be a list")
    seen = set()
    for i, d in enumerate(ds):
        where = f"directives[{i}]."
        if not isinstance(d, dict) or (unknown := set(d) - DIRECTIVE_KEYS):
            raise ValueError(f"directives[{i}] holds only {sorted(DIRECTIVE_KEYS)}"
                             + (f", not {sorted(unknown)}" if isinstance(d, dict) else ""))
        for k in ("id", "statement", "reason", "scope"):
            _text(d, k, where)
        if d["id"] in seen:
            raise ValueError(f"{where}id {d['id']!r} is used twice")
        seen.add(d["id"])
        if d.get("severity") not in ("prohibited", "discouraged"):
            raise ValueError(f"{where}severity must be prohibited or discouraged")
        if "predicate" in d:
            if not isinstance(d["predicate"], str):
                raise ValueError(f"{where}predicate must be an expression over lever names and paths")
            try:
                _parse(d["predicate"])
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"{where}predicate: {e}")
        pats = d.get("forbidden_patterns")
        if pats is not None:
            if d["severity"] != "prohibited":
                raise ValueError(f"{where}forbidden_patterns apply to prohibited directives only")
            if not isinstance(pats, list) or not pats or not all(isinstance(p, str) and p for p in pats):
                raise ValueError(f"{where}forbidden_patterns must list regular expressions")
            for p in pats:
                try:
                    re.compile(p)
                except re.error as e:
                    raise ValueError(f"{where}forbidden_patterns: {p!r}: {e}")
    paths = reg.get("protected_paths") or []
    if not isinstance(paths, list) or not all(isinstance(p, str) and p.strip() for p in paths):
        raise ValueError("protected_paths must list globs over the worktree")
    return {"brief": brief, "directives": ds, "protected_paths": paths}


# the allowed region: a restricted Python expression
_OPS: dict[type, Any] = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_,
        ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b}


def _dotted(node) -> str | None:
    """`H3.warmup_frac` or `optimizer.lr` (an attribute chain of names) as one name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and (base := _dotted(node.value)):
        return f"{base}.{node.attr}"
    return None


@functools.lru_cache(maxsize=None)
def _parse(src: str) -> tuple[ast.Expression, frozenset]:
    """The parsed predicate and the names it reads; anything beyond comparisons, and/or/not,
    arithmetic, `in`, names and constants is refused."""
    tree = ast.parse(src, mode="eval")
    names = set()

    def walk(n):
        if (name := _dotted(n)) is not None:
            names.add(name)
        elif isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
            for v in n.values:
                walk(v)
        elif isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            walk(n.operand)
        elif isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            walk(n.left), walk(n.right)
        elif isinstance(n, ast.Compare) and all(type(o) in _OPS for o in n.ops):
            for v in (n.left, *n.comparators):
                walk(v)
        elif isinstance(n, (ast.List, ast.Tuple, ast.Set)):
            for v in n.elts:
                walk(v)
        elif not isinstance(n, ast.Constant):
            raise ValueError(f"{ast.dump(n)[:40]}...: only comparisons, and/or/not, arithmetic, "
                             "`in`, lever names or paths and constants are allowed")
    walk(tree.body)
    return tree, frozenset(names)


def _eval(n, env: dict):
    if (name := _dotted(n)) is not None:
        return env[name]
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.BoolOp):
        vals = (_eval(v, env) for v in n.values)
        return all(vals) if isinstance(n.op, ast.And) else any(vals)
    if isinstance(n, ast.UnaryOp):
        return _OPS[type(n.op)](_eval(n.operand, env))
    if isinstance(n, ast.BinOp):
        return _OPS[type(n.op)](_eval(n.left, env), _eval(n.right, env))
    if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
        return [_eval(v, env) for v in n.elts]
    left = _eval(n.left, env)
    for op, c in zip(n.ops, n.comparators):
        right = _eval(c, env)
        if not _OPS[type(op)](left, right):
            return False
        left = right
    return True


def env(levers: dict, paths: dict) -> dict:
    """A config's names: its lever names, and the config path each declared lever controls."""
    return {**levers, **{paths[n]: v for n, v in levers.items() if n in paths}}


def holds(predicate: str, names: dict) -> bool:
    """The config lies in the allowed region. A predicate reading a name the config lacks doesn't
    apply to it; one that can't be evaluated (a type mismatch) fails closed."""
    tree, reads = _parse(predicate)
    if not reads <= set(names):
        return True
    try:
        return bool(_eval(tree.body, names))
    except Exception:  # noqa: BLE001 (TypeError, ZeroDivisionError, ...): outside the region
        return False


def violated(directives: list[dict], levers: dict, paths: dict) -> dict | None:
    """The first prohibited directive whose predicate the config breaks."""
    names = env(levers, paths)
    return next((d for d in directives if d["severity"] == "prohibited" and "predicate" in d
                 and not holds(d["predicate"], names)), None)


def compat(directives: list[dict], levers: dict, paths: dict, declared: dict) -> dict:
    """`compat:<id>` of each discouraged directive: its predicate if it has one, else every lever of
    the hypotheses declaring it (`declared`: id -> {lever: baseline}) at its baseline."""
    names = env(levers, paths)
    return {d["id"]: holds(d["predicate"], names) if "predicate" in d else
            all(levers[n] == b for n, b in declared.get(d["id"], {}).items() if n in levers)
            for d in directives if d["severity"] == "discouraged"}


def grid(levers: dict) -> list[dict]:
    """The points registration checks: each graded lever's endpoints, baseline and quartiles (in log
    space for a log range), every option of a categorical or bool."""
    # ponytail: a grid over one hypothesis's box (others at baseline), so a region carved out between
    # grid points, or a predicate over two hypotheses' levers, slips past registration; the per-trial
    # check still refuses such trials before they run. Reparameterise non-box regions if it bites
    axes = []
    for n, lv in levers.items():
        if lv["kind"] in GRADED:
            lo, hi = lv["low"], lv["high"]
            f = (lambda q: math.exp(math.log(lo) + q * (math.log(hi) - math.log(lo)))) if lv.get("log") \
                else (lambda q: lo + q * (hi - lo))
            vals = {lo, hi, lv["baseline"], *(f(q) for q in (0.25, 0.5, 0.75))}
            if lv["kind"] == "int":
                vals = {min(max(round(v), lo), hi) for v in vals}
            axes.append([(n, v) for v in sorted(vals)])
        else:
            axes.append([(n, v) for v in options(lv)])
    # ponytail: the full product, 6^d points; sample it if hypotheses grow past ~6 levers
    return [dict(p) for p in itertools.product(*axes)]


def forbidden(directives: list[dict], added: list[tuple[str, str]]) -> tuple[dict, str, str] | None:
    """The first added line (path, text) matching a prohibited directive's forbidden pattern."""
    for d in directives:
        for p in d.get("forbidden_patterns", []) if d["severity"] == "prohibited" else []:
            for path, line in added:
                if re.search(p, line):
                    return d, path, line
    return None


STOPWORDS = frozenset("the and for with from into that this than then its are was were use using "
                      "not any all our their them more less can may via per each".split())


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


SIMILAR = 0.5  # share of the shorter text's content words the two texts have in common


def similar(a: str, b: str) -> bool:
    """Lexical overlap (the overlap coefficient of content words) at or above SIMILAR."""
    # ponytail: lexical, so a paraphrase with new words slips past; embed the texts if laundering
    # gets past it (the per-trial predicate and diff scans still hold)
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= SIMILAR


def protected(worktree: Path, files: list[str], runner: str, globs: list[str]) -> list[str]:
    """The worktree files the protected globs match (the runner and levers.json always)."""
    pats = [runner, "levers.json", *globs]
    return sorted(f for f in files if any(fnmatch.fnmatch(f, g) or f.startswith(g.rstrip("/") + "/")
                                         for g in pats))


def manifest(worktree: Path, files: list[str]) -> dict:
    return {f: hashlib.sha256((worktree / f).read_bytes()).hexdigest() for f in files}
