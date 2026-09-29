"""Enforcement decisions for the hooks: what an agent may write, run in Bash, and read.

Pure over paths and command text; the CLI (`check …`) supplies the run's state, logs every block
as `hook_blocked`, and the hook adapters only map the answer to an exit code. The Bash parser is
best-effort ("obvious" writes); the protected-path manifest is the backstop for what it misses.
"""
import fnmatch
import os
import re
import shlex
from pathlib import Path

from . import directives

READ_ONLY = {"registration-reviewer", "interplay-reviewer", "round-analyst"}
RECORDING = READ_ONLY | {"hypothesis-generator"}  # subagents that end with `record <kind>`
HISTORY = {"commit", "reset", "rebase", "checkout", "switch", "stash", "merge", "pull", "push",
           "cherry-pick", "revert", "am"}  # git commands that move a worktree's HEAD or branch
INSTALLS = {("pip", "install"), ("uv", "pip", "install"), ("uv", "pip", "sync"), ("uv", "add"),
            ("uv", "sync"), ("conda", "install"), ("mamba", "install"), ("micromamba", "install"),
            ("poetry", "add"), ("poetry", "install"), ("pipx", "install")}
PREFIXES = {"env", "sudo", "command", "exec", "nohup", "time"}  # run the next word as the command


def role(agent: str | None) -> str:
    """The plugin role from a hook's agent_type (`boautoresearch:round-analyst`, or scoped further);
    the main session has none: the orchestrator."""
    return agent.rsplit(":", 1)[-1] if agent else "orchestrator"


def resolve(cwd: Path, path: str) -> Path:
    """Absolute, its directories' symlinks resolved; not the file's own (a venv's python is one)."""
    p = Path(os.path.abspath(os.path.join(cwd, os.path.expanduser(path))))
    return Path(os.path.realpath(p.parent)) / p.name if p.name else Path(os.path.realpath(p))


def raw_read(p: Path, bo: Path) -> bool:
    """A path into a run's raw logs: its log.db, its artifacts, or a directory holding them."""
    try:
        rel = p.relative_to(bo).parts
    except ValueError:
        return False
    return len(rel) <= 1 or rel[1] == "artifacts" or rel[1].startswith("log.db")


def write_refusal(p: Path, root: Path, runner: str, globs: list[str], round_running: bool) -> str | None:
    bo = root / ".bo-research"
    if p == root / "run.yaml":  # the draft of a run.yaml before its init
        return None
    try:
        rel = p.relative_to(bo).parts
    except ValueError:
        if p == root or root in p.parents:
            return (f"{p} is in the user's checkout, which the research never touches: code changes "
                    "happen in the run worktree (.bo-research/<run>/worktree)")
        return None
    if len(rel) >= 3 and (bo / rel[0] / rel[1] / ".git").is_file():  # a run worktree (research or distilled)
        worktree, f = bo / rel[0] / rel[1], "/".join(rel[2:])
        if rel[2] == ".git":
            return f"{p} is the worktree's git link: commits go through the harness"
        if directives.protected(worktree, [f], runner, globs):
            return (f"{f} is a protected path (the runner, levers.json and the registry's protected "
                    "paths): the research may never change how the score is measured")
        if round_running:
            return ("a round is running: the worktree is frozen until it ends, so running trials never "
                    "import half-edited code")
        return None
    return (f"{p} belongs to the harness (the log, artifacts, registry, run.yaml, generated files and "
            "run venv under .bo-research/): only `boautoresearch` commands write there")


def _lex(text: str) -> list[str]:
    lex = shlex.shlex(text, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    return list(lex)


def _command_text(cmd: str) -> str:
    """The command as the shell runs it: a trailing heredoc's body (stdin data, e.g. a record) left
    out — only when the first line opens one heredoc whose delimiter is the last line and no other."""
    first, _, rest = cmd.partition("\n")
    try:
        toks = _lex(first)
    except ValueError:
        return cmd
    ends = [toks[i + 1].lstrip("-") for i, t in enumerate(toks[:-1]) if t in ("<<", "<<-")]
    lines = [ln.strip() for ln in rest.split("\n")]
    return first if len(ends) == 1 and rest and lines[-1] == ends[0] and lines.count(ends[0]) == 1 else cmd


def _segments(cmd: str) -> list[tuple[list[str], list[str]]]:
    """Simple commands as (words, output-redirect targets); ValueError when it won't parse."""
    toks = _lex(_command_text(cmd).replace("\n", " ; "))
    segs: list[tuple[list[str], list[str]]] = []
    words: list[str] = []
    targets: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t and set(t) <= set("();<>|&"):
            if ">" in t:  # an output redirect, unless it duplicates a descriptor (`2>&1`, `>&-`)
                nxt = toks[i + 1] if i + 1 < len(toks) else ""
                if not (t.endswith("&") and re.match(r"^(\d+|-)$", nxt)):
                    targets.append(nxt)
                i += 2
                continue
            if "<" in t or ">" in t:  # an input redirect / descriptor: its word is not a command word
                i += 2
                continue
            segs.append((words, targets))  # an operator ends the simple command
            words, targets = [], []
        else:
            words.append(t)
        i += 1
    segs.append((words, targets))
    out = []
    for words, targets in segs:
        while words and (re.match(r"^\w+=", words[0]) or os.path.basename(words[0]) in PREFIXES):
            words = words[1:]
            while words and words[0].startswith("-"):  # the prefix's own flags (`env -i`)
                words = words[1:]
        if words or targets:
            out.append((words, targets))
    return out


def _program(words: list[str]) -> tuple[str, list[str]]:
    """The program's name and its arguments; `python -m X` runs X."""
    prog = os.path.basename(words[0]) if words else ""
    if re.match(r"^python[\d.]*$", prog) and words[1:2] == ["-m"] and len(words) > 2:
        return words[2], words[3:]
    return re.sub(r"^(pip)[\d.]*$", r"\1", prog), words[1:]


def _written(prog: str, args: list[str]) -> list[str]:
    """The files an obvious shell write changes."""
    files = [a for a in args if not a.startswith("-")]
    if prog in ("rm", "mv", "tee", "touch", "truncate"):
        return files
    if prog in ("cp", "ln", "install"):
        return files[-1:]
    if prog == "sed" and any(a.startswith("--in-place") or re.match(r"^-[a-zA-Z]*i", a) for a in args):
        return files[1:]  # the first is the script
    return []


def raw_paths(bo: Path) -> list[str]:
    """What the orchestrator must not read: each run's directory, log and artifacts (with a trial
    directory standing in before the first trial) and every file under them."""
    out = [str(bo)]
    for run in (d for d in bo.iterdir() if d.is_dir()) if bo.is_dir() else ():
        a = run / "artifacts"
        out += [str(run), str(a), str(a / "trial-0"), *(str(run / n) for n in ("log.db", "log.db-wal", "log.db-shm"))]
        out += [os.path.join(d, f) for d, ds, fs in os.walk(a) for f in ds + fs]
    return out


def raw_glob(word: str, cwd: Path, raw: list[str]) -> bool:
    """A glob that could match a raw path, from `cwd`: brace and variable expansions count as `*`,
    and `*` crosses `/` (as `**` does). A lone `$VAR` is no glob: Bash calls share no variables and
    the orchestrator's compute none (COMPUTED), so one naming the log was set in plain sight."""
    if re.fullmatch(r"\$\w+|\$\{\w+\}", word):
        return False
    pat = re.sub(r"\{[^{}]*\}|\$\{[^}]*\}|\$\w+", "*", os.path.expanduser(word))
    if not re.search(r"[*?[]", pat):
        return False
    pat = pat if pat.startswith(("/", "*")) else os.path.join(os.path.realpath(cwd), pat)  # a leading $VAR may be absolute
    return any(fnmatch.fnmatchcase(p, os.path.normpath(pat)) for p in raw)


def names_raw(text: str, cwd: Path, raw: list[str]) -> bool:
    """Text naming a run's raw logs, or holding a glob that could match them: shell and inline code
    alike, so no parser has to follow what runs it."""
    return bool(re.search(r"log\.db|artifacts/trial-", text)) or any(
        raw_glob(w, cwd, raw) for w in re.split(r"[\s'\"<>|&;()=:]+", text) if w)


COMPUTED = ("the orchestrator's Bash computes no words at run time (no $(..), backticks, $'..' or <(..)): "
            "write paths out in plain sight, and read the run through probes")


RAW_READ = ("the orchestrator reads the run through probes (`boautoresearch status|summary|trials|verdict`) "
            "or asks the round analyst, never the raw log.db or artifacts: a long-lived context doesn't "
            "fill up with raw logs")


def bash_refusal(cmd: str, cwd: Path, agent: str, bo: Path, probes: set[str], write_check) -> str | None:
    """Why `cmd` is refused for this agent, else None. `write_check(path) -> reason | None`."""
    # ponytail: obvious writes only (python -c, xargs, find -delete, perl -i and globbed names slip
    # through); the protected-path manifest catches those at the next round or commit. The
    # orchestrator's raw reads don't rely on the parser: they are refused on the names (below).
    try:
        segs = _segments(cmd)
        toks = _lex(_command_text(cmd).replace("\n", " ; "))
    except ValueError as e:
        return f"the command can't be parsed ({e}): run it in a simpler form"
    r = role(agent)
    raw = raw_paths(bo) if r == "orchestrator" else []
    if r == "orchestrator" and re.search(r"\$\(|`|\$'|[<>]\(", _command_text(cmd)):
        return COMPUTED
    # fail closed: a command that isn't only harness calls (and `cd`) names no raw log anywhere (its
    # words with the quoting undone, inline code, a heredoc it's fed), nor a glob that could match one
    if r == "orchestrator" and any(_program(w)[0] not in ("boautoresearch", "cd") for w, _ in segs) and (
            names_raw(cmd, cwd, raw) or any(names_raw(t, cwd, raw) for t in toks)):
        return RAW_READ
    for words, targets in segs:
        prog, args = _program(words)
        if r in READ_ONLY:
            harness = prog == "boautoresearch" and bool(args) and (args[0] in probes or args[0] == "record")
            if targets or "$(" in cmd or "`" in cmd or not (harness or prog == "cd"):
                return (f"{r} is read-only: its Bash runs only `boautoresearch` probes ({', '.join(sorted(probes))}) "
                        "and `boautoresearch record` (a record body goes in with `--file -` and a heredoc)")
        if prog == "cd":
            cwd = resolve(cwd, args[0]) if args else Path.home()
            continue
        inner = (" ".join(args) if prog == "eval" else args[args.index("-c") + 1]
                 if prog in ("bash", "sh", "zsh") and "-c" in args[:-1] else None)
        if inner is not None and (why := bash_refusal(inner, cwd, agent, bo, probes, write_check)):
            return why
        plain = [prog, *(a for a in args if not a.startswith("-"))]
        if tuple(plain[:2]) in INSTALLS or tuple(plain[:3]) in INSTALLS:
            return ("installs go through the harness: `boautoresearch add-dependency <requirement>` "
                    "installs into the run venv and commits the freeze")
        for f in targets + _written(prog, args):
            if why := write_check(resolve(cwd, f)):
                return why
        if prog == "git":
            where, i = cwd, 0
            while i < len(args) and args[i].startswith("-"):
                if args[i] == "-C" and i + 1 < len(args):
                    where = resolve(where, args[i + 1])
                i += 2 if args[i] in ("-C", "-c") else 1
            if i < len(args) and args[i] in HISTORY and ".bo-research" in where.parts:
                return (f"`git {args[i]}` in a run worktree is refused: commits go through the harness "
                        "(commit-lever, commit-change, add-dependency); read-only git (diff, log, show) is fine")
        if prog == "boautoresearch" and args[:1] == ["record"] and (
                {"--agent-id", "--actor"} & {a.split("=")[0] for a in args}):
            return ("leave out --agent-id and --actor: the hook adds a subagent's own (the orchestrator "
                    "records as itself), so no one records in another agent's name")
        if r == "orchestrator" and prog != "boautoresearch" and any(
                raw_read(resolve(cwd, w), bo) or raw_glob(w, cwd, raw) for w in args + targets):
            return RAW_READ
    return None


def with_agent_id(cmd: str, agent_id: str, agent: str) -> str | None:
    """A subagent's `boautoresearch record …` with its agent id and role added (so SubagentStop can
    find its record), else None."""
    text = _command_text(cmd)  # never inside a heredoc body (a record may quote the command)
    new = re.sub(r"(\bboautoresearch\s+record)(?=\s)",
                 lambda m: f"{m.group(1)} --agent-id {shlex.quote(agent_id)} --actor {shlex.quote(role(agent))}",
                 text, count=1)
    return new + cmd[len(text):] if new != text else None
