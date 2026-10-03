#!/usr/bin/env python3
"""Check process.py's output against the reference summaries in expected_output/.

usage:
  python check.py OUTPUT WORKLOAD   compare one output file with WORKLOAD's reference
  python check.py --all             run process.py on every workload and compare each output

The references were written by the original process.py. The dashboard parses the
summary, so a faster process.py must produce byte-identical output. Exits 0 when
everything matches and 1 when anything differs.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKLOADS = HERE / "workloads"
EXPECTED = HERE / "expected_output"


def reference_for(workload: str | Path) -> Path:
    """expected_output/access-2026-09-08.txt for workloads/access-2026-09-08.log.gz."""
    name = Path(workload).name
    for suffix in (".gz", ".log"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return EXPECTED / f"{name}.txt"


def first_difference(output: Path, reference: Path) -> str | None:
    """None when the files are byte-identical, else a one-line description."""
    if not reference.is_file():
        return f"no reference {reference}"
    if not output.is_file():
        return f"no output {output}"
    got, want = output.read_bytes(), reference.read_bytes()
    if got == want:
        return None
    got_lines, want_lines = got.splitlines(), want.splitlines()
    for i, (g, w) in enumerate(zip(got_lines, want_lines), 1):
        if g != w:
            return f"line {i}: got {g.decode(errors='replace')!r}, expected {w.decode(errors='replace')!r}"
    return f"got {len(got_lines)} lines, expected {len(want_lines)}"


def check_one(output: str | Path, workload: str | Path) -> bool:
    problem = first_difference(Path(output), reference_for(workload))
    print(f"{'ok' if problem is None else 'MISMATCH'}  {Path(workload).name}" + ("" if problem is None else f"  {problem}"))
    return problem is None


def check_all() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for workload in sorted(WORKLOADS.glob("*.log.gz")):
            output = Path(tmp) / "summary.txt"
            run = subprocess.run([sys.executable, str(HERE / "process.py"), str(workload), str(output)])
            if run.returncode != 0:
                print(f"MISMATCH  {workload.name}  process.py exited {run.returncode}")
                ok = False
                continue
            ok = check_one(output, workload) and ok
    return ok


def main(argv: list[str]) -> int:
    if argv == ["--all"]:
        return 0 if check_all() else 1
    if len(argv) == 2:
        return 0 if check_one(argv[0], argv[1]) else 1
    print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
