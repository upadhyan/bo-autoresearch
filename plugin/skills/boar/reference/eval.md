# BOAR eval contract

The eval turns a lever config into a metric value. It lives in `<run dir>/eval/` and is frozen when the reviewer accepts it: the harness records the sha256 of the directory (ignoring `__pycache__/` and `*.pyc`) and keeps a copy of it, and from then on the same files must serve every lever that any later round adds.

## Interface

- `eval/run` is an executable file (shebang line, `chmod +x`). The harness runs it from the target repo root, with stdin closed and these environment variables:
  - `BOAR_CONFIG`: the path to a JSON file mapping every lever name to its value. It lists every lever ever defined, pinned ones at their default. `boar eval check` passes the defaults of the hypotheses proposed so far, possibly `{}`.
  - `BOAR_SPLIT`: `dev` or `holdout`.
  - `BOAR_RUN_DIR`: the run directory (informational).
  - `BOAR_DIAG_DIR`: a fresh, empty temporary directory for the target's diagnostics. The eval leaves it in the target's environment and never reads it; see [Diagnostics](#diagnostics).
- One invocation is one repeat. The harness repeats it `repeats` times per trial and takes the median.
- The last non-empty line of stdout is a single JSON object:

  ```json
  {"metric": 41.7, "guards_ok": true, "metrics": {"peak_rss_mb": 812}}
  ```

  - `metric`: a finite number. Report it even when a guard fails: infeasible trials still steer the sampler.
  - `guards_ok`: `true` only when every guard in the spec held on this invocation.
  - `metrics`: optional extra numbers, kept in the trial records and never optimized.

  Print anything else on earlier lines; stdout and stderr are saved under `trials/<n>/`.
- Outcomes: `guards_ok: false` makes the trial infeasible. A non-zero exit, a crash, a hang, or a last line that isn't such an object makes it failed. A trial still running at 3 × `trial_target_s` (all repeats together) is killed as hung. At finalize each holdout run is a single invocation, killed at 3 × `trial_target_s` / `repeats`, so a holdout invocation must stay well under that too; `boar eval check` runs dev only and doesn't time holdout.

## Levers reach the target code, never the measurement

The eval is frozen before most levers exist, so it handles the config generically:

- Forward the whole config untouched: start the target with the inherited environment so `BOAR_CONFIG` stays set, or pass the file path through. eval/ holds no per-lever code.
- The target code reads each lever where it is used and falls back to the lever's default, so without `BOAR_CONFIG`, or with a lever missing from it, the target behaves exactly as before. One small reader, added to the target code on the boar branch in R1, serves every lever:

  ```python
  import json, os

  _BOAR = json.load(open(os.environ["BOAR_CONFIG"])) if os.environ.get("BOAR_CONFIG") else {}

  def lever(name, default):
      return _BOAR.get(name, default)

  # at the point of use:  workers = lever("parse_workers", 1)
  ```

- The measuring code (timers, memory sampling, the metric computation, the guard checks) never reads `BOAR_CONFIG`. If the eval imports the target in-process, keep the measuring code in eval/ and let only the target's own modules call the reader.

## Diagnostics

The target code, not the eval, measures each hypothesis's observable. Instrumentation added in R1 writes a flat JSON object of numbers to `diag.json` in `$BOAR_DIAG_DIR`, and any other file it likes there, only when that variable is set:

```python
import json, os

def diag(**values):
    where = os.environ.get("BOAR_DIAG_DIR")
    if where:
        with open(os.path.join(where, "diag.json"), "w") as f:
            json.dump(values, f)

# at the end of the run:  diag(cache_hits=hits, passes=passes)
```

- The eval passes its environment through to the target process (the same inheritance that carries `BOAR_CONFIG`), so the variable reaches the target. Its timing, metric and guard code never read `BOAR_DIAG_DIR`, and diagnostics never affect the metric, the guards, the warm start or the incumbent.
- The harness saves repeat k's directory as `trials/<n>/diag-<k>/`; a directory over 20 MB keeps only its `diag.json`. The trial's `diagnostics` are the per-key medians over its repeats; a `diag.json` that isn't a flat object of numbers sets the trial's `diag_error` and never fails the trial.
- Instrumentation runs in scored trials and ships in the final code: counters, not profilers.

## Workloads

- **Dev and holdout**: two disjoint sets of workloads, both drawn from the spec's target population. BO only ever sees dev; holdout is used once, at finalize. Store them in eval/ (for example `eval/data/dev/` and `eval/data/holdout/`) so they freeze with it, or generate them with a generator in eval/ from fixed seeds that differ between the splits.
- **Self-contained**: eval/ holds everything it reads except the target code under test. Copy workloads and references in rather than linking to them: the hash covers only eval/ and records a symlink, not what it points at, so `boar eval check` fails a symlink that leads out of eval/, and the reviewer rejects inputs read from the target repo or anywhere else.
- **Typical cases**: a case where a gain wouldn't carry over to the rest of the population is the wrong case. Several typical cases beat one extreme one.
- **Size**: the smallest eval that is still representative, aimed at about `trial_target_s` per trial counting all repeats, which is about `trial_target_s / repeats` per invocation. Spend that time on representativeness (more typical cases, enough work per case that noise stays small), never on padding. `boar eval check` refuses a baseline trial over 2 × `trial_target_s`.
- **No natural size** (training, for example): run the workload for a fixed wall-clock time, then measure the result. That keeps trials comparable.
- **Guards**: check every guard in the spec on every invocation and set `guards_ok` from them: outputs against references stored in eval/ (computed once from the baseline), accuracy against its floor, peak RSS against the RAM ceiling.
- **Determinism**: derive every seed the workload uses from `BOAR_SPLIT` and `BOAR_REPEAT` (the repeat index, 1, 2, …; take 1 when it is unset, as in a hand run), so the spread across repeats includes the seed noise. With both fixed, a run must reproduce.
- **Runtime writes** (outputs, logs, caches, scratch files) go to a temporary directory (`mktemp -d`), never into the repo or eval/: every trial and holdout run must find the round's commit and the accepted eval as they were. The eval runs from the repo root, so a relative output path lands in the repo. `boar eval check` fails an eval whose baseline run writes into either. After acceptance, a trial that writes into either counts as failed, its error naming what it wrote; a holdout run that does keeps its measurement and the report lists it. The harness restores the changed files, moves new ones aside (to `trials/<n>/repo-writes/` or `holdout/<i>/repo-writes/`), and puts eval/ back from the accepted copy. The eval passed its check at baseline, so such a write usually comes from a lever's code, which calls for `fix` in R3. If every non-baseline trial names the same path, first check `git show --stat` of round 1's commit: an eval output left by a manual run may have been swept in, and the next R1 should `git rm --cached` it and add it to .gitignore.

## Check it

1. Run it by hand from the repo root, once per split, with a config file holding `{}`: `BOAR_CONFIG=<file> BOAR_SPLIT=dev <run dir>/eval/run`, then the same with `BOAR_SPLIT=holdout`. Confirm the last line parses, the guards pass, each invocation (holdout too) takes about `trial_target_s / repeats` and holdout's stays well under 3 × that, and `git status` shows nothing the runs wrote (delete anything they did: round 1's commit would sweep it onto the run branch).
2. Run `boar eval check`: one baseline trial on dev that checks the contract, the guards at baseline, the timing, that the run left the repo and eval/ as they were (a file that was already uncommitted counts as written even when rewritten with the same bytes), and that no symlink leads out of eval/. It also warns about every untracked file it finds: round 1's commit sweeps those in, so delete any that an eval run wrote, such as a cache primed by a manual run. It runs in a detached worker; `boar wait` prints `finished: eval check PASS|FAIL …` with any problems (details in `eval_check.json` and `eval_check.log`). A pass sends the eval to review. Any later edit to eval/ needs a fresh passing check before the reviewer can accept it.

The reviewer accepts the eval only when dev and holdout don't overlap and both are typical of the target population; the workloads (or their seeds and generator) and the guard references live inside eval/; it writes only to a temporary directory; the metric measures what the spec names; the eval's guards cover every guard in the spec; and the lever config reaches only the target code, never the code that does the measuring; and the eval passes its environment through, so `BOAR_DIAG_DIR` reaches the target, while its metric and guard code never read it.
