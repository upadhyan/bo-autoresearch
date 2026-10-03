# Using BOAR: "hey Claude, optimize this"

You have code that's too slow, or uses too much memory, or trains unstably, and you'd like Claude to work on it
overnight and tell you in the morning what actually helped. This guide walks through one run from start to finish.

## Once: install

```sh
uv tool install /path/to/bo-autoresearch/harness      # puts `boar` on PATH
claude plugin marketplace add /path/to/bo-autoresearch
claude plugin install boar@bo-autoresearch
```

## 1. Get the repo ready

BOAR works on a git repo and puts every change on its own branch, `boar/<run-id>`. Your branch stays as it was.

- Commit or stash your work. `boar init` refuses uncommitted changes and untracked files, so it can't sweep your
  files into the round commits.
- Set a git identity for the repo (`git config user.name …`, `git config user.email …`). Rounds commit while
  you're away.
- If you have ideas, write them down. A loose list in a markdown file is enough.

## 2. Ask

Type `/boar`, then say what you want in plain words. Plain "optimize this" without `/boar` won't start a run: the
skill only runs when you invoke it, because a run takes hours.

```
/boar make the nightly ingest in etl/ faster. It runs on the daily partner CSV drops; the
output tables must not change. My ideas are in notes/ideas.md. rounds=6
```

A good request names:

| What | Why | Example |
|---|---|---|
| The thing to improve, and in which direction | It becomes the metric | "faster", "lower peak RAM", "fewer loss spikes" |
| The real workload | Improvements have to carry over to it, not just to a benchmark | "the daily partner CSV drops" |
| What must not break | It becomes a guard checked on every trial | "output tables must not change", "accuracy ≥ 0.91" |
| Your ideas, if any | Each one gets tested and reported, even the ones you doubt | "notes/ideas.md" |
| A budget, if not the default | The default is 12 rounds of 6 trials of about 10 minutes, roughly 12 hours | "rounds=6", "trial_target_s=300" |

A vague request ("optimize this") still works. Claude reads the code and proposes a metric and workload, so the
spec review in the next step just matters more.

## 3. Approve the spec: the only time you're needed

Claude reads the code and your ideas, then shows you a spec like this:

```markdown
## Goal
Make the nightly ingest in etl/ faster.

## Metric
Wall-clock seconds for `python -m etl.ingest` over one day's drop, median of the runs in a trial.
Direction: min

## Target population
Daily partner CSV drops: 40k–400k rows, 3–7 partners, UTF-8 with the occasional malformed row.

## Guards
- The output tables are identical (row-for-row checksum) to the current code's output.
- Peak RSS stays under 4 GB (the nightly box has 8 GB).

## Scope
Allowed: anything under etl/, new pure-Python or stdlib dependencies. Not allowed: changing the
output schema, the partner file format, or the database.

## Known cheats
Skipping malformed rows instead of repairing them; caching parsed files between runs; reading
fewer partners than the drop contains.

## User ideas
(copied word for word from notes/ideas.md)

## Budget
rounds 6 (default 12); everything else default.
```

Read it carefully. Everything after this point is judged against it:

- **Metric and Direction.** Is this the number you care about?
- **Target population.** Does it describe the real workload? An improvement only counts if it carries over to this.
- **Guards.** Anything that must stay true goes here. A gain that breaks a guard is thrown away.
- **Known cheats.** Add any shortcut you'd consider cheating. The reviewer rejects any proposal that uses one.

Ask Claude to change anything, then approve it. Claude also reminds you that the run needs permissions that won't
prompt (auto mode, or allow rules for `boar`, the eval and edits in the repo). Without them the run stops at the
first permission dialog. Then it runs `boar init`, switches to `boar/<run-id>`, and you can walk away.

## 4. Overnight

You don't need to do anything. Roughly what happens:

1. **Research.** Claude profiles the code, reads it, searches docs and issue trackers, and turns everything into
   hypotheses: "mechanism X moves the metric", each with levers whose default is today's behaviour.
2. **The eval.** Claude builds a fast, representative benchmark with separate dev and holdout workloads. An
   independent reviewer agent checks it against your spec before it's frozen.
3. **Rounds.** In each round Claude implements the levers, then Bayesian optimization tries combinations of them
   against the eval. Claude keeps, repairs or removes each hypothesis based on the results and proposes new
   ones, and the reviewer signs off on every proposal and removal. Your cheat-like ideas end up in the report as
   rejected.
4. **Finalize.** The best config is checked once against the holdout workloads it never saw, alternating with the
   baseline, and a report is written.

To look in from another terminal (in the repo):

```sh
boar status   # phase, round, hypotheses, the incumbent so far
boar next     # what the agent is doing now
```

To stop a run early: `boar abort --reason "…"`. To get back to your own code while a run is going, use a separate
worktree. The run needs its checkout to stay as it is.

If the reviewer rejects the eval or every hypothesis three times in a row, Claude stops and asks you. That's the
only other time it will need you.

## 5. The morning after

Claude ends with a summary: baseline vs best config on holdout, the recommended lever settings, and where the
report is. The report is `.boar/<run-id>/report.md`:

1. **Result**: baseline vs incumbent on holdout and dev, with median, spread and relative change, plus the
   recommended config.
2. **What worked**: the hypotheses in the winning config, and the evidence for each.
3. **What didn't**: what was removed and why, with the reviewer's verdict.
4. **Rejected by review**: proposals the reviewer turned down, such as your cheat-like ideas.
5. **Caveats**: dev/holdout disagreement, noise, drift, untested interactions.
6. **Experimental log**: one row per round, linking to every trial in `trials.jsonl`.

The code is on `boar/<run-id>`. Every change sits behind a lever whose default is the old behaviour, so the branch
is safe to read but isn't a clean patch. To keep what worked, apply the changes behind the recommended levers to
your branch (or ask Claude to write that patch with the report open). `git checkout <your branch>` takes you back.

## Trying it on the toy first

[`examples/toy`](../examples/toy) has a log processor with two real slow paths, one idea that does nothing and one
tempting cheat. A run with `rounds=3 trials_per_round=4` spends only a few minutes per round on trials (most of the
time is Claude's own work), and its README says what a good run should find.
