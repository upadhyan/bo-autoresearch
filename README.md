# bo-autoresearch

BOAR (Bayesian Optimization AutoResearch) is an unattended research loop for measurable improvement
tasks: faster code, lower RAM, higher throughput. You start it in Claude Code with `/boar <request>`
and approve a spec. The agent then researches the problem and turns what it finds into hypotheses
with tunable levers. It runs Bayesian optimization over those levers against a frozen eval, round
after round, and an independent reviewer subagent judges each hypothesis, removal and the eval. At
the end, BOAR writes a report of what worked, what didn't, and every experiment.

- [`design.md`](design.md): the design (source of truth).
- [`harness/`](harness): the `boar` CLI (Python, Optuna). It owns the run state, runs the trials and
  refuses out-of-order steps. Its [README](harness/README.md) has installation, the command reference,
  and the run directory layout. [`docs/`](docs) has a [usage guide](docs/guide.md) and the
  [design notes](docs/design-notes.md) (where the code deliberately differs from the design).
- [`plugin/`](plugin): the Claude Code plugin, with the `/boar` skill, the `boar-reviewer` agent and the
  Stop hook.
- [`examples/toy/`](examples/toy): a toy target with two real slow paths, an idea with no effect and a
  tempting cheat, for the acceptance run.

Quick start:

```sh
uv tool install ./harness
claude plugin marketplace add .
claude plugin install boar@bo-autoresearch
```

Then, in the target repo, run `/boar <request>`.

The previous attempt lives on the `archive/v1` branch.
