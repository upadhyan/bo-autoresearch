<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/boar-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/boar-logo-light.svg">
    <img alt="BOAR logo" src="assets/boar-logo-light.svg" width="400">
  </picture>
</p>

# BOAR: Bayesian Optimization AutoResearch

BOAR is a Claude Code plugin that runs an unattended research loop for measurable improvement tasks:
faster code, lower RAM, higher throughput. You type `/boar <request>` and approve a spec. Claude then
researches the problem and turns what it finds into hypotheses with tunable levers. It runs Bayesian
optimization over those levers against a frozen eval, round after round, and an independent reviewer
subagent judges each hypothesis, removal and the eval. At the end, BOAR writes a report of what worked,
what didn't, and every experiment.

## Requirements

- [Claude Code](https://code.claude.com) on macOS, Linux or WSL.
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`. uv also fetches
  Python 3.10+ if you don't have it.
- git. BOAR works on a git repo.

## Install

```sh
claude plugin marketplace add upadhyan/bo-autoresearch
claude plugin install boar@bo-autoresearch
```

Or, inside Claude Code: `/plugin marketplace add upadhyan/bo-autoresearch`, then
`/plugin install boar@bo-autoresearch`.


To upgrade, 
```sh
claude plugin marketplace update bo-autoresearch
claude plugin update boar@bo-autoresearch
```

## Use

In the repo you want to improve, commit or stash your work and set a git identity
(`git config user.name …`, `git config user.email …`), because rounds commit while you're away. Then
start Claude Code there and type `/boar`, followed by what you want:

```
/boar make the nightly ingest in etl/ faster. It runs on the daily partner CSV drops; the
output tables must not change. My ideas are in notes/ideas.md. rounds=6
```

Claude reads the code and your ideas and shows you a spec: the metric and its direction, the real
workload, the guards that must hold, the scope, and the known cheats. Ask for changes, then approve it.
That's the only time you're needed.

A run takes hours (about 12 with the defaults) and continues unattended, so it needs permissions that
won't prompt: auto mode, or allow rules for `Bash(boar *)`, for whatever the eval and research need,
and for edits in the repo. Without them the run stops at the first permission dialog.

When it's done, Claude summarises the result. The report is `.boar/<run-id>/report.md` and the code is
on branch `boar/<run-id>`. Your own branch is untouched. The [usage guide](docs/guide.md) walks through
a whole run: what to put in a request, how to read the spec, what happens overnight, and how to read
the report.

### Watching from your terminal

The plugin puts `boar` on PATH for Claude only. To use it in your own shell, add this alias, which
always points at the installed plugin's copy:

```sh
alias boar='"$(ls -td ~/.claude/plugins/cache/bo-autoresearch/boar/*/bin/boar | head -1)"'
```

Then, in the repo, `boar status` shows the phase, the hypotheses and the incumbent so far,
`boar next` shows what the agent is doing now, and `boar abort --reason "…"` ends the run.

## Try it on a toy first

[`examples/toy`](examples/toy) is a log processor with two real slow paths, an idea that does nothing
and a tempting cheat. Its README sets up a run that spends only a few minutes per round on trials, and
lists what a good run should find.

## Update or remove

```sh
claude plugin marketplace update bo-autoresearch
claude plugin update boar@bo-autoresearch
```

Or turn on auto-update for the `bo-autoresearch` marketplace in `/plugin`. To remove it:
`claude plugin uninstall boar@bo-autoresearch`.

## More

- [`design.md`](design.md): the design (source of truth), and [`docs/design-notes.md`](docs/design-notes.md):
  where the code deliberately differs from it.
- [`plugin/harness/README.md`](plugin/harness/README.md): the `boar` command reference, the run
  directory layout, and development setup.
- The previous attempt lives on the `archive/v1` branch.
