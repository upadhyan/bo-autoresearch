# Offline BO benchmark

`bench.py` simulates multi-round BOAR runs on a cheap synthetic objective, so an idea about the optimizer can be
compared with the harness in seconds instead of overnight runs. It is dev tooling: it imports the harness and
doesn't ship with the plugin.

## Running it

From the repo root:

```sh
uv run --frozen --project plugin/harness python benchmark/bench.py
uv run --frozen --project plugin/harness python benchmark/bench.py --seeds 20 --rounds 8 --trials 4 --repeats 2
uv run --frozen --project plugin/harness pytest benchmark/tests -q
```

The defaults (100 seeds, the scenario's 6 rounds, 6 trials per round, 3 repeats) take about 6 s on a WSL2 laptop.
Seed s is one run with the harness's `seed` set to s. Rounds past the scenario change nothing, so they share the
last commit. For each variant it prints a table of median [IQR] over seeds:

- **regret (s)**: the true (noise-free) metric of the incumbent after the round, minus that of the best config in
  the round's search space, both at the round's commit. The `all` row is each run's mean over its rounds.
- **eval runs**: eval invocations, re-measurements included. A crash or a broken guard stops a trial's repeats,
  as in the harness. The `all` row is each run's total.
- **re-measured**: trials whose config an earlier trial already measured.

Each variant after the first also gets a table of its difference from the first on each seed, in the `all` row's
regret and eval runs: the mean, a 95% bootstrap CI of the mean, and how many seeds came out lower, tied and higher.

## What is simulated

A target like [`examples/toy`](../examples/toy) with levers of every type and a guard (`PROPOSALS`, `true_metric`),
measured with noise seeded like an eval that reads `BOAR_SPLIT` and `BOAR_REPEAT`, so configs are compared paired
(`measure`). `SCENARIO` sets what changes before each round: hypotheses added, fixed and removed, a crash, and metric
shifts, some of which reorder configs. The round itself is the harness's `warmstart`, `OptunaBackend`, `rounds` and
`control` code.

## Adding a variant

A variant is a subclass of `Harness` that overrides one or more of its three methods:

- `backend(space, seed, n_startup)`: the round's optimizer, with `OptunaBackend`'s methods (`add_warm`,
  `enqueue`, `ask`, `tell`);
- `warm(trials, hyps, r)`: the trials copied into the study, the exclusion counts and the incumbent;
- `queue(hyps, r, incumbent)`: the partial configs queued ahead of sampling, which the round then fills,
  dedupes and enqueues.

Add it to `VARIANTS` and pass `--variant` once per variant to compare them on the same seeds:

```python
class Cold(Harness):
    """No warm start: each round's study starts empty."""

    def warm(self, trials, hyps, r):
        valid, excluded, incumbent = super().warm(trials, hyps, r)
        return [], excluded, incumbent


VARIANTS = {"harness": Harness(), "cold": Cold()}
```

```sh
uv run --frozen --project plugin/harness python benchmark/bench.py --variant harness --variant cold
```

## Ax (#47)

`ax_gp.py` runs the harness on Ax's GP in place of TPE. It needs Ax and CPU-only torch from this directory's own uv
project (`pyproject.toml`, `uv.lock`), which keeps them out of the plugin; without them its tests skip.

```sh
uv run --frozen --project benchmark python benchmark/ax_gp.py --seeds 30 --trials 12 --repeats 2
uv run --frozen --project benchmark pytest benchmark/tests -q
```

The experiment and its results are in [docs/experiments/47-ax-gp.md](../docs/experiments/47-ax-gp.md).
