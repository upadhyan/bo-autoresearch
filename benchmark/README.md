# Offline BO benchmark

`bench.py` simulates multi-round BOAR runs on a cheap synthetic objective, so an idea about the optimizer can be
compared with the harness in seconds instead of overnight runs. It is dev tooling: it imports the harness and
doesn't ship with the plugin. Experiment write-ups go in `docs/experiments/`.

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

## What is simulated

A target like [`examples/toy`](../examples/toy): the seconds a log summariser takes, with one hypothesis per cost.
The levers cover every type: a bool (`dedup_set`), a categorical (`route_table`), log-scale ints (`buffer_kb`, which
does nothing, and `batch`), a log-scale float (`gc_scale`) and a linear float with a bool (`prefetch`,
`prefetch_async`). The guard fails when `prefetch` is above 0.8, where the metric is lowest.

`SCENARIO` sets what happens before each round:

| Round | Change | Commit |
|---|---|---|
| 1 | H1 dedup, H2 route table and H3 buffer active | new |
| 2 | H4 batch added; its first version crashes on batches over 16 | new |
| 3 | H4 marked `fix` in round 2 and repaired; H5 GC added; everything 10% slower | new |
| 4 | H3 removed (its lever back at default); H6 prefetch added; the best batch moves from 16 to 4 | new |
| 5 | nothing | same as round 4 |
| 6 | `route_table` `per_file` now beats `once` | new |

Each repeat's noise has a part every config shares and a part of its own (2% each), drawn from (seed, split,
repeat) and (seed, commit, config, split, repeat). So configs are compared paired, as in an eval that seeds from
`BOAR_SPLIT` and `BOAR_REPEAT`, and a config measured again on the same commit gets the same values.

The round itself is the harness's code, called on in-memory trial records: `warmstart.select` and
`warmstart.incumbent` (rules 1-3), `OptunaBackend` (multivariate TPE, warm trials through `add_warm`),
`rounds.extra_queue`, `_filled`, `_same_as` and `_enqueue` (a queued config runs once per commit), and
`control.round_size`. Hypotheses and levers are in the harness's schema.

## Adding a variant

A variant is a subclass of `Harness` that overrides one or more of its three methods:

- `backend(space, seed, n_startup)`: the round's optimizer, with `OptunaBackend`'s methods (`add_warm`,
  `enqueue`, `ask`, `tell`);
- `warm(trials, hyps, r)`: the warm-start set, the exclusion counts and the incumbent;
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
