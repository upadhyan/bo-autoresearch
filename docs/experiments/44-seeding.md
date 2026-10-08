# #44: seeding a new hypothesis with a deliberate first trial

## Hypothesis

From round 2 on, the copied warm-start trials use up `n_startup_trials`, so a round samples nothing at random, and
every copied trial has the new hypothesis's levers at their default. TPE then rarely tries a new lever off default.
Queuing one trial per newly active hypothesis, its levers at a proposed setting (its seed) and every other lever at
the incumbent, should find the wins of new hypotheses sooner.

## Setup

The offline benchmark ([benchmark/](../../benchmark/README.md)) on two scenarios:

- `default` (`SCENARIO`): H1-H3 start in round 1; H4 (int) in round 2, crashing above 16; H5 (float) in round 3; H6
  (float and bool, the guard breaks above `prefetch` 0.8) in round 4; then metric shifts.
- `staggered`: H3 in round 1, then H1 (bool), H2 (categorical), H4, H5 and H6, one per round; the world never changes.
  Added because in `default` the only bool or categorical lever that starts after round 1 is `prefetch_async`, which
  pays only with `prefetch` moved too, while TPE almost never samples a numeric lever exactly at its default (design
  note 17). In the NeuralSGT run 22 of 35 accepted hypotheses started after round 1.

Variants (`bench.VARIANTS`):

| variant | queue |
|---|---|
| `harness` | the incumbent (round 1: the baseline), then `rounds.extra_queue`'s configs |
| `seed-good`, `seed-bad`, `seed-mixed` | also one trial per hypothesis activated this round, from round 2 on: its levers at the seed, the rest at the incumbent |
| `seed-mixed-r1` | the same, from round 1 on |
| `extra-sampled`, `extra-sampled-r1` | the control: as many trials as `seed-mixed(-r1)`, the seeds' slots sampled by TPE |

A seed is the proposal's setting of each lever it names; a bool lever it doesn't name is seeded at its non-default
value, a numeric or categorical one not at all (a categorical lever has several non-default choices, and taking the
first would hang on the order they were listed). A seed is an extra trial, deduped like every queued config.

Each seed against the baseline, in `default`'s world in the round its hypothesis starts (true metric, seconds):

| hypothesis | levers (default) | good seed | | bad seed | |
|---|---|---|---|---|---|
| H1 | `dedup_set` (false) | true (bool rule) | 5.9 better | true (no worse setting) | 5.9 better |
| H2 | `route_table` (per_request) | once | 4.2 better | per_file (the weaker choice) | 3.6 better |
| H3 | `buffer_kb` (8) | 64 | no effect | 4096 | no effect |
| H4 | `batch` (1) | 8 | 1.5 better | 64 | crash (`staggered`: 1.2 better) |
| H5 | `gc_scale` (1.0) | 2.0 | 0.24 better | 0.05 | 2.85 worse |
| H6 | `prefetch` (0.0), `prefetch_async` (false) | 0.5, true | 0.83 better | 1.0, true | guard broken |

`seed-mixed` takes the good seed for odd-numbered hypotheses and the bad one for even ones: from round 2 on that is
H4 bad, H5 good, H6 bad in `default`, and H1 good, H2 bad, H4 bad, H5 good, H6 bad in `staggered`.

Budgets: `--trials 6 --repeats 2` (the NeuralSGT run's) and `--trials 12 --repeats 2`, 6 rounds, seeds 0-499. From
the repo root:

```sh
for s in default staggered; do for t in 6 12; do
  b="uv run --frozen --project plugin/harness python benchmark/bench.py --seeds 500 --trials $t --repeats 2 --scenario $s"
  $b --variant harness --variant seed-good --variant seed-bad --variant seed-mixed --variant extra-sampled
  $b --variant extra-sampled --variant seed-mixed
  $b --variant seed-mixed --variant seed-mixed-r1
  $b --variant extra-sampled-r1 --variant seed-mixed-r1
done; done
```

Regret is each run's mean over its 6 rounds; differences are paired per seed (`bench.compare`).

**Decision rule**, fixed before the runs:

- **Yes** only when, on both scenarios at both budgets, `seed-mixed` − `harness` and `seed-mixed` − `extra-sampled`
  both have a regret CI entirely below zero. The second is the cost test: the eval runs the seeds add must buy more
  than the same runs given to TPE. Otherwise **no**.
- `seed-good` and `seed-bad` bound the best and worst case; they don't enter the rule.
- On yes, round 1 is seeded too when `seed-mixed-r1` − `seed-mixed` and `seed-mixed-r1` − `extra-sampled-r1` both have
  a regret CI entirely below zero at both budgets on `default` (`staggered` has only H3 in round 1).
- A deciding CI that ends within a tenth of its width of zero is run again on 2000 seeds.

## Results

## Decision
