# Equivalence check: what threshold should it use?

Question: the harness's equivalence check (after every code change; spec #21, "Equivalence check") falsely
fails some true no-op changes and starts an unneeded epoch. What should the rule be instead?

Numbers below come from `docs/research/equivalence_check_mc.py` (400k draws per cell, seed 0, normal noise,
true σ = 1 so shifts are in σ units). δ = 2σ is the harness's suggested δ (`cli._suggested_delta`); the dogfood
benchmark runs δ = 2σ and δ = 4σ (spec #21, "Repeats": σ = δ/4 and δ/2).

## 1. What the check does today

`cli._equivalence_check` (harness/boautoresearch/cli.py:1398):

- Configs: the baseline, plus the incumbent when it differs from the baseline.
- For each: **n = 2** new replicates at the current fidelity on the new commit; their mean is compared with the
  **logged mean** of every eligible trial at that config in the current epoch (`study.eligible`: all kinds count).
  Typically m = 5 at the baseline (the calibration round's `replicates_k` = 5 baseline replicates) and m = 3 at
  the incumbent (its trial + `CONFIRMATIONS` = 2).
- Pass iff `|mean_new − mean_logged| ≤ 2σ̂` (+ a float tolerance) at **every** config; otherwise a new epoch.
- σ̂ = `cli._sigma`: the calibration round's **sample sd of the k = 5 baseline replicates** (`calibration.sigma`),
  i.e. **estimated, df = 4**, and fixed for the whole epoch. (Per-round re-estimates in `_reestimate_noise` are only
  logged and flagged; they don't feed `_sigma`.)

Note the tolerance is 2σ̂ on a difference whose standard error is σ·√(1/n + 1/m), not σ.

## 2. Its actual false-alarm rate

Under a true no-op, Δ/(σ̂·√(1/2 + 1/m)) is Student-t with σ̂'s df (the new replicates and the logged mean are
independent of σ̂; for the baseline, the sample mean and sd of the same normal replicates are independent). So
P(fail at one config) = P(|t_df| > 2/√(1/2 + 1/m)):

| σ̂ | baseline (m=5) | incumbent (m=3) | either config (MC) |
|---|---|---|---|
| known σ | 1.7% | 2.9% | 4.5% |
| **df = 4 (today: k = 5)** | 7.5% | 9.4% | **14.9%** |
| df = 2 (k = 3, as in the test fixtures) | 13.9% | 16.0% | ~28% |
| df = 20 (pooled, see §4) | 2.7% | 4.1% | 6.5% |

The "3–6%" in `docs/agents/harness-design-notes.md` (#9/#28 note) is the known-σ figure. With the estimated σ the
harness actually uses, **a true no-op starts a new epoch about 15% of the time**, and across 10 no-op changes in
an epoch the chance of at least one spurious epoch is **52%** (MC, σ̂ held fixed for the epoch).

Two things make it worse in practice:

- **Winner's curse at the incumbent.** The incumbent is chosen as the best config, so its logged mean is
  optimistically biased (Smith & Winkler 2006, the optimizer's curse), and fresh replicates regress. With a 0.5σ
  bias the current rule's rate rises from 14.9% to 17.9%.
- Every false epoch discards the epoch's evidence: k baseline replicates to re-measure σ and a burn-in restart
  (`max(10·d, 20)` fresh trials per hypothesis) — far more than any extra replicates a better check would cost.

## 3. Options

Power = P(check fails) when the objective shifted by 1σ, 2σ, 4σ at both configs (δ/2 and δ at δ = 2σ; δ/2 and δ
at δ = 4σ are the 2σ and 4σ columns). Trials = mean equivalence trials per check (two configs) under a no-op.
Each cell: σ̂ with df = 4 (today) / df = 20 (pooled).

| Rule | False alarm | Power 1σ | Power 2σ | Power 4σ | Trials |
|---|---|---|---|---|---|
| **today**: n=2, \|Δ\| ≤ 2σ̂ | 14.9% / 6.5% | .36 / .27 | .74 / .74 | .99 / 1.0 | 4 |
| (a) n=3, 2σ̂ | 10.9% / 3.5% | .32 / .22 | .73 / .74 | 1.0 / 1.0 | 6 |
| (a) n=5, 2σ̂ | 7.8% / 1.8% | .30 / .18 | .72 / .74 | 1.0 / 1.0 | 10 |
| (b) n=2, t-test, α=1% | 0.9% / 1.0% | .03 / .08 | .12 / .39 | .50 / .98 | 4 |
| (b) n=4, t-test, α=1% | 0.9% / 1.0% | .05 / .12 | .19 / .61 | .69 / 1.0 | 8 |
| (b) n=2, t-test, α=5% | 4.6% / 4.9% | .14 / .22 | .40 / .69 | .90 / 1.0 | 4 |
| (c) TOST ε=δ=2σ, n=2 | 81% / 79% | .92 / .93 | .99 / 1.0 | 1.0 / 1.0 | 4 |
| (c) TOST ε=δ=2σ, n=6 | 53% / 36% | .83 / .80 | .99 / 1.0 | 1.0 / 1.0 | 12 |
| (c) TOST ε=δ=4σ, n=2 | 7.7% / 1.3% | .22 / .10 | .59 / .48 | .99 / 1.0 | 4 |
| (d) 2, +4 if stage 1 fails, α=1% | 0.8% / 0.7% | .05 / .11 | .23 / .60 | .78 / 1.0 | 4.67 / 4.27 |
| (d) 2, +4 if stage 1 fails, α=5% | 3.4% / 2.1% | .18 / .19 | .57 / .71 | .98 / 1.0 | 4.68 / 4.27 |
| (d) 2, +8 if stage 1 fails, α=5% | 3.6% / 2.1% | .20 / .20 | .61 / .72 | .99 / 1.0 | 5.35 / 4.53 |

Rule definitions (per config c with m_c logged trials; the check fails if any config fails; "t-test" is a
two-sided test at α/2 per config, Bonferroni over the two configs):

- (b) fail iff `|Δ_c| > t(1 − α/4, df) · σ̂ · √(1/n + 1/m_c)`.
- (c) TOST (Schuirmann 1987): pass iff the 1−2α (90%) CI of Δ_c lies inside ±ε: `|Δ_c| + t(0.95, df)·σ̂·√(1/n+1/m_c) < ε`.
- (d) stage 1 = today's rule. A config that fails it gets n₂ more replicates; it fails only if the pooled
  (2 + n₂)-replicate mean fails (b) at α.

### Reading the table

- **(a) More replicates alone barely helps while σ̂ has df = 4.** The t_4 tail, not the replicate count, drives
  the false alarms: even n = ∞ leaves P(|t_4| > 2/√(1/m)) = 1.1% at the baseline and 2.6% at the incumbent. Linear cost for a slow gain.
- **(b) Setting the threshold from α** fixes the false-alarm rate exactly (that is the point of using the
  t quantile with σ̂'s df), but with df = 4 the quantile is huge (t(0.9975, 4) = 5.60) and power collapses.
- **(c) TOST is the wrong tool for this decision.** TOST makes *equivalence* the claim that needs proof
  (Schuirmann 1987; Lakens 2017): it controls P(pass | |shift| ≥ ε) ≤ α. Here "no shift, no epoch" is the default
  and a false epoch is the expensive error, so TOST with ε = δ = 2σ fails 79–81% of true no-ops at n = 2 and still
  36–53% at n = 6. It only becomes usable when ε ≥ ~4σ, which is then just a loose fixed threshold. TOST fits if
  the user wants a guarantee that shifts ≥ δ are caught ≥ 95% of the time, and that needs tens of replicates per
  config at δ = 2σ.
- **(d) Two-stage** keeps today's cheap path (≥ 85% of no-ops pass on the first 2 replicates) and spends extra
  replicates only when stage 1 is borderline — the idea behind group-sequential designs (Pocock 1977): look early,
  stop when the answer is clear. Because an epoch starts only if *both* looks agree, the overall false-alarm rate
  is below α, for +0.3–0.7 trials per check on average.
- **σ̂'s df is the biggest single lever.** Every rule improves sharply from df = 4 to df = 20. The harness already
  has the data: within-config replicate groups in the epoch (the k baseline replicates, 2 confirmations per
  incumbent, replicate-floor replicates, earlier equivalence trials). `_reestimate_noise` already pools exactly
  this within one round; pooling it over the epoch gives df ≈ 4 + 2 per confirmed config + …, reaching ~20 after
  a few rounds.

With pooled σ̂, (d) at α = 5% matches today's power at δ = 2σ (0.71 vs 0.74), exceeds it at 4σ, and cuts false
epochs from 6.5–15% to 2.1–3.4% per check (from 52% to 17% over 10 checks). At α = 1% false epochs fall to
~0.7% per check (4–6% over 10 checks) but power at a shift of exactly δ = 2σ drops to 0.60 (0.23 while df = 4).
A shift of δ/2 is missed by every cheap rule, including today's (power 0.27–0.36).

## 4. Recommendation

Decision for the user: **replace "2 replicates within 2σ" with a two-stage t-test on a pooled σ̂, and pick α.**
My pick is **α = 5%** (keeps today's power to catch a shift of δ, cuts false epochs ~5×); choose **α = 1%** if
spurious epochs matter more to you than catching a shift of exactly δ = 2σ.

Exact rule:

1. **σ̂ and df:** pooled within-config sd over the current epoch's eligible trials at the check's fidelity:
   `σ̂² = Σ_c Σ_i (y_ci − ȳ_c)² / df`, `df = Σ_c (n_c − 1)` over configs with ≥ 2 trials (the calibration round's
   k baseline replicates alone give df = k − 1). σ̂ = 0 keeps today's float-tolerance path.
2. **Stage 1** (unchanged): 2 replicates of the baseline and of the incumbent; a config passes if
   `|Δ_c| ≤ 2σ̂`.
3. **Stage 2**, only for a config that failed stage 1: 4 more replicates of it; it fails if
   `|Δ_c| > t(1 − α/4, df) · σ̂ · √(1/6 + 1/m_c)` with Δ_c from all 6 new replicates and m_c the logged trials.
   Constants: α = 0.05 → t(0.9875, df) = 3.50 (df 4), 2.63 (10), 2.42 (20), 2.24 (∞);
   α = 0.01 → t(0.9975, df) = 5.60 (4), 3.58 (10), 3.15 (20), 2.81 (∞).
4. A new epoch starts iff some config fails stage 2.

Optional, independent: compute the incumbent's logged mean without the trial that selected it (its replicates
only), which removes most of the winner's-curse bias (at 0.5σ bias the α = 1% rule goes from 0.7% to 1.5%, α = 5%
from 2.1% to 4.0%).

Spec #21 wording change ("Equivalence check" bullet), from:

> After any code change, the harness runs the incumbent and the baseline, 2 replicates each. If either disagrees
> with its logged mean by more than 2σ, a new epoch starts: …

to:

> After any code change, the harness runs the incumbent and the baseline, 2 replicates each. A config that
> disagrees with its logged mean by more than 2σ gets 4 more replicates, and a new epoch starts only if their
> combined mean still differs from the logged mean in a two-sided t-test at α = 5% (split across the two
> configs), with σ pooled from the epoch's replicates: …

The harness-design-notes "~3–6%" note should read "~15% with R0's df = 4 σ̂" if the rule stays as is.

Out of scope but same pattern: the distilled-branch verification (`2σ̂·√(2/r)`, r = 3) is a z = 2 rule on a
df = 4 σ̂, so it falsely fails an identical branch P(|t_4| > 2) ≈ 12% per attempt, not 4.6%.

## Sources

- Schuirmann, D. J. (1987). A comparison of the Two One-Sided Tests Procedure and the Power Approach for assessing
  the equivalence of average bioavailability. *J. Pharmacokinetics and Biopharmaceutics* 15, 657–680.
  https://doi.org/10.1007/BF01068419 — TOST; equivalent to the 1−2α CI inside the bounds.
- Lakens, D. (2017). Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. *Social
  Psychological and Personality Science* 8(4), 355–362. https://doi.org/10.1177/1948550617697177 — bounds from the
  smallest effect size of interest; the null is non-equivalence.
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of clinical trials. *Biometrika* 64(2),
  191–199. https://doi.org/10.1093/biomet/64.2.191 — interim looks with early stopping.
- Smith, J. E. & Winkler, R. L. (2006). The optimizer's curse: skepticism and postdecision surprise in decision
  analysis. *Management Science* 52(3), 311–322. https://doi.org/10.1287/mnsc.1050.0451 — selected alternatives'
  estimates are biased upward.
- Student-t pivot for a mean difference with an independent pooled variance estimate: any statistics text, e.g.
  NIST/SEMATECH e-Handbook of Statistical Methods §7.3.1, https://www.itl.nist.gov/div898/handbook/prc/section3/prc31.htm
- Harness source: `harness/boautoresearch/cli.py` (`_equivalence_check`, `_sigma`, `_estimate_noise`,
  `_reestimate_noise`, `CONFIRMATIONS`), `calibration.py` (`sigma`), `study.py` (`eligible`, `incumbent`).
