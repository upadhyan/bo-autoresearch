# Irrelevance statistic: what "matters by at least δ" should measure

Research note on the fixed reject form in [#21](https://github.com/upadhyan/bo-autoresearch/issues/21) ("Verdict engine"), whose reasoning is in [#8](https://github.com/upadhyan/bo-autoresearch/issues/8) ("Reject-condition form"). It builds on [#3](https://github.com/upadhyan/bo-autoresearch/issues/3) (`research/lever-importance-interactions`), which chose total-order Sobol on a noise-aware GP posterior.

Question: the `irrelevant` reject compares √V_T, the square root of the unnormalised group total-order Sobol variance, with δ, the user's minimum meaningful effect in objective units. √V_T is a standard deviation, while δ is a change in the objective. What statistic answers "does this lever group change the objective by at least δ, counting interactions" on δ's own scale? Is a separate irrelevance test needed next to Δ at all?

Sources were read on 2026-09-29. Anything marked **Proposal** is this note's recommendation, not a sourced fact. Throughout, `u` is the hypothesis's lever group, `~u` is every other lever, `b_u` is the group's baseline and `f` is a latent (noise-free) posterior sample path of the verdict GP over the in-search box.

## Gist

- √V_T measures how much `f` *varies* as the group moves, not how *far* it moves. The ratio of √V_T to the actual end-to-end change depends on the shape of the effect: it is at most 1/2 (a step), 1/√12 ≈ 0.29 for a linear effect, 1/6 for a pure product interaction, and it tends to 0 for a narrow spike. **No constant rescaling of δ can fix this**, so the user is right that a δ/√12 threshold would be cherry-rigging. It is exact for one shape only.
- The statistic on δ's scale that still counts interactions is the **total-effect range**: the largest change in `f` from moving the group anywhere in its box, with the other levers held at the worst-case setting,
  `M_u = sup over x_~u of [ max over x_u of f(x_u, x_~u) − min over x_u of f(x_u, x_~u) ]`.
  It is the largest range among the group's ICE curves, the oscillation of `f` in `u`. For `a·x`, it gives `a`. For `a·x_A·x_B`, it gives `a`.
- **Δ ≤ M_u holds on every posterior sample path** (proof below). So "UB(M_u) < δ" implies "UB(Δ) < δ": with M_u, `irrelevant` is a *strict sub-case* of `no-improvement`, and retain reduces to LB(Δ) > δ. A separate irrelevance *test* adds no reject power. What it adds is a *label* with a different action: drop the lever's key and keep its trials, versus freeze and filter the trials to baseline. That label is justified *only* by a sup-norm bound, because it bounds every individual trial's error by M_u < δ.
- **Proposal:** replace √V_T with M_u in the verdict engine. Define `irrelevant` as `no-improvement` **and** UB(M_u) < δ. Retain on LB(Δ) > δ alone. Keep √V_T and S_T as telemetry. Apply simultaneous rejects and freezes sequentially, one at a time, because Δ cannot see *substitute* levers.

## Key facts

1. **√V_T is a root-mean-square conditional standard deviation.** The total effect is `V_T,u = E_{x~u}[ Var_{x_u}( f | x_~u ) ]` ([Homma & Saltelli 1996](https://doi.org/10.1016/0951-8320(96)00002-6); [Sobol 2001](https://doi.org/10.1016/S0378-4754(00)00270-6)). Jansen's estimator writes it as half the mean squared difference between two runs that differ only in the group, `(1/2N) Σ (f(a_j) − f(ab_j^(i)))²` ([Lo Piano et al. 2021, Eq. 6](https://arxiv.org/pdf/2203.00639), citing Jansen 1999; [Saltelli et al. 2010](https://doi.org/10.1016/j.cpc.2009.09.018)). So √(2V_T) is an RMS *pairwise difference*, and √V_T is 1/√2 of that. Both are averages over the box.
2. **The √V_T : range ratio depends on the shape of the effect.** For a random variable confined to `[m, M]`, `Var ≤ (M − m)²/4` (Popoviciu's inequality; see [Bhatia & Davis 2000](https://www.researchgate.net/publication/265464069_A_Better_Bound_on_the_Variance) and [Lim & McCann 2020](https://arxiv.org/pdf/2001.11851)). Applied inside the expectation, this gives **√V_T,u ≤ M_u / 2** for every `f`. That is the only universal relation, and it goes one way only. Worked values on `[0,1]` (checked numerically):
   - step of height `a` at 0.5: √V_T = a/2;
   - linear `a·x`: √V_T = a/√12 ≈ 0.29a;
   - product `a·x_A·x_B`: √V_T,A = a/6 (because `E[x_B²] = 1/3`);
   - spike of height `a` and width ε: √V_T ≈ a√ε → 0.

   In every case M_u = a. The product case matters most here. **The interaction that total-order Sobol was chosen to protect is the case √V_T under-reports most** (an interaction worth 6δ can be rejected `irrelevant`), because the variance is averaged over settings of B where A barely matters.
3. **Equivalence and ROPE tests put the margin on the scale of the parameter being tested.** TOST sets bounds "based on the smallest effect size of interest", often in raw units ([Lakens 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/); [Schuirmann 1987](https://doi.org/10.1007/BF01068419)). The ROPE is a range of values of the *same parameter* judged practically equivalent to the null, and "accept" means the whole 95% HDI lies inside it ([Kruschke 2018](https://journals.sagepub.com/doi/10.1177/2515245918771304)). If δ is "the smallest objective change the user cares about", the tested parameter must be an objective *change*, a contrast, and not a dispersion. √V_T < δ is a ROPE on the wrong parameter.
4. **Rescaled dispersion is an ad hoc fix, even in the literature that uses it.** Greenwell, Boehmke & McCarthy (2018) score importance by the "flatness" of the partial dependence: the sample SD for continuous predictors, and the range divided by four for categorical ones, "an estimate of the standard deviation" ([arXiv 1805.04755, §3](https://arxiv.org/pdf/1805.04755)). The factor 4 is a small-sample convention, not a derivation. Their measure is also a PDP (main-effect) measure, and they acknowledge PDPs mislead under interactions and point to ICE curves (Goldstein et al. 2015).
5. **Partial-dependence ranges miss interactions. ICE ranges do not.** The PDP averages the other inputs out. ICE plots show one curve per setting of the others, and so expose heterogeneity that the average hides ([Goldstein, Kapelner, Bleich & Pitkin 2015](https://arxiv.org/abs/1309.6392)). For `x_A·x_B` on `[−1,1]²`, the PDP of A is flat, so its range is 0, while the largest ICE range is 2. **M_u is the largest ICE range**, taken over the whole box rather than the observed rows.
6. **Morris elementary effects are an average absolute difference.** μ* averages `|EE_i|` so that effects of opposite sign cannot cancel. Campolongo et al. (2007) introduce it as a screening proxy for the total index ([Campolongo, Cariboni & Saltelli 2007](https://doi.org/10.1016/j.envsoft.2006.10.004); [Morris 1991](https://doi.org/10.1080/00401706.1991.10484804)). Lo Piano et al. note that the total-index estimator "resembles the method of Morris" ([2021, §2](https://arxiv.org/pdf/2203.00639)). Un-normalised by step size, μ* is on δ's scale, but it is an *average*: a lever worth 5δ in 10% of the box scores about 0.5δ.
7. **Derivative-based measures bound V_T from above, which is the wrong direction for a reject.** For `x_i ~ U[a,b]`, `D_i^tot ≤ (b − a)² ν_i / π²` with `ν_i = E[(∂f/∂x_i)²]` ([Sobol & Kucherenko 2009](https://doi.org/10.1016/j.matcom.2009.01.023)). This generalises to `D_i^tot ≤ C(μ_i) ν_i` with a Poincaré constant for other input laws ([Lamboni, Iooss, Popelin & Gamboa 2013, Thm 3.1](https://arxiv.org/pdf/1202.0943)). GP derivatives are GPs, so ν has closed-form posterior estimators ([De Lozzo & Marrel 2016](https://doi.org/10.1137/15M1013377)). `(b−a)·√ν` equals `a` for a linear effect, but it overstates oscillating effects and needs differentiable, continuous levers, so it cannot handle categorical or bool levers. It is a conservative screen, not an "at least δ" test.
8. **Moment-independent and Shapley measures aren't on δ's scale either.** Borgonovo's δ_i is an L1 distance between output densities, so it is dimensionless and lies in [0, 1] ([Borgonovo 2007](https://doi.org/10.1016/j.ress.2006.04.015)). Shapley effects divide Var(f) among inputs and lie between the first-order and total indices ([Owen 2014](https://doi.org/10.1137/130936233)). They are the right tool for *dependent* inputs ([Song, Nelson & Staum 2016](https://doi.org/10.1137/15M1048070)), but they are still in variance units. ARD lengthscales are ruled out for relevance, as covered in #3 ([Paananen et al. 2019](https://arxiv.org/abs/1712.08048)).
9. **Functionals of GP sample paths give valid posterior bounds for those functionals.** Treating each sensitivity index as a random variable under the GP posterior and taking quantiles across sample paths is the Oakley & O'Hagan / Marrel approach ([Oakley & O'Hagan 2004](https://doi.org/10.1111/j.1467-9868.2004.05304.x); [Marrel et al. 2009](https://arxiv.org/abs/0802.1008)). It applies to any functional, a sup included: compute M_u on each path and take the 95th percentile. No simultaneous-band correction is needed on top, because the sup is inside the functional. Pathwise (random-feature) posterior samples make each path a cheap deterministic function that can be optimised ([Wilson et al. 2020](https://arxiv.org/abs/2002.09309)).
10. **Sup-type statistics pay a log factor in power.** The maximum of N roughly independent Gaussian posterior values sits about `√(2 log N)` posterior SDs above the mean. GP-UCB's union bound over a finite set uses `β_t = 2 log(|D| t² π² / 6δ)` for the same reason ([Srinivas et al. 2010, Thm 1](https://arxiv.org/abs/0912.3995)). UB(M_u) and UB(Δ) are therefore inflated where the posterior is wide, especially in corners the sampler rarely visits. This is conservative for rejects, but it costs trials.
11. **Repeated looks.** Bayesian quantities keep their interpretation under optional stopping when the prior is right ([Rouder 2014](https://doi.org/10.3758/s13423-014-0595-4)). With default or pragmatic priors, which here means MAP-fitted GP hyperparameters, frequentist error rates under repeated looks are not controlled ([de Heide & Grünwald 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8219595/)). The harness's confirming check after max(5·d, 10) more trials is a heuristic safeguard, not an α-spending guarantee. Its false-reject rate has to be measured, for example on the dogfood benchmark.

## Detail

### Why Δ ≤ M_u on every sample path

Fix a posterior path `f`. Let `(x*_u, x*_~u)` maximise `f` with the group free, and let `x°_~u` maximise `f(b_u, ·)`. Then

`Δ = f(x*_u, x*_~u) − f(b_u, x°_~u) ≤ f(x*_u, x*_~u) − f(b_u, x*_~u) ≤ max_{x_u} f(·, x*_~u) − min_{x_u} f(·, x*_~u) ≤ M_u`.

The first inequality holds because `x°_~u` is optimal at baseline. Also, Δ ≥ 0 whenever the free candidate set contains the baseline slice. Because the inequality holds path by path, every quantile of Δ is at most the same quantile of M_u, so `UB(Δ) ≤ UB(M_u)` and `LB(Δ) ≤ LB(M_u)`. Consequences:

- `UB(M_u) < δ ⇒ UB(Δ) < δ`: every `irrelevant` is also a `no-improvement`.
- `LB(Δ) > δ ⇒ LB(M_u) > δ`: the second retain condition is redundant.
- The only verdict region M_u adds is "UB(Δ) < δ **and** UB(M_u) < δ", which is a label on a reject that already happens.

With a finite candidate set the inequality holds only if M_u is evaluated on a set that contains `x*_~u` and `b_u`. **Proposal:** define `irrelevant := no-improvement ∧ UB(M_u) < δ` so that the subset relation holds by construction. A discretisation error in M_u can then change a label but can never cause a reject.

### What still distinguishes `irrelevant` from `no-improvement`

- **Meaning.** `irrelevant` means that moving the group cannot change the objective by δ for *any* setting of the other levers. `no-improvement` means the group may matter, and may even hurt away from baseline, but it cannot *raise* the best achievable objective by δ.
- **Action on the log (#21 "Mapping rules").** "A lever removed as credibly irrelevant has its key dropped", so its off-baseline trials stay in the study as if the lever didn't exist. "Any other removed lever is filtered to its baseline", so those trials leave the study. Dropping the key is safe only if every kept trial's value changes by less than δ when the lever is put back at baseline, that is, `|f(x) − f(b_u, x_~u)| ≤ M_u < δ` for every trial. **√V_T < δ gives no per-trial bound**: a linear lever with √V_T = 0.9δ moves individual trials by about 3.1δ. So the current key-drop rule is licensed by M_u, not by √V_T.
- **Reporting.** #8 asks the final report to list `no-improvement` separately from `irrelevant`. That still holds: "matters but doesn't help" is actionable ("harmful", "only matters in bad regions"), while "irrelevant" is not.

A lever that matters only where the other levers are bad (for example, divergence at a very high learning rate) gets a large M_u and so lands in `no-improvement`, not `irrelevant`. That is correct: its off-baseline trials really did differ, so they shouldn't be kept with the key dropped.

### Interaction with the rest of the verdict engine

- **Interaction survival.** Complements (A helps only with B): both Δ_A, with B optimised, and M_A, at the best B, register the joint gain. Neither is averaged down, unlike √V_T (a/6). **Substitutes** (A alone gains g, B alone gains g, both together gain g) are the failure mode. Δ_A = Δ_B = 0 because the other lever is optimised in both terms, so two co-active hypotheses can both reach `no-improvement` at the same check, and freezing both loses g. This is true of the current spec too. M_A = M_B = g ≥ δ, so neither would be labelled irrelevant. **Proposal:** apply confirmed rejects and freezes among co-active hypotheses one at a time, smallest UB(Δ) first, and recompute the others after each. A confirmed reject already ends the round, so the natural implementation is "one no-improvement reject per round-end".
- **Freezing a single lever** (#21: "when its own upper bound is below δ"). **Proposal:** freeze lever i on UB(Δ_i) < δ, with i free versus i at baseline and everything else optimised. The loss at the optimum is then below δ, which is the property that matters. Separately, mark it key-droppable on UB(M_i) < δ. Freeze one lever per check (same substitute argument). Freezing on √V_T,i < δ would freeze a lever worth up to 3.5δ.
- **Burn-in and confirmation.** Unchanged. Sup statistics are sensitive to extrapolation in corners, so the range-coverage gate matters more. The candidate set for M_u and Δ should be the same quasi-random set, plus the baseline slice, the incumbent and the Δ argmaxes.
- **Noise and the σ floor.** Compute M_u and Δ on *latent* paths `f`, never on noisy predictive draws `y`. Otherwise the max-minus-min picks up about `2σ·√(2 log N)` of pure noise. The σ² floor stops the GP from interpolating noise. Without it, spurious wiggles would inflate both M_u (safe) and Δ (a false *retain*, since retain is now LB(Δ) > δ alone). With the floor, posteriors are wider, so there are fewer verdicts and more "active": conservative.
- **Power.** Rejects no longer come from √V_T, so the reject set shrinks to exactly {UB(Δ) < δ}. That removes the false-irrelevant cases but also slows rejects of truly flat groups, which √V_T used to reject sooner. The size of this cost is not sourced; see open questions.

## Options

| Option | On δ's scale? | Counts interactions? | Categorical / bool | Relation to Δ | Main cost |
|---|---|---|---|---|---|
| **A. Status quo: √V_T < δ** | No; ratio to effect is shape-dependent, 0 to 1/2 | Yes, but averaged (product term: a/6) | Yes | Can fire while LB(Δ) > δ (linear, a ∈ (δ, 3.5δ)); retain blocked for such levers | False `irrelevant` rejects; key-drop not licensed |
| B. √V_T < δ/√12 or δ/2 | Exact for one shape only (linear or step) | Averaged | Yes | Same shape problem | Cherry-rigging (user rejects it); δ/2 via Popoviciu is sound only as a *sufficient* bound in the M direction |
| C. √(2V_T) (Jansen RMS pair difference) | Is a difference, but RMS: linear → 0.41a | Averaged | Yes | No ordering | Localised effects diluted |
| D. Mean ICE range, E_{x~u}[R_u] / μ*-style | Yes, as an average | Averaged | Yes | No ordering | 5δ in 10% of the box scores ~0.5δ |
| E. PDP range | Yes | **No** (main effect only; x_A·x_B → 0) | Yes | No ordering | Myopic failure #3 warned against |
| F. DGSM (b−a)√ν | Linear: exact; oscillating: overstates | Yes (derivative anywhere) | **No** | No ordering | Upper bound on V_T, the wrong direction; continuous only |
| G. Borgonovo δ / Shapley | No (dimensionless / variance) | Yes | Yes | None | Wrong scale; Shapley only needed for dependent inputs |
| **H. Sup ICE range M_u (recommended)** | **Yes; exact oscillation** | **Yes, worst-case over x_~u** | **Yes (max − min over options)** | **Δ ≤ M_u pathwise → irrelevant ⊂ no-improvement** | Sup inflation (fact 10); discretisation (mitigated by the definition); corners |
| I. Trimmed sup: q-quantile of R_u over x_~u | Yes | Mostly; misses effects confined to < 1−q of the box | Yes | No ordering for q < 1 | New knob q; partly reintroduces dilution |
| J. Drop the irrelevance test; Δ only, one reject type | Yes | Yes (via optimisation) | Yes | — | Loses the key-drop licence and the report split; every reject filters trials |

## Recommendation (Proposal)

Adopt **H**. The spec edits to #21 "Verdict engine" (and #8 items 1, 4, 5):

1. Replace √V_T with **M_u**: the largest range of the latent posterior over the group, over settings of the other active levers in the in-search box. Compute it pathwise on the same candidate set as Δ, augmented with the baseline slice and the Δ argmaxes, and take one-sided 95% bounds across paths.
2. `no-improvement`: UB(Δ) < δ. The action is unchanged: freeze and filter to baseline.
3. `irrelevant`: `no-improvement` **and** UB(M_u) < δ. The action is to drop the key and keep the trials.
4. **Retain**: LB(Δ) > δ. LB(M_u) is recorded but implied.
5. Freezing a lever: UB(Δ_i) < δ; key-drop when UB(M_i) < δ as well. Freeze one lever per check.
6. Apply confirmed rejects among co-active hypotheses one per round-end, smallest UB(Δ) first (substitutes).
7. √V_T and normalised S_T become telemetry in the verdict record, next to M_u and Δ.

If a second, faster irrelevance screen is still wanted, the only sound one on this scale is the Popoviciu direction: √V_T,u ≤ M_u/2, so an LB(√V_T) > δ/2 *proves* M_u > δ. That can skip the M computation for clearly-mattering groups. It is a bound, not a rescaled threshold, and it can never cause a reject.

## Open questions

- **Power cost.** How many extra trials does a flat group need to reach UB(Δ) < δ, compared with the old UB(√V_T) < δ? That needs a run of the dogfood benchmark (#38) with flat, linear, product and substitute levers.
- **Candidate-set size and the sup log factor** (fact 10): pick N for M_u and Δ, and decide whether local refinement from the top candidates is worth it.
- **False-reject rate across repeated looks with the confirming check** (fact 11): measure it on the benchmark rather than assuming 5%.
- **Whether `irrelevant` should still require an explicit range-coverage pass in every corner.** The M_u sup is taken over the whole box, including regions the sampler avoided.
- **Substitute detection beyond sequential application**, for example by recording the pairwise M of a jointly frozen set.
