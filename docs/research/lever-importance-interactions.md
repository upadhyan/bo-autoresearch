# Lever importance and interaction detection from short, noisy BO rounds

Resolves issue #3. Feeds the **Reject-condition form** decision on map #1.

Question: what machine-checkable ways are there to decide whether a **lever** matters, including through interactions with other levers, using the few dozen noisy, non-uniformly sampled **trials** of one **round**?

Sources were read on 2026-09-24. Optuna claims are checked against source on `master` (commit `984b37b946`, version string `5.1.0.dev`; latest PyPI release is 5.0.0). Paper claims are checked against the PDFs.

## Key facts for Reject-condition form

1. **The quantity to reject on is the total-order (Sobol) index, not a main effect.** A main effect averages the other levers out. For `f = x1 * x2` on `[-1, 1]^2`, both main effects are exactly zero and all variance is in the pairwise term. The total-order index `S_T(i) = 1 - Var(E[f | x_~i]) / Var(f)` counts every term containing `i` (Homma & Saltelli 1996). A rejection that looks only at the main effect is the myopic failure named on map #1.
2. **None of Optuna's three evaluators exposes interaction terms, and none returns an uncertainty.** All return one scalar per param (`optuna/importance/*`). `FanovaImportanceEvaluator` computes a per-tree std and then throws it away (`get_importance(i)[0]`). Optuna's evaluators cannot, on their own, support a claim that a lever doesn't matter.
3. **The default evaluator, PED-ANOVA in local mode, is explicitly unsafe for rejection.** Its authors write that it is "not appropriate to discard HPs with low HPI only in the local space" because such params "are likely to have more interaction effects" (Watanabe et al. 2023, App. E.2). Local mode also scores a lever that TPE has already converged on as unimportant.
4. **Minimum trials are well above one round.** PED-ANOVA needs at least 5 trials in the top quantile (Optuna docstring). At the default `target_quantile=0.1` that means at least 41 trials, and at 30 trials it means `target_quantile >= 0.17`. On a 12-D benchmark its ratios only converged at about 10^4 points, and Optuna's fANOVA had not converged by 10^4 (Watanabe et al. 2023, Fig. 15). For GP surrogates the standard floor is about `10 * d` runs, and that is for deterministic, space-filling designs (Loeppky, Sacks & Welch 2009). Noisy BO data needs more.
5. **BO sampling biases every surrogate-based method, in method-specific ways.** fANOVA integrates over the uniform box (Hutter et al. 2014, Def. 5), but its RF is only accurate where BO sampled, and Optuna's docs recommend `RandomSampler`. PDP and ICE assume independence, and BO breaks that by correlating params (Moosbauer et al. 2021). MDI uses training-set impurity and favours high-cardinality params (scikit-learn docs). Optuna's fANOVA integrates log-scale params uniformly in raw space (`transform_log=False`).
6. **GP ARD lengthscales are not a reject statistic.** They conflate non-linearity with relevance (Paananen et al. 2019). In Optuna they are a private MAP point estimate under a heuristic prior (`optuna/_gp/prior.py`), so they come with no uncertainty. They also cannot separate a main effect from an interaction.
7. **Pairwise terms are available off-Optuna.** `automl/fanova` gives `quantify_importance((i, j))` and `get_most_important_pairwise_marginals`, each with a std across trees. SALib `sobol.analyze` gives `S1`, `S2` and `ST`, each with a `_conf` value. SHAP gives per-trial `(n, d, d)` interaction matrices (`TreeExplainer.shap_interaction_values`, Lundberg et al. 2018), but with no uncertainty. ICE and 2-way PDPs show interactions only visually.
8. **The one principled machine check is Sobol indices on a noise-aware GP posterior.** Fit a GP with a noise term, draw posterior function samples, compute `S_T(lever)` (and `S2` for named pairs) over the round's box for each sample, and read off a credible interval (Oakley & O'Hagan 2004; Marrel et al. 2009). Where BO never sampled, the posterior reverts to the prior, which widens the interval instead of hiding the gap.
9. **A reject condition needs a third outcome, "inconclusive".** The verdict should be `reject` only when all hold: the upper credible bound of `S_T(lever)` is below τ, `n_trials >= max(10*d, floor)`, the lever's sampled values cover its range, and the surrogate passes a held-out fit check. Otherwise the verdict is `inconclusive`, which carries the hypothesis forward and is never a rejection.

| Method | Pairwise terms? | Uncertainty? | Min trials (sourced) | Bias under TPE/GP sampling | Safe as a reject statistic? |
|---|---|---|---|---|---|
| Optuna `PedAnovaImportanceEvaluator` (default) | No (1-D only in Optuna; higher order is in the paper but exponential) | No | ≥5 in top quantile → ≥41 at default γ′=0.1 | Local mode removes sampling bias by design but measures "importance during the search"; converged levers score low | **No** (authors say so) |
| Optuna `FanovaImportanceEvaluator` | No (the private tree code could; the API doesn't) | No (std computed, then dropped) | Code: 2. No convergence by 10^4 on 12-D | Uniform-measure integral over an RF that only saw BO's region; raw-space integration for log params | No |
| Optuna `MeanDecreaseImpurityImportanceEvaluator` | No | No | Code: 2 | Training-set impurity; favours high-cardinality params | No |
| `automl/fanova` (original) | Yes, any order | std across trees | Not stated; paper demo used 100 random points | Same RF-extrapolation issue | Secondary signal only |
| GP ARD lengthscales (Optuna GP) | No (a product kernel models interactions but doesn't report them) | No (MAP) | ~10·d | Conflates non-linearity with relevance | No |
| PDP / ICE (sklearn) | 2-way PDP visual; ICE spread visual | GP-PDP bands (Moosbauer) | Surrogate-dependent | Independence assumption broken by BO | Visual audit only |
| SHAP interaction values on tree surrogate | Yes, per trial | No | Surrogate-dependent | `tree_path_dependent` uses BO's sample distribution as background | Secondary signal only |
| **Sobol S1/S2/ST on GP posterior samples** | **Yes (S2), plus total ST** | **Yes (credible interval over posterior)** | ~10·d lower bound | Uniform measure is explicit; unexplored regions widen intervals | **Yes, with preconditions** |

## Detail

### 1. What "doesn't matter" has to mean

Functional ANOVA writes the objective surface as a sum of terms over subsets of levers, `f(θ) = Σ_U f_U(θ_U)`. Its variance splits the same way, `V = Σ_U V_U`, and `F_U = V_U / V` is the fraction explained by subset `U` (Hutter et al. 2014, §3.1, Eqs. 4–6). The main effect `f_{j}` averages over all instantiations of the other levers. Terms with `|U| > 1` "capture exactly the interaction effects between all variables in U" (§3.1).

Two consequences for reject conditions:

- A lever can have `F_{j} = 0` and still matter a lot through `F_{j,k}`. Hutter et al.'s own Online LDA example shows this: 65% of the variance is batch size alone and 18% is its interaction with κ, and "κ is much more important for small batch sizes than for large ones — an interaction effect that cannot be captured by single marginals" (§4.1).
- The statistic that rules out every route by which lever `j` can matter is the **total-order index**, the sum of `F_U` over all `U ∋ j`. Equivalently, `S_T(j) = E_{x_~j}[Var_{x_j}(f | x_~j)] / Var(f)` (Homma & Saltelli 1996). Only a small `S_T` licenses "this lever doesn't matter".

Watch the naming: `automl/fanova`'s "total importance" for a set `U` is the variance of the joint marginal of `U` (all sub-effects of `U`). It is **not** the Sobol total-order index, which also includes interactions with levers outside `U` (`fanova/fanova.py`, `__compute_marginals` / `quantify_importance`).

### 2. Optuna's importance evaluators

Common facts, from `optuna/importance/__init__.py` and the evaluators:

- `get_param_importances` defaults to `PedAnovaImportanceEvaluator` on master. It returns `dict[str, float]`, normalized to sum to 1 by default. With `normalize=True`, if every raw value is zero it returns `1/n_params` for every param. So "equal importances" can mean "nothing could be estimated".
- All three evaluators return one scalar per param. None returns pairs or intervals.
- fANOVA and MDI assess only params present in every completed trial, so they drop conditional levers. PED-ANOVA handles conditional params via regimes (Conditional PED-ANOVA, arXiv 2601.20800).

#### 2a. `FanovaImportanceEvaluator`

- **Model:** sklearn `RandomForestRegressor(n_estimators=64, max_depth=64, min_samples_split=2, min_samples_leaf=1)` (`_fanova/_evaluator.py`, `_fanova/_fanova.py`). With `min_samples_leaf=1` at n≈30, trees can put every trial in its own leaf, so objective noise is fitted and then attributed to levers as variance.
- **Measure:** the marginal is taken under the uniform density over the configuration box, `â_U(θ_U) = (1/||Θ_T||) ∫ ŷ dθ_T` (Hutter et al. 2014, Def. 5, Eq. 1). Optuna builds the box with `_SearchSpaceTransform(..., transform_log=False)`, so a `log=True` lever such as a learning rate over `[1e-5, 1e-1]` is integrated uniformly in raw space. That space is dominated by its top decade.
- **Interactions:** the paper's Algorithm 2 computes `F_U` for all `|U| ≤ K`. Optuna's private `_FanovaTree.get_marginal_variance(features)` accepts several columns, so `V_{ij} = V({i,j}) − V_i − V_j` is computable, but only via private API. The public evaluator loops over single params only.
- **Uncertainty:** the paper computes "means and standard deviations across the trees" (§3.2). Optuna's `_Fanova.get_importance` returns `(mean, std)`, and the evaluator keeps only `[0]`. Across-tree std also measures bootstrap disagreement, not posterior uncertainty about `f`.
- **Minimum trials:** the code needs 2 (`len(trials) <= 1` returns zeros), and raises if every tree has zero variance. The paper's ground-truth demo used 100 random points out of a 288-point grid (§4.1). Watanabe et al. found Optuna fANOVA had not converged at 10^4 points on JAHS-Bench-201 (Fig. 15).
- **BO bias:** Optuna's docstring says reliability depends on RF accuracy and recommends "an exploration-oriented sampler such as RandomSampler". The RF is piecewise-constant outside the sampled region, so the uniform-box integral over unexplored space is extrapolation (see also Watanabe et al. 2023, App. B.5, citing Moosbauer et al. 2021).

#### 2b. `PedAnovaImportanceEvaluator` (default)

- **Definition:** local HPI is `γ′² / γ² · D_PE(p_d(·|D_γ′) ‖ p_d(·|D_γ))`, the Pearson divergence between the 1-D KDE of the top-γ′ trials and the KDE of the top-γ trials (Watanabe et al. 2023, Thm. 1, Alg. 1). Optuna computes `pdf_local @ ((pdf_top / pdf_local − 1) ** 2)` on a 50-point grid, using Scott's-rule Parzen estimators with `prior_weight=1.0` (`_ped_anova/evaluator.py`, `scott_parzen_estimator.py`). Unlike fANOVA, it builds the grid in log space for log params.
- **Defaults:** `target_quantile=0.1` (γ′), `region_quantile=1.0` (γ), `evaluate_on_local=True`. So by default it compares the top 10% with **all observed trials**, not with the uniform box.
- **Interactions:** the Optuna implementation is per-param only. The paper generalizes to higher orders (App. C.2, Eq. 47) but lists as a limitation that order-`s` terms cost `O(Π_{d∈s} n_d²)` (App. E.1). It also notes that the exact `v_s / v_0` percentage is unavailable in high dimension.
- **Uncertainty:** none.
- **Minimum trials:** the docstring says "it is preferable to include at least 5 trials above `target_quantile`". The quantile filter keeps `ceil(γ′·n)` trials plus ties, so γ′=0.1 needs n ≥ 41. Each regime needs at least 2 trials or it is dropped with a warning. The KDE prior carries weight 1 against the γ′·n top trials: 25% of the mass with 3 top trials. That shrinks small-n importances toward zero, which is the direction that produces false rejections. Paper App. D.2 / Fig. 15: ratios "start to converge from 10^4 data points" on 12-D JAHS.
- **BO bias:** the paper claims the Lebesgue-split (quantile) local space "remove[s] the sampling bias caused by a nonuniform sampler" (§3.1.1). For post-hoc HPO analysis it recommends γ=1 with `p_d(x_d|D)` instead of uniform (App. E.2), and Optuna's defaults do this. The cost is that it measures "HPI during the search". If TPE has already concentrated a lever near its optimum, `p(x|top) ≈ p(x|all)` and the lever scores low *because* it matters and was found. The paper says "it is also not appropriate to discard HPs with low HPI only in the local space because low local HPI just implies that the HPs have a similar trend both in the global and local spaces and the HPs are likely to have more interaction effects" (App. E.2).

#### 2c. `MeanDecreaseImpurityImportanceEvaluator`

- It uses the same RF as fANOVA and returns `forest.feature_importances_`, summed over one-hot columns (`_mean_decrease_impurity.py`).
- scikit-learn: impurity-based importance "is strongly biased and favor[s] high cardinality features", and it is computed from training-set statistics (permutation-importance user guide). An integer lever with 3 values is penalized against a float lever.
- It gives no interactions, no uncertainty, and has no measure-theoretic meaning. It is unfit for rejection.

### 3. Alternatives

#### 3a. GP ARD lengthscales

- Optuna's `GPSampler` uses a Matérn-5/2 kernel with one inverse squared lengthscale per param (`optuna/_gp/gp.py`, `GPRegressor.length_scales`). It fits these by MAP with the prior `-(0.1/ℓ⁻² + 0.1·ℓ⁻²)`, which the source comments call "picked by heuristics" (`optuna/_gp/prior.py`). It also fits a noise variance. The model lives in the private `optuna._gp` module and becomes active after `n_startup_trials=10`.
- Paananen et al. (AISTATS 2019) show ARD relevance is unreliable: a lever with a strong but near-linear effect gets a long lengthscale, so it looks irrelevant, while non-linearity shortens lengthscales. They propose sensitivity-based measures (KL, VAR) on the posterior predictive instead.
- A single ARD kernel models interactions multiplicatively, so a pure-interaction lever still gets a short lengthscale. But the lengthscale cannot say whether the lever acts through a main effect or an interaction, and MAP gives no uncertainty.

#### 3b. Partial dependence and ICE

- PD is the expected response as a function of the lever with the other levers marginalized out (Friedman 2001). ICE draws one curve per sample (Goldstein et al. 2015). scikit-learn: "Both PDPs and ICEs assume that the input features of interest are independent from the complement features", and PDPs "can obscure a heterogeneous relationship created by interactions. When interactions are present the ICE plot will provide many more insights". 2-way PDPs "show the interactions among the two features" (scikit-learn user guide, partial dependence).
- BO violates the independence assumption because its samples cluster and correlate params. Moosbauer et al. (NeurIPS 2021) show that BO's sampling bias corrupts PDPs of HPO surrogates. Their fix is to use the GP posterior for PD confidence bands and to split the space into sub-regions where the PD is trustworthy.
- For `f = x1·x2`, the PD of each lever is flat. PD alone therefore reproduces the myopic rejection. ICE spread, a 2-way PD, or Friedman's H-statistic (Friedman & Popescu 2008) flag the interaction, but the H-statistic has no built-in uncertainty.

#### 3c. SHAP interaction values on a surrogate

- The pairwise SHAP interaction value is the Shapley interaction index `Φ_ij = Σ_{S⊆N∖{i,j}} |S|!(M−|S|−2)! / (2(M−1)!) · ∇_ij(S)`. The main effect is `Φ_ii = φ_i − Σ_{j≠i} Φ_ij`, the matrix is symmetric, and each row sums to the lever's SHAP value (Lundberg, Erion & Lee 2018, Eqs. 3–6; the tree algorithm runs in `O(TMLD²)`).
- The library is `shap.TreeExplainer(model).shap_interaction_values(X)`, which returns `(n_samples, n_features, n_features)` and whose rows "sum to the SHAP value" (SHAP docs). `feature_perturbation="tree_path_dependent"` uses "the number of training examples that went down each leaf" as the background. For BO data that makes the sampler's distribution the reference measure. `"interventional"` needs a background dataset, and a uniform sample of the search box could be passed to realign it with the uniform measure.
- The values are local, per trial. A global number such as `mean |Φ_ij|` is a heuristic with no variance-decomposition meaning and no uncertainty. The method inherits all the tree-surrogate problems of §2a.

#### 3d. Sobol indices on the GP posterior (recommended primary statistic)

- Oakley & O'Hagan (JRSS-B 2004) and Marrel et al. (RESS 2009) compute Sobol indices from a GP emulator. They treat each index as a random variable under the GP posterior, which gives a mean and a credible interval. Marrel et al. find that the full-stochastic-process approach converges better and is more robust than using the predictor mean alone.
- In practice: draw `K` posterior function samples, or use the posterior mean plus sampled GP hyperparameters. Estimate `S1`, `S2`, and `ST` on each with SALib's `sobol.analyze`, which costs `N·(2d+2)` surrogate evaluations. These are cheap because they are surrogate calls, not trials. Then take quantiles across the `K` samples. SALib's own `*_conf` values cover only Monte-Carlo estimator error, not surrogate uncertainty, so the outer loop over posterior samples is what makes the interval honest.
- Why this suits the use case:
  - The integration measure is chosen explicitly: the round's lever box, in log space for log levers.
  - The noise term keeps trial noise out of the lever variance.
  - Unexplored regions revert to the prior and widen the interval rather than silently contributing zero.
  - `ST` covers every interaction, and `S2` names the partner lever, which supports the reason "L matters only via M".
- Limits:
  - Aim for about 10·d trials as a floor. That figure comes from deterministic, space-filling designs (Loeppky, Sacks & Welch 2009), and BO rounds are neither.
  - A single stationary kernel can be misspecified.
  - Categorical levers need a suitable kernel. Optuna's GP handles categoricals, but through private API.

### 4. Implications for the Reject-condition form (proposal, not decided)

- Pre-register the reject condition against **`S_T(lever)`, the total-order index**, and optionally `S2(lever, partner)`. Never pre-register it against a main effect or an Optuna importance.
- Make the verdict three-valued: `reject`, `keep`, `inconclusive`. Emit `reject` only when all of these hold:
  1. The upper credible bound of `S_T` is below a pre-registered τ.
  2. `n_trials` meets a floor, at least 10·d.
  3. The lever's sampled values cover its range. For example, some trials fall in each third of the range on the lever's native scale.
  4. The surrogate passes a held-out or LOO fit check.
  If any fails, emit `inconclusive`, and the hypothesis survives the round.
- Warm start maps old trials to the lever's "off" value (CONTEXT.md), creating a point mass. Coverage checks and the integration measure should count only the round's own trials for that lever, or treat "off" as its own category.
- Optuna evaluators can go in the round summary as descriptive telemetry, and PED-ANOVA local HPI is useful for "what drove the good trials". They must not be reject statistics.
- The dogfood benchmark on map #1, with planted useful, useless and interaction-only levers, should assert that the rule returns `inconclusive` or `keep`, never `reject`, for the interaction-only lever at the round size v1 uses.

## Sources

- Hutter, Hoos, Leyton-Brown (2014). *An Efficient Approach for Assessing Hyperparameter Importance*. ICML, PMLR 32. http://proceedings.mlr.press/v32/hutter14.html
- Watanabe, Bansal, Hutter (2023). *PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces*. IJCAI. https://arxiv.org/abs/2304.10255
- Conditional PED-ANOVA (KDD 2026), cited by Optuna. https://arxiv.org/abs/2601.20800
- Lundberg, Erion, Lee (2018). *Consistent Individualized Feature Attribution for Tree Ensembles*. https://arxiv.org/abs/1802.03888. Also Lundberg et al. (2020), *From local explanations to global understanding with explainable AI for trees*, Nature Machine Intelligence 2:56–67.
- Moosbauer, Herbinger, Casalicchio, Lindauer, Bischl (2021). *Explaining Hyperparameter Optimization via Partial Dependence Plots*. NeurIPS. https://arxiv.org/abs/2111.04820
- Paananen, Piironen, Andersen, Vehtari (2019). *Variable selection for Gaussian processes via sensitivity analysis of the posterior predictive distribution*. AISTATS. https://arxiv.org/abs/1712.08048
- Oakley & O'Hagan (2004). *Probabilistic sensitivity analysis of complex models: a Bayesian approach*. JRSS-B 66(3):751–769.
- Marrel, Iooss, Laurent, Roustant (2009). *Calculations of Sobol indices for the Gaussian process metamodel*. RESS 94:742–751. https://arxiv.org/abs/0802.1008
- Homma & Saltelli (1996). *Importance measures in global sensitivity analysis of nonlinear models*. RESS 52:1–17.
- Loeppky, Sacks, Welch (2009). *Choosing the Sample Size of a Computer Experiment: A Practical Guide*. Technometrics 51(4):366–376. https://www.tandfonline.com/doi/abs/10.1198/TECH.2009.08040
- Friedman (2001). *Greedy function approximation: a gradient boosting machine*. Ann. Statist. 29(5). Friedman & Popescu (2008). *Predictive learning via rule ensembles*. Ann. Appl. Statist. 2(3). Goldstein, Kapelner, Bleich, Pitkin (2015). *Peeking Inside the Black Box*. JCGS 24(1).
- Optuna source, `master` @ `984b37b946`: `optuna/importance/{__init__.py,_fanova/*,_ped_anova/*,_mean_decrease_impurity.py}`, `optuna/_gp/{gp.py,prior.py}`, `optuna/samplers/_gp/sampler.py`. https://github.com/optuna/optuna
- `automl/fanova`, `fanova/fanova.py`. https://github.com/automl/fanova
- scikit-learn user guide: permutation importance (https://scikit-learn.org/stable/modules/permutation_importance.html) and partial dependence and ICE (https://scikit-learn.org/stable/modules/partial_dependence.html).
- SHAP `TreeExplainer` docs. https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
- SALib `sobol.analyze` docs. https://salib.readthedocs.io/en/latest/api/SALib.analyze.html
