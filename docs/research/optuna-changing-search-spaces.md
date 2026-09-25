# Optuna mechanics for changing search spaces

Resolves [#2](https://github.com/upadhyan/bo-autoresearch/issues/2). Feeds **Warm-start policy** and **Runner harness API** (map: #1).

**Version checked:** Optuna **v5.0.0** (latest on PyPI and GitHub on 2026-09-24), tag commit `01ddd17ab9f36c3d2817aed9afe376e1cb047d35`. Source links below point at that tag (`SRC` = `https://github.com/optuna/optuna/blob/v5.0.0`). Behaviour claims marked **[probe]** were also confirmed by running a script against optuna 5.0.0 with SQLite storage (see [Probe](#probe)).

## Key facts for Warm-start policy / Runner harness API

1. **Every model-based sampler needs identical distributions.** The multivariate TPE (the v5 default for single-objective), GPSampler and CmaEsSampler each build their joint model only over the *intersection search space*. A param stays in that space only if every finished trial in the study has an **identical** distribution for it, bounds included. Change any bound, or add or drop a lever, and that param leaves the joint model for the rest of the study. TPE then models it one dimension at a time. GP and CMA-ES sample it **uniformly at random**.
2. **So use one study per round, re-expressed.** Create a fresh study per round in the same SQLite file. Warm-start it with `create_trial` + `add_trials`, rewriting each prior trial to the current round's *exact* `distributions` and param set. This is the only way the joint model sees the prior data.
3. **`create_trial` validates hard.** Every param value must lie inside the distribution you pass, and `params` and `distributions` must have the same keys. Otherwise it raises `ValueError`. So for a narrowed lever you must drop or re-label the out-of-range prior trials. For an added lever you must impute a value (the baseline the code had before the lever existed). For a dropped lever you must delete the key.
4. **`enqueue_trial` validates softly.** Out-of-range fixed values only warn, and **the trial really runs with the out-of-range value**. Keys the objective never suggests are ignored. Levers left out are sampled normally. The harness must bounds-check enqueued seeds itself.
5. **SQLite enforces three rules within one study; InMemory enforces none.** Within a single study, `RDBStorage` raises `ValueError` if a param changes kind (int↔float↔categorical), changes its `log` flag, or changes its categorical `choices`. Float/int bounds and `step` may change freely. `InMemoryStorage` skips these checks when trials come from `add_trial`. Test the harness against SQLite, not in-memory storage.
6. **Out-of-range priors bias TPE towards the edge.** If old trials that sit outside a narrowed range are kept in the study, TPE turns them into kernels truncated to the new bounds. In the probe, the optimum at x=15 with the range narrowed to [0,5] gave samples of 4.85–5.0. That is one more reason to rebuild priors in a fresh study.
7. **`group=True` keeps the oldest distribution and separates new levers.** TPE's `group=True` keys subspaces by param name and keeps the *first-seen* distribution, so widened bounds are never explored through the joint model. It also puts a lever added in a later round into its **own subspace**, so its interactions with the older levers are not modelled. Re-expressing into a fresh study avoids both problems.
8. **Use `trial.set_constraint`; GP needs the same constraint count everywhere.** `constraints_func` is **deprecated in v5.0.0**; use `trial.set_constraint(name, value)` or `create_trial(constraints=...)`. TPE sums the positive violations per trial and tolerates missing or renamed constraints. GPSampler raises `ValueError` unless every completed trial has the **same number** of constraints. CmaEsSampler ignores constraints and only warns. The warm-start constraint set must match the current round's set.
9. **Warm-start trials count toward `n_startup_trials`.** The count is taken over all COMPLETE trials (TPE also counts PRUNED). If warm-start trials reach the threshold, the round skips random start-up. Copying `system_attrs` verbatim also copies CMA-ES's pickled optimizer (`cma:*` keys), which carries the old dimensionality. Copy only params, values, constraints and `user_attrs` for lineage.
10. **Crash-safe resume keeps finished trials, not sampler state.** `RDBStorage` commits each write, so `create_study(..., load_if_exists=True)` resumes every finished trial. The sampler's RNG state is lost unless you pickle the sampler. A killed trial stays `RUNNING` forever unless heartbeat is on (`heartbeat_interval`, **`optimize()` only**, not ask/tell), or the harness fails stale `RUNNING` trials itself at startup. SQLite suits a single process only.

---

## 1. `add_trial` / `create_trial` / `enqueue_trial` under changed distributions

### What each primitive checks

| Primitive | What it validates | Source |
|---|---|---|
| `optuna.trial.create_trial` | Runs `FrozenTrial._validate()`. `params` and `distributions` must have the same keys, and each value must be contained in its own distribution, or it raises `ValueError`. A COMPLETE trial needs a value and no NaN. The `constraints=` kwarg calls `set_constraint`. | [`trial/_frozen.py#L519-L637`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_frozen.py#L519-L642), [`_validate` L300-L337](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_frozen.py#L300-L337) |
| `Study.add_trial` / `add_trials` | Calls `trial._validate()` and checks the objective count, then `storage.create_new_trial(template_trial=...)`. It does **not** check the trial against the study's other trials. | [`study/study.py#L907-L977`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/study/study.py#L907-L977) |
| RDB `create_new_trial` with template | Writes each param through `_set_trial_param_without_commit`, which loads the **first** stored distribution of that param name *in the same study* and runs `check_distribution_compatibility`. | [`storages/_rdb/storage.py#L567-L572`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_rdb/storage.py#L567-L572), [`#L625-L649`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_rdb/storage.py#L625-L649) |
| InMemory `create_new_trial` with template | Deep-copies the template. It runs **no** compatibility check and doesn't register the distribution. Only `set_trial_param` checks. | [`storages/_in_memory.py#L160-L172`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_in_memory.py#L160-L172), [`#L207-L214`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_in_memory.py#L207-L214) |
| `Study.enqueue_trial` | Adds a `WAITING` trial with `system_attrs={"fixed_params": params}`. It does no validation against any distribution. `skip_if_exists` compares the key set and values. | [`study/study.py#L842-L905`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/study/study.py#L842-L905), [`_should_skip_enqueue` L1094-L1122](https://github.com/optuna/optuna/blob/v5.0.0/optuna/study/study.py#L1094-L1122) |

The compatibility rule itself is in [`distributions.py#L623-L658`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/distributions.py#L623-L658). It checks three things. The distribution class must match (Float, Int or Categorical). The `log` flag must match for Float and Int. Categorical choices must be equal. Bounds and `step` are **not** checked.

### Case by case

The table assumes SQLite (RDB) storage and **the same study**. In a *new* study, only `create_trial`'s own validation applies.

| Change between rounds | `add_trial(create_trial(...))` into the same study | `enqueue_trial` / `suggest_*` in the same study |
|---|---|---|
| Float/int bounds **widened** | OK **[probe]** | OK |
| Float/int bounds **narrowed** | OK if the value lies inside the distribution you declare. Otherwise `ValueError` from `_validate` **[probe]** | An out-of-range fixed value warns `Fixed parameter x ... is out of range` and **is used anyway** ([`trial/_trial.py#L633-L646`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_trial.py#L633-L646)) **[probe: x=20 ran against [0,10]]** |
| `step` changed | OK (not checked) | OK |
| `log` flag changed | `ValueError: Cannot set different log configuration...` **[probe]** | Same error at `suggest_*` |
| Int ↔ float ↔ categorical | `ValueError: Cannot set different distribution kind...` **[probe]** | Same error |
| Categorical choices changed (added, removed or reordered) | `ValueError: CategoricalDistribution does not support dynamic value space.` **[probe]** | Same error at `suggest_categorical` **[probe]** |
| **New** param (lever added) | Allowed. Old trials simply lack the key. | A key missing from `fixed_params` is sampled normally **[probe]** |
| **Dropped** param (lever removed) | Allowed. The new trials lack the key. | An `enqueue_trial` key never suggested is ignored; it stays only in `system_attrs["fixed_params"]` **[probe]** |
| **Fixed** param (a lever pinned to one value) | Declare a single-point distribution (`low == high`) or leave the key out. Every sampler skips `single()` distributions in the relative space and `Trial` returns the lone value ([`trial/_trial.py#L615-L616`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_trial.py#L615-L616)). Changing a lever from a range to a point still changes its distribution, so it leaves the intersection space. | `enqueue_trial({"lr": 1e-3})` or `PartialFixedSampler` ([`samplers/_partial_fixed.py#L66-L105`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_partial_fixed.py#L66-L105)) removes the lever from the relative space and returns the fixed value. |

In `FrozenTrial._suggest`, used when re-running a frozen trial, an out-of-range value also only warns ([`trial/_frozen.py#L336-L357`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_frozen.py#L336-L357)).

The official FAQ says only that, when a search space changes, "the behavior when altered is defined by each sampler individually" ([`docs/source/faq.rst#L266-L277`](https://github.com/optuna/optuna/blob/v5.0.0/docs/source/faq.rst#L266-L277), [rendered FAQ](https://optuna.readthedocs.io/en/stable/faq.html#what-happens-when-i-dynamically-alter-a-search-space)). The sampler sections below give that behaviour.

## 2. How each sampler handles a dynamic space

### How dispatch works for all samplers

For each `suggest_*`, `Trial._suggest` uses the first rule that applies ([`trial/_trial.py#L602-L631`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_trial.py#L602-L631)):

1. A value already set in this trial is reused.
2. A value in `fixed_params` (from an enqueued trial) is used.
3. A `single()` distribution returns its only value.
4. A relatively sampled value is used, but only if the param is in `relative_search_space` and the value is *contained* in the current distribution ([`#L648-L665`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_trial.py#L648-L665)).
5. Otherwise the sampler's `sample_independent` is called.

For TPE, GP and CMA-ES, the relative search space is built by `IntersectionSearchSpace`. It keeps a param only when every finished trial has `trial.distributions.get(name) == distribution`, with bounds included ([`search_space/intersection.py#L14-L55`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/search_space/intersection.py#L14-L55); the docstring says params "with dynamic value ranges are excluded"). The space is cached and only ever shrinks as new trials arrive.

**[probe]** Take trial 1 with `{x:[0,10], y:[0,1]}` and trial 2 with `{x:[0,5], y:[0,1], z:[0,1]}`. The intersection is `['y']`.

### TPESampler

- **`multivariate` defaults to `True` for single-objective studies in v5** (`None` means auto; see [`samplers/_tpe/sampler.py#L159-L168`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L159-L168) and [`_is_multivariate` L414-L429](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L414-L429)).
- **When `multivariate=True, group=False`:** the relative space is the intersection space with `include_pruned=True`, minus single-point distributions ([`#L390`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L390), [`#L454-L459`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L454-L459)). Params outside it fall back to `sample_independent`. That is still TPE, one dimension at a time, not random. The code's own warning text says "`multivariate=True,group=False` does not support dynamic search space (but `multivariate=True,group=True` works)" ([`#L507-L539`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L507-L539)). **[probe]** After narrowing `x` from [0,20] to [0,5] in one study, the relative space was `[]`.
- **When `group=True` (experimental):** `_GroupDecomposedSearchSpace` partitions params by co-occurrence across COMPLETE and PRUNED trials, and each subspace gets its own multivariate TPE ([`search_space/group_decomposed.py#L22-L67`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/search_space/group_decomposed.py#L22-L67), [`sampler.py#L441-L452`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L441-L452), [`#L464-L478`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L464-L478)). This is the built-in answer for **conditional params**, where a lever only exists under some branch. It has three effects when levers change between rounds:
  - The subspace keeps the distribution of the **first** trial that introduced the name, because `add_distributions` reuses `search_space[name]`. If the bounds were widened, relative samples come only from the old, narrower range. If they were narrowed, a relative value outside the new range fails `_is_relative_param` and falls back to independent sampling.
  - **[probe]** After rounds `{a,b}` then `{a,b,c}`, the subspaces were `[['a','b'], ['c']]`. The new lever `c` is modelled independently of `a` and `b`, so **no interaction between new and old levers is learned**. This matters for the planted-interaction dogfood test.
  - Dropping a lever splits its subspace in the same way.
- **Which trials are used:** `n_startup_trials` counts all COMPLETE and PRUNED trials in the study, whatever their params ([`#L493-L521`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L493-L521)). Next, the good/bad split (below/above γ) is made over **all** trials. Then `_get_internal_repr` keeps only the trials whose params contain every key in the space being sampled ([`#L564-L574`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L564-L574), [`#L576-L618`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L576-L618)). A newly added lever is therefore modelled only from the trials that have it, but the split is still based on all trials. With `constant_liar=True` (the default), RUNNING trials go into the "above" set ([`#L740-L776`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L740-L776)).
- **Prior values outside the current bounds:** they become kernel centres. `compute_sigmas` uses the current `low` and `high` as endpoints, and the kernels are truncated normals on the current `[low, high]` ([`samplers/_tpe/parzen_estimator.py#L154-L221`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/parzen_estimator.py#L154-L221)). Samples always stay in bounds, and nothing errors. But good trials beyond an edge pile probability mass at that edge. **[probe]** With 20 priors on [0,20] and the optimum at 15, narrowing to [0,5] produced the samples `[4.85, 4.99, 4.99, 5.0, 5.0]`. A categorical value that is no longer among the choices can't reach this point, because the storage rejects the changed choices first.

### GPSampler

- The relative space is the intersection space (COMPLETE, WAITING and RUNNING trials; pruned trials excluded) minus single-point distributions ([`samplers/_gp/sampler.py#L253-L254`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L253-L254), [`#L297-L306`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L297-L306)). Only **COMPLETE** trials train the GP. Pruned trials are ignored ([`#L379-L394`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L379-L394)).
- Every param outside the intersection, whether conditional, changed or new, goes to `independent_sampler`, which is **`RandomSampler` by default**. The docstring says this sampler is used "for conditional parameters" ([`#L191-L194`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L191-L194)). The docs' feature table marks GP's conditional-space support as partial (▲) ([`docs/source/reference/samplers/index.rst#L30`](https://github.com/optuna/optuna/blob/v5.0.0/docs/source/reference/samplers/index.rst#L30)).
- The fitted-GP cache is dropped when the dimension changes ([`#L438-L444`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L438-L444)). Normalisation uses the current distribution's bounds ([`_gp/search_space.py#L62-L88`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/_gp/search_space.py#L62-L88)). Because intersection membership requires identical distributions, out-of-range priors can't reach the GP from inside one study.
- **Practical upshot:** GP only benefits from a warm start when the prior trials are re-expressed to the current round's exact distributions in a fresh study.

### CmaEsSampler

- The relative space is the intersection space, **float and int params only**. Categoricals are always sampled independently ([`samplers/_cmaes.py#L372-L388`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L372-L388)).
- The optimizer is pickled into trial `system_attrs` under `cma:`, `sepcma:` or `cmawm:` keys, and the most recent one is restored ([`#L322-L328`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L322-L328), [`#L489-L505`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L489-L505)). If the restored optimizer's dimension differs from the current space, it logs "`CmaEsSampler` does not support dynamic search space" and returns `{}`, which means **random sampling** ([`#L410-L422`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L410-L422)). Copying these `system_attrs` into a new round's study would restore a stale optimizer.
- **The built-in warm start is `source_trials=`** (experimental). It fits the initial mean, σ and covariance from source trials using `cmaes.get_warm_start_mgd` ([`#L516-L546`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L516-L546), docstring [`#L194-L205`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L194-L205)). A source trial counts as compatible when its **param-name set exactly equals** the current space. Bounds are not compared ([`_is_compatible_search_space` L677-L681](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L677-L681)). If no source trial is compatible, it raises `ValueError("No compatible source_trials")`. **[probe]** The same names with narrowed bounds work, but adding one lever raises the error. `source_trials` cannot be combined with `x0`, `sigma0` or `use_separable_cma` ([`#L345-L354`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L345-L354)).
- CMA-ES does not support multiple objectives ([`#L396`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L396)).

### Summary

| | Joint model is built over | Params outside it go to | Conditional params | Built-in warm-start hook |
|---|---|---|---|---|
| TPE (`multivariate`, default) | Params with identical distributions in all finished trials | 1-D TPE | Partial, via 1-D fallback | none (use `add_trials`) |
| TPE `group=True` | Name-based co-occurrence groups, first-seen distribution | 1-D TPE | **Yes** (designed for it) | none |
| GP | Params with identical distributions (COMPLETE, RUNNING, WAITING) | `RandomSampler` | Partial | none (use `add_trials`) |
| CMA-ES | Float/int params with identical distributions; dimension frozen in the pickled optimizer | `RandomSampler` | Partial | `source_trials` (name-matched) |

## 3. `constraints_func` and constraints per sampler

- **v5.0.0 deprecates `constraints_func`** on TPE, GP and NSGA-II, with removal scheduled for v7.0.0. The replacement is `Trial.set_constraint(key, value)` ([TPE docstring `#L216-L232`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L216-L232), [`#L404-L408`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L404-L408); [GP `#L208-L224`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L208-L224)). Constraints are stored per name as `system_attrs["constraints:<key>"]`, alongside the old list format ([`study/_constrained_optimization.py#L14-L52`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/study/_constrained_optimization.py#L14-L52); [`trial/_trial.py#L736-L785`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_trial.py#L736-L785)). `create_trial(constraints={...})` attaches constraints to warm-start trials. A value `> 0` means the constraint is violated. A setting of the same key a second time is ignored, with a warning.
- The deprecated `constraints_func` still runs in `after_trial`, for COMPLETE and PRUNED trials only ([`samplers/_base.py#L240-L266`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_base.py#L240-L266)).
- **TPE:** a trial is infeasible when its summed positive violations are `> 0`. Infeasible trials are ranked after the complete and pruned ones in the good/bad split ([`sampler.py#L740-L776`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L740-L776), [`#L857-L866`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L857-L866)). Trials with no constraints count as feasible. **[probe]** A mix of none, `mem` and `lat` across trials works.
- **GP:** the constrained path turns on when *any* completed trial has constraints ([`_is_constrained_optimization`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/study/_constrained_optimization.py#L17-L20)). It then fits one GP per constraint (constrained LogEI), and **every completed trial must have the same number of constraints**, or it raises `ValueError` ([`samplers/_gp/sampler.py#L466`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L466), [`#L509-L511`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L509-L511), [`#L626-L638`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L626-L638)). **[probe]** One trial without constraints among two with `mem` raised `ValueError: The number of constraints must be the same for all trials.` Note that the check compares counts, not names.
- **CMA-ES:** it doesn't use constraints and warns "Constraints have been set, but CmaEsSampler will not use them" ([`_cmaes.py#L662-L674`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_cmaes.py#L662-L674)).
- **Docs:** constraints are **soft only**, and Optuna "DOES NOT support" hard constraints. The samplers that support them are TPE, NSGA-II, NSGA-III, GP, BoTorch and AutoSampler ([`docs/source/faq.rst#L452-L512`](https://github.com/optuna/optuna/blob/v5.0.0/docs/source/faq.rst#L452-L512), [rendered](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-optimize-a-model-with-some-constraints)).

## 4. Crash-safe resume with RDB/SQLite storage

- **Durability:** every RDB storage operation runs in a scoped session that commits on exit ([`storages/_rdb/storage.py#L76-L80`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_rdb/storage.py#L76-L80)). Params, values, attributes and states are persisted as they are set. `create_study(study_name=..., storage="sqlite:///x.db", load_if_exists=True)` resumes the study ([`tutorial/20_recipes/001_rdb.py#L50-L55`](https://github.com/optuna/optuna/blob/v5.0.0/tutorial/20_recipes/001_rdb.py#L50-L55)).
- **Sampler state is not stored.** The tutorial says so and recommends pickling the sampler for seeded reproducibility ([`001_rdb.py#L58-L72`](https://github.com/optuna/optuna/blob/v5.0.0/tutorial/20_recipes/001_rdb.py#L58-L72)). TPE and GP rebuild everything from trials, so a resume loses only RNG continuity. CMA-ES resumes its optimizer from `system_attrs`. TPE (with `constant_liar`) and GP also store in-flight relative params in `system_attrs` ([`samplers/_gp/sampler.py#L403-L412`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L403-L412)).
- **Killed trials:** without a heartbeat, a crashed trial stays `RUNNING` forever. TPE's constant liar then keeps penalising its region ([`_tpe/sampler.py#L204-L215`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L204-L215)), and GP includes it as a running trial. `RDBStorage(heartbeat_interval=..., grace_period=..., heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(max_retry=N))` marks stale trials `FAIL` just before each new `ask` inside `optimize()` ([`study/_optimize.py#L186-L197`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/study/_optimize.py#L186-L197), [`storages/_heartbeat.py#L156-L186`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_heartbeat.py#L156-L186)). The retry callback re-adds the trial as `WAITING` with the same params and a `retry_history` ([`storages/_callbacks.py#L40-L96`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_callbacks.py#L40-L96)). There are three caveats:
  - The heartbeat is **experimental** and **does not work with ask/tell**. With ask/tell you must `tell(t, state=FAIL)` stale `RUNNING` trials yourself ([`storages/_rdb/storage.py#L143-L158`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_rdb/storage.py#L143-L158); [`faq.rst#L581-L625`](https://github.com/optuna/optuna/blob/v5.0.0/docs/source/faq.rst#L581-L625), [rendered](https://optuna.readthedocs.io/en/stable/faq.html#can-i-monitor-trials-and-make-them-failed-automatically-when-they-are-killed-unexpectedly)).
  - `failed_trial_callback` and `RetryFailedTrialCallback` are deprecated aliases since v4.9 ([`storage.py#L167-L169`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_rdb/storage.py#L167-L169), [`_callbacks.py#L135-L139`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_callbacks.py#L135-L139)).
- **SQLite limits:** the docs "would never recommend SQLite3 for parallel optimization", for three reasons ([`faq.rst#L552-L566`](https://github.com/optuna/optuna/blob/v5.0.0/docs/source/faq.rst#L552-L566), [`storage.py#L188-L189`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/storages/_rdb/storage.py#L188-L189)):
  - Enqueued trials use `SELECT ... FOR UPDATE`, which SQLite doesn't support.
  - Concurrent writers get "database is locked" errors. You can raise `engine_kwargs={"connect_args": {"timeout": 20.0}}`.
  - SQLite doesn't work over NFS.

  If more than one process ever writes, use `JournalStorage(JournalFileBackend(...))`. This is fine for our single-machine v1 as long as one runner process writes at a time.
- **Moving whole studies:** `optuna.copy_study(from_study_name=..., from_storage=..., to_storage=...)` copies a study verbatim, `system_attrs` included ([`study/study.py#L1434`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/study/study.py#L1434)). That makes it the wrong tool for re-expressing trials between rounds.

## Implications for our design (input for the decision tickets, not decisions)

- **Warm-start policy.** Use one Optuna study per round, named e.g. `<run>/round-<k>`, in the run's SQLite file.
  - Build round k+1's warm-start set from the log by rewriting each prior COMPLETE trial into the new lever set:
    - **Kept lever, same range:** copy it as is.
    - **Narrowed lever:** drop the trial if its value falls outside the new range.
    - **Widened lever:** copy it, declaring the new distribution.
    - **Added lever:** impute the value the code had before the lever existed. Mark it in `user_attrs`.
    - **Removed lever:** delete the key. The trial is only valid if the value the code now hard-codes equals the trial's value, otherwise drop it.
    - **Changed categorical choices:** drop the trial or map it, because only a fresh study permits new choices.
  - Copy only `params`, `distributions`, `value`, `constraints` and lineage `user_attrs`. Never copy `system_attrs`.
  - Decide explicitly whether warm-start trials should count toward `n_startup_trials`.
- **Runner harness API.**
  - Record constraints with `trial.set_constraint(name, v)`, never `constraints_func`.
  - Keep the constraint set constant within a research run, or GP breaks.
  - Validate every `enqueue_trial` seed against the current distributions before enqueueing.
  - Prefer `study.optimize` over ask/tell if we want the heartbeat. Otherwise the harness must fail stale `RUNNING` trials at startup.
  - Run tests on SQLite, not InMemory storage.
- **Sampler choice.** Only the fresh-study plus re-expression route gives TPE, GP or CMA-ES a joint model over new and old levers. TPE `group=True` separates new levers from old ones, which works against the goal of detecting interactions.

## Probe

The probe ran optuna 5.0.0 on Python 3.9 with SQLite storage. The condensed script below settles the questions that are faster to run than to read:

```python
import optuna
from optuna.distributions import FloatDistribution as F, CategoricalDistribution as C
from optuna.trial import create_trial
from optuna.search_space import intersection_search_space
from optuna.search_space.group_decomposed import _GroupDecomposedSearchSpace
ct = lambda p, d, v=1.0, **kw: create_trial(params=p, distributions=d, value=v, **kw)

s = optuna.create_study(storage="sqlite:///probe.db")
s.add_trial(ct({"x": 5.0, "c": "a"}, {"x": F(0, 10), "c": C(["a", "b"])}))
s.add_trial(ct({"x": 2.0}, {"x": F(1, 3)}))                 # ok: bounds may change
s.add_trial(ct({"c": "a"}, {"c": C(["a", "b", "z"])}))      # ValueError on SQLite (ok on InMemory)

print(intersection_search_space([
    ct({"x": 1.0, "y": 1.0}, {"x": F(0, 10), "y": F(0, 1)}),
    ct({"x": 1.0, "y": 1.0, "z": .5}, {"x": F(0, 5), "y": F(0, 1), "z": F(0, 1)})]))  # {'y': ...}

g = optuna.create_study()
g.add_trial(ct({"a": .1, "b": .1}, {"a": F(0, 1), "b": F(0, 1)}))
g.add_trial(ct({"a": .1, "b": .1, "c": .1}, {"a": F(0, 1), "b": F(0, 1), "c": F(0, 1)}))
print([sorted(sp) for sp in _GroupDecomposedSearchSpace(True).calculate(g).search_spaces])  # [['a','b'], ['c']]
```

Observed output, abridged:

```
[ok]   rdb: same study, float bounds narrowed / widened
[err]  rdb: same study, categorical choices changed: ValueError: CategoricalDistribution does not support dynamic value space.
[err]  rdb: same study, int -> float: ValueError: Cannot set different distribution kind to the same parameter name.
[err]  rdb: same study, log flag changed: ValueError: Cannot set different log configuration to the same parameter name.
[ok]   mem: (all of the above succeed on InMemoryStorage via add_trial)
[err]  create_trial value outside its own distribution: ValueError: The value 20.0 of parameter 'x' isn't contained in ...
[ok]   enqueue out-of-range + extra key + missing key
       warn: Fixed parameter x with value 20.0 is out of range for distribution FloatDistribution(high=10.0, ...)
       trial 0 {'x': 20.0, 'new': 0.085...} fixed_params= {'x': 20.0, 'gone': 1}
intersection: ['y']
group subspaces: [['a', 'b'], ['c']]
[ok]   TPE narrowed to [0,5] with priors up to 19 -> [4.85, 4.99, 4.99, 5.0, 5.0]
[ok]   TPE relative space after narrowing -> []
[err]  GP with mixed constraint counts: ValueError: The number of constraints must be the same for all trials.
[ok]   TPE with mixed/renamed constraints
[ok]   CMA source_trials same names, narrowed bounds
[err]  CMA source_trials with an added lever: ValueError: No compatible source_trials
```
