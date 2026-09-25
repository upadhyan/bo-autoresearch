# BO Autoresearch

A research loop where an orchestrating agent proposes falsifiable hypotheses and exposes them as search levers. Bayesian optimization, not agent tokens, does the searching.

## Language

### Hypotheses

**Hypothesis**:
A falsifiable claim about what limits the objective. It has a concise rationale, one or more levers, and reject conditions that are fixed before it enters the loop.
_Avoid_: Idea, experiment, feature

**Lever**:
A searchable parameter exposed by a hypothesis. It is a continuous or integer range or a categorical. A boolean is allowed only for a mechanism that cannot be graded.
_Avoid_: Knob, flag, toggle, switch

**Reject condition**:
A checkable rule, pre-registered before the hypothesis enters the loop, whose satisfaction falsifies the hypothesis.
_Avoid_: Kill criterion, stopping rule

**Directive**:
A user-stated limit on what the research may try, with a severity of **prohibited** (never tried), **discouraged** (allowed only with a stated reason, at lower priority) or **preferred** (a soft pull, e.g. downstream compatibility).
_Avoid_: Constraint (reserved for objective feasibility), guardrail, scope rule

### Search

**Constraint**:
A feasibility condition on a trial's outcome, such as runtime < X. It is evaluated after the trial runs.
_Avoid_: Using it for limits on what may be tried (see Directive)

**Objective**:
The single scalar the research is optimizing, produced by one trial.
_Avoid_: Score, target, metric (when meaning the optimized value)

**Trial**:
One evaluation of the objective at a single assignment of lever values. It always leaves an artifact.
_Avoid_: Run, experiment

**Round**:
One Optuna study run under a fixed hypothesis set. Changing the hypothesis set ends the round.
_Avoid_: Phase, generation, iteration

**Warm start**:
Seeding a new round's study with prior trials mapped into the new round's search space. For example, a newly added lever takes its "off" value in the old trials.
_Avoid_: Transfer, resume
