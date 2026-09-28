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

**Baseline**:
The value of a lever at which its code does exactly what the code did before the lever existed. It always lies inside the lever's range.
_Avoid_: Off, default, zero

**Reject condition**:
A checkable rule, pre-registered before the hypothesis enters the loop, whose satisfaction falsifies the hypothesis.
_Avoid_: Kill criterion, stopping rule

**Minimum meaningful effect**:
The smallest change in the objective, in the objective's own units, that the user counts as worth having. It is set once by the user and is the threshold for every verdict.
_Avoid_: Tolerance, epsilon, τ

**Predicted direction**:
The way a hypothesis claims each of its graded levers moves the objective. It is pre-registered, but contradicting it flags the verdict rather than rejecting the hypothesis.
_Avoid_: Expected sign, prior

**Verdict record**:
The harness's per-round statement of a hypothesis's verdict and the evidence behind it. Any interpretation of a verdict must cite it.
_Avoid_: Report, judgement

**Retained**:
The positive verdict on a hypothesis: its levers demonstrably matter and demonstrably improve the objective beyond the minimum meaningful effect, interactions included. It is re-judged every round, so it can later become rejected.
_Avoid_: Accepted, confirmed, proven

**Inconclusive**:
The terminal verdict on a hypothesis that was tested but never gathered enough evidence to be retained or rejected.
_Avoid_: Failed, neutral

**Parked**:
Withdrawn from the loop without a verdict, with a stated reason. It is never evidence against the hypothesis, and a parked hypothesis can return unchanged.
_Avoid_: Shelved, dropped, rejected

**Pruned**:
Vetoed by a prohibited directive before registration, so it is never tested.
_Avoid_: Rejected (which requires a test)

**Revival**:
A rejected hypothesis re-entering the loop as a new version, with fresh reject conditions, because a newly added hypothesis might interact with it.
_Avoid_: Retry, resurrection

**Narrowing**:
Shrinking an active hypothesis's lever ranges on evidence. The hypothesis and its reject conditions stay the same.
_Avoid_: Pruning, refinement

**Fidelity-sensitive**:
Declared at registration for a hypothesis whose mechanism may only pay off at higher fidelity. It can never be rejected at a proxy fidelity.
_Avoid_: Slow-burn, long-horizon

**Rival**:
A hypothesis offering a competing explanation to another, through separate levers. Rivals are tested together, and a verdict on one says nothing about whether it explains the other.
_Avoid_: Conflict (too broad), alternative

**Masking**:
A declared exclusion between two hypotheses with different lever shapes: when the masking hypothesis's mechanism is on, the masked hypothesis's lever runs at its baseline value.
_Avoid_: Mutual exclusion (which, for same-shape levers, is a merge into one categorical lever)

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

**Replicate**:
A trial that repeats an earlier trial's lever assignment with a new seed. It stays a trial of its own and is never averaged in.
_Avoid_: Rerun, repeat, duplicate

**Fidelity**:
The cost level at which a trial evaluates the objective, e.g. training length or data fraction. All trials in a round share one fidelity.
_Avoid_: Budget (reserved for trial counts), proxy (as a noun)

**Reference fidelity**:
The fidelity whose objective the user actually cares about. A proxy fidelity is valid only once checked against it.
_Avoid_: Full run, real run

**Round**:
One Optuna study run under a fixed hypothesis set and fidelity. Changing either ends the round.
_Avoid_: Phase, generation, iteration

**Warm start**:
Seeding a new round's study with the eligible trials from the experiment log, rewritten into the new round's search space. For example, a newly added lever takes its baseline value in the old trials.
_Avoid_: Transfer, resume

**Experiment log**:
The append-only record of every trial and event in a research run. It is the single source of truth, and each round's study is derived from it.
_Avoid_: Study (an Optuna study is disposable), database, history

**Epoch**:
A stretch of the experiment log over which the objective is the same function. A code change that alters the objective starts a new epoch, and trials from earlier epochs become telemetry.
_Avoid_: Version, era, generation
