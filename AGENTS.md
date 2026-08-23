# FSRL Research Instructions

## Scientific north star

The project-level question is:

> How does a shared relational learning system transform sparse and partially
> encoded evidence into a stable, coherent, yet individualized global
> relational structure?

Behavioral reproduction is a competence gate, not the scientific endpoint.
Build toward an explanatory chain linking external evidence, subject-specific
effective evidence, meta-learned recurrent assembly, episode-local fast-weight
state, relational representation, and individualized behavior.

## Build claims from the full evidence

- Organize the working theory from all independently supported positive
  results. For each result, retain its protocol, estimand, controls, seed scope,
  provenance, and exact claim boundary.
- Do not erase a positive result merely because another link in the proposed
  mechanism failed. State which link remains supported and which link must be
  replaced.
- Do not respond to a valid negative result only by asking what smaller claim
  can still survive. Ask what assumption or missing mechanism prevents the
  supported positive phenomena from arising together, and design the next
  discriminating test around that diagnosis.
- Treat convergent negative fingerprints as an opportunity for a unified
  mechanism. Prefer one hypothesis that explains several discrepancies over
  unrelated patches that fit each metric separately.

## Use negative results to revise the route, not abandon the question

- First verify competence, implementation, measurement, and protocol fidelity.
  A failed competence gate is non-interpretable; a valid result below a frozen
  gate is mechanistic evidence.
- Keep the project-level scientific question separate from the claim supported
  by one checkpoint or experiment. A failed candidate should not defensively
  shrink the scientific question. However, the experiment-level conclusion
  must change when evidence falsifies a causal link; never retain an unsupported
  statement by relabeling or rhetorical qualification.
- Preserve negative results and use them to reject assumptions, redirect the
  mechanism, or identify a missing component. Do not explain them away, filter
  seeds, move thresholds, refit nuisance parameters, or select checkpoints to
  manufacture a pass.
- The next desired positive result must be mechanistically specified. It should
  show how the already-supported phenomena can be produced under the revised
  theory, not merely make a failed scalar metric numerically acceptable.

## Required research loop

For every substantive stage, record the following in order:

1. Scientific question and current theory.
2. Registered protocol, estimands, controls, competence gates, and stop rules.
3. Experiment or read-only analysis.
4. Positive results, negative results, and uncertainty.
5. Which causal links are supported, rejected, or still unidentified.
6. Revised theory explaining how the positive results could arise while
   respecting the negative constraints.
7. The next decisive test, including what each possible outcome would imply.

Keep exploratory and confirmatory work separate. Prefer analyses of existing
artifacts before retraining. Validate new mechanisms or workflow changes with
one to three seeds, then freeze the protocol before a formal multi-seed run.
Never tune on formal confirmation seeds. If diagnostics expose a missing
mechanism, preserve the frozen candidate as a baseline and register a new
candidate rather than editing the old confirmation contract after seeing its
results.

## Current FSRL direction

The held-out graph transfer, fast-weight necessity, stable subject-specific
evidence bottleneck, and antisymmetric subjective-rank geometry form the current
positive evidence backbone. The immediate unresolved question is how
`D_s -> P_T^(s) -> pi_s` performs global assembly.

The completed registered slope/Hodge diagnostic rules out the stable-omission
evidence model as the source of the excessive distance slope. It instead shows
that both pilot networks transform evidence into an almost purely additive
logit potential, while the human choice field retains a stronger learned-pair
residual. Treat this as a policy-level mixed-code signal, not yet proof of a
local memory store.

The completed registered prefix/LOO diagnostic further shows that retained
support trials have immediate causal effects on disjoint pairs, and that
leaving out one retained relation differentially changes third-party item
potentials. The final neural potential aligns more closely with the exact
posterior expected-rank/Hodge potential than with a MAP order. Episode-specific
content requires the fast-weight pathway and is already present in hidden
dynamics at query onset; the output direction selects a modestly more additive
component. Do not describe this as independent correctness-propagating updates:
remote absolute effects are robust, but their correctness-aligned contribution
is unresolved or negative.

The completed registered support-write localization assigns most
relation-specific write direction and downstream policy change to eligibility,
not to the matched DA difference. The modeled DA magnitude retains a weaker
state- and innovation-sensitive gain signal after an explicitly exploratory
exposure adjustment, but it is not an exact Bayesian surprise variable. Alpha
causally amplifies norm-matched writes in the frozen query policy, yet it does
not reproducibly beat scalar and shuffled controls in posterior-update
direction. Keep amplification and directional mapping as distinct claims.

Trial updates positively align with exact posterior expected-rank innovations,
especially at the first relation exposure, but do not outperform a fixed
true-order direction across the whole episode. Late attenuation occurs in
write generation without literal clipping or replicated functional saturation.
The no-prior-relation history control supports relation-specific assimilation
by the fourth exposure, not a monotonic redundancy effect from the second
exposure onward.

Before scaling formal seeds or changing the architecture, prioritize a
registered within-exposure factor-swap diagnostic. Exchange DA scalars and
eligibility matrices among matched trials while holding the pre-query state and
alpha expression map fixed; test whether eligibility transfers update direction
and DA transfers magnitude. Contrast natural and no-prior-relation histories to
localize late assimilation to DA, eligibility, or their interaction. Use more
than one preregistered alpha permutation or a local recurrent-sensitivity
analysis before interpreting alpha's functional amplification as systematic.

The target is to explain how bidirectional global redistribution accumulates
into a correctly aligned potential and how human behavior retains additional
local or conjunctive structure. Do not tune the slope or add a memory module
before the remaining gain-direction-history interaction is localized.

## Repository workflow

- Work on `dev`; reserve `main` for stable releases.
- Use GPU for neural training and evaluation when it materially helps. CPU is
  appropriate for lightweight tests, data checks, bootstrap summaries, and
  exact enumeration when it is the more efficient implementation.
- At the end of an experiment, preserve positive and negative outputs, run the
  scoped validation, commit the intended files, verify that the worktree is
  clean, and push to `origin/dev`.
- Do not include unrelated or incomplete changes merely to make the worktree
  clean.
