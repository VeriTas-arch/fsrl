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

The completed support-write and causal factor-swap diagnostics establish a
functional division of labor. Matched `delta E` transfers a donor relation's
natural neural-potential direction almost exactly when recipient DA, baseline,
alpha, and effective-write norm are held fixed. High versus low natural DA
factors change downstream policy magnitude while leaving direction almost
unchanged. Treat eligibility as evidence-specific content/direction and DA as
state-dependent gain; do not identify modeled DA with exact Bayesian surprise.

Actual alpha places retained writes in systematically high-gain recurrent
directions: it beats all 32 registered norm-matched permutation nulls in both
pilot seeds. This supports a meta-learned sensitivity map, not a unique
posterior-direction map. Keep gain placement and direction alignment separate.

Do not describe the network as performing sequential approximate Bayesian
updates. Trial updates align most strongly with exact innovation on first
exposure, then diverge while the terminal state remains posterior-like. The
working algorithm is a meta-learned state-dependent iterative relaxation:
`E_t = E_theta(e_t, P_(t-1))`, `d_t = g_theta(e_t, P_(t-1))`, and accumulated
interacting writes converge toward a distributional expected-rank potential.

At exposure 4, no-prior-relation history increases eligibility-generated write
norm, while DA is suppressive or unresolved. The completed registered 2-by-2
history state-by-factor closure shows that accumulated `P` has a reproducible
positive effect on local recurrent expression and a smaller positive
history-matched interaction in both pilots. The total factor-generation main
effect remains unresolved. Therefore do not attribute assimilation solely to
eligibility, DA, or a scalar learning rate: history primarily changes the
recurrent sensitivity landscape in which a nearly direction-preserving write
is expressed, with an additional matched nonlinear advantage.

Formal seeds 2001--2010 are complete under both frozen v1 contracts, with no
filtering. All competence and integrity gates pass. Six of seven primary
mechanism links confirm across network seeds: global remote reassembly,
eligibility direction transfer, alpha sensitivity placement, history-dependent
expression and interaction, and terminal expected-rank-over-MAP projection.
Within the unresolved DA link, both magnitude contrasts confirm, but direction
preservation fails its prospectively fixed threshold. The complete chain is
therefore not confirmed.
Never edit `confirmation_v1`, `mechanism_confirmation_v1`, or their result
artifacts after observing these outcomes; revisions require a separately frozen
v2 contract and new seeds.

Treat the DA-direction failure as implementation heterogeneity, not permission
to filter seed 2009 or erase the six positive links. Nine networks use the
development-like eligibility-direction/DA-gain factorization; one competent
network preserves the invariant global mechanism while isolated eligibility
transfer and DA direction preservation fail. The next implementation-level
test should distinguish a co-adapted complete-write code from content that is
distributed into baseline-dependent recurrent expression.

The formal behavioral conjunction fails in all ten networks. The common
negative fingerprint is an excessive symbolic-distance slope together with
lower learned and overall accuracy, while nonlearned accuracy, stable coherent
errors, causal necessity, geometry, and global reassembly transport. Build the
next candidate from those positives: preserve the confirmed fast-weight global
expected-rank channel and test a separate episode-local learned-pair residual
channel. Require selective causal ablations of the two channels; do not tune a
temperature, choose checkpoints, or add an unconstrained memory module merely
to move the failed scalars.

Start v2 only with one to three development seeds. Freeze estimands, competence
gates, and outcome-contingent interpretations before any new formal population.
The current evidence and next decisive test are recorded in
`docs/formal_confirmation_v1.md`.

## Repository workflow

- Work on `dev`; reserve `main` for stable releases.
- Use GPU for neural training and evaluation when it materially helps. CPU is
  appropriate for lightweight tests, data checks, bootstrap summaries, and
  exact enumeration when it is the more efficient implementation.
- Run formal confirmation and mechanism commands only through
  `python -m fsrl.formal_runtime`. This entry point requires a visible GPU and
  bounds PyTorch intra-op and inter-op work to one CPU thread; do not also
  change NumPy/BLAS thread settings, bypass the entry point, or mix bounded and
  unbounded seed artifacts in one formal aggregate.
- Formal meta-training uses one contiguous CPU-to-GPU input transfer per trial
  and `torch.compile(net, fullgraph=True)` with the default mode. Treat the
  compiler configuration and implementation source hashes as part of the
  execution lock; validate any further optimization on one to three development
  seeds before changing it.
- At the end of an experiment, preserve positive and negative outputs, run the
  scoped validation, commit the intended files, verify that the worktree is
  clean, and push to `origin/dev`.
- Do not include unrelated or incomplete changes merely to make the worktree
  clean.
