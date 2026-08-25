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

## Task-faithful information boundary

- Be strict at the experimental observation and behavioral-response interfaces,
  while allowing explicitly hypothesized abstraction inside the cognitive or
  neural computation. For the Liu task, the information-preserving support
  representation is item identity plus signed relative bar difference; random
  absolute bar height is a nuisance, but displayed magnitude is not.
- If a model receives information unavailable to participants, such as true
  rank, query labels, test feedback, or a neural-derived target, the comparison
  is invalid. If a model discards available information, such as replacing
  signed metric evidence by its sign, treat that operation as a cognitive
  bottleneck hypothesis requiring independent evidence rather than as a
  description of the task.
- Keep observation, cognitive encoding, relational inference, and test readout
  as separate contracts. A global-order latent state is an admissible
  result-supported abstraction; ordinalization, compression, omission, or
  noisy retention of the displayed metric evidence are competing encoding
  theories and must be prospectively distinguished.
- Do not use learning-stage choice or reward noise in the Liu paradigm:
  participants passively observed four presentations of each support relation,
  made no learning response, and received no test feedback.

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
expected-rank channel and search for the mechanism that preserves direct local
experience. Require selective causal ablations; do not tune a temperature,
choose checkpoints, or add an unconstrained memory module merely to move the
failed scalars.

The post-formal two-seed hidden-residual audit has rejected the simplest
readout-only route. Retained relations cause a small direct-enriched Hodge
residual in response hidden states, but it does not contain one cross-subject,
cross-relation correctness direction that is suppressed by `W_out`; the current
output direction reads the direct residual at least as well. Do not describe
this as an existing dual channel or fit a second response-state readout and call
it a mechanism. Next localize relation-conditioned traces in the generated
write and terminal fast-weight/support state, then test whether query dynamics
route or discard them. Only consider a new persistent local mechanism if these
earlier-state audits also fail.

The completed relation-trace localization returns a replicated non-monotonic
pattern. Generated cumulative effective writes and terminal relation-LOO
effective fast-weight matrices do not contain the registered shared
same-relation, held-out-subject prototype geometry. Nevertheless, the first
fast-weight-sensitive query transition contains replicated relation identity in
both its full direct hidden influence and its direct Hodge residual. Raw-state
controls also fail, step 0 and stable-omitted traces are exactly zero, and the
post-response trace persists. Treat this as a mixed operator-code result, not
as proof that persistent local information is absent and not as an existing
correctness channel. `H>A` again has the strongest response identity while the
prior audit shows that its residual is correctness-opposed.

The next local-fidelity test should freeze the full state-by-query operator
factorial `A_(q,e) = (alpha elementwise-multiplied by delta P_q) h_e^0` on
development seeds 1901 and 1902. Separate state-relation identity, query-basis
identity, and their matched interaction, and require exact step-1
preactivation/hidden reconstruction plus stable-omitted controls. If state
identity or congruent binding is present, search for a fidelity transformation;
if only query identity is present, a new local storage mechanism becomes more
plausible; if operator identity is lost through recurrent expression, localize
the routing failure. Do not train v2 or fit a new readout before this test.

That frozen state-by-query factorial is now complete. In both pilots,
preactivation action `A_(q,e)` has strong state-relation identity when query
basis is fixed, every relation preferentially binds its exactly matching query
over both shared-endpoint and disjoint nulls, and the nonlinear hidden effect
preserves the identity. Cross-query state identity fails below chance, and the
step-0 query key alone does not pass the held-out-subject identity rule. The
local state is therefore a basis-dependent, query-addressable synaptic
operator, not a shared static embedding or a query-invariant relation vector.
The operating point adds a smaller matched gain advantage but does not generate
identity de novo. Do not claim rank-one key-value memory, episodic retrieval,
or correctness from this result.

Storage and access are now supported under the frozen functional estimands.
The completed operator-output semantics audit holds `W_out` fixed and compares
direct Hodge-residual correctness across `W_out^T A`, `W_out^T J_b A`, and the
exact `W_out^T [tanh(b+A)-tanh(b)]` over all 28 query edges, with remote and
stable-omission controls. Aggregate correctness is positive at all three
stages in both pilots. The local Jacobian strongly improves raw and normalized
correctness over `A`, but exact finite-amplitude tanh strongly degrades both.
The prospectively registered `H>A` relation is strongly correct at `A` and
`J_b A` and strongly incorrect at exact `H` in both seeds. The missing link is
therefore relation-conditioned nonlinear expression, not operator value
generation or the first-order operating point. A global scalar local gain is
insufficient because it would amplify the wrong-sign `H>A` response. Do not
fit a new readout or add a memory module.

The final frozen amplitude-path gate is complete on the two pilots. `H>A` is
the only relation whose mean direct correctness crosses zero in either seed,
at `lambda=0.55--0.60` and `0.60--0.65`; the robust transition intervals
overlap. Most retained subjects cross (`49/55` and `45/55`), but their brackets
are broad. `F>A` also crosses in a sizeable minority without a mean crossing,
while the other relations are much less affected. The analytic quadratic
curvature coefficient is correctness-opposed overall and much more negative
for `H>A` than for the other relations. Treat this as finite-amplitude
relation- and subject-conditioned curvature, not generic saturation.

Read-only local-fidelity localization on seeds 1901 and 1902 is now complete.
Do not add adaptive amplitude points or more post hoc diagnostics on those
seeds. Global clipping is not the selected mechanism because the other seven
relation means remain correct at natural amplitude and subject thresholds vary.

The first registered v2 gate-only sufficiency test is complete on the new
development seed 2101. It froze a competent v1 backbone and trained only
`gamma=1/(1+beta*r)` with unsigned hidden-space
`r=||K2||/(||J_bu||+epsilon)` on the generic held-out-graph distribution.
`beta` remains nonzero and all frozen tensors and integrity controls pass. The
conditioned gate preserves fast-weight necessity, nonlearned inference,
remote/third-party reassembly, query binding, and terminal expected-rank over
MAP. Preserve these positive results.

The candidate nevertheless fails local rescue and state-conditioning
specificity. H>A direct correctness moves toward zero but remains robustly
negative. Aggregate retained direct correctness decreases relative to v1, the
other seven relations decline substantially, and matched-global and shuffled
controls exceed the conditioned aggregate. Online gamma is negatively, not
positively, associated with offline crossing midpoint; H>A and F>A receive
among the largest gamma values despite the most crossings. The symbolic-
distance slope does not improve. Reject unsigned curvature magnitude as the
sufficient online control variable; do not describe this as evidence against
the preserved operator/global backbone or against all state-conditioned
control.

After the unsigned-risk failure, the policy-relative signed-curvature test was
prospectively registered on the same frozen seed-2101 backbone. It used
`j=W_margin^T J_bu`, `k=W_margin^T K2`, and a relative-scale safeguarded
`r_opp=relu(-j*k)/d`, followed by the same bounded monotone gamma. The exact
equation, safeguards, generic-only adaptation, matched-global, shuffled, and
sign-reversed controls, and outcome rules were frozen before implementation;
no true labels, relation identity, H>A, LOO state, or offline crossings entered
the gate.

The completed policy-relative signed-curvature v2.1 test is also a valid
negative. Its source, artifact, competence, and global-mechanism gates pass,
but opposition-minus-v1 local rescue, H>A rescue, and superiority to both
matched-global and shuffled controls fail. The gate assigns almost no
attenuation to the critical relations: only 2.50 percent of retained H>A and
2.68 percent of retained F>A oriented direct-query states have `j*k<0`, with
mean gamma 0.99944 and 0.99949. The diagnostic crossing correlation is weak
and non-significant. The sign-reversed same-beta control is substantially worse
in aggregate and on the other seven relations while moving H>A only
nonspecifically toward zero. Treat sign as informative but the scalar local
quadratic-opposition statistic as insufficient.

Close the scalar amplitude-gating family. Do not run seeds 2102/2103, retune
beta or `tau`, add caps or more elaborate risk statistics, change `tanh`, or
begin end-to-end co-adaptation to rescue it. The subsequent separately frozen
candidate reused the seed-2101 backbone and explicitly preserved the `J_b u`
first-order branch through a policy residual, with global-mechanism gates and
matched-magnitude and shuffled controls. The v2.1 result and amplitude-family
closure are in `docs/policy_opposition_gate_pilot_v2_1.md`.

The completed first-order policy-residual v2.2 test is a valid mixed negative.
Generic-only adaptation selects eta 0.09398. The natural residual causes a
robust, state-query-specific movement toward correctness for H>A (+0.03596)
and F>A (+0.00779); matched-magnitude and shuffled controls do not reproduce
the H>A movement. Preserve this as positive evidence that the existing
first-order residual contains causal information for the two hardest
relations. It is not sufficient: H>A remains robustly negative, aggregate
direct-correctness improvement is unresolved, both aggregate control-
specificity contrasts fail, and the six already-correct relations decline.
All competence, fast-weight necessity, nonlearned inference, remote/third-
party reassembly, query-binding, and expected-rank-over-MAP gates remain
supported.

Close the family in which the existing first-order signal is rescued by a
low-capacity response-expression correction. Do not run 2102/2103, select eta
on Liu, add relation-specific residual scalars, change `tanh`, or begin end-to-
end co-adaptation to rescue it. The next justified family is a separately
persistent local component that selectively preserves direct experience
alongside the confirmed global expected-rank channel. Retain the H>A/F>A
first-order residual fingerprint as a constraint and possible routing
diagnostic, not as proof that the residual is a sufficient local channel. The
v2.2 result and stop/go decision are in
`docs/policy_residual_pilot_v2_2.md`.

The first separately persistent local-component pilot is now complete on the
same frozen seed-2101 backbone. Its fixed shared address is a normalized
antisymmetric tensor product of the two normal item cues; the normal encoded
signed support value is accumulated in an episode-local trace, and only one
positive local gain is adapted on generic held-out graphs. No item/relation ID,
H>A/F>A flag, label, residual target, posterior, or Liu value enters the
module or adaptation.

Preserve the strong positive causal decomposition. Dual-intact increases
retained direct correctness by 0.30039 and makes H>A robustly positive; a
query-address derangement does not reproduce it. Local influence is strongly
direct-enriched. With `P_T` removed, learned-pair accuracy remains above chance
while nonlearned accuracy is near chance and remote reassembly collapses. With
the local trace removed, the system is exactly v1 and all confirmed global
mechanism gates remain intact. This supports functionally separable global
inferential and local conjunctive persistent states on this backbone; it does
not prove unique tensor-product coding or biologically separate stores.

The original v2.3 pilot is nevertheless a registered mixed negative because the
sampled learned-accuracy rescue is only +0.00146 with a bootstrap interval
crossing zero, and the symbolic-distance slope does not improve. Nine of ten
primary flags pass, but the all-primary conjunction fails. Do not run
2102/2103 without a separately frozen replication contract, retune the gain on
Liu, relax the original behavioral rule, alter the fixed address after seeing
the result, or begin end-to-end co-adaptation. Preserve the double dissociation
rather than calling the local trace ineffective, but do not retroactively call
the discovery pilot a full behavioral rescue.

Before another trainable local candidate, freeze a read-only attribution of
the behavioral non-rescue. Partition learned errors and signed margins by
retained versus stable-omitted evidence and measure whether the frozen local
correction crosses decision/probability thresholds on retained relations. If
omitted relations dominate, revise how weak evidence can persist locally while
remaining unavailable to global assembly; if retained low-margin relations
dominate, test a shared value/expression transform while preserving the now-
supported storage and query-address mechanism. Do not tune either route before
this attribution. The v2.3 result is in
`docs/conjunctive_local_trace_pilot_v2_3.md`.

That registered read-only attribution is now complete. Stable-omitted learned
cells account for 71.36 percent of v1 exact learned error mass, with a bootstrap
lower bound of 62.81 percent, despite being only 296 of 1232 learned
orientation cells. Their local self trace is exactly zero. Among retained
cells, 827 of 936 already have v1 exact correct probability at least 0.99. The
local trace nevertheless increases retained exact probability by 0.00682 with
a positive bootstrap lower bound and removes 20.15 percent of retained exact
error mass; the existing sampled learned-accuracy contrast remains unresolved.
Treat the large direct-causal rescue and tiny sampled change as an endpoint-
sensitivity result, not a contradiction.

The self/cross decomposition further supports the local representation. Every
retained self contribution is correctly signed, with mean 0.33691. Cross-talk
is directionally negative but its absolute magnitude is 28.41 percent of self,
below the registered one-third materiality threshold. Stable-omitted cells
receive no self signal, and cross-talk alone slightly worsens them. Do not
increase the local gain as a repair.

The full `P`-off/local-intact retained exact-probability point estimate is
0.66768, but its lower bound is 0.64435 and narrowly misses the frozen 0.65
retained-sufficiency rule. Therefore do not relabel the result as a formal
dual-evidence-access PASS. Pure-local diagnostics are strong, but they cannot
replace that registered condition. The frozen attribution outcome is
`confirmation_estimand_sensitivity`; retained value-conversion limitation and
material address interference are not supported.

The exact distance-slope decomposition separates the behavioral fingerprint.
Nonlearned queries contribute 81.05 percent of the v1 slope. Both learned
contributions become slightly smaller with the local trace, but the nonlearned
contribution increases enough to cancel them. Treat direct learned fidelity
and excessive global symbolic-distance dependence as two mechanistically
distinct problems; do not require one local memory component to repair both.

Do not retroactively change v2.3, train a value transform inside its frozen
contract, alter its stable-omission rule, or move the 0.65 threshold. The
required replication contract was prospectively frozen
with the mechanism unchanged and exact retained probability, direct causal
rescue, self/query specificity, and the `P`/`L` double dissociation as four
independent confirmation links. The attribution is in
`docs/local_behavior_attribution_v2_3.md`.

That two-backbone replication is now complete on mandatory seeds 2102 and
2103. Both artifact sets were trained and generic-gain-adapted before either
Liu evaluation, then jointly hash-locked. All competence and integrity gates
pass. Each of the four primary links passes separately within both networks;
participants were bootstrapped within network and never pooled across
networks. The registered outcome is `replicated_mechanism`, not a network-
population prevalence estimate.

Retained exact probability improves by 0.01194 and 0.00889 with positive
participant-bootstrap lower bounds. Direct causal correctness improves by
0.32957 and 0.28515. Natural query addresses beat deranged addresses by
0.34062 and 0.29472, and own-write contributions are robustly correctly
signed. Under `P`-off/local-intact, retained-minus-omitted exact probability
and sampled learned-minus-nonlearned accuracy are positive while remote
reassembly collapses. Under `L`-off, both networks exactly restore v1 and all
global mechanism gates. Treat the retained-evidence `L_T` versus global-
inference `P_T` causal decomposition as the working project mechanism. Do not
claim unique tensor-product coding, biological stores, or population
prevalence.

Preserve the original seed-2101 sampled learned-accuracy failure and 0.65
historical-threshold miss. The replication seeds show small positive sampled
learned-accuracy changes, but nonlearned accuracy declines slightly and the
symbolic-distance slope remains unchanged or slightly worse. Nonlearned
queries remain the largest exact-slope source. Stable-omitted cells still hold
66.65 and 71.33 percent of v1 exact learned error mass in the two networks,
and their own local write is exactly zero. Thus the local mechanism is
replicated while full human behavior remains unsolved.

The registered zero-parameter `dual_evidence_access_v2.4` pilot is now complete
on the same frozen seed-2102 and seed-2103 backbones. Global assembly retains
`s_G=m_t z_sr`, while the unchanged local trace receives
`s_L=m_t[z_sr+(1-z_sr)p_sr]`, where `p_sr` is the existing reliability that
generated the stable global admission. No gain, address, state update, query
read, recurrent computation, activation, or output readout changes. Every
competence and integrity gate passes after a source-locked estimator-replay
repair, and all four primary links pass separately within both networks; no
participants are pooled across networks.

Preserve the positive differential-access result. Omitted exact probability
improves by 0.07432 and 0.06024, and omitted relation-LOO direct correctness by
0.23711 and 0.20515. Natural evidence routing beats a blockwise scalar-
multiset derangement, and natural query addressing beats the canonical query
derangement. Under `P`-off, omitted exact probability remains above chance and
improves over shared access, while nonlearned accuracy stays near chance and
remote influence collapses. Together with the v2.3 double dissociation, this
supports different evidence-admission rules feeding two causally distinct
computations: selective effective evidence enters `P_T` for global assembly,
whereas broader weak observed evidence enters `L_T` for direct fidelity.

Retain the qualified negatives. Newly admitted omitted writes cause a small
but reproducible retained-probability cost (-0.00182 and -0.00135), although
both seeds pass the frozen -0.005 noninferiority rule, retained own writes are
exactly unchanged, and retained direct causal effects are numerically
unchanged. The symbolic-distance slope becomes slightly larger, and
nonlearned/global policy remains its dominant exact source. Do not claim that
retained decisions are literally unaffected, that `p_sr` is the uniquely
correct human evidence model, that the two states are biological stores, or
that network-population prevalence is established.

Before a population claim, freeze the unchanged v2.4 equation and four-link
contract on new independent backbones; train and generic-adapt all mandatory
artifacts before inspecting any Liu result. Do not refit `p_sr`, `lambda_L`,
the retained noninferiority margin, or either routing control. Treat the
excessive nonlearned/global distance slope as a separate `P_T` policy question;
begin that family with read-only localization rather than modifying v2.4. The
v2.3 replication is in
`docs/conjunctive_local_trace_replication_v2_3.md`, and the differential-access
result and execution-repair audit are in
`docs/dual_evidence_access_pilot_v2_4.md`.

That fresh-backbone confirmation is now complete on seeds 2104 and 2105. Both
1000-step backbones and both 500-step generic-only gains were generated before
one joint artifact lock, and neither Liu evaluation ran until that lock was
committed and pushed. Every competence and integrity gate passes. All four
v2.4 links confirm independently in both fresh networks; participants were
never pooled across networks. The registered outcome is
`fresh_backbone_confirmation_pass`, not a network-population prevalence
estimate.

Promote `P_T/L_T` plus differential evidence admission into the working main
computational model. Omitted exact probability improves by 0.06118 and 0.05241
and omitted direct causal correctness by 0.19402 and 0.20817. Matched evidence
and query routing beat both derangements. Under `P`-off, omitted direct
fidelity persists while nonlearned inference stays near chance and remote
influence collapses; `L`-off exactly restores v1 and all global gates.

Preserve the now four-network retained trade-off. Fresh-seed retained exact
costs are -0.00114 and -0.00143, both within the frozen -0.005 noninferiority
margin, while retained own writes and direct causal effects remain numerically
unchanged. This repeats the 2102/2103 pattern and should be attributed to
cross-talk from newly admitted omitted writes, not erased as noise. The
symbolic-distance slope also remains excessive and slightly increases.

Close local-model development unless new contradictory evidence appears. The
separately frozen, read-only `P_T` slope localization is now complete on the
pure global `L`-off branches of seeds 2104 and 2105. Every source, artifact,
runtime, and exact-identity gate passes, and an independent replay is byte-
identical. Participants were bootstrapped within network and never pooled.

The excessive nonlearned slope is already in the additive policy potential.
The additive component supplies more than 100 percent of the positive margin
slope in both networks; the Hodge residual is small and significantly
negative. The normalized neural potential is strongly posterior-aligned but
has significantly *less* distance geometry than the exact-posterior expected-
rank comparator. The fixed sigmoid link is also significantly compressive,
not amplifying. Meanwhile neural exact-probability slope exceeds the posterior
comparator in both networks. Track B's frozen within-neural decomposition
therefore identified multiplication by the natural additive-potential
amplitude, rather than normalized geometry, residual corruption, or sigmoid
amplification, as the leading algebraic contributor inside the neural field.
The subsequent audit below rejects the stronger cross-comparator hypothesis
that this natural neural norm is globally over-sharp.

Do not describe normalized neural and posterior geometry as equivalent. The
subsequent separately frozen amplitude-provenance audit is now complete on the
same pure-global 2104/2105 branches. Its original protocol, prospectively
repaired decision axes, implementation lock, and two byte-identical GPU
executions all precede interpretation. Every source, artifact, reconstruction,
Hodge, posterior, denominator, and bootstrap-finiteness gate passes.

The registered outcome is `comparator_sensitive_unresolved`. The frozen
Track-B probability-slope excess remains positive in both networks, but the
stronger global-over-amplitude premise is rejected in common policy-margin
units. Mean `Y=log(a_N/a_post)` is -0.87606 and -0.84235 with wholly negative
participant-bootstrap intervals; neural additive policy-potential amplitude
is smaller, not larger, than the exact-posterior log-odds additive potential.
The mandatory additive probability-potential contrast is positive only as a
point estimate and unresolved in both networks. A through-origin scalar map of
the two additive potentials explains only 82.38 and 82.29 percent of neural
potential energy, below the frozen 90 percent rule, and its scale is 0.34296
and 0.35450 rather than greater than one. Do not call this a constant units
mismatch.

Preserve the positive `P_T` result. Query-time `P=0` subtraction assigns nearly
all signed allocation along the intact additive-potential direction to the
`P_T`-induced contrast (`phi_P` 1.00086 and 0.99827), while the baseline
allocation is unresolved around zero and was not tested for equivalence.
Track B therefore still identifies where the excessive slope resides inside
the neural policy; the signed allocation is consistent with and preserves its
episode-state provenance. What fails is the extra inference that the neural
additive-potential norm is oversized relative to the posterior comparator. Do
not use the stopped layer elasticities to claim a drive, recurrent, readout,
or co-adapted source.

Revise the global-policy problem from scalar overgain to a confidence-
allocation/comparator-shape question: a smaller neural additive-potential norm
can still have a steeper probability-distance slope when confidence is
distributed differently across pairs and distances. The probability-space
allocation remains unresolved. Do not fit a temperature, normalize `P_T`,
normalize `W_out`, or perform any scalar scale correction.

The subsequently frozen same-unit neural/posterior additive-by-residual field
reassembly is now complete on those same 2104/2105 backbones. This is a
sequential field-level sufficiency diagnostic, not an independent confirmation
or a realizable recurrent intervention. Its direct-margin amplitude bridge was
prospectively repaired after static implementation review but before either
checkpoint replay: exact reconstruction is anchored to `a_N_from_margin`,
while the already registered hidden/readout bridge retains its GPU tolerance.
Both formal executions are byte-identical, and every source, artifact,
endpoint, Hodge, norm-match, factorial-identity, and bootstrap-finiteness gate
passes.

The registered field-source fingerprint is `mixed_or_unresolved` separately in
both networks. Posterior-additive replacement materially reduces the neural-
posterior probability-slope difference (`Delta_A` 0.04080 and 0.04192), but its
remaining closure contrast is also materially positive (`C_A` 0.01694 and
0.01684), so additive replacement is not sufficient. Posterior-residual
replacement is unresolved and leaves a large positive closure contrast.
Neither natural single-component replacement nor the both-components rule
passes. Preserve the reductions without converting the largest one into a
sufficiency claim.

The symmetric additive-source main effect is materially positive (`A` 0.05417
and 0.05594), whereas the residual-source main effect is unresolved. The
registered probability-link interaction is materially negative (`I` -0.02674
and -0.02804): the neural-versus-posterior additive-source effect is smaller
under the neural residual than under the posterior residual. The corresponding
pre-sigmoid margin interaction is exactly zero. Treat `I` only as fixed-sigmoid
field-reassembly moderation, not recurrent, circuit, or biological coupling.
`A` and `R` partition `D`; `I` is a separately registered contrast of the same
four cells. None constitutes an independent evidence source or confirmation
link.

The sole posterior-to-neural norm match further separates shape from the
natural source replacement. At the neural full-28-edge additive norm with the
neural residual fixed, posterior additive shape materially reduces the slope
(`Q_shape` 0.03368 and 0.03474), and moving that posterior-shape field from the
participant-wise neural target to its natural norm reduces the cohort mean
(`Q_amp` 0.00712 and 0.00718); the scale direction is not uniform across
participants. But the norm-matched cell remains materially above the posterior
anchor (`C_shape` 0.02406 and 0.02403); shape is informative but not sufficient.
The norm match fixes complete pre-sigmoid field energy, not the 20-pair subset
norm or probability amplitude. It cannot trigger a scalar or model
intervention.

The revised constraint is therefore a comparator-relative additive allocation
difference with fixed-sigmoid residual context dependence and an unresolved
closure gap. Preserve the exact-posterior comparator boundary: it is not the
human posterior or ground-truth neural geometry. Do not add additional hybrid,
norm, mask, temperature, or comparator variants on 2104/2105. Do not modify
v2.4 or mix local cross-talk into this global analysis. Before any
implementable or causal follow-up, freeze the unchanged three-part field
fingerprint on one to three new development backbones: positive additive-source
effect, positive matched-norm shape reduction without assumed closure, and
negative probability-link interaction. A comparator-adequacy test requires a
separate prospective criterion.

That prospectively frozen replication is now complete on the two mandatory
fresh backbones 2106 and 2107. Both 1000-step backbones were generated before
one joint artifact lock, and neither Liu evaluation ran until that lock was
committed and pushed. Every competence, source, artifact, endpoint, Hodge,
norm-match, factorial-identity, and finiteness gate passes. A second complete
GPU evaluation is byte-identical. Participants were bootstrapped separately
within each network and never pooled; this is not network-population inference.

The registered outcome is `replicated_field_fingerprint`. The prerequisite
neural-versus-posterior slope anchor is positive in both networks. All three
primary links replicate independently: the additive-source main effect is
materially positive (`A` 0.05548 and 0.05375), the neural-norm posterior-shape
reduction is materially positive (`Q_shape` 0.03454 and 0.03325), and the
fixed-sigmoid interaction is materially negative (`I` -0.02779 and -0.02648).
Promote their conjunction, rather than any one component, into the stable
comparator-relative global-policy field fingerprint.

Preserve the replicated secondary boundaries. Natural posterior-additive
replacement and neural-norm posterior-shape replacement remain materially
above the posterior anchor in both networks (`C_A` 0.01667 and 0.01687;
`C_shape` 0.02372 and 0.02414). Shape/allocation contributes but is not
sufficient. The residual-source main effect remains unresolved in both
networks and was not a primary gate; do not call it zero, small, absent, or
equivalent. The pre-sigmoid interaction remains zero to floating-point error,
so negative `I` is fixed-link probability-field context dependence, not a
recurrent, circuit, or biological interaction.

That prospectively frozen read-only allocation audit is now complete on the
same locked 2106/2107 backbones. The exact bridge from the equal-energy field
difference through the fixed sigmoid to the slope ledger is verified for every
participant and bootstrap draw: `sum_e q_(s,e) = Q_shape,s`. Both complete GPU
executions are byte-identical, all source, artifact, fingerprint, identity, and
finiteness gates pass, and participants remain separate within network. This is
sequential localization, not another independent confirmation or network-
population inference.

The registered outcome is `policy_effective_allocation_localized`, with scope
`structural_only`. After removing an intercept and linear distance, the pair
vectors replicate across networks in both the correct-signed field mismatch
(`corr(r_delta)=0.99477`, bootstrap lower bound 0.40845) and the exact slope
bridge (`corr(r_q)=0.95800`, lower bound 0.33947). The six-level exact slope
ledger also replicates (`corr=0.99986`, lower bound 0.99267): positive
`Q_shape` is concentrated at distances 1 and 2 and partially offset at
distances 3 through 6. Treat this as stable comparator-relative structural
allocation at the policy-effective locations, not pairwise significance,
component sufficiency, a causal result, or a population-prevalence claim.

Posterior uncertainty and effective evidence coverage are reproducibly
associated with the allocation but fail the frozen policy-effectiveness rule.
For both variables in both networks, the simultaneous pair-fixed-effect field
coefficient is positive while the exact-bridge coefficient is negative. Do not
call either axis null, absent, or irrelevant; the registered result is a stable
field-to-policy sign reversal. The two predictors are highly correlated,
posterior uncertainty is comparator-derived, and coverage is observational, so
the coefficients are conditional associations rather than causal or total
effects.

Preserve the structural positive while revising the route: the audit does not
authorize a state-conditioned `P_T -> g_N` generation analysis, parameter
inspection, neural intervention, fifth allocation axis, new interaction, norm
change, or temperature fit. The only registered next stage is a prospectively
defined comparator-adequacy test with an external criterion fixed before any
alternative comparator is introduced. The exact posterior remains a frozen
comparator, not the human posterior or ground-truth neural geometry; do not
search comparator families or select the alternative that best removes the
current fingerprint.

That prospectively defined external comparator-adequacy audit is now complete.
It uses the public Liu trial-level human choice field as the only external
standard, retains all 77 eligible participants including correct rankers, and
does not match human to virtual subjects. Human participants alone are
bootstrapped; the current 77-subject exact-posterior field remains fixed. No
neural field, second comparator, temperature fit, checkpoint load, or neural
replay enters the audit.

The registered outcome is `comparator_externally_inadequate`; both necessary
criteria fail. On the fixed 20 nonlearned pairs, human distance slope is
`0.05398` with participant-bootstrap 95 percent interval
`[0.04731, 0.06069]`, whereas the current posterior slope is `0.01066` and is
`inadequate_below`. The secondary six-level ledger shows that the posterior is
already much more confident than humans at distances 1 through 4, especially
distances 1 and 2, while distance-5 and distance-6 differences are unresolved.
This explains the location of the previous `q` fingerprint without turning
near distances into a retrospective primary.

Human distance-residualized pair structure is highly reliable: odd/even-block
`r_HH=0.96976` and Spearman-Brown-corrected `rho_H=0.98465`. The current
posterior nevertheless has `r_PH=-0.04214`; its corrected ceiling ratio is
`-0.04247` with a 90 percent interval `[-0.32253, 0.36888]`, far below the
frozen 0.80 adequacy floor. Treat this as failure to capture a reliable human
pair fingerprint, not as a robust negative correlation.

Preserve all earlier positive network-internal and comparator-relative results,
but close intervention under this comparator. Do not inspect or modify `P_T`,
`W_out`, temperature, evidence admission, norm matching, or the v2.4 local
mechanism to remove the current neural-posterior difference. The only justified
route is prospective comparator-theory reassessment. Any future comparator
must have independent theoretical motivation and pass a separately frozen
human-only external adequacy contract before it can define a neural target; do
not search alternatives against the neural field.

The completed human-only comparator-theory test preserves the displayed signed
relative magnitudes and coherent eight-item order space, while giving each
participant one episode-stable Bernoulli access variable per support relation.
Only one global access probability, one constructive temperature, and one
response-lapse probability were fit on the 40-person preregistered cohort. The
parameters were then committed and hash-locked before the 37-person replication
cohort was opened. Derivation selected `rho=0.78988`, `tau=0.11349`, and
`epsilon=0.05073`; all 27 registered optimizer starts converged, no parameter
hit a bound, and all source-isolation and integrity gates pass.

The registered outcome is `distance_adequate_pair_inadequate`. The candidate
nonlearned slope is `0.06004`, inside the held-out human 95 percent interval
`[0.04221, 0.06079]` around `S_H=0.05143`. Preserve this positive evidence:
metric observations plus stable imperfect access and constructive global-order
commitment are sufficient for the human distance gradient under this split.
They are not sufficient for the reliable distance-residualized pair field.
`r_CH=0.23168`, corrected `eta_pair=0.23337`, and its 90 percent lower bound is
`-0.00896`, far below the frozen `0.80` floor.

Individual qualification is mixed and cannot rescue the failed pair primary.
The candidate passes the 80-percent stable-error and inter-subject ranking-
diversity axes. Its self-consistency miss is numerically tiny
(`0.999989` versus the human ceiling `1.0`) but fails the exact registered
interval. More materially, it overproduces perfectly stable errors
(`0.91867` versus human `0.75000`, human 95 percent upper bound `0.89655`).
Treat stable imperfect access as a sufficient source of distance dependence and
some individualized global structure, not as a human-adequate comparator or a
literal omission mechanism.

Close comparator search on the existing 37-person holdout. Do not add
relation-specific access, refit `rho/tau/epsilon`, run an ordinal limit,
compression law, Bradley--Terry model, or neural-informed comparator on these
responses. Neural intervention remains closed. The next decisive test is a new
behavior experiment that counterbalances displayed magnitude placement over an
unchanged signed support graph. The original Liu data do not independently
identify pair identity, graph position, true rank distance, and displayed
magnitude.

That behavior experiment is now prospectively registered, but no participant
collection is authorized. It is a within-participant two-list crossover with
disjoint image sets and four counterbalance cells. Both conditions preserve the
same eight signed support relations and magnitude multiset
`{1,2,3,3,3,4,5,7}`. Assignment A uses low-to-high order
`A-B-C-D-E-F-G-H`; assignment B uses `D-B-E-A-C-F-H-G`. Both are exact scalar
embeddings, all eight support-edge magnitudes change, and the complete query
graph splits prospectively into seven nonlearned order-flip pairs, thirteen
nonlearned same-direction pairs, and eight learned pairs. Do not replace
assignment B with an arbitrary edge shuffle or introduce metric frustration.

The registered hierarchy first tests assignment-following preference on the
seven flip pairs, then magnitude-linked confidence over the thirteen
same-direction nonlearned pairs only if the flip effect is equivalent to zero,
and finally the learned-pair magnitude slope only if both earlier effects are
equivalent. The symmetric SESOI values are `0.10` probability for the flip
contrast and `0.02` probability per gap unit for each slope. Equivalence has
practical-interpretation priority over a tiny directional effect; intervals
that merely include zero remain unresolved. Preserve the fixed outcome tree and
do not select a favorable pair subset, interaction, transform, or rank score
after collection.

Sample-size planning uses the existing 77 participants only for empirical
participant-profile variance and pair covariance. A conservative independent-
list construction gives null standard deviations `0.20534`, `0.05985`, and
`0.03233` for the flip, nonlearned-confidence, and learned-relation estimands.
Under the frozen balanced four-cell OLS and symmetric TOST equations, the
minimum analyzable size is 100, exactly 25 per cell; 96 fails the 0.90
all-axis power rule. Recruitment may replace only prospectively ineligible or
incomplete cases up to 30 enrolled per cell. Never use an observed condition
contrast to replace, exclude, or extend recruitment.

The pure synthetic contract validator passes every registered assignment,
pair-partition, task-interface, sample-size, power, and outcome-tree gate while
opening no participant response file. This is implementation validation, not
new behavioral evidence. Before the first new participant, separately review
and freeze acquisition code, image and randomization manifests, analysis code,
platform timing, consent/ethics status, source hashes, and write-once result
paths. Until that explicit authorization, do not begin collection, inspect a
pilot, continue comparator modeling, or reopen neural intervention. The
registration is in `benchmarks/magnitude_placement_behavior_v1.json`; the
synthetic validation is in
`results/magnitude_placement_behavior_v1_validation.json`.

The binary-codebook v1.1 collection-readiness stage is now complete. The
deterministic renderer, 120-slot counterbalanced manifest, immutable raw schema
and writer, importer, frozen analysis, all five synthetic outcome branches,
and four-cell/two-session dry run pass from source frozen before artifact
generation. This is an implementation result only. No human response was
created or opened, every external ethics/consent/recruitment/privacy/platform
requirement remains pending, and collection status remains `NO_GO`. Preserve
the v1 and v1.1 registrations and artifacts; do not recruit, pilot, inspect new
human data, or reopen the study without a later explicit user `GO`.
The revised protocol, readiness contract, repair, frozen randomization, and
readiness result are in
`benchmarks/magnitude_placement_behavior_v1_1.json`,
`benchmarks/magnitude_placement_behavior_v1_1_collection_readiness.json`,
`benchmarks/magnitude_placement_behavior_v1_1_collection_readiness_repair1.json`,
`benchmarks/magnitude_placement_behavior_v1_1_randomization.json.gz`, and
`results/magnitude_placement_behavior_v1_1_collection_readiness.json`.

## Near-term model-only scope: 2026-08-25 through 2026-09-25

For this period, do not require the current model to establish that its
internal computation is the mechanism used by human brains. Keep three levels
separate: behavioral phenomenon reproduction, model-internal computational
mechanism, and prospective human-mechanism validation. Work only on the first
two levels. Park new human-experiment implementation and data collection,
human comparator search, MEG/neural alignment, symbolic-distance tuning,
temperature or `W_out`/`P_T` calibration, and activation-function changes.

The model question is now:

> How does a meta-learned plastic network transform sparse relational evidence
> into a stable global structure while preserving direct experience?

Advance it in this order:

1. Build a frozen behavioral reproduction map for v2.4 covering learned and
   nonlearned accuracy, symbolic-distance and serial-position effects,
   difficult-pair bimodality, stable within-subject errors, self-consistent
   versus inconsistent errors, Hodge-reconstructed subjective ranking, and
   inter-subject ranking diversity or similarity. Classify every row as
   `reproduced`, `qualitatively_reproduced_quantitatively_mismatched`, or
   `not_reproduced`; report uncertainty and provenance. Do not tune the model
   in response to a mismatch. Treat symbolic-distance dependence as
   qualitatively reproduced with exact human calibration unresolved unless a
   frozen audit establishes otherwise.
2. Compress the confirmed global chain to
   `e_t -> (E_t,d_t) -> delta P_t -> P_T -> g_ij ~= s_i-s_j -> choice`,
   retaining support-write direction, DA magnitude but not identity, alpha
   sensitivity placement, remote reassembly, support-stage potential
   formation, query readout, and `P`-off necessity. The target description is
   `P_T implements state-dependent iterative relational reassembly`.
3. Compress the confirmed local chain to
   `direct observed relation -> L_T -> query-addressed contribution -> direct
   fidelity`. Preserve the `P`-off/`L`-on versus `L`-off double dissociation
   and v2.4 differential admission equations. Describe `P_T` as supporting
   relational abstraction/generalization and `L_T` as preserving
   experience-specific fidelity, without claiming unique code, biological
   stores, or human implementation.
4. Freeze the asymmetric algorithmic organization established by the completed
   compression sequence: `a_t` is an exact local edge-state algorithm, whereas
   `P_t` remains the interacting global implementation that emits a
   near-additive output but is not closed by the tested reduced states. Stop
   searching for a symmetric reduced global state. First test whether this
   functional asymmetry transports across matched Liu support topologies, then
   vary presentation order, sparsity, and item count one axis at a time. Only
   after the Liu-style task class is internally secure should list linking or
   adjacent transitive inference begin. Use one to three development seeds
   before freezing any larger run.
5. Only after the reproduction map and reduced mechanism are stable, perform
   read-only ancestry comparisons with the Miconi active/passive reinstatement
   mechanism. Do not treat analogy as identity or modify the frozen v2.4 model
   merely to increase similarity.

The four near-term deliverables are: a Liu-phenomenon/model reproduction map; a
main causal mechanism figure centered on evidence-to-`P_T` global assembly and
evidence-to-`a_T` local fidelity; the frozen asymmetric algorithmic theory; and
Liu-style structural generalization. List linking, mechanism ancestry, and
prospective human falsification are later layers, not present model acceptance
gates.

The first deliverable is now complete under a prospectively frozen, read-only
contract on v2.4 fresh backbones 2104 and 2105. Six of nine Liu phenomena are
reproduced independently in both networks: learned accuracy, nonlearned
accuracy, difficult-pair bimodality, 80-percent-stable errors,
Hodge-reconstructed individualized ranking, and inter-subject ranking
diversity. Three are directionally reproduced but quantitatively mismatched:
the symbolic-distance slope is too steep in both networks; the independently
registered serial-position endpoint contrast is too weak in both; and seed
2104 overproduces self-inconsistent subjects while seed 2105 is calibrated.
Nothing is classified `not_reproduced`.

Retain these three mismatches together. The opposite directions of the
distance and serial-position amplitude discrepancies reject a single global
temperature, output-gain, or `P_T`-amplitude repair as the next route. Do not
tune v2.4 against the map. Proceed to the main `P_T/L_T` causal mechanism
figure and reduced algorithm using confirmed links, while carrying all three
mismatches as behavioral shape constraints. The contract and result are in
`benchmarks/model_behavior_reproduction_map_v1.json` and
`results/model_behavior_reproduction_map_v1.json`; the interpretation is in
`docs/model_behavior_reproduction_map_v1.md`.

The second near-term deliverable is now frozen as a synthesis of existing
evidence, not a new experiment. The main model has two differentially admitted
episode-local states. `P_T` implements state-dependent iterative relational
reassembly: eligibility commonly carries relation direction, modeled DA
modulates magnitude, `alpha` places writes in high-sensitivity directions,
accumulated state changes their expression, and interacting writes produce an
expected-rank-like additive global field. `L_T` accumulates a broader weak
signed signal on a conjunctive address and supplies query-matched direct
fidelity. `P`-off and `L`-off establish the functional double dissociation.

Keep parameter-level and biological boundaries explicit. The
eligibility-direction/DA-gain split is not universal because of competent seed
2009; isolated remote updates are not independently correctness-propagating;
the address and `p_sr` are sufficient implementations rather than unique
codes; and `P_T/L_T` are not identified biological stores. The main mechanism
figure, equations, causal-link provenance, and negative constraints are in
`docs/model_mechanism_synthesis_v1.md`. The next active deliverable is a
read-only reduced-algorithm compression test, not further internal probing or
model tuning.

The first registered reduced-algorithm test is now complete and is a valid
negative for the global potential-only transition. The local tensor state
compresses exactly to edge coefficients plus its fixed Gram kernel:
`ell_q=(K a_T)_q`, with maximum reconstruction error below `8.89e-15` across
development and Liu preservation backbones. Promote this edge-addressed local
memory as the current algorithmic form of `L_T`; do not claim `K=I`, the
tensor-product key, or a biological store is unique.

Do not promote `s_t` to a closed Markov state under the frozen rank-2
state-by-evidence transition. In leave-one-backbone-out tests on 2101--2103,
the candidate is no better than the accumulator, prefix and terminal rollout
gates fail, and it recovers only 25--35 percent of full remote magnitude.
The full 576-parameter bilinear diagnostic is also worse than the accumulator,
so do not increase bilinear rank or refit v1. Preserve the positive boundary:
full fields remain more than 0.996 Hodge-additive, reduced/full remote
influence correlations are 0.81--0.82, and untouched Liu potential
correlations are about 0.91. This supports `s_t` as a strong output observable,
not the registered closed learning state.

The strict reduced double-dissociation and nine-row behavior-preservation
conjunctions also fail. Global-only nonlearned inference and local-only learned
fidelity remain strong, but exact-K local cross-talk retains 30--32 percent of
intact remote magnitude and several behavior categories change. Do not count a
newly human-calibrated serial-position category as a repair; human behavior was
not an optimization target.

The next admissible compression family is one separately frozen scalar
confidence/history augmentation, reusing the v1 trajectory artifact without
checkpoint re-extraction:
`c_(t+1)=c_t+0.5*||x_t||^2` and
`s_(t+1)=Pi[s_t+A x_t+B(c_t x_t)]`. Compare it with the accumulator and the
failed v1 candidate under the same held-out backbones, remote LOO, untouched
Liu double dissociation, and behavior categories. If it fails, do not add
alternative coverage, time, block, or interaction features after inspection;
move to a separately registered higher-dimensional state question. The v1
contract, result, and interpretation are in
`benchmarks/dual_state_reduced_algorithm_v1.json`,
`results/dual_state_reduced_algorithm_v1.json`, and
`docs/dual_state_reduced_algorithm_v1.md`.

That scalar-history v2 test is now complete and is a valid mixed negative.
Preserve its positive links. The frozen cumulative effective-evidence energy
reduces held-out one-step MSE by about 14 percent in every development
backbone, with all paired episode intervals below zero, and restores the
relation-LOO remote magnitude ratio from 25--35 percent in v1 to 78--90
percent. The strict reduced P/L double dissociation also passes independently
in 2104 and 2105. This supports history-dependent global gain as a real part
of the reduced computation.

The scalar is not a sufficient state. NRMSE, rollout, and behavior-category
gates fail in all required networks, and remote field MSE is worse than both
the accumulator and v1 even though total remote magnitude is restored. The
failure is therefore allocation, not simply missing total gain. Close scalar
history feature engineering: do not try count, absolute magnitude, time,
block, normalization, nonlinear scalar transforms, or mixtures.

The next and only task-derived confidence candidate should retain where
evidence accumulated while holding fitted capacity at 128 parameters:
`q_(t+1)=q_t+x_t elementwise-multiplied by x_t` and
`s_(t+1)=Pi[s_t+A x_t+B(q_t elementwise-multiplied by x_t)]`. Freeze it
separately, reuse the v1 trajectory artifact, and compare it with the
accumulator and v2 under the same remote, rollout, double-dissociation, and
behavior gates. If it fails, close task-derived confidence ledgers and move to
a separately registered `P_t`-derived latent-state audit. The v2 contract,
result, and report are in `benchmarks/dual_state_reduced_algorithm_v2.json`,
`results/dual_state_reduced_algorithm_v2.json`, and
`docs/dual_state_reduced_algorithm_v2.md`.

That capacity-matched item-history v3 test is now complete and is also a valid
mixed negative. Item allocation is informative: it reduces one-step MSE by
about ten percent versus the accumulator in every held-out backbone and avoids
the scalar v2 remote-MSE inflation. It is still worse than v2 on one-step
prediction, recovers only 20--25 percent of full remote magnitude, fails all
rollout gates, weakens Liu potential alignment and nonlearned accuracy, and
fails both strict preservation conjunctions.

Together v2 and v3 establish a constrained trade-off. A global scalar restores
remote amount but misallocates the field; an item ledger improves remote error
but loses the amount. Do not combine them after seeing complementary failures.
Close all hand-built count, coverage, energy, time, block, pair-ledger, and
scalar-plus-item history variants.

The next active step is a read-only `P_t` latent sufficiency audit. Re-extract
the exact registered generic episodes and require the reconstructed `s_t`
trajectory to match the frozen v1 NPZ before using `P_t`. Because independent
backbones have unaligned fast-weight coordinates, first test prospectively
fixed latent dimensions within each backbone on held-out episodes; do not pool
or call PCA coordinates a shared algorithm. Ask whether a low-dimensional
`P_t` projection predicts residual updates and remote LOO effects beyond
`(s_t,x_t)`. Only replicated dimensional sufficiency should motivate a later
cross-network invariant-state construction. The v3 contract, result, and
report are in `benchmarks/dual_state_reduced_algorithm_v3.json`,
`results/dual_state_reduced_algorithm_v3.json`, and
`docs/dual_state_reduced_algorithm_v3.md`.

That functional fast-weight latent-sufficiency audit is now complete and is a
valid replicated negative under its registered linear readout. Exact v1
natural and LOO trajectories reassemble in all three backbones, all integrity
gates pass, and rank 7 numerically reconstructs the full functional-`P`
prediction. Nevertheless, the full residual `alpha elementwise-multiplied by
P_t` readout generalizes catastrophically: held-out one-step MSE is 57--141
times the `(s_t,x_t)` baseline, and remote correlations are negative while
remote magnitude is inflated 28--118 fold. Rank 1 is already robustly harmful
in every network and error increases through rank 7. Therefore no information
fraction or `k_min` is defined under the frozen rules.

Do not interpret this as absence of information or causality in `P_t`. Reject
the narrower claim that one fixed cross-episode linear supervised projection
of residual functional `P_t` exposes a transferable low-dimensional state,
even within a backbone. Do not tune the ridge, change `P_t` normalization, add
PCA or nonlinear kernels, or proceed to autonomous latent closure after this
failure. Preserve the stronger synthesis from v1--v3: near-additive relational
geometry is a low-dimensional output code, while its generating global
learning state remains high-dimensional and interacting.

The current reduced theory is deliberately asymmetric. The local branch has
an exact edge-state algorithm `a_(t+1)=a_t+s_t^L e_(r_t)` with read
`ell=K a_T`; the global branch remains the frozen meta-learned plastic system
whose high-dimensional state produces a near-additive potential observable.
Consolidate and test claims at this boundary rather than forcing a symmetric
low-dimensional global algorithm. This model-only audit does not reopen human
experiment design. Its contract, repair, result, and report are in
`benchmarks/functional_fast_weight_latent_sufficiency_v1.json`,
`benchmarks/functional_fast_weight_latent_sufficiency_v1.repair1.json`,
`results/functional_fast_weight_latent_sufficiency_v1.json`, and
`docs/functional_fast_weight_latent_sufficiency_v1.md`.

The prospectively frozen `liu_support_topology_transport_v1` development test
is complete with registered outcome `LIU_STRUCTURAL_MECHANISM_TRANSPORTED`.
All three matched, pairwise non-isomorphic alternative graphs pass all eight
links independently in seeds 2101--2103: nine of nine graph-by-backbone cells
and 72 of 72 link decisions pass, with no participant pooling or network
majority rescue. Intact learned and nonlearned competence, near-additive and
transitive global structure, individualized stable rankings, P-off collapse,
remote/third-party P-LOO, a-off direct loss, local-only nontransitivity, and
exact edge-ledger compression all transport.

Preserve the quantitative topology variation. The high-hub graph has lower
nonlearned performance and Hodge-to-true alignment and a smaller local direct
benefit than the other graphs, but every registered functional boundary still
passes. Therefore claim transport of the asymmetric computation, not invariant
performance across graphs, arbitrary graph generalization, graph identities
known to be absent from procedural meta-training, or network-population
prevalence.

Attempt 1 is preserved because its original outcome was
`TOPOLOGY_DEPENDENT_OR_UNRESOLVED`: two graphs exceeded the `1e-6` local-read
exactness threshold by only `0.24--0.31e-6` when sequential float32 GPU sums
were compared with grouped float64 Gram sums. The separately registered repair
recomputed the already established exact-`a_T` identity in one common float64
representation while retaining GPU differences as diagnostics. The repaired
state/read errors are at most `4.44e-16` and `9.77e-15`, and every nonrepair
value is exactly equal to attempt 1. The contract, repair, preserved attempt,
final result, and report are in
`benchmarks/liu_support_topology_transport_v1.json`,
`benchmarks/liu_support_topology_transport_v1.repair1.json`,
`results/liu_support_topology_transport_v1.attempt1.json`,
`results/liu_support_topology_transport_v1.json`, and
`docs/liu_support_topology_transport_v1.md`.

Carry the excessive symbolic-distance slope, weak serial-position endpoint,
and original seed-2104 inconsistency as known limitations. Do not tune
temperature, `W_out`, `P_T`, `a_T`, evidence admission, or graph-specific
parameters on the alternative graphs. Human-mechanism validation, MEG, list
linking, Miconi ancestry, and new global compression remain deferred. Continue
Liu internal validity in the fixed order: presentation order, sparsity, then
item count. The next protocol must vary presentation order only while holding
the transported graph/model contract fixed.

That presentation-order transport is now complete on development seeds
2101--2103. Blockwise-random, relation-clustered, and exact-reverse schedules
were made from every subject's same 32 physical support trials, with only
position-aligned relation gains rebuilt and no admission resampling. All eight
registered links pass independently in all nine schedule-by-backbone cells;
the outcome is `LIU_PRESENTATION_ORDER_MECHANISM_TRANSPORTED`.

Preserve the exact algorithmic contrast. In common float64 arithmetic, both
nonbaseline terminal `a_T` ledgers and every one of their 28 Gram reads are
exactly equal to blockwise random for every participant and backbone. The
global `P_T` output is quantitatively order-sensitive: relation clustering has
only `.829--.837` field correlation and `.872--.878` exact decision agreement
with baseline, whereas reversal has `.959--.965` correlation and
`.943--.953` agreement. Relation clustering also lowers learned exact accuracy,
but every competence, construction, individualization, remote reassembly, and
P/a double-dissociation gate still passes. Treat this as an exactly commutative
local edge ledger beside a state-dependent iterative global computation, not
as behavioral or `P_T` invariance. The contract, result, and report are in
`benchmarks/liu_presentation_order_transport_v1.json`,
`results/liu_presentation_order_transport_v1.json`, and
`docs/liu_presentation_order_transport_v1.md`.

The model-side theory is now formally consolidated as **asymmetric
algorithmic organization of relational memory**. Keep three evidence levels
separate: the frozen 9-row phenomenology map (six reproduced, three qualitative
but quantitatively mismatched, none absent); the replicated functional
decomposition (`P_T` for remote/global assembly and edge state `a_T` for
query-addressed direct fidelity); and the algorithmic asymmetry (exact local
edge-plus-Gram compression versus a near-additive global output whose learning
dynamics are not closed by any registered reduced state).

Stop the global reduced-algorithm search. The current network implements global
reassembly through a high-dimensional interacting `P_T`, but do not call that
dimensionality intrinsic or claim that no nonlinear, trajectory-dependent, or
episode-coordinate-dependent compact state can exist. Those possibilities are
logically open but are not a licensed near-term search program. Preserve v2 as
evidence for history-dependent update amount, v3 as evidence for relational
allocation structure, and the functional-`P` failure as evidence against a
stable fixed cross-episode linear coordinate.

The immediate model question is now Liu-style evidence sparsity. Freeze
connected eight-item graphs with `|E| = 7, 8, 9, 10` and a prospective matched
design before execution, then test the same competence, construction,
individualization, remote reassembly, exact-local, and P/a double-dissociation
links on only one to three development backbones. Distinguish transport of the
functional asymmetry from the quantitative prediction that sparse evidence
increases reliance on `P_T` while dense evidence directly covers more queries
through `a_T`. Item-count transport follows only after sparsity. List linking,
classic transitive inference, Miconi ancestry, MEG, human-mechanism validation,
and new global-compression work remain deferred. The consolidated synthesis
and its source-hash registry remain in
`docs/asymmetric_algorithmic_organization_v1.md` and
`benchmarks/asymmetric_algorithmic_organization_v1.json`.

That sparsity test is now complete. Its two nested graph families have matched
rank-distance multisets at `E=7,8,9,10`, preserve every common physical trial
and stable admission, and exactly replay both frozen `E=8` source cells. All
integrity gates pass and an independent GPU execution is byte-identical.

The registered outcome is `SPARSITY_DEPENDENT_OR_UNRESOLVED`: 23 of 24 cells
and 191 of 192 links pass. Both families transport at `E=7,8,9`. At `E=10`,
five of six cells pass; balanced-branched seed 2103 fails only the stable-error
prevalence part of `individualized_stable_structure` (point `.8630`, lower
bound `.7808` versus the frozen `.80` boundary). Every competence,
constructive-global, remote-reassembly, local-direct, P-off-scope, and exact
local link passes in all 24 cells. Preserve transport over the registered
seven-to-nine range and the E=10 causal positives, but do not claim complete
seven-to-ten sparsity transport or move/pool the failed threshold.

The secondary density-allocation prediction is rejected. All-query causal
dependence on `P_T` has a positive rather than negative density slope in every
family/backbone (`+.00293` to `+.00748` probability per added relation; five
intervals exclude zero). The `a_T` all-query dependence slope is unresolved in
all six analyses, even though its observed-relation direct benefit remains
positive in every cell. Therefore `P_T` is not merely recruited to compensate
for sparse direct coverage; more compatible evidence can strengthen global
construction. `a_T` remains a direct-fidelity path whose cross-talk prevents a
monotonic total-policy gain. The contract, result, and report are in
`benchmarks/liu_evidence_sparsity_transport_v1.json`,
`results/liu_evidence_sparsity_transport_v1.json`, and
`docs/liu_evidence_sparsity_transport_v1.md`.

Do not begin item-count transport yet. First freeze a read-only localization on
the existing sparsity artifact only, separating general density-linked
convergence of subject Hodge orders/stable error sets, an E=10
balanced-family-specific effect, and a lone bootstrap-boundary miss. Do not
rerun networks, add densities, alter the 80-percent stable-error definition, or
pool participants/backbones for that localization.

The fresh confirmation is in
`docs/dual_evidence_access_confirmation_v2_4.md`, the slope localization is in
`docs/global_policy_slope_localization_v1.md`, the amplitude audit is in
`docs/global_policy_amplitude_provenance_v1.md`, and the field reassembly is in
`docs/global_policy_field_reassembly_v1.md`; the fresh-backbone fingerprint
replication is in
`docs/global_policy_field_fingerprint_replication_v1.md`, and the equal-energy
allocation localization is in
`docs/global_policy_allocation_audit_v1.md`, and the human-external comparator
adequacy result is in
`docs/global_policy_comparator_adequacy_v1.md`, and the held-out human-only
metric constructive-comparator result is in
`docs/human_metric_constructive_comparator_v1.md`.

Start any later v2 replication with only one to three development seeds. Freeze
estimands, competence gates, and outcome-contingent interpretations before any
new formal population.
Keep the seed-2009 complete-write/operator localization separate from the human
local-fidelity question. The current formal evidence is recorded in
`docs/formal_confirmation_v1.md`; the response-state negative and its next
decisive test are in `docs/hidden_residual_audit_v1.md`; the subsequent mixed
operator-code result is in `docs/relation_trace_localization_v1.md`; its
functional resolution is in `docs/state_query_operator_binding_v1.md`; the
value-to-expression localization is in `docs/operator_output_semantics_v1.md`;
the final amplitude/curvature architecture-selection gate is in
`docs/operator_amplitude_path_v1.md`; and the first v2 intervention and its
negative result are in `docs/curvature_gate_pilot_v2.md`. The signed
policy-opposition follow-up and amplitude-gating family closure are in
`docs/policy_opposition_gate_pilot_v2_1.md`; the first-order residual test and
low-capacity expression-family closure are in
`docs/policy_residual_pilot_v2_2.md`; and the persistent conjunctive trace and
causal local/global double dissociation are in
`docs/conjunctive_local_trace_pilot_v2_3.md`; and the registered explanation of
its direct-causal versus sampled-behavior mismatch is in
`docs/local_behavior_attribution_v2_3.md`; and the independent two-backbone
replication of the local/global decomposition is in
`docs/conjunctive_local_trace_replication_v2_3.md`; and the registered
differential evidence-access sufficiency result is in
`docs/dual_evidence_access_pilot_v2_4.md`; and its blind fresh-backbone
confirmation is in `docs/dual_evidence_access_confirmation_v2_4.md`.
The registered global-policy slope localization is in
`docs/global_policy_slope_localization_v1.md`; and the registered common-unit
amplitude-provenance result is in
`docs/global_policy_amplitude_provenance_v1.md`; and the registered
additive-by-residual field-reassembly result is in
`docs/global_policy_field_reassembly_v1.md`; and its prospectively frozen
fresh-backbone fingerprint replication is in
`docs/global_policy_field_fingerprint_replication_v1.md`; and the registered
equal-energy allocation localization is in
`docs/global_policy_allocation_audit_v1.md`; and the human-external comparator
adequacy result is in
`docs/global_policy_comparator_adequacy_v1.md`; and the registered human-only
metric constructive-comparator derivation and confirmation are in
`docs/human_metric_constructive_comparator_v1.md`.

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
