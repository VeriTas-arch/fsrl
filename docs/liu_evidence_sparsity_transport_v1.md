# Liu evidence-sparsity transport v1

## Registered outcome

`SPARSITY_DEPENDENT_OR_UNRESOLVED`

The complete conjunction passes in 23 of 24 prospectively registered
family-by-density-by-backbone cells. This is 191 of 192 primary-link decisions.
Both graph families transport independently at `E=7`, `E=8`, and `E=9`. At
`E=10`, five of six cells pass; seed 2103 on the balanced-branched family fails
only `individualized_stable_structure` because the participant-bootstrap lower
bound for 80-percent-stable error prevalence is `.7808`, below the frozen `.80`
boundary. Its point estimate is `.8630`.

Every competence, constructive-global, `P_T` remote-reassembly, `a_T` direct,
P-off/local-scope, and exact-local link passes in every cell. The registered
full sparsity-range claim nevertheless does not pass, and the E=10 failure must
not be repaired by moving the stable-error threshold or pooling networks.

The strongest supported statement is:

> The frozen `P_T/a_T` functional asymmetry transports across both registered
> graph families from seven through nine observed relations. At ten relations,
> competence and the causal division remain present in every backbone, but the
> complete individualized-stability conjunction is heterogeneous.

This is a development result on two graph families and three frozen backbones.
It is not arbitrary-sparsity or item-count generalization, a
network-population prevalence estimate, or a human/biological mechanism.

## Frozen density design

Two independent graph families were centered on the source-correct Liu cycle
and the already transported balanced-branched graph. Within each family,
`E=7,8,9,10` forms one nested sequence. Every density step removes or adds one
rank-distance-three relation; therefore both families have the same
rank-distance multiset at a given density.

| E | Rank-distance multiset | Support trials | Direct query coverage |
| ---: | --- | ---: | ---: |
| 7 | `1,2,3,3,4,5,7` | 28 | 7/28 |
| 8 | `1,2,3,3,3,4,5,7` | 32 | 8/28 |
| 9 | `1,2,3,3,3,3,4,5,7` | 36 | 9/28 |
| 10 | `1,2,3,3,3,3,3,4,5,7` | 40 | 10/28 |

Every observed relation retains four passive presentations with signed metric
evidence and no support choice or feedback. Within a family, every common
relation has the same four physical trials, relative within-block order, and
episode-stable admission across densities. Added relations are generated once
and then filtered from the lower-density schedules. Item count, query set,
backbone, local gain, cue codes, query schedules, v2.4 admission equation,
`P_T`, `a_T`, `tanh`, `W_out`, and temperature are fixed.

Density intentionally changes the number of observed relations, total support
trials, total evidence, and event rate within the normalized support phase. The
result cannot distinguish those quantities from one another.

## Provenance and execution

- Initial registration commit: `41847ef2f77dbc8d766fcf7ed9202dd6bcf5d52a`.
- Mechanical registration clarification commit:
  `6f2d7cdfa7c2c45549f76de1a0b08caca6696864`.
- Implementation/source-lock commit:
  `3859d9e55bfc3bbe9d8a37a6ce633be7e383a081`.
- Frozen backbones/local gains: seeds 2101, 2102, and 2103.
- Virtual participants: 77 within every family, density, and network.
- Participant bootstrap: 10,000 samples within each cell; never pooled.
- Runtime: NVIDIA GeForce RTX 5090, PyTorch `2.13.0+cu130`, CUDA 13.0, with
  PyTorch intra-op and inter-op CPU threads both fixed to one.
- No training, gain adaptation, compilation, parameter update, density-specific
  calibration, or result-dependent rerun occurred.

All source, artifact, graph, nested-schedule, nested-admission, GPU-runtime,
tensor-freeze, orientation, finite-value, and exactness gates pass. The Liu
cycle and balanced-branched `E=8` cells exactly reproduce their frozen source
schedule hashes and complete shared metric projections in all three networks.
An independent full execution is byte-identical to the registered result; both
have SHA-256
`baad32a2b737cc26a6e9c6a02ccb82da0a51c953d3d80bbfad115c74499d1b41`.

## Primary results

The table gives point-estimate ranges over seeds 2101--2103. Decisions use
within-cell participant-bootstrap intervals, never these across-seed ranges.

| Family | E | Learned exact | Nonlearned exact | Hodge tau to true | Inter-subject tau | Stable error >=80% | Global P-LOO remote | Intact - a-off learned probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Liu cycle | 7 | .9425--.9481 | .7922--.8033 | .6688--.6883 | .5036--.5200 | .9079--.9467 | .3253--.3627 | .0199--.0266 |
| Liu cycle | 8 | .9448 | .8091--.8240 | .7059--.7226 | .5401--.5500 | .9054--.9333 | .3123--.3570 | .0145--.0221 |
| Liu cycle | 9 | .9495--.9538 | .8134--.8250 | .7199--.7403 | .5398--.5726 | .8919--.9178 | .2747--.2977 | .0130--.0208 |
| Liu cycle | 10 | .9506--.9558 | .8182--.8268 | .7338--.7551 | .5807--.5939 | .8873--.9041 | .2497--.2683 | .0122--.0201 |
| Balanced branched | 7 | .9276--.9406 | .8108--.8169 | .6929--.7032 | .5288--.5323 | .9054--.9467 | .3119--.3457 | .0117--.0135 |
| Balanced branched | 8 | .9334--.9367 | .8117--.8208 | .7013--.7143 | .5328--.5429 | .9178--.9467 | .2977--.3277 | .0116--.0138 |
| Balanced branched | 9 | .9408--.9495 | .8257--.8346 | .7421--.7532 | .5763--.5850 | .8889--.9315 | .2501--.2657 | .0074--.0096 |
| Balanced branched | 10 | .9416--.9506 | .8290--.8369 | .7375--.7662 | .5810--.5986 | .8630--.9296 | .2374--.2445 | .0084--.0109 |

Across all 24 cells, the worst registered bounds for the other seven links
remain clearly on their passing side:

| Link component | Worst bound | Registered boundary |
| --- | ---: | ---: |
| intact learned exact accuracy | lower .9054 | lower > .50 |
| intact nonlearned exact accuracy | lower .7619 | lower > .50 |
| intact Hodge fraction | lower .9868 | lower >= .95 |
| a-off Hodge fraction | lower .9947 | lower >= .95 |
| transitive-triplet fraction | lower .9951 | lower >= .95 |
| Hodge tau to true | lower .6224 | lower > 0 |
| inter-subject tau | upper .6449 | upper < .80 |
| P-off/a-on nonlearned probability | upper .4767 | upper <= .55 |
| P-off local minus 0.25 global remote | upper -.0373 | upper < 0 |
| global P-LOO remote influence | lower .2271 | lower > 0 |
| global P-LOO third-party fraction | lower .1620 | lower > 0 |
| intact minus a-off learned probability | lower .00357 | lower > 0 |
| P-off/a-on learned probability | lower .6279 | lower > .50 |
| P-off learned minus nonlearned probability | lower .1740 | lower > 0 |

The only failed bound is balanced-branched E=10 seed 2103 stable-error
prevalence: point `.8630`, interval `[.7808,.9315]`. It still has 73 eligible
non-correct subjects, inter-subject tau `.5955` with upper bound `.6449`, and
passes all competence and causal mechanism links. This is not a failed model
or a collapsed double dissociation; it is a registered uncertainty/heterogeneity
boundary on stable individual errors.

## Preserved positive mechanism

At every density and in every network, the intact system remains competent on
observed and unobserved pairs, its global field remains near-additive and
transitive, and its Hodge order remains positively aligned with true rank.
Global relation LOO retains disjoint and third-party effects. Removing `a_T`
causes a positive observed-relation probability loss. With `P_T` removed and
`a_T` intact, observed-relation probability remains above chance while
nonlearned probability stays near chance and remote influence collapses.

Therefore the E=10 individualization miss does not erase the supported causal
chain:

`broader local admission -> a_T -> direct observed-relation fidelity`,

`selective global admission -> P_T -> coherent remote/nonlearned assembly`.

It instead limits the stronger claim that competence, causal division, and the
full stable-individualization conjunction all transport through ten observed
relations in both graph families and all three backbones.

The exact edge-ledger algorithm also remains intact. Across all 24 cells, the
largest common-float64 tensor/ledger error is `1.11e-15` and the largest
all-query Gram-read error is `8.88e-15`.

## Rejected density-allocation prediction

The separately registered secondary prediction does not pass in any of the six
family-by-backbone analyses. The participant-level slope of all-query global
dependence,

`mean(p_intact - p_P-off/a-on)`,

is positive in every analysis (`+.00293` to `+.00748` probability per added
relation), not negative as predicted. Five of six intervals exclude zero on
the positive side; the remaining interval is `[-.00058,+.00667]`. More observed
evidence therefore tends to increase, not reduce, causal dependence on the
global recurrent computation.

The all-query local-dependence slope,

`mean(p_intact - p_a-off)`,

ranges from `-.00029` to `+.00055` and every interval crosses zero. The local
trace continues to provide a robust positive effect on directly observed
relations, but its mean effect over all 28 queries is slightly negative or near
zero because address-kernel cross-talk affects nonlearned pairs. Merely covering
more direct relations does not produce the preregistered monotonic increase in
total local policy dependence.

This revises the sparsity theory. `P_T` is not simply a compensatory mechanism
used when direct coverage is missing. Additional mutually compatible evidence
can strengthen the coherent global construction and make the intact policy more
dependent on `P_T`. Conversely, `a_T` remains a direct-fidelity mechanism, not
an increasingly dominant all-query policy as density rises.

## Secondary behavior and carried limitations

Across cells, sampled learned accuracy is `.9180--.9381`, sampled nonlearned
accuracy is `.7818--.8228`, symbolic-distance slope is `.0462--.0533`, and the
serial endpoint contrast is `.0270--.0878`. These were mandatory reports, not
acceptance gates. The excessive symbolic-distance slope, weak original serial
endpoint, and seed-2104 inconsistency remain known limitations and were not
tuned.

## Theory update and next test

The positive evidence now supports a broad functional mechanism from seven to
nine observed relations across two topology backgrounds. The valid E=10
negative suggests that evidence density may change the *individualization* of
the constructed ranking before it removes competence or the causal division:
inter-subject tau generally rises and stable-error prevalence generally falls
at higher density. That pattern is descriptive here; it was not a registered
monotonic primary estimand.

Do not proceed directly to item-count transport. The next step is a separately
frozen read-only localization using this result only. It should distinguish:

1. a genuine density-linked convergence of subject-specific Hodge orders and
   stable error sets;
2. an E=10 balanced-family-specific effect; and
3. a bootstrap-boundary miss with no replicated participant-level density
   change.

Freeze paired within-subject order/error estimands, family contrasts, and
outcome interpretations before reading any additional raw decomposition. Do
not rerun networks, add density levels, tune the stable-error cutoff, pool
participants or backbones, or begin item-count, list-linking, human, MEG,
Miconi-ancestry, or new global-compression work until that localization is
resolved.
