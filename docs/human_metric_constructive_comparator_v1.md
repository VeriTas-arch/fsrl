# Human metric constructive comparator v1

## Conclusion

The prospectively selected, human-only, metric-preserving comparator returns
`distance_adequate_pair_inadequate` on the untouched 37-person replication
cohort.

The result contains a real positive. A three-parameter model with episode-stable
imperfect access to the eight displayed metric relations, exact construction
over all `8!` coherent orders, one committed order per participant, and a small
test lapse reproduces the held-out human nonlearned symbolic-distance gradient:

```text
candidate S_C = 0.06004
human S_H     = 0.05143
human 95% CI  = [0.04221, 0.06079]
```

It does not recover the reliable distance-residualized pair field:

```text
r_CH                  = 0.23168
r_HH                  = 0.97162
rho_H                 = 0.98561
eta_pair              = 0.23337
eta_pair 90% interval = [-0.00896, 0.49126]
registered floor      = 0.80
```

The candidate also passes two of four individual qualification axes: the
prevalence of at least one 80-percent-stable error and inter-subject ranking
diversity. It fails exact self-consistency by a numerically tiny amount and more
materially overproduces 100-percent-stable errors. Because pair adequacy is a
necessary primary, the candidate is not a provisional external comparator and
cannot define a neural intervention target.

Close comparator search on the existing replication responses. The next
decisive study is a new magnitude-placement behavior experiment, not another
comparator fit on the same holdout.

## Task-faithful information boundary

The [source experiment](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003756)
presented eight fixed nonadjacent support pairs. Participants saw ticked bars,
were explicitly instructed to attend to their relative difference, observed
each pair once in each of four learning blocks without making a response, and
then chose the higher item for all 28 pairs in ten no-feedback test blocks.

The registered comparator therefore keeps the observation interface fixed:

```text
O_r = (item_i, item_j, signed displayed magnitude m_r).
```

Random absolute bar level is removed as task-irrelevant nuisance. Displayed
magnitude is not removed. Internal abstraction is permitted only as an explicit
cognitive hypothesis:

```text
task-faithful observation
    -> episode-stable effective access
    -> constructive coherent order
    -> repeated no-feedback choices.
```

This distinguishes an information-preserving task abstraction from a sign-only
ordinal bottleneck. The latter remains a possible theory, but it was neither a
control nor a fallback candidate here.

## Frozen model

For participant `s` and support relation `r`, the candidate assigns one access
state for the complete episode:

```text
z_sr ~ Bernoulli(rho).
```

`z_sr=1` retains the complete displayed metric relation; `z_sr=0` supplies no
effective global evidence. It does not replace magnitude by sign. The four
identical-magnitude presentations are summarized at the stable relation level,
so the order energy contains one term for each retained relation:

```text
E_z(pi) = sum_r z_r [delta_pi(r) - m_r]^2,
delta_pi(r) = [position_pi(lower_r) - position_pi(higher_r)] / 7.
```

All 256 stable masks and all 40,320 high-to-low orders are enumerated exactly:

```text
q(pi | z,tau) = softmax_pi[-E_z(pi)/tau]

q(pi | rho,tau)
  = sum_z rho^sum(z) (1-rho)^(8-sum(z)) q(pi | z,tau).
```

Each participant commits once to `pi_s`. Conditional on this order, each test
response has source-correct probability `1-epsilon` if the order gives the
source-correct orientation and `epsilon` otherwise. Thus repeated responses and
different query pairs are independent only conditional on one shared committed
order. The derivation likelihood integrates that order once per participant;
it is not a product of cohort-marginal pair probabilities.

There are exactly three cohort-level parameters:

```text
rho      relation access probability
tau      constructive order temperature
epsilon  response lapse
```

No relation, item, distance, subject, cohort, learned/nonlearned, or query-
specific parameter is present.

## Derivation firewall and result

The scientific contract was committed and pushed as `22f3fde` before the new
runner or any candidate parameter existed. The implementation and source lock
were committed and pushed as `49ca86c` before loading the 40-person derivation
cohort.

Attempt 1 completed numerical optimization but failed before opening an
artifact because strict JSON serialization rejected a `numpy.bool_` gate. No
parameter or scientific result was written, printed, inspected, or used to
change the contract. The failure is preserved in
`results/human_metric_constructive_comparator_v1_attempt1_noninterpretable.json`.
Repair 1 changes only gate-scalar conversion to native Python `bool`; it was
separately frozen and pushed as `9279a6a` before replay.

The replay loaded only the 40 preregistered participants. For every participant,
all ten correctness responses for all 28 pairs entered one exact committed-order
likelihood. The 27 fixed L-BFGS-B starts all converged. All solutions within
`1e-6` log-likelihood units of the selected solution give a 28-pair field within
`1.24e-6` maximum absolute difference. The selected parameters are:

| Parameter | Estimate | Registered bound | At bound |
| --- | ---: | ---: | --- |
| `rho` | 0.78988 | [0.0001, 0.9999] | no |
| `tau` | 0.11349 | [0.0001, 4.0] | no |
| `epsilon` | 0.05073 | [0.0001, 0.49] | no |

The summed derivation log likelihood is `-1538.94356`. All source, runtime,
cohort-isolation, completeness, optimizer, and probability gates pass. The
parameter artifact and lock were committed and pushed as `f56a22b` before the
replication responses were loaded.

## Held-out primary results

The confirmation uses only the 37 retained replication participants. Candidate
parameters and its 28-pair field remain fixed. Human participants alone are
resampled in one 10,000-draw bootstrap; the candidate is not refit or resampled.

### 1. Distance adequacy passes

For the inherited 20 nonlearned pairs and fixed OLS weights:

| Estimand | Point | Human participant-bootstrap rule | Status |
| --- | ---: | ---: | --- |
| Human `S_H` | 0.05143 | 95% [0.04221, 0.06079] | external interval |
| Candidate `S_C` | 0.06004 | must lie inside human 95% interval | pass |

The candidate lies near the upper edge of the external interval but is inside
it under the prospectively frozen inclusive rule. Preserve the exact claim:

> Metric observations plus stable imperfect access and constructive commitment
> to one global order are sufficient for the held-out human nonlearned distance
> gradient under this cohort split.

This is not evidence that relations are literally omitted, that these are the
unique sufficient components, or that the comparator captures the complete
human choice field.

### 2. Pair adequacy fails

Only an intercept and linear symbolic distance are removed from the fixed
20-vectors. Replication odd/even blocks provide a high human noise ceiling:

| Estimand | Point | 90% bootstrap | 95% bootstrap |
| --- | ---: | ---: | ---: |
| Human split-half `r_HH` | 0.97162 | [0.88943, 0.98405] | [0.86912, 0.98667] |
| Corrected reliability `rho_H` | 0.98561 | [0.94148, 0.99196] | [0.92998, 0.99329] |
| Candidate-human `r_CH` | 0.23168 | [-0.00886, 0.48232] | [-0.04375, 0.53329] |
| Corrected `eta_pair` | 0.23337 | [-0.00896, 0.49126] | [-0.04435, 0.54360] |

The registered pair rule requires the 90-percent lower bound of `eta_pair` to
be at least `0.80`. It fails. The positive point correlation is descriptive and
does not establish a stable pair match; its interval includes zero. Thus one
i.i.d. stable-access probability can generate a human-like global distance
trend without capturing the reliable relation-specific deviations from that
trend.

## Individual qualification

The candidate predictive point is estimated from the frozen 200,000-participant
simulation. Human intervals resample the 37 replication participants. All
definitions match the source-locked human benchmark.

| Mandatory axis | Human point [95%] | Candidate | Status |
| --- | ---: | ---: | --- |
| Mean self-consistency | 1.00000 [1.00000, 1.00000] | 0.999989 | fail |
| At least one 80%-stable error among error rankers | 0.93750 [0.84375, 1.00000] | 0.99881 | pass |
| At least one 100%-stable error among error rankers | 0.75000 [0.59375, 0.89655] | 0.91867 | fail |
| Mean inter-subject Kendall tau | 0.55847 [0.48778, 0.65963] | 0.52502 | pass |

The self-consistency failure is an exact preregistered endpoint failure but its
absolute discrepancy is only about `1.06e-5`: 13 of 198,871 eligible simulated
participants are classified as self-inconsistent. It should not be inflated
into evidence for a material coherence defect. The 100-percent-stable-error
failure is more informative. Conditional on becoming an erroneous ranker, the
candidate makes perfectly repeated errors too often, even though its fitted
lapse reduces cohort accuracy.

Mandatory descriptive accuracies point in the same direction:

| Metric | Human replication | Candidate prediction |
| --- | ---: | ---: |
| Overall accuracy | 0.85994 | 0.81039 |
| Learned accuracy | 0.91453 | 0.87385 |
| Nonlearned accuracy | 0.83811 | 0.78500 |

These were not additional primary gates and do not redefine the registered
outcome.

## Provenance and integrity

- All source, task-interface, cohort-isolation, parameter-lock, replay,
  probability, human-benchmark, bootstrap, simulation, and finiteness gates
  pass.
- Derivation used exactly 40 participants; confirmation used exactly 37. No
  participant, trial, block, pair, order, mask, optimizer start, or predictive
  participant was filtered after execution.
- The runner loaded no checkpoint, virtual-subject schedule, neural field,
  `P_T`, `L_T`, `Delta g`, `q` ledger, ordinal comparator, or alternate model.
- Both phases used `python -m fsrl.formal_runtime` on an NVIDIA GeForce RTX 5090
  with one PyTorch intra-op and one inter-op CPU thread.
- The complete canonical confirmation and `/tmp` replay are byte-identical.
  SHA-256:

```text
8c77ee6c4020705895a39f824b1dba4e3af2a426f086217e8e0e032ac2988b4b
```

## Revised theory and stop/go

The evidence now supports a narrower but useful positive chain:

```text
task-displayed metric relations
    -> stable imperfect effective access
    -> coherent constructive order distribution
    -> human-like nonlearned distance gradient
       + plausible ranking diversity
       + high stable-error prevalence.
```

It rejects the stronger sufficiency chain:

```text
one global i.i.d. access probability
    -> reliable human pair-specific field
       + calibrated perfectly stable errors.
```

The failure is not a reason to shrink the project question or erase the
distance success. It identifies what a successful theory must add: the form of
effective evidence must be structured enough to explain which relations acquire
which confidence, while not simply making erroneous global orders too
deterministic. Existing Liu observations cannot tell whether that structure
comes from metric encoding, graph position, pair identity, or another coupled
factor because those variables were not independently manipulated.

Therefore:

1. do not promote this candidate to the project comparator;
2. do not reopen `P_T -> g_N` or any neural intervention;
3. do not add relation-specific `rho`, refit the three parameters, or run an
   ordinal/compression/Bradley--Terry alternative on the same 37 participants;
4. prospectively design a new two-list behavior experiment with the same signed
   graph and magnitude multiset but counterbalanced magnitude placement across
   support edges.

That experiment can distinguish three mechanisms that the current data cannot:

```text
magnitude placement changes global order
    -> metric-dependent global construction

global order is equivalent but confidence changes
    -> ordinal structure plus metric confidence modulation

both order and confidence are equivalent
    -> evidence for internal ordinalization.
```

Equivalence margins, directional estimands, counterbalancing, exclusion rules,
and sample size must be frozen before data collection. Until then, ordinal
encoding remains a hypothesis rather than a task description or a selected
comparator.
