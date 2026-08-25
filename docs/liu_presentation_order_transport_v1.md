# Liu presentation-order transport v1

## Registered outcome

`LIU_PRESENTATION_ORDER_MECHANISM_TRANSPORTED`

All three prospectively frozen schedules pass all eight registered links
independently in all three mandatory development backbones. This is `9/9`
schedule-by-backbone cells and `72/72` link decisions, with no participant
pooling across schedules or networks and no across-network majority rule.

The result supports this model-side statement within the registered scope:

> On the source-correct Liu graph, the functional division between interacting
> global state `P_T` and exact local edge state `a_T` survives blockwise-random,
> relation-clustered, and reversed presentations of the same physical evidence.
> `P_T` is quantitatively order-sensitive, whereas `a_T` is exactly
> presentation-order invariant under its frozen additive update.

This is a development transport result across three fixed schedules and three
frozen backbones. It is not evidence for arbitrary timing, spacing, recency, or
graph-by-order robustness; a network-population prevalence estimate; behavioral
invariance across schedules; or a human/biological mechanism.

## Frozen manipulation

For each of 77 virtual participants, every condition contains exactly the same
32 passive support observations: eight source-correct Liu relations, four
presentations per relation, the same higher/lower items, signed displayed
magnitudes, left-right orientations, and relation-specific admission values.
Only their temporal positions differ:

| Schedule | Frozen construction |
| --- | --- |
| blockwise random | Original four blocks, each containing every relation once in independently randomized order |
| relation clustered | Stable sort of the same 32 trials by registered relation index, giving four consecutive presentations per relation |
| reverse | Exact reversal of the original 32-trial list |

After each permutation, `block_index=floor(new_trial_index/8)` is reassigned as
metadata. The model still receives the registered trial-index time grid.
Relation-specific gains are rebuilt from the unchanged subject-relation values;
no evidence admission is resampled. Checkpoint, local gain, graph, cue codes,
query schedules, `P_T` update, `a_T` write/read, `tanh`, `W_out`, and temperature
are fixed.

## Provenance and execution

- Preregistration commit: `ff48f0d1aed863a79986b39be53116f6cd727d83`.
- Implementation/source-lock commit:
  `2b062b5c1a2a409f39b98f7221488dbbb8cd3e96`.
- Frozen backbones/local gains: seeds 2101, 2102, and 2103.
- Virtual participants: 77 within every schedule and network.
- Participant bootstrap: 10,000 samples within each schedule and network;
  never pooled.
- Runtime: NVIDIA GeForce RTX 5090, PyTorch `2.13.0+cu130`, CUDA 13.0, with
  PyTorch intra-op and inter-op CPU threads both fixed to one.
- No training, gain adaptation, compilation, parameter update, schedule-specific
  calibration, or result-dependent rerun occurred.

All source, artifact, schedule-multiset, gain-reassembly, GPU-runtime,
tensor-freeze, orientation, finite-value, and exactness gates pass. An
independent execution is byte-identical to the registered result; both have
SHA-256 `d13f0e1ff448cc370a0f0d00c633a8e8a9914968c4f7ea3e73e706d834ca5aba`.

## Primary results

The table gives ranges of point estimates over seeds 2101--2103. Every decision
used its within-cell participant-bootstrap interval, not an across-seed range.

| Schedule | Learned exact accuracy | Nonlearned exact accuracy | Intact Hodge fraction | Hodge tau to true | Inter-subject tau | Stable error >=80% | Global P-LOO remote | Intact - a-off learned probability | P-off/a-on learned probability | P-off/a-on nonlearned probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blockwise random | .9448 | .8091--.8240 | .9886--.9899 | .7059--.7226 | .5401--.5500 | .9054--.9333 | .3123--.3570 | .0145--.0221 | .6605--.7137 | .4457--.4555 |
| relation clustered | .8571--.8831 | .7844--.7877 | .9884--.9906 | .6169--.6299 | .4748--.4882 | .9481--1.0000 | .3342--.3481 | .0220--.0268 | .6605--.7137 | .4457--.4555 |
| reverse | .9367--.9513 | .8110--.8188 | .9881--.9894 | .7004--.7189 | .5074--.5298 | .9041--.9583 | .3134--.3580 | .0151--.0210 | .6605--.7137 | .4457--.4555 |

The worst registered bound across all nine cells remains on the passing side of
every threshold:

| Link component | Worst bound | Registered boundary |
| --- | ---: | ---: |
| intact learned exact accuracy | lower .8279 | lower > .50 |
| intact nonlearned exact accuracy | lower .7552 | lower > .50 |
| intact Hodge fraction | lower .9863 | lower >= .95 |
| a-off Hodge fraction | lower .9941 | lower >= .95 |
| transitive-triplet fraction | lower .9951 | lower >= .95 |
| Hodge tau to true | lower .5677 | lower > 0 |
| inter-subject tau | upper .6002 | upper < .80 |
| stable-error prevalence | lower .8356 | lower >= .80 |
| P-off/a-on nonlearned probability | upper .4736 | upper <= .55 |
| P-off local minus 0.25 global remote | upper -.0534 | upper < 0 |
| global P-LOO remote influence | lower .2939 | lower > 0 |
| global P-LOO third-party fraction | lower .1920 | lower > 0 |
| intact minus a-off learned probability | lower .0105 | lower > 0 |
| P-off/a-on learned probability | lower .6357 | lower > .50 |
| P-off learned minus nonlearned probability | lower .1832 | lower > 0 |

### Functional transport

Every schedule retains competent learned and nonlearned decisions, a nearly
additive and transitive global field, positive Hodge-order alignment to true
rank, and coherent but individualized stable rankings. Removing `a_T` causes a
positive direct learned-pair probability loss. With `P_T` removed and `a_T`
intact, learned probability remains above chance, nonlearned probability stays
below chance, and remote influence remains below one quarter of intact global
remote reassembly. Conversely, the `a`-off branch retains robust disjoint and
third-party relation-LOO effects.

Thus the registered functional asymmetry is not tied to the original
blockwise-random presentation sequence. It also does not imply that behavior or
the global state is numerically order invariant.

### Exact local algorithm

Within every schedule, the common-float64 edge ledger reconstructs the frozen
tensor state to at most `5.55e-16` and all 28 Gram reads to at most
`8.44e-15`. More strongly, both nonbaseline schedules have exactly zero
difference from blockwise random in the terminal ledger and every query read
for every participant and backbone.

The local computation is therefore exactly commutative under the registered
schedule permutations:

`a_(t+1) = a_t + s_t^L e_(r_t)`,

`ell = K a_T`.

This proves invariance of the frozen local edge-plus-Gram algorithm, not
uniqueness of its tensor-product address or biological order invariance.

### Quantitative global order sensitivity

The high-dimensional recurrent `P_T` computation is not order invariant, as
prospectively allowed. Relative to blockwise random, its frozen `a`-off output
field changes most under relation clustering:

| Schedule versus baseline | Field Pearson | Centered RMSE | Exact decision agreement | Hodge-potential tau | Change in remote LOO |
| --- | ---: | ---: | ---: | ---: | ---: |
| relation clustered | .8288--.8365 | 1.0769--1.1569 | .8715--.8785 | .7393--.7662 | -.0101--.0315 |
| reverse | .9593--.9646 | .4356--.4750 | .9434--.9527 | .8915--.8970 | -.0004--.0011 |

Relation clustering reduces learned exact accuracy and Hodge-to-true alignment
more than exact reversal, while still preserving all competence and mechanism
gates. This supports a state-dependent iterative global computation whose
quantitative allocation depends on temporal trajectory, alongside an exactly
additive local evidence ledger. The result does not identify which recurrent
state variable causes this order sensitivity.

## Secondary behavior and limitations

| Schedule | Sampled learned accuracy | Sampled nonlearned accuracy | Symbolic-distance slope | Serial endpoint contrast |
| --- | ---: | ---: | ---: | ---: |
| blockwise random | .9222--.9284 | .8003--.8088 | .0498--.0509 | .0419--.0529 |
| relation clustered | .8492--.8640 | .7794--.7808 | .0480--.0519 | .0522--.0785 |
| reverse | .9154--.9292 | .7994--.8052 | .0490--.0497 | .0466--.0509 |

These quantitative schedule effects were not acceptance gates and were not
calibrated to new human data. The excessive symbolic-distance slope, weak
serial-position endpoint, and original seed-2104 inconsistency remain known
limitations. Nothing here licenses tuning temperature, `W_out`, `P_T`, `a_T`,
evidence admission, or schedule-specific parameters.

## Theory update and next test

Together with topology transport, the working model chain now survives both
which sparse Liu relations are observed and how a fixed evidence multiset is
ordered:

`Liu-style sparse evidence -> differential admission ->`

- interacting `P_T` for coherent, individualized, remote/global assembly; and
- exact additive `a_T` for query-addressed direct-experience fidelity.

The next Liu internal-validity axis is evidence sparsity, not list linking.
Freeze connected eight-item graphs with `|E| = 7, 8, 9, 10` before execution,
while keeping item count, four presentations per observed relation, task
interface, evidence, and model fixed. The test should distinguish transport of
the functional asymmetry from the prospective quantitative prediction that
sparser evidence increases reliance on `P_T` and denser evidence increases the
fraction of queries directly covered by `a_T`. Item count follows only after
sparsity. List linking, classic transitive inference, Miconi ancestry, MEG, and
human-mechanism validation remain deferred.
