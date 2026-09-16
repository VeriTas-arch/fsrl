# Exact P/L retained cross-talk decomposition

## Result

All locked inputs and exact-algebra checks passed. The gain-free omitted-write cross-talk was identical across seeds, and its source sum reconstructed the dual-minus-shared local and probability effects within the frozen tolerances.

The parent result is unchanged: seed 3006 remains a formal retained-fidelity failure, and the three-seed P/L replication remains `competent_alternative_organization`.

## Seed decomposition

| seed | local gain | retained mean effect | bootstrap lower | shared-total sensitivity | P-only sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3004 | 0.280311 | -0.2180% | -0.4289% | 0.038501 | 0.047889 |
| 3005 | 0.287231 | -0.1927% | -0.3667% | 0.032027 | 0.040846 |
| 3006 | 0.289113 | -0.2755% | -0.5051% | 0.031642 | 0.042294 |

The exact probability operating point is the shared total margin (`P + shared L`); the P-only sensitivity is shown separately and was not used for reconstruction.

## Gain x operating-point factorial

Each cell is the retained mean effect, with `pass` referring only to the historical -0.005 lower-bound reference. It does not revise the parent decision.

| operating seed | gain 3004 | gain 3005 | gain 3006 |
| --- | ---: | ---: | ---: |
| 3004 | -0.2180% (pass) | -0.2242% (pass) | -0.2258% (pass) |
| 3005 | -0.1877% (pass) | -0.1927% (pass) | -0.1941% (pass) |
| 3006 | -0.2658% (pass) | -0.2735% (fail) | -0.2755% (fail) |

Registered factorial classification: `joint_gain_by_operating_point_boundary`.

The Shapley splits below attribute the difference in retained mean effect exactly within the fixed factorial; they do not explain how training produced either component.

| comparison | gain component | operating-point component | total |
| --- | ---: | ---: | ---: |
| 3004->3006 | -0.0088% | -0.0488% | -0.0575% |
| 3005->3006 | -0.0017% | -0.0811% | -0.0828% |

## Source pattern

The source analysis is descriptive rather than dichotomized. Participant-weighted medians were:

- top-one absolute share: 0.768
- top-two absolute share: 1.000
- effective source count: 1.522
- cancellation ratio: 0.824
- sources needed for 80% absolute mass: 1.500

## Interpretation boundary

The retained cost is completely routed through omitted writes and nonorthogonal fixed addresses. Across-network variation is not a change in address geometry: it enters through the learned scalar gain and the frozen shared-margin operating point. This establishes deterministic component attribution, not a causal account of joint training.

No new training, quantitative-human fitting, or N transport was run. Given that the dual-store computation is already functionally informative but structurally non-necessary, this diagnostic does not justify further P/L repair by itself; the clean P/L model can remain the functional-decomposition model while work returns to the single-P structural-sufficiency mainline.
