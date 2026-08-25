# Constructive-ranking mechanism development

This document records the implementation contract and current evidence on the
`dev` branch. The tracked candidate is a single-seed development result, not a
replication distribution.

## Current claim boundary

The current development result is consistent with this narrow candidate
mechanism:

> A population-shared meta-learned update rule uses episode-local recurrent
> plasticity to select an episode-specific, query-order-invariant ranking policy
> from sparse relation evidence with stable subject-level omissions.

For coherent-error virtual subjects, the **antisymmetric relational component**
of frozen query hidden states is closer to that subject's inferred ranking than
to the true ranking or other subjects' rankings. The context-averaged hidden
geometry is negative, so the result does not establish that the whole hidden
state is organized by subjective rank. Ablated networks can retain a generic
transitive response scaffold, so the result also does not show that plastic
state creates transitivity from nothing.

## Versioned components

- `benchmarks/liu_v1.json` freezes the eight-item, 32-observation passive phase,
  280 no-feedback queries, and reported human targets.
- `fsrl.constructive.ExactRankingPosterior` enumerates all `8! = 40320` orders.
- `fsrl.meta_tasks.GenericRankingTaskGenerator` samples connected 7--10-edge
  ranking graphs and rejects the Liu graph. Item codes and ranks vary by
  episode.
- `stable_omission` samples which support relations a virtual subject retains
  once per episode and holds that mask fixed across all four blocks. The older
  continuous `stable_attenuation` mode remains available as a negative-result
  control.
- `fsrl.meta_train` uses query labels only in the supervised outer loss. Query
  inputs contain no label or feedback, ordinary hidden state and eligibility
  reset on every query, and the post-support fast weight `P_T` is frozen.
- `benchmarks/qualification_v2.json` requires held-out provenance, intact
  competence, exact query-order invariance, chance-level nonlearned accuracy,
  and low decision agreement after write-off, `alpha=0`, reset, or cross-subject
  shuffle. It is a developmental gate revised after inspecting v1 ablations.
- `fsrl.behavioral` samples the registered 280-query protocol from frozen logits
  and measures rank class, circular triads, stable errors, beta pair classes,
  symbolic distance, and inter-subject Kendall structure.
- `benchmarks/human_fit_v1.json` freezes a four-value global-temperature grid
  and uses only overall accuracy for selection. The grid was frozen after
  inspecting `T=1.0` and `T=0.5`, but before `T=0.75` and `T=0.25`; it is not a
  confirmation-stage preregistration. All metrics other than overall accuracy
  were excluded from the selection rule. Because the accuracy target is an
  approximate figure read and no tolerance is registered, this step reports no
  formal PASS.
- `fsrl.geometry` reconstructs item vectors from half the orientation contrast
  `h(i,j)-h(j,i)` using a vector Hodge solve. The older context-average estimator
  is retained as a negative control.

## Tracked candidate and current result

The candidate checkpoint is
`checkpoints/dev-v2-seed1801-step1000/net.dat` (SHA-256
`0fb9f063ba8e35b0d94c5a7ed5b6bf8c80d1ed963baebe3d823176aa7653d690`).
Its machine-readable summary is `results/dev_v2_seed1801_step1000.json`.

- Causal qualification: **developmental-gate PASS**. Intact
  overall/nonlearned accuracy is
  `0.860/0.840`; all four interventions reduce nonlearned accuracy to
  `0.488--0.497` and pair-decision agreement with intact to `0.485--0.512`.
  Query-order maximum absolute logit change is exactly `0`.
- Human comparison: **descriptive only**. The registered rule selects
  temperature `0.25`: overall/learned/nonlearned accuracy is
  `0.854/0.895/0.837`, rank classes are `9/59/9`, symbolic-distance slope is
  `0.0426`, stable-error prevalence is `0.853` at at least 80% and `0.735` at
  100%, and beta classes are `11` high-accuracy, `17` bimodal, `0` ordinary.
- Geometry: **exploratory-gate PASS for the antisymmetric estimator**. Across 59
  self-consistent incorrect subjects, mean subjective-minus-true Spearman is
  `0.224`, subjective-minus-other is `0.444`, and the one-sided sign-test
  `p=8.77e-11`.
- Context-average control: **negative**. Subjective-minus-true is `0.048` and
  the sign-test `p=0.217`.

## Negative evidence that changed the design

The first continuous-attenuation checkpoint achieved high intact accuracy but
mostly produced low-confidence stochastic errors. It did not reproduce stable
wrong ranks or beta bimodality. This motivated fixed relation omission, which
creates repeatable informational differences without injecting a rank label.

The first causal gate required ablations to lose transitivity. That was too
strong: the frozen base network can retain a generic scalar, transitive response
scaffold even when it no longer selects the intact subject-specific ordering.
Version 2 therefore tests both chance-level correctness and decision mismatch
to intact. The claim is necessity for the episode-specific policy, not creation
of every transitive tendency. Because this revision used v1 diagnostics, it
must be validated on new seeds before being treated as confirmatory.

The first geometry estimator averaged an item's hidden state across both
left/right roles and partners. It was non-significant and can cancel an
antisymmetric relational signal. The version-2 Hodge estimator is primary; the
failed context-average result remains reported rather than being overwritten.
Because v2 was proposed after this negative result on the same checkpoint, its
current pass is exploratory and requires a fresh-seed confirmation.

## Reproduce the registered checks

Use the repository environment. The code selects CUDA only when
`torch.cuda.is_available()` is true; restricted execution environments can hide
the GPU and silently cause CPU execution, so verify device visibility in the
same execution context as training:

```bash
direnv exec . python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")'
direnv exec . python -m fsrl.test_runtime
```

Evaluate the tracked checkpoint:

```bash
direnv exec . python -m fsrl.liu_eval \
  --checkpoint checkpoints/dev-v2-seed1801-step1000/net.dat \
  --output artifacts/runs/dev-v2/liu-causal.json \
  --batch-size 77 --cue-mode permuted_shared \
  --subject-encoding stable_omission \
  --cue-seed 1802 --support-seed 1803 --subject-encoding-seed 1804 \
  --order-seed 1805 --order-schedules 8

direnv exec . python -m fsrl.qualification \
  --result artifacts/runs/dev-v2/liu-causal.json \
  --output artifacts/runs/dev-v2/qualification.json
```

Generate four behavioral files using temperatures `1.0`, `0.75`, `0.5`, and
`0.25`, with otherwise identical arguments:

```bash
direnv exec . python -m fsrl.behavioral \
  --checkpoint checkpoints/dev-v2-seed1801-step1000/net.dat \
  --output artifacts/runs/dev-v2/behavior-temp025.json \
  --batch-size 77 --cue-seed 1802 --support-seed 1803 \
  --subject-encoding-seed 1804 --choice-seed 1806 \
  --temperature 0.25 --subject-encoding stable_omission
```

Pass all four outputs to the registered selector, then use only the selected
behavior for geometry:

```bash
direnv exec . python -m fsrl.human_fit \
  --behavior artifacts/runs/dev-v2/behavior-temp100.json \
  --behavior artifacts/runs/dev-v2/behavior-temp075.json \
  --behavior artifacts/runs/dev-v2/behavior-temp050.json \
  --behavior artifacts/runs/dev-v2/behavior-temp025.json \
  --output artifacts/runs/dev-v2/human-fit.json

direnv exec . python -m fsrl.geometry \
  --checkpoint checkpoints/dev-v2-seed1801-step1000/net.dat \
  --behavior artifacts/runs/dev-v2/behavior-temp025.json \
  --output artifacts/runs/dev-v2/geometry.json
```

## Remaining work before a paper-level claim

1. Freeze numeric human targets from source data rather than figure reads and
   register tolerances before any further fit.
2. Run a predeclared multi-seed replication and report the seed distribution,
   including failures; the tracked checkpoint is only seed 1801.
3. Add matched controls separating fixed relation omission from alternative
   stable subject bottlenecks.
4. If the external `G7-F` checkpoint becomes available, audit it separately.
   No matching checkpoint was found in any fetched remote branch, and legacy
   G7 artifacts cannot satisfy held-out-training provenance without metadata.
