# Frozen confirmation protocol

`07591af56ccd693307ab90cc72b03335cb4cd6c9` is the development freeze. Its
single checkpoint and all `liu-constructive-ranking-v1` outputs remain
developmental evidence; they are not eligible confirmation seeds.

## Source correction and human gate

Liu, Wang, and Luo define the objective order as `A < ... < H`. The development
protocol reversed that order without reflecting the retained graph. The
confirmation protocol therefore uses `benchmarks/liu_v2.json`, with `H` highest
and the eight source relations directed accordingly. Meta-training excludes both
the source-correct rank graph and its rank-axis reflection. This is stricter than
the v1 held-out declaration and prevents the reflected development graph from
entering the confirmation distribution.

`benchmarks/liu_human_exact_v1.json` is regenerated from the two public OSF
trial files before any confirmation training. It records SHA-256 checks for the
raw trials and the released Figure 2d/3b source data, exact combined accuracies,
the published 8/64/5 rank classes, stable-error prevalence, and a fixed 10,000
sample participant bootstrap.

Two source-level ambiguities are preserved rather than silently tuned away:

- A transparent majority-vote circular-triad reconstruction matches 75/77
  released Figure 3b labels. IDs 30 and 38 contain majority ties and are labeled
  inconsistent in the released figure, but the paper does not specify the tie
  rule that produces those two labels. Released labels are the reproduction
  target; reconstructed labels remain in the artifact as a diagnostic.
- The released Figure 2d Beta parameters for 27/28 pairs exactly match clipping
  endpoints to 0.01/0.99 before fixed-support MLE. B-H instead exactly matches
  0.001/0.999. Both the uniform recomputation and the released classifications
  are retained.

The human gate must pass before training:

```bash
direnv exec . python -m fsrl.human_benchmark
direnv exec . python -m fsrl.confirmation validate
```

## Frozen neural procedure

The machine-readable formal contract is `benchmarks/confirmation_v1.json`. It
was frozen only after the two-seed workflow pilot described below. It fixes:

- fresh formal seeds 2001–2010, all mandatory and never filtered;
- the existing architecture and optimizer configuration;
- exactly 1,000 outer steps and only the final checkpoint;
- stable omission during meta-training;
- the v2 causal and antisymmetric-geometry gates;
- a common 77-subject evaluation cohort and common evaluation seeds;
- query temperature 0.25, with no seed-specific refit;
- exact-posterior temperature 0.05 for probabilities (MAP itself is invariant
  to this positive temperature).

Run one registered seed on a CUDA-visible host with:

```bash
direnv exec . python -m fsrl.formal_runtime confirmation run-seed --seed 2001
```

After all ten registered seeds finish, aggregate without optional omissions:

```bash
direnv exec . python -m fsrl.formal_runtime confirmation aggregate \
  --output results/confirmation_v1.json
```

Aggregation fails if any seed is absent. It reports causal, behavioral,
geometry, and joint pass proportions plus full seed-level distributions. The
behavioral criterion is fixed before training: every registered scalar estimate
must lie inside its human participant-bootstrap 95% interval. Standardized
discrepancies and Beta class counts are also reported.

## Two-seed workflow pilot

Seeds 1901 and 1902 are registered separately in `benchmarks/pilot_v1.json`.
They were deliberately limited to two after resource and protocol validation.
The pilot exposed and fixed a v1 assumption in deterministic causal scoring:
canonical item order had been treated as true rank order. Because this was
corrected after pilot training began, neither pilot seed may be counted toward
formal confirmation.

Both pilot seeds passed the frozen causal and antisymmetric-geometry gates.
Both failed the deliberately strict behavioral rule for exactly one reason:
symbolic-distance slopes 0.0466 and 0.0476 exceeded the human bootstrap upper
bound 0.0449. The other eight registered scalar comparisons fell within their
human intervals. The full, unfiltered pilot aggregate is
`results/pilot_v1.json`.

The observed workload consists of many small sequential recurrent CUDA kernels.
A single process uses about 5 GB on an RTX 5090 and remains dispatch-bound.
Formal execution therefore uses `fsrl.formal_runtime`, which requires a visible
GPU and fixes PyTorch intra-op and inter-op CPU work to one thread. The
unbounded default created 74 operating-system threads and consumed about 17.6
CPU cores while GPU utilization was only about 23%. The bounded training runtime
averaged about one CPU core while retaining GPU execution. NumPy and BLAS thread
settings deliberately remain unchanged: forcing every CPU library to one thread
changed byte-level assembly and factor-swap development results, whereas the
PyTorch-only bound reproduced all three frozen mechanism results exactly.

Formal training also batches all time-step inputs for one trial into one
contiguous CPU-to-GPU transfer and compiles `RetroModulRNN` with
`torch.compile(net, fullgraph=True)`. The mode argument is deliberately omitted,
so PyTorch uses its default mode and Inductor backend. A source-locked execution
record in every checkpoint and seed summary identifies this configuration, the
PyTorch/CUDA/device environment, and the training implementation hash.

The resource defect was found after seeds 2001--2003 completed and seed 2004
started, before any scientific outcome was inspected. That execution was
aborted and its ignored artifacts were archived under
`output/aborted-confirmation-v1-unbounded-threads-20260824/`; none are eligible
for aggregation. A seed-2004 replay with the bounded runtime reproduced the
first 184 training-log records byte for byte (identical SHA-256) and completed
those steps in 28.26 seconds at 105% CPU. Trial-level input batching preserved
all 184 records byte for byte and reduced the run to 23.98 seconds. Adding
full-graph default compilation reduced it further to 17.47 seconds. Complete
compiled development seeds 1901 and 1902 took 88.52 and 87.49 seconds for 1,000
steps. Across both seeds, task and accuracy branches never changed, the maximum
training-statistic difference was `9.5367431640625e-7`, and the maximum
checkpoint parameter difference was `2.682209014892578e-7`. Joint reruns of all
three frozen mechanism analyses differed by at most `1.430511474609375e-6`,
below the registered `3.814697265625e-6` tolerance, with unchanged confirmation
decisions. All ten formal seeds must therefore be rerun from scratch through the
bounded compiled entry point. Future implementation changes still require one
to three development seeds and numerical-equivalence checks before formal use.

## Matched controls and algorithmic interpretation

Each checkpoint is evaluated under stable omission, presentationwise omission,
blockwise omission, and a uniform no-bottleneck control. The last uses the
cohort's mean registered reliability as one deterministic gain for every subject
and relation. Because every relation occurs exactly once in each support block,
blockwise and presentationwise resampling have the same statistical granularity
in this task. They are retained to satisfy the planned control family but cannot
be treated as two independent mechanistic contrasts.

For stable omission, `fsrl.algorithmic` enumerates all 40,320 rankings from the
exact realized evidence and compares the frozen neural order with the MAP set,
posterior pair probabilities, and posterior-supported neural choices. This is a
descriptive algorithmic analysis, not a qualification gate. A close match would
support approximate global inference; a systematic mismatch would identify a
learned neural inductive bias. Neither result alone changes the fast-weight
necessity claim.

## Claim boundary

The confirmation tests whether a shared meta-learned relational prior, combined
with a stable subject-specific evidence bottleneck, constructs an episode-local
fast-weight state that is necessary for subject-specific query-policy selection.
It does not test whether fast plasticity creates transitivity from nothing, nor
whether the full hidden-state manifold represents subjective rank. The registered
geometry claim remains limited to the antisymmetric relational component.
