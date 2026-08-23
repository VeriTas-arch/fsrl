# Constructive-ranking mechanism development

This document records the implementation contract on the `dev` branch. It is a
development protocol, not a positive scientific result.

## Claim boundary

The implemented pipeline tests the following candidate mechanism:

> A population-shared meta-learned update rule uses episode-local recurrent
> plasticity to assemble sparse, stably distorted relation evidence into a
> query-order-invariant global ranking state.

The current code establishes executable estimands and qualification gates. It
does **not** yet establish that a trained checkpoint passes those gates, matches
the Liu et al. human distribution, or reproduces subjective neural geometry.

## Registered components

- `benchmarks/liu_v1.json` freezes the eight-item, 32-observation passive
  learning phase and 280 no-feedback queries, together with the human targets.
- `fsrl.constructive.ExactRankingPosterior` enumerates all `8! = 40320` global
  orders. Relation reliability changes the likelihood; one committed order is
  read without query-by-query state changes.
- `fsrl.meta_tasks.GenericRankingTaskGenerator` samples connected 7--10-edge
  ranking graphs and rejects the Liu graph. Item codes and true rank
  permutations vary by episode.
- `fsrl.subject_encoding.SubjectEncodingState` is sampled once per virtual
  subject. Its baseline, item salience, and distance slope only attenuate
  observed relation evidence; it contains neither the true nor subjective
  order, and remains fixed across all four support blocks.
- `fsrl.meta_train` uses supervised outer-loop cross entropy. Query labels enter
  only the outer loss, never the episode inputs or plastic update. There is no
  query feedback, and query fast weights are frozen.
- `fsrl.liu_eval` resets ordinary hidden state and eligibility on every query
  while reusing a fixed post-learning `P_T`. It evaluates intact, plastic-write
  off, `alpha=0`, post-learning reset, and cross-subject `P_T` shuffle. Accuracy
  is balanced across both orientations of every pair, so a fixed left/right
  response bias cannot pass the gate.
- `benchmarks/qualification_v1.json` registers the GO/NO-GO thresholds. The
  checkpoint's sibling `config.json` must match its SHA-256 and record that the
  Liu graph was held out.

The `permuted_shared` cue mode gives each batch member a different item-to-code
mapping from the same codebook. It is required for cross-subject `P_T` shuffle
to be a meaningful mismatch intervention; it is not itself a model of human
individuality. The stable encoding state is the explicit individuality
bottleneck in this version.

## Run the pipeline

Use the repository environment for every command:

```bash
env CUDA_VISIBLE_DEVICES=-1 direnv exec . \
  python -m unittest discover -s tests -v
```

For a short infrastructure check:

```bash
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  direnv exec . python -m fsrl.meta_train \
  --output-dir output/dev-smoke \
  --outer-steps 1 --batch-size 2 --hidden-size 8 --cue-size 8 \
  --seed 51 --save-every 1
```

For an actual training run, remove the smoke-test size overrides and register
the chosen seed and output directory:

```bash
direnv exec . python -m fsrl.meta_train \
  --output-dir output/meta-seed-1 \
  --outer-steps 30000 --batch-size 32 --hidden-size 200 \
  --cue-size 15 --seed 1 --save-every 500
```

Run all causal conditions using the zero-feedback checkpoint:

```bash
direnv exec . python -m fsrl.liu_eval \
  --checkpoint output/meta-seed-1/net.dat \
  --output output/meta-seed-1/liu-causal.json \
  --batch-size 77 --cue-mode permuted_shared \
  --subject-encoding stable_bottleneck \
  --cue-seed 1 --support-seed 100 --subject-encoding-seed 300 \
  --order-seed 200 --order-schedules 8
```

Apply the registered gate:

```bash
direnv exec . python -m fsrl.qualification \
  --result output/meta-seed-1/liu-causal.json \
  --output output/meta-seed-1/qualification.json
```

The qualification command exits with status 1 when any gate fails. An
infrastructure smoke run is expected to fail the scientific gate.

## GO/NO-GO sequence

1. **Plastic-state necessity:** intact nonlearned performance and transitive
   structure must pass their lower bounds; all four interventions must fall
   below their registered upper bounds.
2. **Strict query-order invariance:** maximum paired logit change across query
   orders must be at most `1e-6`.
3. **Held-out graph generalization:** the checkpoint hash must match registered
   training metadata and `liu_graph_held_out` must be true.
4. **Human fitting:** only after 1--3 pass, compare correct-ranker counts,
   stable self-consistent errors, pair-level modes, symbolic-distance effects,
   and inter-subject Kendall structure against the registered human targets.
5. **Neural geometry:** only after the behavioural fit passes, test whether
   erroneous virtual subjects' item geometry aligns more closely with their
   committed subjective order than with ground truth. This analysis is not yet
   implemented because it is downstream of the competence and causal gates.

## Outstanding external artifact

No checkpoint identified as `G7-F` is registered in the fetched repository
branches. If that checkpoint is supplied, run `fsrl.liu_eval` with
`--subject-encoding none` first to audit the original G7 mechanism semantics.
That legacy audit can establish plastic-state necessity and query-order
invariance, but it cannot satisfy the new model's held-out-graph provenance
gate unless matching training metadata is also supplied.
