# Maintained causal evaluation

This directory owns frozen rollout, causal interventions, metrics, sampling,
and registered checkpoint evaluation. Current evaluation may use the maintained
backend, while a frozen study must retain the backend named by its execution
record.

## Run the causal qualification suite

```bash
direnv exec . python -m fsrl.evaluation \
  --checkpoint artifacts/runs/relational_model/seed-1/net.pth \
  --output artifacts/runs/relational_model/seed-1/evaluation.json \
  --evaluation-backend batched_sequence
```

The current high-level default is `batched_sequence`. It runs one compiled
sequence per support trial and transfers the complete query batch together.
Frozen studies with `legacy_stepwise` records are not silently upgraded. Each
result retains its execution profile and observed runtime snapshot.

Outputs remain runtime artifacts until a registered protocol, provenance
record, outcome, and exact claim boundary promote the required summary into a
study. See the [artifact contract](../../artifacts/README.md).

## Backend parity and performance

Before a large prospective run, compare the maintained and legacy backends:

```bash
direnv exec . python -m fsrl.evaluation.performance \
  --warmups 1 --repeats 3 \
  --output /tmp/fsrl-frozen-evaluation-benchmark.json
```

The benchmark is an engineering diagnostic, not scientific evidence and not a
hardware-independent threshold. Exact and bounded equivalence claims remain in
their versioned contracts under the
[relational-model workflow](../../workflows/relational_model/README.md).
