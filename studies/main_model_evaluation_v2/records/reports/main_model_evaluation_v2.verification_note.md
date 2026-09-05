# Publication-order verification note

The scientific result, portable arrays and reports were published in commit
`032bd09`. They remain byte-preserved.

The original locked command
`direnv exec . python -m tools.provenance.main_model_evaluation_v2 verify-record`
exited with `RuntimeError: saved main-model evaluation report differs`. Before
that final comparison it had reconstructed all 1,200 fit-by-cohort evaluations,
matched the registered integer arrays exactly and matched every saved summary.
The failed invocation is not reclassified as a successful run.

The cause is serialization order, not a scientific or numerical disagreement.
`publish()` rendered dictionaries in their computation insertion order, with
`mean` before `interval`. The result JSON was then serialized with sorted keys;
the original verifier loaded that order and rendered `interval` before `mean`,
so an exact text comparison necessarily failed even though all values agreed.

The separately versioned read-only audit is:

```bash
direnv exec . python -m tools.provenance.verify_main_model_evaluation_v2
```

It was added and pushed in commit `8b8d018`, outside the frozen scientific
source inventory. On the complete registered matrix it returned PASS with all
three statements true: arrays exactly reconstructed, summaries exactly
reconstructed, and report bytes exactly reconstructed. It restores the freshly
rebuilt summary's original insertion order only for rendering; it does not sort
report lines, alter a value, relax a tolerance, rewrite a frozen record, add a
cohort or compute a new estimand.

Regression coverage reproduces the sorted-JSON mismatch and requires changed
summary values or changed report text to fail. This is an evidence-packaging
audit, not a revised result, new model evaluation, confirmation of the exposed
core threshold or authorization for training.
