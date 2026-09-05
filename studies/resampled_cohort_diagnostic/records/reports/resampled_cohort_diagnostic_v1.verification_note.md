# Publication-order verification note

The scientific result and report were published in commit
`1d822ce33c483702be612507d9495b557a610360` after independent reconstruction of
all 400 cohorts for each of the three fits. Those records remain byte-preserved.

The subsequent portable command
`direnv exec . python -m fsrl.infra.formal_runtime resampled-cohort-diagnostic verify-record`
exited with `RuntimeError: cohort diagnostic report differs`. It had already
passed every shard's independent recurrence/behavior reconstruction and exact
summary comparison before reaching the final report comparison. This failed
verification invocation is retained here, not reclassified as a successful run.

The cause is serialization order, not numerical disagreement. `publish()`
rendered the report from the summary's original dictionary insertion order,
whereas the result JSON was written with sorted keys. `verify_record()` renders
the loaded sorted dictionaries: continuous rows move and the displayed Wilson
dictionaries reorder their keys. The frozen source therefore cannot reproduce
its own report text through that original command, even when all values match.

The separately versioned read-only audit is:

```bash
direnv exec . python -m tools.provenance.verify_resampled_cohort_v1
```

Run it from a clean, pushed `dev` descendant. It keeps the original source and
input lock checks, reconstructs all saved shards and their numerical witnesses,
and compares all newly reconstructed summary values to the saved result. Only
then does it render using the freshly reconstructed summary's original order,
exactly as publication did, and compare the full report text. It does not
sort report lines, discard text differences, relax numerical tolerances,
rewrite any report or modify a scientific implementation source.

Regression coverage in `tests/infra/test_resampled_report_audit.py` reproduces
the sorted-JSON mismatch and requires both changed statistics and changed report
text to fail. The audit is an evidence-packaging correction, not a new
scientific estimand, new evaluation cohort, revised result or main-model gate.

For the readable scientific interpretation, see
[the original interpretation](resampled_cohort_diagnostic_v1.interpretation.md).
