# Provenance and repository audits

This directory contains versioned, repository-only provenance, migration, and
quality tools. Reusable runtime APIs belong in `fsrl.infra`; these commands
audit the checkout and do not become package APIs.

## Registry and file integrity

Check registered evidence, external data, runtime manifests, and the generated
record catalog:

```bash
direnv exec . python -m fsrl.infra.study_registry check
direnv exec . python -m fsrl.infra.file_contracts check
direnv exec . python -m tools.provenance.build_record_catalog_v2
```

The registry check validates ownership, hashes, migration chains, generated
human views, and current locators. Structural PASS is not scientific support.

The current claim graph and paper figures have separate checks:

```bash
direnv exec . python -m fsrl.workflows check \
  workflows/relational_model/workflow.toml
direnv exec . python -m fsrl.workflows.paper_figures check
```

## Migration audits

Audit historical sources, active locators, and Git-backed provenance without
rewriting frozen records:

```bash
direnv exec . python tools/provenance/migrate_flat_records_v1.py audit
direnv exec . python tools/provenance/rewrite_runtime_locators_v1.py audit
direnv exec . python tools/provenance/rewrite_active_record_paths_v1.py check
direnv exec . python tools/provenance/index_source_provenance_v1.py check
direnv exec . python -m tools.provenance.backfill_run_manifests_v1
```

A physical evidence move requires a new append-only migration map and exact
source-commit, byte, hash, and active-locator verification. Never rewrite an
older migration to make the current layout appear direct.

## Frozen cohort publication

The frozen Resampled cohort publication has a read-only reconstruction audit:

```bash
direnv exec . python -m tools.provenance.verify_resampled_cohort_v1
```

It checks all saved numerical outputs and reconstructs the original pre-JSON
report order. See the [verification note](../studies/resampled_cohort_diagnostic/records/reports/resampled_cohort_diagnostic_v1.verification_note.md)
for the original verifier's serialization-order failure and preserved boundary.

## Refactor equivalence

The current closing audit is version 3:

```bash
direnv exec . python -m tools.provenance.audit_refactor_equivalence_v3
```

Versions 1 and 2 remain executable because they audit their own pinned
candidates:

```bash
direnv exec . python -m tools.provenance.audit_refactor_equivalence_v2
direnv exec . python -m tools.provenance.audit_refactor_equivalence_v1
```

None of these commands claims fresh cross-commit replay beyond the coverage and
exclusions in its versioned contract.
