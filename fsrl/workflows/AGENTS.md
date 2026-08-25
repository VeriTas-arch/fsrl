# Workflow implementation guide

This file applies to `fsrl/workflows/`, the code that validates and renders
repository-level scientific workflow and figure contracts.

Navigation: [package guide](../AGENTS.md) ·
[workflow manifest guide](../../workflows/AGENTS.md) ·
[synthesis guide](../../synthesis/AGENTS.md)

- Validators must fail on dangling studies, records, JSON pointers,
  implementation paths, test paths, dependency edges, verification commands,
  or figure references.
- Verification commands are structured argument lists with declared resources;
  do not store opaque shell pipelines as the machine contract.
- Rendering is deterministic and generated README files remain human views of
  machine-readable authorities.
- Frozen-evidence commands operate on versioned snapshot roots and verify
  hashes. They must not mutate snapshots or select a different historical
  source silently.
- Paper-figure checks validate source data, panel specifications, render
  manifests, and declared outputs without treating a visually plausible image
  as sufficient provenance.

Schema changes require validator tests for both accepted and rejected inputs,
regeneration of affected human views, and repository registry/figure checks.
