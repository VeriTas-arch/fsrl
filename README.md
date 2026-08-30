# Meta-training plastic networks for relational learning

This repository develops and audits a plastic recurrent model that transforms
sparse, partially retained evidence into global relational structure and
query-specific direct fidelity. It began from the code accompanying Miconi &
Kay; the upstream source, supplied checkpoints, and maintained teaching route
now live in an isolated reproduction capsule.

## Start here

- [Current interpretation](synthesis/manuscript/relational_model/README.md) is
  the reader-first account of the model, causal evidence, retained negative
  results, and exact claim limits.
- [Relational model mainline](workflows/relational_model/README.md) is the
  generated human view of the machine-readable claim-to-code-to-evidence
  route.
- [Study registry](studies/README.md) owns atomic questions, estimands,
  protocols, results, and claim boundaries.
- [Current synthesis](synthesis/README.md) organizes diagnostic history,
  closed candidate families, unresolved boundaries, figures, and reporting.
- [Research discussions](discussions/README.md) records literature connections
  and proposed tests without becoming scientific evidence.
- [Code and setup](fsrl/README.md) describes the maintained package,
  installation, and architecture.
- [Tests](tests/README.md) and [provenance tools](tools/README.md) document the
  repository validation routes.
- [Original-paper reproduction](reproductions/relational_learning_2024/README.md)
  isolates the external source and teaching workflow from the current model.

The current synthesis has been reviewed against the workflow and registry.
That review state does not promote evidence or alter an atomic study outcome.
For an auditable claim, follow interpretation → workflow → study-owned record.

## Repository map

| Area | Directories | Responsibility |
| --- | --- | --- |
| Code and execution | [`fsrl/`](fsrl/README.md), [`tests/`](tests/README.md) | Maintained package, training, evaluation, analysis, workflows, and regression contracts |
| Evidence and interpretation | [`studies/`](studies/README.md), [`workflows/`](workflows/relational_model/README.md), [`synthesis/`](synthesis/README.md), [`discussions/`](discussions/README.md) | Atomic evidence, current claim graph, reporting, and non-authoritative interpretation |
| Inputs and reproduction | [`data/`](data/README.md), [`reproductions/`](reproductions/relational_learning_2024/README.md) | Immutable external data and isolated external-paper materials |
| Operations | [`artifacts/`](artifacts/README.md), [`tools/`](tools/README.md) | Ignored execution outputs, provenance checks, and versioned migrations |

Top-level `workflows/` contains repository claim contracts, while
`fsrl/workflows/` contains the code that validates and renders them. Likewise,
`tools/` contains repository-only migration utilities; reusable runtime and
provenance APIs remain in `fsrl/infra/` and are included in the package.

## Current scientific snapshot

The registered evidence supports a working computational division:

- a meta-learned fast-weight state assembles global relational structure;
- a causally distinct query-addressed state preserves direct evidence with a
  broader admission rule;
- registered ablations separate global inference from direct local fidelity;
- global-policy slope, some transport boundaries, and compression into a
  minimal algorithm remain qualified or unresolved.

This is navigation, not a substitute for registered estimands and controls.
The model-level evidence and one-factor transport program are frozen for
reporting; do not treat documentation or a successful command as authorization
for a new scientific estimand.

## Quick start

The repository uses Python 3.12 and an editable installation. After creating
the `ipex` Conda environment described in the
[package setup guide](fsrl/README.md#development-installation):

```bash
direnv allow
direnv exec . python -m pip install -e ".[reproduction,test]"
direnv exec . python -m fsrl.infra.test_runtime
```

`pyproject.toml` is the single dependency and tool-configuration authority.
Use the dedicated guides for
[prospective training](fsrl/training/README.md),
[maintained evaluation](fsrl/evaluation/README.md),
[repository validation](tests/README.md),
[provenance audits](tools/README.md), and
[runtime artifact ownership](artifacts/README.md).

## Provenance boundary

Pre-refactor paths and later reporting-snapshot moves are preserved through
append-only maps under `studies/migrations/`. Active historical identifiers
resolve through `fsrl.infra.study_registry.resolve_record`; frozen execution
source is verified from Git blobs and witness commits rather than copied into
the maintained package.

The generated record catalog provides stable logical IDs and typed format
metadata. Study records and immutable synthesis snapshots remain authoritative;
generated README files are human navigation views.

## Source paper

The project derives from
[Neural mechanisms of relational learning and fast knowledge reassembly in
plastic neural networks](https://thomasmiconi.github.io/NN.pdf), by Thomas
Miconi and Kenneth Kay, Nature Neuroscience 2024. Use the
[reproduction capsule](reproductions/relational_learning_2024/README.md) for
source provenance, teaching commands, and exact historical-replay guidance.
