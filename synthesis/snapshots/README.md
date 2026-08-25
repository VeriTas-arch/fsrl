# Historical reporting snapshots

This directory contains immutable, versioned reporting objects. A snapshot
preserves the claim graph, exact evidence selection, execution records,
environment contract, presentation view, and artifact identities that were
frozen together. It is not the current model workflow and does not receive
scientific edits.

## Available snapshots

- [`reporting_v1/`](reporting_v1/README.md) is the historical v1 reporting
  freeze. Its internal Liu identifiers, paths, commands, and tag names are
  preserved as provenance.

The historical README inside a snapshot describes its original checkout.
Use the maintained commands below from the current repository:

```bash
direnv exec . python -m fsrl.workflows.frozen_evidence status
direnv exec . python -m fsrl.workflows.frozen_evidence verify
direnv exec . python -m fsrl.workflows.frozen_evidence doctor
direnv exec . python -m fsrl.workflows.frozen_evidence restore-test-artifacts
direnv exec . python -m fsrl.workflows.frozen_evidence summarize \
  --output-dir /tmp/fsrl-reporting-v1-summary
direnv exec . python -m fsrl.workflows.frozen_evidence replay \
  --stage behavioral_competence
```

There is intentionally no `replay --all`. Historical stages differ in source
commit, artifacts, runtime, and GPU cost, so stage selection remains explicit.
The replay command uses a detached worktree and does not switch or modify the
current `dev` checkout.

Any future frozen reporting object belongs in a new sibling directory such as
`reporting_v2/`; it must not overwrite `reporting_v1/`.
