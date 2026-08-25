# Runtime artifacts

This is the repository's only ignored runtime-output root:

- `runs/<workflow>/` keeps each training or analysis run together with its
  checkpoints, logs, evaluations, and temporary previews;
- `reproductions/<capsule>/` keeps regenerated outputs from an external-paper
  reproduction separate from current-model runs.

Do not recreate top-level `output/`, `figures/`, or `checkpoints/` directories.
A runtime artifact does not become scientific evidence merely because a command
completed.

Promote only the smallest registered evidence object needed for a study into
`studies/<study-id>/records/`, together with its protocol, provenance, outcome,
and exact claim boundary. Cross-study figure source data may instead be
promoted into `synthesis/figures/`.
