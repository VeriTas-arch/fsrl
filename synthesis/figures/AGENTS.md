# Report and paper figure guide

This file applies to `synthesis/figures/`.

Navigation: [synthesis guide](../AGENTS.md) · [figure portal](README.md) ·
[paper alignment](paper_alignment/README.md) ·
[current workflow](../../workflows/relational_model/README.md)

- Cross-study report figures belong here; study-specific diagnostic figures
  remain with their owning study.
- `README.md` is hand-maintained navigation. Figure specifications, source-data
  tables, and render manifests are the machine-readable authorities.
- Every promoted figure needs a stable figure ID, panel purpose, source study
  and estimand, source-data table, deterministic generation command, output
  paths, and render manifest.
- Prefer tracked source tables over extracting values from rendered images.
  Figures summarize registered values; they do not create new evidence.
- Keep paper-comparator panels explicitly distinct from project-model panels.
  A visual match is not an inferential or mechanistic claim.
- Store editable vector output plus the formats needed for review. Avoid
  duplicating the same rendered figure under multiple current locations.
- Cosmetic edits must not alter data, panel membership, axes, thresholds, or
  uncertainty conventions silently.

Run `direnv exec . python -m fsrl.workflows.paper_figures check` after any
figure-spec, source-data, renderer, manifest, or promoted-output change.
