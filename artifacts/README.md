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

Prospective executions use `runs/<workflow-id>/<execution-id>/run.json`; the
manifest owns portable relative locators, hashes, formats, lifecycle state,
resolved configuration, and producer provenance. Do not encode scientific
status or hyperparameters only in directory names. Historical workflow roots
may carry backfilled `run.json` inventories, but those inventories do not infer
ownership, promote evidence, or authorize moving the recorded source bytes.

Format roles are intentionally narrow:

- JSON contains manifests, contracts, and compact result summaries;
- JSON Lines contains append-only events or training metrics;
- CSV contains rectangular external or figure-source tables;
- NPZ contains typed numerical arrays and requires a JSON array inventory;
- deterministic JSON.GZ is the compatibility view for large frozen JSON: its
  decompressed bytes and JSON Pointer semantics must match the source exactly;
- new PyTorch state dictionaries use `.pth`; `.pt` is reserved for exported or
  scripted PyTorch programs. Historical `.dat` files remain frozen sources and
  receive byte-identical `.pth` compatibility views;
- SVG is the canonical editable figure, PDF is the publication rendering, and
  PNG is an optional preview unless a figure contract requires all three.

Prospective bulk numerical output should use compact JSON summaries plus NPZ,
not monolithic JSON. Historical sources are different: run the reversible view
materializer rather than redesigning their study-specific structure:

```bash
direnv exec . python -m tools.provenance.materialize_historical_file_views_v1 --apply
direnv exec . python -m tools.provenance.materialize_historical_file_views_v1
```
