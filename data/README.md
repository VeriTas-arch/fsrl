# External inputs

`data/` contains tracked, immutable inputs obtained outside this repository.
Each dataset belongs under `external/<source>/` with source provenance and
content hashes. Do not place checkpoints, generated arrays, cached transforms,
analysis results, or figures here; those remain under `artifacts/` until a
registered evidence or figure workflow promotes the required compact object.

Each prospective dataset also has a machine-readable `dataset.toml`. It owns
stable file IDs, source URLs, formats, byte counts, hashes, and tabular schemas;
the original source files remain byte-preserved and are never rewritten to fit
the repository's preferred representation.

## Available datasets

- [Liu, Wang, and Luo (2026) behavioral source data](external/liu2026/README.md)
  contains the byte-preserved public trial-level and published-panel files used
  by the current relational-model reporting workflow.
