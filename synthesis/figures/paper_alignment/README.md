# Published behavioral figure alignment

This suite redraws the released human behavioral data and places the frozen
`dual_access_matched` model from independent networks 2104 and 2105 on matched
estimands. It is a reporting layer, not a new model fit.

The static contract is `figure_spec.json`. `source/model_subject_pair_accuracy.csv`
is produced by a minimal read-only GPU replay and is accepted only when every
stored behavioral field reconstructs the frozen v2.4 result. Rendering is then
CPU-only and produces one source table alongside every SVG, PDF, and PNG.

## Figure set

- **Group behavioral profiles** — published panels 1E, 1F, and 1G.
  [PNG](figure_01_group_behavior/figure_01_group_behavior.png) ·
  [SVG](figure_01_group_behavior/figure_01_group_behavior.svg) ·
  [PDF](figure_01_group_behavior/figure_01_group_behavior.pdf) ·
  [source data](figure_01_group_behavior/source_data.csv)
- **Pair-level structure** — published panels 2A, 2B, 2D, and 2E.
  [PNG](figure_02_pair_structure/figure_02_pair_structure.png) ·
  [SVG](figure_02_pair_structure/figure_02_pair_structure.svg) ·
  [PDF](figure_02_pair_structure/figure_02_pair_structure.pdf) ·
  [source data](figure_02_pair_structure/source_data.csv)
- **Stable local-error fingerprints** — published panel 2H with adapted layout.
  [PNG](figure_02h_error_fingerprints/figure_02h_error_fingerprints.png) ·
  [SVG](figure_02h_error_fingerprints/figure_02h_error_fingerprints.svg) ·
  [PDF](figure_02h_error_fingerprints/figure_02h_error_fingerprints.pdf) ·
  [source data](figure_02h_error_fingerprints/source_data.csv)
- **Reconstructed global rankings** — published panels 3B, 3C, 3D, and 3E.
  [PNG](figure_03_global_rankings/figure_03_global_rankings.png) ·
  [SVG](figure_03_global_rankings/figure_03_global_rankings.svg) ·
  [PDF](figure_03_global_rankings/figure_03_global_rankings.pdf) ·
  [source data](figure_03_global_rankings/source_data.csv)

`manifest.json` hashes every rendered output and records the generator and
environment. `figure_spec.json` records panel mappings, fixed selection rules,
frozen inputs, and explicit exclusions.

```bash
direnv exec . python -m fsrl.workflows.paper_figures replay
direnv exec . python -m fsrl.workflows.paper_figures render
direnv exec . python -m fsrl.workflows.paper_figures check
```

The figures intentionally retain the model's excessive symbolic-distance
slope, weak serial-position endpoint contrast, and seed-specific
self-inconsistency mismatch. Q-learning controls and MEG panels are outside the
suite; a hidden-state RSA would be a new model prediction rather than a
reproduction of the paper's sensor-time measurements.
