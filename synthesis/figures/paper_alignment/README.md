# Published behavioral figure alignment

This suite redraws the released human behavioral data and places the frozen
`dual_access_matched` model from independent networks 2104 and 2105 on matched
estimands. It is a reporting layer, not a new model fit.

The static contract is `figure_spec.json`. `source/model_subject_pair_accuracy.csv`
is produced by a minimal read-only GPU replay and is accepted only when every
stored behavioral field reconstructs the frozen v2.4 result. Rendering is then
CPU-only and produces one source table alongside every SVG, PDF, and PNG.

## Figure set

| Figure | Published panels | Outputs |
| --- | --- | --- |
| Group behavioral profiles | 1E, 1F, 1G | [PNG](figure_01_group_behavior/figure_01_group_behavior.png), [SVG](figure_01_group_behavior/figure_01_group_behavior.svg), [PDF](figure_01_group_behavior/figure_01_group_behavior.pdf), [source data](figure_01_group_behavior/source_data.csv) |
| Pair-level structure | 2A, 2B, 2D, 2E | [PNG](figure_02_pair_structure/figure_02_pair_structure.png), [SVG](figure_02_pair_structure/figure_02_pair_structure.svg), [PDF](figure_02_pair_structure/figure_02_pair_structure.pdf), [source data](figure_02_pair_structure/source_data.csv) |
| Stable local-error fingerprints | 2H, layout adapted | [PNG](figure_02h_error_fingerprints/figure_02h_error_fingerprints.png), [SVG](figure_02h_error_fingerprints/figure_02h_error_fingerprints.svg), [PDF](figure_02h_error_fingerprints/figure_02h_error_fingerprints.pdf), [source data](figure_02h_error_fingerprints/source_data.csv) |
| Reconstructed global rankings | 3B, 3C, 3D, 3E | [PNG](figure_03_global_rankings/figure_03_global_rankings.png), [SVG](figure_03_global_rankings/figure_03_global_rankings.svg), [PDF](figure_03_global_rankings/figure_03_global_rankings.pdf), [source data](figure_03_global_rankings/source_data.csv) |

`manifest.json` hashes every rendered output and records the generator and
environment. `figure_spec.json` records panel mappings, fixed selection rules,
frozen inputs, and explicit exclusions.

```bash
direnv exec . python -m fsrl.paper_figure_alignment replay
direnv exec . python -m fsrl.paper_figure_alignment render
direnv exec . python -m fsrl.paper_figure_alignment check
```

The figures intentionally retain the model's excessive symbolic-distance
slope, weak serial-position endpoint contrast, and seed-specific
self-inconsistency mismatch. Q-learning controls and MEG panels are outside the
suite; a hidden-state RSA would be a new model prediction rather than a
reproduction of the paper's sensor-time measurements.
