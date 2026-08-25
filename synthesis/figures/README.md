# Figure workflow

This directory is reserved for cross-study figures intended for a report or
paper. The architecture distinguishes three layers:

1. exact historical outputs remain under each study's `records/` directory;
2. reproducible study-level figures may be promoted to `studies/<id>/figures/`;
3. cross-study figures and their machine-readable source tables live here.

Every promoted figure should have a source-data file, a generation command, and
study/estimand provenance. Prefer a stable figure ID whose directory contains
the rendered panel, source table, generation script or command, and a manifest
mapping every panel to study IDs and frozen estimands. Historical presentation
assets remain under `synthesis/records/` until the second curation pass decides
whether to regenerate or retire them. Do not copy an image here merely to make
it easier to find.

## Current suites

- [Published behavioral figure alignment](paper_alignment/README.md) redraws
  released human results and places the frozen two-network model on the same
  estimands. Its manifest records the exact sources, exclusions, and rendered
  outputs.
