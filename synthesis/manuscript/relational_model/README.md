# Relational-model working manuscript

This directory contains the active Typst manuscript used for internal display
and discussion of the maintained relational-model claim graph. It is a
reporting layer, not a new evidence source. Scientific claims remain owned by
`workflows/relational_model/workflow.toml` and the study-level frozen records;
the historical `reporting_v1` snapshot is read-only.

The draft is intentionally paper-first. Its structure can later be compressed
into slides without changing the scientific narrative or claim boundaries.

## Build

From the repository root:

```bash
typst compile --root . \
  synthesis/manuscript/relational_model/main.typ \
  synthesis/manuscript/relational_model/relational_model.pdf
```

For live editing:

```bash
typst watch --root . \
  synthesis/manuscript/relational_model/main.typ \
  synthesis/manuscript/relational_model/relational_model.pdf
```

The document uses no Typst Universe packages. Existing registered SVG figures
are included directly, and the remaining schematics and tables are native
Typst content.
