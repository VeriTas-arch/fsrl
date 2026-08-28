# Relational-learning 2024 reproduction capsule

This capsule keeps the original paper reproduction separate from the maintained
FSRL research model. It has three explicit layers:

- `upstream/` retains the released notebooks and assets as byte-preserved
  source artifacts. Its Python exports carry provenance-tracked maintenance
  repairs and are included in editor analysis, linting, and formatting.
- `checkpoints/` contains the supplied active and passive network weights plus
  byte-identical `.pth` views used by the maintained reproduction.
- `cli.py`, `training.py`, `episode.py`, `task.py`, and `figures.py` are the
  maintained readable reproduction route. They reuse only the checkpoint-
  compatible plastic RNN and shared logging/configuration from `fsrl`.

`source_manifest.toml` retains the imported SHA-256 for repaired Python exports
and locks every current upstream file and supplied checkpoint by SHA-256. These
files are reproduction inputs, not evidence for the current P/L model and not
part of the diagnostic study registry.
`checkpoint_views.toml` separately binds each maintained `.pth` view to its
immutable `.dat` source, byte count, and SHA-256.

Verify the capsule without executing the upstream code:

```bash
direnv exec . python -m reproductions.relational_learning_2024.verify
```

## Run the readable reproduction

Train the rewarded active/passive task implementation:

```bash
direnv exec . python -m reproductions.relational_learning_2024.cli \
  --seed 1 --nbiter 30000
```

Runtime outputs go to
`artifacts/reproductions/relational_learning_2024/training/` by default.
The maintained adapter writes one canonical `net.pth` state dict and one
`training_metrics.npz` numeric archive. It does not create legacy `.dat`
checkpoints or numeric `.txt` logs. The byte-locked supplied `.dat` files remain
read-only provenance inputs; maintained code loads their byte-identical `.pth`
views through the current checkpoint boundary.

Render the maintained teaching approximations of the original figure panels:

```bash
direnv exec . python -m reproductions.relational_learning_2024.figures \
  --figures all \
  --model-path reproductions/relational_learning_2024/checkpoints/net_active.pth
```

These teaching figures are not automatically report-facing results. Promoted
paper figures for the current project remain under `synthesis/figures/` and
must carry source data plus panel-level provenance.

## Historical replay

The exact upstream notebooks retain their original relative-path assumptions.
Use a detached worktree at the relevant historical commit when exact historical
execution is required. Do not add `upstream/` to `PYTHONPATH` or import it from
the maintained model.
