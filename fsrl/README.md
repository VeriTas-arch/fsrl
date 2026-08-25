# Maintained code architecture

`fsrl` now has one dependency direction for new work:

```text
workflows
  -> training / evaluation
       -> tasks / core
            -> runtime / provenance
analysis
  -> typed outputs from tasks or evaluation
```

The packages have deliberately narrow ownership:

- `core/` owns the checkpoint-compatible plastic RNN, the confirmed local
  trace, typed states, input-channel ABI, and the first-class P/L rollout and
  intervention API.
- `tasks/` owns task protocols, generic sparse-graph generation, and evidence-
  admission equations. It does not own study registry paths.
- `training/` owns generic backbone optimization and checkpoint loading.
- `evaluation/` owns frozen rollout and causal intervention interfaces.
- `analysis/` owns reusable pure estimators such as Hodge decomposition and
  participant bootstrap utilities.
- `workflows/` validates and renders the schema-driven mainline under the
  repository-level `workflows/` directory.
- `runtime.py` and `provenance.py` make device, thread, compilation, hashing,
  and exclusive-output choices explicit.

The maintained command-line boundaries are `python -m fsrl.training` and
`python -m fsrl.evaluation`. Formal studies continue to use their registered
runner and frozen execution contract.

## Compatibility and diagnostic code

`model.py`, `meta_train.py`, `meta_tasks.py`, `liu_eval.py`,
`ranking_protocol.py`, and `conjunctive_local_trace.py` are small compatibility
adapters. Historical study runners may continue importing those names, but new
code should import the canonical packages above.

The remaining large flat modules are study-owned diagnostic runners. They are
kept runnable because their exact historical versions are referenced by frozen
contracts and Git provenance. They are not a library layer: stable packages
must never import a pilot, audit, confirmation, replication, or transport
runner. `tests/test_core_architecture.py` enforces that boundary. A later
curation pass can retire or relocate individual runners after their study
records have a replacement replay route.

Eponymous protocol names remain only where they identify frozen historical
records or compatibility APIs. New workflow and package names describe the
scientific computation instead.

## External-paper reproduction

The original rewarded relational-learning implementation is not part of this
package. Its upstream snapshot, supplied weights, and maintained teaching route
live in `reproductions/relational_learning_2024/`. This prevents editor and
import discovery from treating upstream notebooks or exports as active model
source.
