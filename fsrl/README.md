# Maintained code architecture

`fsrl` now has one dependency direction for new work:

```text
workflows
  -> experiments
       -> analysis / training / evaluation
            -> tasks / core
                 -> infra / paths
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
- `evaluation/` owns frozen rollout, ordered query-field reconstruction, and
  causal intervention interfaces.
- `analysis/` owns reusable pure estimators such as Hodge decomposition,
  policy transforms, and participant bootstrap utilities.
- `infra/` owns runtime policy, provenance, the study registry, and
  bounded execution helpers.
- `experiments/` owns maintained evidence-producing runners, grouped by
  assembly, confirmation, local fidelity, global policy, reduction, transport,
  and human-program scope.
- `workflows/` validates and renders the schema-driven mainline under the
  repository-level `workflows/` directory.
- `paths.py` is the only repository-root path contract; package modules do not
  infer the checkout root from their own directory depth.

The maintained command-line boundaries are `python -m fsrl.training` and
`python -m fsrl.evaluation`. Maintained formal workflows run through
`python -m fsrl.infra.formal_runtime`; historical commands continue
to replay from their exact Git commit.

## Experiments and historical replay

The package root contains only `__init__.py` and the explicit `paths.py`
contract. Reusable code lives in the stable packages above; evidence-producing
code lives below `experiments/`. Tests mirror the same ownership structure.

Frozen contracts may still name former flat modules. Those identities are
verified against Git blobs and witness commits rather than recreated as a large
set of compatibility files. Full historical execution uses a detached
worktree. Stable packages never import `experiments/`; the explicit dependency
layer graph and non-flat roots are enforced by
`tests/core/test_architecture.py`.

Eponymous protocol names remain only where they identify frozen historical
records or compatibility APIs. New workflow and package names describe the
scientific computation instead.

## External-paper reproduction

The original rewarded relational-learning implementation is not part of this
package. Its upstream snapshot, supplied weights, and maintained teaching route
live in `reproductions/relational_learning_2024/`. This prevents editor and
import discovery from treating upstream notebooks or exports as active model
source.
