# Maintained code architecture

## Development installation

The repository uses Python 3.12 and an editable local installation. Its
`.envrc` selects the shared Conda environment named `ipex`:

```bash
conda create -n ipex python=3.12 pip
direnv allow
direnv exec . python -m pip install --upgrade pip
```

For a CUDA-specific build, install PyTorch first using the command from the
[PyTorch installation guide](https://pytorch.org/get-started/locally/). Then
install the maintained package, reproduction dependencies, and test tools:

```bash
direnv exec . python -m pip install -e ".[reproduction,test]"
direnv exec . python -m pip check
direnv exec . python -c "import fsrl; print(fsrl.__file__)"
```

`pyproject.toml` is the single dependency and tool-configuration authority.
When dependencies are already present, refresh packaging metadata without
dependency resolution:

```bash
direnv exec . python -m pip install --no-deps -e .
```

Ordinary source edits are visible immediately through the editable install.
The distribution is named `fsrl-relational-model`; the import namespace is
`fsrl`. Registered studies, synthesis records, and external data remain
repository-owned and are not bundled into the wheel.

See [training](training/README.md) and [evaluation](evaluation/README.md) for
the maintained command-line interfaces.

## Package architecture

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
- [`training/`](training/README.md) owns generic backbone optimization and
  checkpoint loading.
- [`evaluation/`](evaluation/README.md) owns frozen rollout, ordered query-field
  reconstruction, and causal intervention interfaces. Registered checkpoint
  loading and the shared global/local query bundle live here rather than in an
  experiment runner.
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
The wheel contains the `fsrl` namespace and the two transport contracts read
from package-relative paths; repository evidence and workflow data remain
outside the wheel by design. Importing the installed public packages does not
resolve repository records; registered protocols and evidence are resolved
only when their repository-backed operations are called.

Frozen contracts may still name former flat modules. Those identities are
verified against Git blobs and witness commits rather than recreated as a large
set of compatibility files. Full historical execution uses a detached
worktree. Stable packages never import `experiments/`; the explicit dependency
layer graph and non-flat roots are enforced by
`tests/core/test_architecture.py`.

Eponymous protocol names remain only where they identify frozen historical
records or compatibility APIs. New workflow and package names describe the
scientific computation instead.

## Equivalence and compatibility boundaries

Current code is allowed to change structure without imitating a historical
checkout. The boundary is explicit:

- registered records are selected by stable `record_id`; the catalog exposes
  historical registered identity and current materialized identity separately;
- exact historical replay runs at its recorded Git commit, while current
  outputs are accepted through versioned semantic assertions;
- current semantic replay rules are versioned in checked-in contract JSON whose
  source points to one registered protocol record. The active topology and item-
  count transport decisions use this boundary now; protocol-specific rules in
  other frozen study runners remain historical-execution code until migrated
  one study at a time, and are not current public APIs;
- new checkpoints are plain state dictionaries written as `.pth`; the current
  loader accepts only `.pth`. Historical `.dat` inputs are confined to the
  explicit frozen-replay adapter and can be materialized as byte-identical
  `.pth` views;
- structured declarations use TOML, contracts/results/metadata use JSON,
  append-only runtime logs use JSONL, and dense numeric arrays use NPZ. CSV is
  reserved for external or tabular interchange. New model weights do not use
  `.dat`, `.pt`, pickle, or NPY; `.pt` remains reserved for a serialized program
  if one is ever explicitly registered;
- `RetroModelConfig`, `PlasticRNNState`, protocol IDs, and explicit holdout
  signatures are the current model/task APIs. The former CamelCase state
  methods, v1-default protocol adapter, and `exclude_liu_graph` switch are
  available only through historical Git/reproduction sources, not the
  maintained package. `TrainConfig` still carries the historical
  episode/evaluation field names, but model construction crosses one typed
  `RetroModelConfig` boundary instead of spreading its short-key dictionary;
- high-level training and causal evaluation default to the current versioned
  sequence/batched execution. Historical stepwise execution remains an
  explicit profile and is never silently substituted for a frozen lock.
- production global/local query paths call `GlobalLocalRelationalSystem`; the
  direct local correction, global fast-weight intervention, and policy residual
  therefore cross one maintained P/L readout boundary.
- `basedpyright` checks the complete maintained package at its incremental
  baseline and applies strict mode to new contract and pure-computation
  modules. The complexity budget rejects new C901 violations or increases in
  registered legacy hotspots without forcing frozen runners into a cosmetic
  rewrite.

Equivalence is layered: bytes for immutable evidence and normalized legacy
views; exact tensors/state dictionaries for adapters; numerical trajectory and
gradient parity for execution refactors; semantic decision parity for result
documents; and unchanged claim boundaries at the workflow layer.

## External-paper reproduction

The original rewarded relational-learning implementation is not part of this
package. Its upstream snapshot, supplied weights, and maintained teaching route
live in `reproductions/relational_learning_2024/`. This prevents editor and
import discovery from treating upstream notebooks or exports as active model
source.
