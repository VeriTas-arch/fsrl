# Meta-training plastic networks for relational learning

This repository develops and audits a plastic recurrent model that transforms
sparse, partially retained evidence into global relational structure and
query-specific direct fidelity. It began from the code accompanying Miconi &
Kay; its upstream snapshot, supplied checkpoints, and maintained teaching
reproduction now live in one isolated capsule under
`reproductions/relational_learning_2024/`.

## Start here

- [Current synthesis](synthesis/README.md) gives the shortest human reading
  route through the model evidence.
- [Study registry](studies/README.md) lists every registered positive, negative,
  mixed, unresolved, transported, and deferred study.
- [Frozen evidence overlay](synthesis/frozen/README.md) is the historical
  machine-verifiable reporting object.
- [Figure workflow](synthesis/figures/README.md) defines how study outputs
  become report- or paper-facing figures.
- [Maintained model workflow](workflows/relational_model/README.md) connects
  code, registered evidence, verification commands, and report outputs in one
  machine-checked and human-readable route.
- [Code architecture](fsrl/README.md) defines stable package ownership,
  compatibility adapters, and the boundary around historical study runners.

The first reorganization pass is intentionally marked
review_state = "indexed". It establishes ownership, navigation, and provenance
without pretending that the current prose is already the final paper argument.

~~~text
studies/                 experiment-level questions and exact records
  <study-id>/
    README.md             generated human capsule
    study.toml            authoritative metadata and record hashes
    records/              byte-preserved reports, contracts, locks, results
synthesis/                current cross-study account and frozen overlay
artifacts/runs/           ignored training, evaluation, and checkpoint runs
artifacts/reproductions/  ignored regenerated external-paper outputs
data/external/            tracked immutable external source datasets
fsrl/                     executable model and analysis code
tests/                    regression and scientific-contract tests
workflows/                schema-driven maintained research routes
reproductions/            isolated external-paper reproduction capsules
~~~

The pre-refactor docs/, benchmarks/, results/, research/liu/, and
mainlines/liu_v1/ paths are recorded in
studies/migrations/flat-records-v1.json. Active code resolves those frozen
identifiers through fsrl.study_registry.resolve_record; the files themselves
have one authoritative current location.

Frozen execution locks that identify historical Python files are indexed by
`(path, sha256)` in synthesis/source-provenance.toml and verified against Git
blobs plus witness commits. The maintained project implementation and tests
live in `fsrl/` and `tests/`. The separate reproduction capsule has its own
byte-locked upstream inputs and runnable teaching code; full historical replay
still uses a detached Git worktree.

## Current scientific snapshot

The evidence registry currently supports a working computational division:

- a meta-learned fast-weight state assembles global relational structure;
- a causally distinct query-addressed state preserves direct evidence with a
  broader admission rule;
- registered ablations separate global inference from direct local fidelity;
- global-policy slope, some transport boundaries, and the final compression
  into a minimal algorithm remain qualified or unresolved.

This is a navigation-level summary, not a substitute for the per-study
estimands, controls, seeds, outcomes, and claim boundaries.

## Environment setup

The repository uses Python 3.12 and an editable local installation. Its
`.envrc` selects the shared Conda environment named `ipex`:

~~~bash
conda create -n ipex python=3.12 pip
direnv allow
direnv exec . python -m pip install --upgrade pip
~~~

For a CUDA-specific build, install PyTorch first using the command from
<https://pytorch.org/get-started/locally/>. Then install the maintained `fsrl`
namespace, reproduction dependencies, and test tools in editable mode and
verify the environment:

~~~bash
direnv exec . python -m pip install -r requirements.txt
direnv exec . python -m pip check
direnv exec . python -c "import fsrl; print(fsrl.__file__)"
~~~

When dependencies are already installed, refresh only the local package
metadata without dependency resolution:

~~~bash
direnv exec . python -m pip install --no-deps -e .
~~~

Ordinary Python source edits are visible immediately through the editable
installation; rerun the refresh only after packaging metadata changes. The
distribution is named `fsrl-relational-model`, while its import namespace
remains `fsrl`. This first packaging stage is repository-bound: registered
studies, synthesis records, and external data remain outside a standalone
wheel and are resolved from this checkout.

## Validation

Check the study registry and generated human views:

~~~bash
direnv exec . python -m fsrl.infra.study_registry check
~~~

Audit the one-time physical migration against its frozen Git sources and check
that compatibility rewrites and the Git-backed source index remain reproducible:

~~~bash
direnv exec . python tools/provenance/migrate_flat_records_v1.py audit
direnv exec . python tools/provenance/rewrite_runtime_locators_v1.py audit
direnv exec . python tools/provenance/rewrite_active_record_paths_v1.py check
direnv exec . python tools/provenance/index_source_provenance_v1.py check
~~~

Run tests through the timeout-bounded entry point. It creates an independent
process group and cleans up that group on timeout or interruption:

~~~bash
direnv exec . python -m fsrl.infra.test_runtime
~~~

To run one unittest module with a shorter timeout:

~~~bash
direnv exec . python -m fsrl.infra.test_runtime --timeout 60 \
  --framework unittest -- tests.infra.test_study_registry -v
~~~

Runtime outputs belong under `artifacts/runs/<workflow>/`, including checkpoints,
logs, evaluations, and previews owned by that run. External-paper teaching
outputs belong under `artifacts/reproductions/<capsule>/`. The former top-level
`output/`, `figures/`, and `checkpoints/` layouts are retired. Promoting an
output into a study requires a registered protocol, exact provenance, a result
status, and a claim boundary.

## Original-paper teaching reproduction

To generate the teaching figures extracted from the upstream main.py:

~~~bash
direnv exec . python -m reproductions.relational_learning_2024.figures \
  --figures all \
  --model-path reproductions/relational_learning_2024/checkpoints/net_active.dat
~~~

Outputs are written to the ignored
`artifacts/reproductions/relational_learning_2024/figures/` directory by
default. They are not automatically part of the research evidence or
paper-figure registry.

## Original README

This repository derives from the code for
[Neural mechanisms of relational learning and fast knowledge reassembly in
plastic neural networks](https://thomasmiconi.github.io/NN.pdf), by Thomas
Miconi and Kenneth Kay, Nature Neuroscience 2024.

The original notebooks, appendix files, and supplied active/passive parameter
files are retained together under
`reproductions/relational_learning_2024/`. Their byte hashes are recorded in
that capsule's `source_manifest.toml`.

For exact original-paper replay, use the capsule's upstream notebook in a
detached historical worktree and follow its figure-specific settings. For new
project-level results, use the registered study, workflow, and synthesis layers
above instead.
