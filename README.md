# Meta-training plastic networks for relational learning

This repository develops and audits a plastic recurrent model that transforms
sparse, partially retained evidence into global relational structure and
query-specific direct fidelity. It began from the code accompanying Miconi &
Kay; its upstream snapshot, supplied checkpoints, and maintained teaching
reproduction now live in one isolated capsule under
`reproductions/relational_learning_2024/`.

## Start here

- [Relational model mainline](workflows/relational_model/README.md) is the
  shortest claim-to-code-to-evidence route and the current machine-readable
  scientific workflow.
- [Current synthesis](synthesis/README.md) organizes diagnostic history,
  closed candidate families, unresolved boundaries, and reporting context.
- [Study registry](studies/README.md) lists every registered positive, negative,
  mixed, unresolved, transported, and deferred study.
- [Historical reporting snapshots](synthesis/snapshots/README.md) provides the
  maintained replay guide for immutable reporting objects.
- [Figure workflow](synthesis/figures/README.md) defines how study outputs
  become report- or paper-facing figures.
- [Code architecture](fsrl/README.md) defines stable package ownership,
  compatibility adapters, and the boundary around historical study runners.
- [Analysis-file contract](artifacts/README.md) defines runtime manifests,
  format roles, and the non-destructive historical-conversion boundary.

The first reorganization pass is intentionally marked
review_state = "indexed". It establishes ownership, navigation, and provenance
without pretending that the current prose is already the final paper argument.

~~~text
studies/                 experiment-level questions and exact records
  <study-id>/
    README.md             generated human capsule
    study.toml            authoritative metadata and record hashes
    records/              byte-preserved reports, contracts, locks, results
synthesis/                current cross-study account, figures, and snapshots
artifacts/runs/           ignored workflow/execution runs with local manifests
artifacts/reproductions/  ignored regenerated external-paper outputs
data/external/            tracked immutable external source datasets
fsrl/                     executable model and analysis code
tests/                    regression and scientific-contract tests
workflows/                schema-driven maintained research routes
reproductions/            isolated external-paper reproduction capsules
~~~

The pre-refactor flat paths and the later synthesis-snapshot relocation are
recorded in versioned maps under `studies/migrations/`. Active code resolves
every historical identifier through `fsrl.infra.study_registry.resolve_record`;
the files themselves have one authoritative current location.

`studies/catalogs/record-catalog-v2.json` provides stable logical IDs and typed
format metadata for every registered record and retired Git-backed asset. It is
a generated compatibility view: original record bytes, hashes, and claim
pointers remain authoritative.

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

`requirements/constraints-py312.txt` records the tested direct dependency
snapshot. It makes fresh CPU research environments comparable without
pretending that a CUDA wheel URL is portable across hosts.

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
direnv exec . python -m fsrl.infra.file_contracts check
direnv exec . python -m tools.provenance.build_record_catalog_v2
~~~

Audit the one-time physical migration against its frozen Git sources and check
that compatibility rewrites and the Git-backed source index remain reproducible:

~~~bash
direnv exec . python tools/provenance/migrate_flat_records_v1.py audit
direnv exec . python tools/provenance/rewrite_runtime_locators_v1.py audit
direnv exec . python tools/provenance/rewrite_active_record_paths_v1.py check
direnv exec . python tools/provenance/index_source_provenance_v1.py check
direnv exec . python -m tools.provenance.backfill_run_manifests_v1
~~~

Run tests through the timeout-bounded entry point. It creates an independent
process group and cleans up that group on timeout or interruption:

~~~bash
direnv exec . python -m fsrl.infra.test_runtime
direnv exec . basedpyright
direnv exec . ruff check fsrl tests tools --select B905
~~~

To run one unittest module with a shorter timeout:

~~~bash
direnv exec . python -m fsrl.infra.test_runtime --timeout 60 \
  --framework unittest -- tests.infra.test_study_registry -v
~~~

Prospective runtime outputs belong under
`artifacts/runs/<workflow>/<execution-id>/`, including one `run.json` plus the
checkpoints, logs, evaluations, and previews owned by that execution. Historical
workflow roots retain their original layout and may carry additive backfilled
manifests. External-paper teaching
outputs belong under `artifacts/reproductions/<capsule>/`. The former top-level
`output/`, `figures/`, and `checkpoints/` layouts are retired. Promoting an
output into a study requires a registered protocol, exact provenance, a result
status, and a claim boundary.

## Prospective optimized training

New development training can opt into the schema-v3 GPU execution path:

~~~bash
direnv exec . python -m fsrl.training \
  --output-dir artifacts/runs/relational_model/seed-1 \
  --seed 1 \
  --device cuda \
  --optimized-execution
~~~

On CUDA, this mode compiles complete recurrent trial sequences with
`fullgraph=True` and defaults to `mode="reduce-overhead"`. The training loop
marks one explicit CUDA Graph iteration boundary per outer step. Execution
schema v3 records that boundary and the effective compile mode; the runtime
record also includes the device, CUDA capability, matrix precision,
determinism flags, and both PyTorch and BLAS thread limits. The older
`--compile-model` path remains the `mode="default"`, byte-replay-compatible
single-cell execution used by registered historical backbones. The maintained
training CLI writes `net.pth`. Frozen runners prefer a byte-identical `.pth`
view and use an explicitly named legacy adapter only when an untouched
registered checkpoint contract still points to `.dat`.

Task sampling preserves the historical RNG stream while vectorizing cue-code
similarity checks. Each trial sequence is assembled in one preallocated NumPy
array before the existing batched host-to-device transfer.

Use `--compile-mode default` to retain the non-CUDA-Graph prospective profile.
The CUDA Graph profile trades extra capture warmup and reserved device memory
for lower steady-state launch overhead. Use the default mode when memory or
short-run startup latency matters more than long-run throughput.
The `max-autotune` modes are explicit opt-ins because selecting different
matrix kernels can change floating-point reduction order and therefore the
optimizer trajectory; the selected mode is preserved in checkpoint
provenance.

Prospective evaluations can separately opt into one compiled sequence per
support trial and one device transfer for the complete query batch:

~~~bash
direnv exec . python -m fsrl.evaluation \
  --checkpoint artifacts/runs/relational_model/seed-1/net.pth \
  --output artifacts/runs/relational_model/seed-1/evaluation.json \
  --evaluation-backend batched_sequence
~~~

The default remains `legacy_stepwise`, so frozen studies are not silently
upgraded. The batched result carries its execution profile and observed runtime
snapshot. Before a large run, benchmark exact parity and throughput on the
visible CUDA device with a small number of repeats:

~~~bash
direnv exec . python -m fsrl.evaluation.performance \
  --warmups 1 --repeats 3 \
  --output artifacts/runs/runtime/frozen-evaluation-benchmark.json
~~~

This benchmark is explicitly an engineering diagnostic, not scientific
evidence and not a hardware-independent performance threshold.

## Original-paper teaching reproduction

To generate the teaching figures extracted from the upstream main.py:

~~~bash
direnv exec . python -m reproductions.relational_learning_2024.figures \
  --figures all \
  --model-path reproductions/relational_learning_2024/checkpoints/net_active.pth
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
