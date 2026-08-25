# Meta-training plastic networks for relational learning

This repository develops and audits a plastic recurrent model that transforms
sparse, partially retained evidence into global relational structure and
query-specific direct fidelity. It began from the code accompanying Miconi &
Kay; the unchanged educational/upstream material remains under archive/ and
addons/.

## Start here

- [Current synthesis](synthesis/README.md) gives the shortest human reading
  route through the model evidence.
- [Study registry](studies/README.md) lists every registered positive, negative,
  mixed, unresolved, transported, and deferred study.
- [Frozen evidence overlay](synthesis/frozen/README.md) is the historical
  machine-verifiable reporting object.
- [Figure workflow](synthesis/figures/README.md) defines how study outputs
  become report- or paper-facing figures.

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
artifacts/                ignored runtime outputs and regenerated data
data/external/            tracked source datasets
fsrl/                     executable model and analysis code
tests/                    regression and scientific-contract tests
archive/                  upstream and historical code
~~~

The pre-refactor docs/, benchmarks/, results/, research/liu/, and
mainlines/liu_v1/ paths are recorded in
studies/migrations/flat-records-v1.json. Active code resolves those frozen
identifiers through fsrl.study_registry.resolve_record; the files themselves
have one authoritative current location.

Frozen execution locks that identify pre-refactor Python files are backed by
exact snapshots under synthesis/frozen/source/. They are provenance evidence;
the active implementation and test surface remains fsrl/ plus tests/.

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

Use Python 3.12.

~~~bash
conda create -n fsrl python=3.12
conda activate fsrl
pip install --upgrade pip
pip install -r requirements.txt
~~~

The default requirements install the standard PyPI build of PyTorch. For a
CUDA-specific build, install PyTorch first using the command from
<https://pytorch.org/get-started/locally/>, then install the remaining
requirements.

## Validation

Check the study registry and generated human views:

~~~bash
direnv exec . python -m fsrl.study_registry check
~~~

Audit the one-time physical migration against its frozen Git sources and check
that compatibility rewrites and source snapshots remain reproducible:

~~~bash
direnv exec . python tools/provenance/migrate_flat_records_v1.py audit
direnv exec . python tools/provenance/rewrite_active_record_paths_v1.py check
direnv exec . python tools/provenance/snapshot_refactor_sources_v1.py check
~~~

Run tests through the timeout-bounded entry point. It creates an independent
process group and cleans up that group on timeout or interruption:

~~~bash
direnv exec . python -m fsrl.test_runtime
~~~

To run one unittest module with a shorter timeout:

~~~bash
direnv exec . python -m fsrl.test_runtime --timeout 60 \
  --framework unittest -- tests.test_study_registry -v
~~~

Runtime outputs belong in artifacts/. Promoting an output into a study requires
a registered protocol, exact provenance, a result status, and a claim boundary.
The pre-existing ignored output/ and figures/ trees remain legacy runner caches
for compatibility; they are not part of the study registry or frozen evidence.

## Teaching figures

To generate the teaching figures extracted from the upstream main.py:

~~~bash
python eval_figures.py --figures all
~~~

Outputs are written to the ignored figures/ directory by default. These are not
automatically part of the research evidence or paper-figure registry.

## Original README

This repository derives from the code for
[Neural mechanisms of relational learning and fast knowledge reassembly in
plastic neural networks](https://thomasmiconi.github.io/NN.pdf), by Thomas
Miconi and Kenneth Kay, Nature Neuroscience 2024.

The original notebooks and simplified scripts are retained under archive/.
Parameter files for the active and passive strategies remain at the repository
root for compatibility with the original evaluation workflow.

For the original paper figures, copy either net_active.dat or net_passive.dat
to net.dat, use the archived evaluation notebook, and follow its figure-specific
settings. For new project-level results, use the registered study and synthesis
workflow above instead.
