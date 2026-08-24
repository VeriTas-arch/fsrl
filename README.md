# Meta-training plastic networks for transitive inference

> [!NOTE]
> For educational purposes, we modified the code to make it easier to read and understand. The original code used for the paper is at <https://github.com/ThomasMiconi/TransitiveInference>, and the original README is kept below for reference.

## Environment setup

Use Python 3.12.

```bash
conda create -n fsrl python=3.12
conda activate fsrl
pip install --upgrade pip
pip install -r requirements.txt
```

## Tests

Run tests through the timeout-bounded entry point. It gives the test process an
independent process group and cleans up that whole group on timeout or
interruption. The default command is `pytest -q` with a five-minute timeout.

```bash
direnv exec . python -m fsrl.test_runtime
```

To run one `unittest` module with a different timeout:

```bash
direnv exec . python -m fsrl.test_runtime --timeout 60 \
  --framework unittest -- tests.test_liu_eval -v
```

The default `requirements.txt` installs the standard PyPI build of PyTorch. If you need a CUDA-specific PyTorch build, install PyTorch first using the command from <https://pytorch.org/get-started/locally/>, then run:

```bash
pip install numpy matplotlib scipy scikit-learn tqdm
```

To generate the teaching figures extracted from `main.py`:

```bash
python eval_figures.py --figures all
```

Outputs are written to `figures/` by default. Use a smaller batch for quick demos:

```bash
python eval_figures.py --figures fig2a fig4b --batch-size 64 --seed 1
```

## Constructive-ranking mechanism development

The `dev` branch adds a registered Liu-style passive-learning benchmark, an
exact 40320-permutation computational model, generic sparse-graph supervised
meta-training with the Liu graph held out, stable subject-level relation
omission, causal fast-weight controls, registered human-behavior analysis, and
subjective-rank geometry tests. A tracked single-seed candidate passes the
causal and antisymmetric-geometry gates and is frozen as development evidence at
`07591af`. See [`docs/development_plan.md`](docs/development_plan.md) for those
results and their claim boundary.

The source-corrected, all-seed protocol now includes an exact OSF human
benchmark, matched bottleneck controls, and a 40,320-order posterior comparison.
See [`docs/confirmation_protocol.md`](docs/confirmation_protocol.md) for the
two-seed pilot, frozen formal contract, and commands. Pilot outputs are reported
without seed filtering in [`results/pilot_v1.json`](results/pilot_v1.json).

A preregistered read-only diagnostic on both pilot checkpoints localizes the
excessive distance slope to the neural transformation rather than the evidence
model. Frozen neural logit fields are nearly pure additive potentials, whereas
human choices retain a stronger learned-pair residual after controlling for
distance. See [`docs/assembly_diagnostics_v1.md`](docs/assembly_diagnostics_v1.md)
and [`results/assembly_diagnostics_v1.json`](results/assembly_diagnostics_v1.json).

The follow-up registered prefix and relation-LOO diagnostic shows that retained
support evidence causally reorganizes disjoint pairs, progressively forms a
distributional expected-rank potential rather than a hard MAP order, and is
already available to hidden dynamics at query onset. The output direction
selects an even more additive component, but query recurrence is not the source
of commitment. See
[`docs/assembly_trajectory_v1.md`](docs/assembly_trajectory_v1.md) and
[`results/assembly_trajectory_v1.json`](results/assembly_trajectory_v1.json).

The registered support-write localization shows that evidence-dependent
eligibility supplies nearly all effective write direction, while the modeled
DA signal has a much smaller, coarse gain role. Alpha makes norm-matched writes
far more effective for the query policy but does not uniquely align them with
the exact posterior innovation. First exposures are strongly innovation
aligned; later updates attenuate without clipping and show relation-specific
assimilation by the fourth exposure. See
[`docs/support_write_localization_v1.md`](docs/support_write_localization_v1.md)
and
[`results/support_write_localization_v1.json`](results/support_write_localization_v1.json).

The registered causal factor-swap diagnostic upgrades that decomposition into
a transferable computation: matched eligibility moves relation-specific update
direction almost exactly to a donor identity, DA changes downstream magnitude
while preserving direction, and actual alpha places every retained write above
all 32 norm-matched permutation nulls in local policy gain. Exposure-4 history
primarily changes eligibility generation, but its full behavioral expression
also depends on the recipient fast-weight baseline. The resulting algorithm is
best described as state-dependent iterative relaxation toward a posterior-like
terminal potential, not sequential exact Bayesian updating. See
[`docs/support_factor_swap_v1.md`](docs/support_factor_swap_v1.md) and
[`results/support_factor_swap_v1.json`](results/support_factor_swap_v1.json).

The final registered pilot closure crosses natural versus no-prior-relation
fast-weight baselines with factors generated under those two histories. In both
pilots, history has a positive effect on recurrent expression and a smaller
positive matched interaction; the standalone total factor-generation effect is
unresolved. This localizes assimilation to a state-dependent sensitivity
landscape rather than a scalar write-strength mechanism and closes exploratory
mechanism pilots. See
[`docs/history_state_factorial_v1.md`](docs/history_state_factorial_v1.md),
[`results/history_state_factorial_v1.json`](results/history_state_factorial_v1.json),
and the frozen formal mechanism contract
[`benchmarks/mechanism_confirmation_v1.json`](benchmarks/mechanism_confirmation_v1.json).
The source-locked formal runner and serial execution commands are documented in
[`docs/mechanism_confirmation_protocol.md`](docs/mechanism_confirmation_protocol.md).

Formal seeds 2001--2010 have now been completed without filtering. All ten pass
the frozen causal, antisymmetric-geometry, fast-weight-necessity, and source-
reproduction gates. Six of seven registered mechanism links confirm across
network seeds: global remote reassembly, eligibility direction transfer, alpha
high-gain placement, history-dependent expression and interaction, and terminal
expected-rank-over-MAP projection. Within the unresolved DA link, both magnitude
contrasts confirm, but direction preservation does not because one competent
network uses a different factorization. All ten fail the strict all-scalar human
behavior conjunction, at minimum because every seed has an excessive symbolic-
distance slope. See [`docs/formal_confirmation_v1.md`](docs/formal_confirmation_v1.md),
[`results/confirmation_v1.json`](results/confirmation_v1.json), and
[`results/mechanism_confirmation_v1.json`](results/mechanism_confirmation_v1.json).

The first post-formal read-only representation audit asks whether the missing
human learned-pair component already exists as a response-hidden residual that
the fixed output suppresses. Retained relations do create a small causal local
Hodge residual in both frozen development networks, but a direction jointly
held out by subject and relation is not selectively local and does not
outperform the current output direction. Thus v2 should not begin by merely
fitting another readout to the same response state. The next audit must localize
relation-conditioned content in earlier support/fast-weight state before a new
persistent mechanism is considered. See
[`docs/hidden_residual_audit_v1.md`](docs/hidden_residual_audit_v1.md) and
[`results/hidden_residual_audit_v1.json`](results/hidden_residual_audit_v1.json).

The next frozen two-seed localization finds a non-monotonic information path.
Generated and terminal effective fast-weight matrices do not carry a shared
same-relation prototype under held-out-subject matching, but the first
fast-weight-sensitive query transition produces replicated relation identity
in both full hidden influence and its direct Hodge residual. This mixed result
rejects a static matrix-label code without proving that persistent relation
information is absent. The next read-only test must factor terminal operator
identity from query-basis identity before v2 adds storage, routing, or fidelity
machinery. See
[`docs/relation_trace_localization_v1.md`](docs/relation_trace_localization_v1.md),
[`benchmarks/relation_trace_localization_v1_1.json`](benchmarks/relation_trace_localization_v1_1.json),
and
[`results/relation_trace_localization_v1_1.json`](results/relation_trace_localization_v1_1.json).

The frozen state-by-query factorial resolves that mixed path. A terminal
relation-LOO matrix does not have a shared static relation prototype, but its
preactivation action has strong held-out-subject state identity within a fixed
query basis. Exact relation matching yields much more normalized gain than
either shared-endpoint or disjoint mismatches in both development seeds, and
the nonlinear hidden response preserves this identity. Cross-query identity
fails, so the code is a basis-dependent, query-addressable synaptic operator,
not a query-invariant relation vector. Storage and access are present; the next
read-only audit must localize where operator output fails to become a
correctness-aligned fidelity residual. See
[`docs/state_query_operator_binding_v1.md`](docs/state_query_operator_binding_v1.md),
[`benchmarks/state_query_operator_binding_v1.json`](benchmarks/state_query_operator_binding_v1.json),
and
[`results/state_query_operator_binding_v1.json`](results/state_query_operator_binding_v1.json).

The subsequent frozen operator-output semantics audit localizes the missing
fidelity link. The bound operator value `W_out^T A` is correctness-aligned in
both development seeds, and the local Jacobian `W_out^T J_b A` strengthens both
raw and normalized correctness. The exact finite-amplitude tanh response then
strongly degrades semantic direction. Aggregate exact-response correctness
remains positive, but the prospectively registered `H>A` relation reverses
from strongly correct at `A` and `J_b A` to strongly incorrect at the actual
response in both seeds. The v2 target is therefore relation-conditioned,
fidelity-preserving nonlinear expression—not a new memory store, another
readout, or a global scalar local gain. See
[`docs/operator_output_semantics_v1.md`](docs/operator_output_semantics_v1.md),
[`benchmarks/operator_output_semantics_v1.json`](benchmarks/operator_output_semantics_v1.json),
and
[`results/operator_output_semantics_v1.json`](results/operator_output_semantics_v1.json).

The final old-seed amplitude path localizes that failure to state-conditioned
finite-amplitude curvature, then closes read-only discovery on seeds 1901 and
1902. The first registered v2 intervention trains a new seed-2101 v1 backbone,
freezes it, and adapts only one curvature-norm gate parameter on the generic
training distribution. This gate preserves fast-weight necessity, remote and
third-party reassembly, query binding, nonlearned inference, and terminal
expected-rank-over-MAP projection. It nevertheless fails the registered local
rescue: H>A becomes less negative but remains robustly wrong-sign, the other
relations lose direct correctness, matched-global and shuffled controls exceed
the conditioned aggregate, and online gamma is negatively rather than
positively associated with offline crossing susceptibility. This rejects
unsigned curvature magnitude as the sufficient control variable and motivates
a separately registered sign/opposition-aware online gate before any new seed
or end-to-end v2 training. See
[`docs/operator_amplitude_path_v1.md`](docs/operator_amplitude_path_v1.md),
[`docs/curvature_gate_pilot_v2.md`](docs/curvature_gate_pilot_v2.md), and
[`results/curvature_gate_pilot_v2.json`](results/curvature_gate_pilot_v2.json).

## Original README

This is the code for the paper [Neural mechanisms of relational learning and fast knowledge reassembly in plastic neural networks](https://thomasmiconi.github.io/NN.pdf), by Thomas Miconi and Kenneth Kay, Nature Neuroscience 2024 (previous preprint [here](https://www.biorxiv.org/content/10.1101/2023.07.27.550739)).

We also include parameter files for two pre-trained networks, representing each of the two strategies (active, list-linking and passive, not list-linking) described in the paper.

The code consists of two notebooks. These notebooks are immediately usable on Google Colab, as-is.

If you just want to understand how the system works, it is **highly** recommended to look at `simple.ipynb` first.

The code actually used for the paper is in `main.ipynb`. This code includes a lot of additional code for running the various experiments from the paper. By contrast, the code in `simple.ipynb` (which only contains one large code cell) is a simplified version that only includes the basic code for meta-training a plastic network for transitive inference. The network structure and experimental settings are essentially idenctical between the two, with only the additional code for the various side experiments removed.

Note that the networks produced by `simple.ipynb` can be used in the EVAL (figure-producing) mode of `main.ipynb`.

Consult the respecitve notebooks for more details.

### To generate the figures from the paper

1. Copy `net_active.dat` to `net.dat` and upload it to where the notebook can access it.

2. In line 207 of `main.ipynb`, set EVAL to `True`

3. Run `main.ipynb` (making sure that `net.dat` is in the path of your notebook)

This produces figures for the active strategy (capable of list-linking). Other figures may need more modifications, consult the relevant cells in `main.ipynb`.

To produce similar figures for the passive strategy (not capable of list-linking), use `net_passive.dat` (and rename it to `net.dat`) instead.

### To train your own networks from scratch

1. In line 207 of `main.ipynb`, set EVAL to `False`

2. Run `main.ipynb`

This will run for 30000 iterations (which might take a few hours) and produce a fully meta-trained plastic network, stored in `net.dat`. You can then use `main.ipynb` (with EVAL set to `True` in line 207) to produce figures for this trained network.
