# FSRL repository agent guide

This file applies to the complete repository. More specific `AGENTS.md` files
under the maintained source, evidence, workflow, synthesis, reproduction, test,
and provenance-tool trees add local rules. Follow the nearest guide before
editing within a subtree.

## Directory index

| Directory | Scope guide | Contents |
| --- | --- | --- |
| [`fsrl/`](fsrl/) | [`fsrl/AGENTS.md`](fsrl/AGENTS.md) | Maintained model, tasks, training, evaluation, analysis, experiments, and infrastructure |
| [`studies/`](studies/) | [`studies/AGENTS.md`](studies/AGENTS.md) | Experiment manifests, frozen records, generated human capsules, and migration ledgers |
| [`workflows/`](workflows/) | [`workflows/AGENTS.md`](workflows/AGENTS.md) | Current cross-study scientific mainlines and their schemas |
| [`synthesis/`](synthesis/) | [`synthesis/AGENTS.md`](synthesis/AGENTS.md) | Human synthesis, report figures, and immutable reporting snapshots |
| [`discussions/`](discussions/) | this guide | Non-authoritative literature, interpretation, and experiment-design discussions |
| [`tests/`](tests/) | [`tests/AGENTS.md`](tests/AGENTS.md) | Unit, scientific-contract, architecture, and repository tests |
| [`reproductions/`](reproductions/) | [`reproductions/AGENTS.md`](reproductions/AGENTS.md) | Isolated external-paper source and teaching reproductions |
| [`tools/`](tools/) | [`tools/AGENTS.md`](tools/AGENTS.md) | Versioned provenance and migration tools |
| `artifacts/` | this guide | Ignored run and reproduction outputs |
| [`data/`](data/README.md) | this guide | Tracked immutable external inputs and their dataset contracts |

The source guide links additional rules for `tasks/`, `experiments/`, `infra/`,
and package workflow code. The synthesis guide links the figure-specific guide.

## Start here

- Current claim-to-evidence route:
  [`workflows/relational_model/README.md`](workflows/relational_model/README.md)
- Diagnostic synthesis and unresolved boundaries:
  [`synthesis/README.md`](synthesis/README.md)
- Non-authoritative literature and design discussions:
  [`discussions/README.md`](discussions/README.md)
- Complete registered evidence ledger: [`studies/README.md`](studies/README.md)
- Maintained code architecture: [`fsrl/README.md`](fsrl/README.md)
- Historical reporting snapshots:
  [`synthesis/snapshots/README.md`](synthesis/snapshots/README.md)

The machine-readable workflow is the authority for the current claim graph.
Study manifests and frozen records are the authority for atomic scientific
facts. Do not turn an `AGENTS.md`, generated README, or conversational summary
into a competing evidence database.

Files under `discussions/` may explain how external work bears on the project
and why a test is proposed, deferred, or rejected. They must link observed
project claims to their workflow or study authority, label inference
separately, and must not register a result, alter a frozen boundary, or
authorize execution.

## Scientific north star

The project asks how a shared relational learning system transforms sparse and
partially encoded evidence into a stable, coherent, individualized global
structure while preserving direct experience. Behavioral reproduction is a
competence gate, not the endpoint.

1. Build the working theory from every independently supported positive result,
   retaining its protocol, estimand, controls, seed scope, provenance, and
   exact claim boundary.
2. Preserve valid negative results. Use them to replace failed causal links or
   assumptions, not to filter seeds, move thresholds, refit nuisance terms, or
   rhetorically preserve a falsified experiment-level claim.
3. Keep exploratory and confirmatory work separate. Start a new mechanism or
   workflow with one to three development seeds, then freeze the protocol and
   all mandatory artifacts before formal evaluation.
4. Prefer read-only analysis of existing artifacts before retraining. Never
   tune on confirmation seeds or pool participants across networks.
5. Keep model computation, human behavior, and biological implementation as
   distinct claim levels. A successful model intervention does not establish a
   human neural mechanism.
6. Preserve task information available to participants and withhold unavailable
   information. Any internal omission, compression, or abstraction is a model
   hypothesis that needs evidence.

The model-level evidence and one-factor transport program are currently frozen
for reporting. Organization, figures, packaging, and reproducibility work may
continue; do not start training, tuning, new evaluation, or a new scientific
estimand unless the user explicitly opens that program.

The user has opened the narrowly scoped `joint_training_strategy` comparison:
same final P/L architecture and evidence rules, matched staged versus joint
optimization on three paired new seeds. Its prospective authority is
[`studies/joint_training_strategy/`](studies/joint_training_strategy/).
Follow its protocol, implementation-lock, all-artifact-lock, then evaluation
sequence. This does not reopen historical experiments, human data collection,
closed candidate families, or unrestricted model tuning.

The user has also opened `minimal_relational_learner`: an independent compact
metric-error score learner, with the confirmed local trace and a separately
trained score-only baseline. Its prospective authority is
[`studies/minimal_relational_learner/`](studies/minimal_relational_learner/).
This is not a repair of joint training or a reconstruction of old P trajectories.
Freeze its own behavior, local-use, history-sensitivity, and complexity criteria;
do not inherit every historical intervention threshold or reopen human fitting.
Its fixed three-pair comparison is now complete and frozen for reporting.
The registered execution gates are not standing permission for another run:
extra training, evaluation, calibration, or candidate repair requires a new
explicitly authorized prospective question.

The user has opened `minimal_learner_diagnostics`: a prospective analysis of
the six already exposed minimal-learner fits, without training or calibration.
Its authority is [`studies/minimal_learner_diagnostics/`](studies/minimal_learner_diagnostics/).
Freeze the diagnostic contract and implementation/input witnesses before
execution. This admission covers encoding/integration references, fixed-readout
analysis, and exact local cross-address attribution only; it does not authorize
an encoding-noise candidate, new training, or main-model promotion.
Its fixed three-stream diagnostic execution is now complete and frozen.
Further analysis axes or model changes require a new prospective authorization;
the completed diagnostic controls are not a standing tuning program.

The user has opened `score_circuit` in an independent worktree: a frozen-parameter
test of a finite-time opponent compartment realization of score-only, not a
behavioral repair. Its prospective authority is [`studies/score_circuit/`](studies/score_circuit/).
Freeze the protocol, qualify on non-Liu fixtures, then jointly lock all three
existing fits, saved inputs and implementation before evaluation. No training,
calibration or main-model promotion is authorized. Keep the original checkout
files untouched; the detached dev-descendant worktree may push scoped, validated
commits directly to `origin/dev` without creating a remote branch.
The complete fixed matrix is now finished and frozen. This conditional circuit
result is not permission for another run, biological-parameter tuning, a new
admission mechanism, human fitting, or main-model promotion. A successor requires
a separately authorized prospective question.

## Repository invariants

1. Structural changes must not alter equations, parameters, seeds, thresholds,
   result values, or frozen claim boundaries incidentally.
2. Existing dirty or untracked files belong to the user. Stage only intended
   paths and never absorb unrelated work to make the tree clean.
3. Negative and superseded studies remain registered evidence. Historical
   source is verified from Git blobs and witness commits, not copied back into
   the active import tree.
4. Generated README files are navigation. Edit their declared TOML or JSON
   authority and rebuild them.
5. Runtime outputs belong under ignored `artifacts/runs/<workflow>/`; external
   teaching outputs belong under `artifacts/reproductions/<capsule>/`.
6. Work on `dev`, keep `main` stable, and push completed, validated, scoped
   commits to `origin/dev` after confirming a clean worktree.
7. Run repository Python and Ruff commands through `direnv exec .` and use
   `rg` or `rg --files` for searches.

## Validation boundary

Run the checks named by every applicable nested guide. The repository-wide
minimum for a completed structural change is:

```bash
direnv exec . python -m fsrl.infra.study_registry check
direnv exec . python -m fsrl.workflows check workflows/relational_model/workflow.toml
direnv exec . python -m fsrl.workflows.paper_figures check
direnv exec . basedpyright
direnv exec . python -m tools.quality.complexity_budget
direnv exec . ruff check fsrl tests tools reproductions
direnv exec . ruff format --check fsrl tests tools reproductions
direnv exec . python -m fsrl.infra.test_runtime
git diff --check
```

A physical evidence move additionally requires a new append-only migration map,
source-commit and byte verification, active-locator checks, and a frozen-evidence
verification. Never rewrite an older migration to make the current layout look
direct.
