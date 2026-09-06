# FSRL repository agent guide

This file applies to the complete repository. Follow the nearest `AGENTS.md`
before editing a subtree. Local guides add domain contracts; task execution and
validation defaults are maintained here.

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

Completed execution gates do not authorize reruns, extra cohorts, new analysis
axes, tuning, human fitting, or main-model promotion. A successor requires a
separately authorized prospective question.

## Task execution

1. Establish scope once: inspect worktree status, applicable guides, and only the
   study records, code, and callers needed for the task. Expand to resolve concrete
   dependencies or questions; routine work needs no repository-wide audit.
2. State the intended outcome and smallest required checks briefly, then complete
   the authorized scope continuously without repeated approval. Pause for a scope
   change, a protocol stop, or a decision that requires user input.
3. Close once: review the final diff, complete affected checks, summarize results
   and limits, and commit intended paths. Batch commits and pushes at required
   scientific freeze points and result archival. Existing registered sequences
   remain binding; do not add a checkpoint for every implementation step.

Use the [experiment guide](fsrl/experiments/AGENTS.md#research-lifecycle) for new
development protocols. Process simplification applies prospectively; it does not
rewrite frozen records, waive registered gates, or authorize scientific work.

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

Choose checks by the affected behavior or contract below and the local guide.
State their coverage and trigger together in the brief task plan. Documentation
inside a source or evidence directory does not itself require code or evidence
validation. Read-only questions need checks only when they help answer the
question.

| Change | Required validation |
| --- | --- |
| Hand-maintained documentation and agent guides | Relevant documentation/link tests and `git diff --check`. |
| Local code or tests | Changed-file Ruff, affected tests, and type checking when typed interfaces or assumptions change. |
| Study, registry, workflow, or figure contracts and generated views | Owning validator and affected documentation/link tests; rebuild affected views from their authorities. |
| Shared APIs, dependencies, runtime/process behavior, cross-package layout, validator schemas, or a release | The [complete engineering suite](tests/README.md#complete-bounded-suite) plus affected domain validators. |
| Historical source, evidence locators, or migration metadata | Affected provenance, migration, and frozen-evidence checks. Physical moves also follow the rule below. |
| Authorized scientific execution | The owning study's protocol, source/input/artifact locks, and scientific gates. Engineering checks do not revalidate an estimand. |

Run the smallest affected checks while iterating and the required broader checks
once the change is ready. Reuse successful checks within the task while their
relevant files, inputs, dependencies, and environment remain unchanged. A broader
passing suite covers an identical standalone check without another invocation.
Rerun only affected checks after changes, failures, or new concrete concerns.
Committing, pushing, or handing off unchanged work does not trigger another test
run or closing audit; check Git status and references for delivery instead.
Honor an explicitly requested audit or registered verification stage without
duplicating unchanged checks it already covers.

Summarize validation in the existing handoff. Do not add validation manifests,
per-stage audit reports, approval records, or tracking machinery unless they
carry required new information under the task or registered protocol.

A physical evidence move additionally requires a new append-only migration map,
source-commit and byte verification, active-locator checks, and a frozen-evidence
verification. Never rewrite an older migration to make the current layout look
direct.
