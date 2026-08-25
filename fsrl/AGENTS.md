# Maintained package guide

This file applies to `fsrl/`. More specific guides under `tasks/`,
`experiments/`, `infra/`, and `workflows/` add subtree rules.

Navigation: [repository guide](../AGENTS.md) · [package architecture](README.md)
· [task guide](tasks/AGENTS.md) · [experiment guide](experiments/AGENTS.md) ·
[infrastructure guide](infra/AGENTS.md) · [workflow-code guide](workflows/AGENTS.md)

## Package ownership

- `core/` owns checkpoint-compatible recurrent equations, typed state, the
  input ABI, the confirmed local trace, and the P/L rollout API.
- `tasks/` owns task protocols, generic graph generation, and evidence
  admission; it follows the task-information boundary in its local guide.
- `training/` owns generic backbone optimization and checkpoint loading.
- `evaluation/` owns frozen rollout and causal intervention interfaces.
- `analysis/` owns reusable estimators and pure transformations.
- `experiments/` owns evidence-producing runners, not reusable core objects.
- `infra/` owns runtime, registry, provenance, logging, and bounded execution.
- `workflows/` validates and renders repository-level workflow and figure
  contracts.
- `paths.py` is the only repository-root path contract.

## Dependency and design rules

1. Follow the explicit dependency graph enforced by
   `tests/core/test_architecture.py`; stable layers must not import
   `experiments` or plotting workflows.
2. Reusable modules must be safe to import and must not train, evaluate, write
   artifacts, or render figures on import.
3. Preserve checkpoint state keys, input layout, numerical order, and frozen
   evaluator semantics unless the task explicitly authorizes a scientific
   change with a new contract.
4. Extract a shared helper only when more than one maintained caller needs it.
   Do not recreate removed flat modules as compatibility wrappers without a
   concrete active consumer.
5. Name code by scientific role, not development chronology. Historical
   version names may remain only where they identify a frozen protocol or
   artifact.
6. New code must be Ruff-formatted. Do not add formatter exclusion lists for
   maintained or historical runner files.

## Validation

Run Ruff on changed package and test files, the smallest affected test modules,
and the broader package suite when dependency boundaries or shared APIs change.
Use the bounded test runtime for the complete suite. A scientific numerical
change also requires its registered workflow gates; import and unit tests alone
do not revalidate an estimand.
