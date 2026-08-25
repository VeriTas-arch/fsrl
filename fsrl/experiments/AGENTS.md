# Evidence-producing experiment guide

This file applies to `fsrl/experiments/`.

Navigation: [package guide](../AGENTS.md) · [study guide](../../studies/AGENTS.md)
· [current mainline](../../workflows/relational_model/README.md) ·
[diagnostic synthesis](../../synthesis/README.md)

## Research lifecycle

For each substantive stage, record in order: the question and current theory;
the prospective protocol, estimands, controls, competence gates, and stop
rules; the analysis or experiment; positive and negative results with
uncertainty; supported, rejected, and unidentified links; the revised theory;
and the next discriminating test.

- Prefer read-only diagnosis before intervention and use one to three
  development seeds before freezing formal work.
- Never edit a frozen candidate, contract, seed set, threshold, or outcome rule
  after seeing its result. Register a successor study instead.
- Preserve competence and integrity gates. A failed competence gate is
  non-interpretable; a valid below-threshold result is evidence.
- Train and adapt every mandatory backbone before inspecting confirmation
  outcomes. Analyze participants within network; do not pool networks as a
  population sample.
- Carry forward supported links when a candidate fails, but close the failed
  causal family according to its registered stop rule.
- Do not add relation labels, hard-case flags, offline targets, posterior
  targets, or evaluation labels to make a candidate pass.

The current model program is frozen for reporting. Read the workflow and study
registry for current boundaries; do not infer current status from runner names.
Starting an experiment requires explicit user authorization for a new
scientific program.

## Implementation boundary

Experiment modules may compose public core, task, training, evaluation, and
analysis APIs. They must not become dependencies of those stable layers.
One-off study logic stays with its experiment family; genuinely reusable pure
estimators move to `analysis/` only when multiple maintained callers need them.
Every promoted output must have one registered study owner.
