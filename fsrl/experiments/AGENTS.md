# Evidence-producing experiment guide

This file applies to `fsrl/experiments/`.

Navigation: [package guide](../AGENTS.md) · [study guide](../../studies/AGENTS.md)
· [current mainline](../../workflows/relational_model/README.md) ·
[diagnostic synthesis](../../synthesis/README.md)

## Research lifecycle

Apply the [root task execution rules](../../AGENTS.md#task-execution) to each new
authorized development study:

1. Before execution, write a compact prospective protocol using the outline
   below, qualify the affected computation, and record source/configuration/input
   identities in the required locks before the corresponding execution or outcome
   exposure. Reference unchanged task, metric, and runtime contracts.
2. Execute continuously with the mandatory correctness, competence, integrity,
   and stop checks.
3. Close once with the canonical machine-readable result, the artifacts needed
   to reconstruct it, and a concise interpretation: positive and negative results
   with uncertainty, supported/rejected/unidentified links, revised theory, and
   the next discriminating question. Update the owning study and generated views.

Use this outline in the study's existing protocol format; it is not a new schema
or an additional document:

```text
Question: current theory, primary contrast/estimand, and intended claim boundary.
Design: development or confirmation, controls, seeds, cohorts, and fixed inputs.
Decision: uncertainty method, competence/integrity gates, and stop/outcome rules.
Execution: command, runtime, source/configuration/RNG identities, and freeze points.
Outputs: required locks, sufficient artifacts, canonical result, and interpretation.
```

Do not copy another study's admission records, review reports, or commit/push
sequence into a new protocol unless the design needs them. Required locks precede
the execution or outcome exposure they protect.

## Scientific gates

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

Development uses one to three seeds, the minimum prospectively fixed cohort count
needed to discriminate the primary contrast, focused parity checks, and compact
sufficient statistics. Promote raw outputs only when needed to reconstruct the
estimand; otherwise keep them in ignored runtime artifacts. Schedule complete
independent numerical reconstruction at registered confirmation or promotion
gates, or for a concrete correctness concern; it is not a default development
step. A new or changed equation needs a focused independent correctness check
before use. Full engineering checks follow the
[root validation boundary](../../AGENTS.md#validation-boundary) and registered gates.

In either tier, freeze the protocol before outcome exposure, lock mandatory
artifacts before evaluation, and do not tune on evaluated seeds or cohorts.
Development results retain their seed scope; extra engineering checks do not
make them confirmatory. The [root scientific boundary](../../AGENTS.md#scientific-north-star)
governs authorization; runner names do not establish study status.

## Implementation boundary

Experiment modules may compose public core, task, training, evaluation, and
analysis APIs. They must not become dependencies of those stable layers.
One-off study logic stays with its experiment family; genuinely reusable pure
estimators move to `analysis/` only when multiple maintained callers need them.
Every promoted output must have one registered study owner.
