# Task and observation contract guide

This file applies to `fsrl/tasks/`.

Navigation: [package guide](../AGENTS.md) ·
[current task stage](../../workflows/relational_model/README.md) ·
[task-fidelity study](../../studies/task_fidelity/README.md)

## Information boundary

- Keep observation, cognitive encoding, relational inference, and test readout
  as separate contracts.
- The ranking support representation preserves item identity and signed
  displayed relative magnitude. Random absolute bar height is nuisance;
  displayed magnitude is not.
- Do not expose true rank, test query labels, test feedback, neural-derived
  targets, or any information unavailable to participants.
- Participants passively observe four presentations of each support relation,
  make no learning response, and receive no test feedback. Do not add
  learning-stage choice or reward noise.
- Ordinalization, omission, attenuation, noisy retention, or differential
  evidence admission are explicit encoding hypotheses. Do not present them as
  properties of the experimental stimulus.

Generic training tasks may vary graphs and presentations, but evaluation
adapters must preserve the registered observation and response interfaces.
Changes to either interface require focused protocol tests and a new scientific
contract when they affect a registered estimand.
