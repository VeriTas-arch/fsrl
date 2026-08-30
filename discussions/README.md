# Research discussions

This directory records maintained discussions that connect external literature
to the FSRL project, make interpretation choices explicit, and preserve the
reasoning behind proposed or rejected experiments.

## Authority boundary

These documents are navigation and reasoning aids, not scientific evidence.

- The current claim graph remains
  [`workflows/relational_model/workflow.toml`](../workflows/relational_model/workflow.toml).
- Atomic results, estimands, protocols, and claim boundaries remain in
  [`studies/`](../studies/README.md).
- The current cross-study account remains in
  [`synthesis/`](../synthesis/README.md).
- A proposal recorded here does not authorize training, evaluation, human-data
  collection, parameter fitting, or a new estimand.
- If a discussion becomes an experiment, register the new study prospectively
  and leave the earlier discussion as non-evidentiary context.

Discussion documents may be revised as literature or registered project
evidence changes. They should separate external facts, verified repository
evidence, interpretation, decisions, and unresolved questions.

## Index

| Discussion | Scope | Current decision |
| --- | --- | --- |
| [Yang and Maass (2026): local order learning](yang_maass_order_learning.md) | Relation between a local rank-learning rule and the frozen FSRL global/local decomposition | Narrow the claim now; retain a model-side stress test as optional and the registered magnitude-placement program as the decisive extension |
