# Scientific workflow manifest guide

This file applies to the repository-level `workflows/` tree.

Navigation: [repository guide](../AGENTS.md) ·
[current mainline](relational_model/README.md) · [study guide](../studies/AGENTS.md)
· [synthesis guide](../synthesis/AGENTS.md)

`relational_model/workflow.toml` is the canonical current claim graph. Its
generated README is the shortest human route from question to evidence, code,
tests, verification commands, and promoted figures.

## Workflow contract

- Each stage states one question, method, current finding, exact boundary, and
  explicit dependencies.
- `implementation` and `tests` name current existing files, not historical
  aliases or broad directories.
- Every material finding or boundary has exact evidence references: study ID,
  record path, optional JSON pointer, semantic use (`defines`, `supports`,
  `constrains`, or `closes`), and a short description.
- Verification commands are argv arrays with a declared CPU or GPU resource.
- Figure references resolve to registered figure IDs and do not duplicate file
  paths ad hoc.
- The set of workflow studies must cover the material evidence behind the
  working claim, including valid negatives that close tempting alternatives.

Do not edit the generated README directly or use the workflow to rewrite study
outcomes. A new scientific result first becomes a study-owned frozen record;
only then may a reviewed workflow change promote or constrain a claim.

Validate with:

```bash
direnv exec . python -m fsrl.workflows check \
  workflows/relational_model/workflow.toml
```
