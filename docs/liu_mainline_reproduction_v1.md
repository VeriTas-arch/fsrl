# Liu Mainline v1 reproduction contract

## Outcome

Liu Mainline v1 is reproduced in three increasingly strong layers:

1. `verify` proves that frozen evidence, historical source blobs, artifact
   members, report pointers, and prospective semantic assertions remain intact.
2. `summarize` deterministically reconstructs the report tables and four main
   SVG figures from frozen JSON fields only.
3. `replay --stage ...` executes one registered historical runner from its
   detached execution-commit worktree with only its required artifacts.

The first two layers are CPU-only and are required in a clean clone. The third
is deliberately stage-specific; neural stages require a visible CUDA GPU.

## Clean-clone verification

From a clone containing the repository bundle and full Git history:

```bash
direnv allow
direnv exec . python -m fsrl.liu_mainline status
direnv exec . python -m fsrl.liu_mainline verify
direnv exec . python -m fsrl.liu_mainline doctor
direnv exec . python -m fsrl.liu_mainline restore-test-artifacts
direnv exec . python -m unittest
direnv exec . python -m fsrl.liu_mainline summarize \
  --output-dir /tmp/fsrl-mainline-clean-summary
```

`verify` reads historical files directly from the Git object database. It does
not checkout those commits and does not compare an old lock to the current
working-tree implementation.

The expected verified inventory is:

| Object | Count |
|---|---:|
| Claim nodes | 7 |
| DAG edges | 12 |
| Historical execution records | 8 |
| Explicit replay-stage names | 9 |
| Bundle members | 33: 27 replay + 6 CPU-test support |
| Report figures | 4 |
| Report metrics with JSON pointers | 26 |

## Replay stages

Every command below creates or reuses
`/tmp/fsrl-mainline/worktrees/<execution_commit>`. Results default to
`/tmp/fsrl-mainline/outputs/<stage>/result.json`; replay refuses to overwrite an
existing result. Use `--output` to choose another new path.

| Stage | Runtime | Scientific role |
|---|---|---|
| `behavioral_competence` | CPU | Reassemble the frozen nine-phenomenon map without checkpoint load |
| `global_reassembly` | GPU | Re-evaluate the fresh-backbone v2.4 causal system, reading the global links |
| `local_direct_fidelity` | GPU | Re-evaluate the same v2.4 system, reading local rescue/specificity links |
| `algorithmic_asymmetry` | CPU | Replay the final registered item-history reduction failure |
| `support_topology_transport` | GPU | Replay three matched N=8 graph alternatives |
| `presentation_order_transport` | GPU | Replay blockwise, clustered, and reverse schedules |
| `evidence_sparsity_transport` | GPU | Replay E=7/8/9/10 transport and retain the E=10 miss |
| `sparsity_individualization_localization` | CPU | Replay the read-only density localization |
| `item_count_transport` | GPU | Replay N=6/8/10 transport |

Example:

```bash
direnv exec . python -m fsrl.liu_mainline doctor \
  --stage global_reassembly
direnv exec . python -m fsrl.liu_mainline replay \
  --stage global_reassembly \
  --output /tmp/fsrl-mainline/outputs/global-reassembly-check.json
```

There is no aggregate replay command. A caller must name one stage explicitly.

## Exact and semantic outcomes

Each execution record contains both contracts:

```json
{
  "replay_policy": {
    "exact": {
      "allowed_environment": "liu-mainline-v1-linux-cuda130",
      "expected_sha256": "..."
    },
    "semantic": {
      "assertions": [
        {
          "json_pointer": "/decision/outcome",
          "operator": "equals",
          "expected": "..."
        }
      ]
    }
  }
}
```

Byte identity is the strongest result. A semantic pass on a different runtime
does not retroactively weaken the exact contract, and an exact mismatch is not
permission to set a new tolerance. The CLI reports both outcomes separately.
It also reports `replay_outcome=exact_and_semantic` or `semantic_only`. Historical
results that serialize checkout-specific absolute paths cannot be byte-identical
inside a detached worktree; GPU reductions can additionally differ in their
last floating-point bits. Those differences remain visible rather than being
normalized away.

## Artifact restoration

The repository bundle is named by its own SHA-256. `artifacts.json` records for
each member:

- logical name;
- bundle member path;
- member SHA-256;
- byte size;
- producing historical stage;
- replay stages that require it.

Replay extracts only matching members. If a target file already exists, its
hash must match; otherwise replay fails rather than replacing it. The bundle is
small enough for the repository backend. Moving it to a release asset or LFS in
a later mainline version must preserve every member identity.

`restore-test-artifacts` applies the same fail-closed extraction rule to the six
seed-1901/1902 files referenced by legacy source-integrity tests. It does not
train or evaluate a model. These files live under ignored `output/` paths, so a
clean clone remains Git-clean after restoration and the full CPU suite.

## Environment reconstruction

`environment.json` records the verified host and the runtime fields embedded in
frozen results. `requirements-lock.txt` separately pins the reconstructable
Python closure and the official CUDA 13.0 PyTorch wheel index. `.envrc` remains
a local convenience and is not the scientific environment definition.

The CPU verifier and summarizer intentionally work when the GPU is hidden by a
sandbox. GPU replay readiness is assessed only when a GPU stage is requested.

## Report provenance

`summarize` writes:

```text
liu_mainline_summary.json
liu_mainline_tables.csv
figure_1_behavioral_target.svg
figure_2_causal_mechanism.svg
figure_3_algorithmic_asymmetry.svg
figure_4_transport_boundaries.svg
```

Each metric carries this chain:

```text
figure metric
 -> table row
 -> source path and SHA-256
 -> exact JSON pointer
 -> frozen evidence file
 -> claim node
```

The command performs no checkpoint load, Torch/model evaluation, participant
resampling, bootstrap, threshold change, or new aggregation beyond deterministic
field extraction and formatting.

## Freeze boundary

Before the annotated `liu-mainline-v1` tag is created, all of the following must
pass on pushed `dev`:

- full repository test suite;
- overlay `verify` and deterministic summarize comparison;
- clean-clone CPU verification;
- at least one GPU fixed-artifact stage replay;
- clean working tree after validation.

After the tag, v1 is immutable. Corrections use an erratum; scientific changes
use Liu v2.

The completed checks and their exact/semantic boundaries are recorded in
`mainlines/liu_v1/validation.json`. This attestation is evidence about the
reproduction process; it does not add a scientific estimand or change any
historical result.
