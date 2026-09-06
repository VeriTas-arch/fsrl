# Verification note: common encoding state v1

The formal publication reconstructed all 400 prospectively locked Liu cohorts,
all 18 fixed/trained condition streams, and every registered cohort endpoint.
The official `verify-record` command returned `passed: true` with outcome
`joint_structure_unresolved`.

Before the successful publication, operator monitoring interrupted preliminary
publication attempts prior to the first Liu shard reconstruction. One captured
stack was in generic bootstrap reconstruction and another was waiting in the
required `git ls-remote` pushed-clean check. No formal result or report existed
after those attempts, no parameters or inputs changed, and no Liu summary or
decision had been exposed. The unchanged command was then completed from the
same clean pushed commit `c83ae30a16bc7828df4e67fb8ce94cbbaaa85020`.

Low host utilization during publication reflected repeated provenance checks
and CPU reference reconstruction, not model training. SSH connection reuse was
applied only as an execution-environment optimization; it did not alter source,
artifacts, estimands, thresholds, or numerical results.

At the user's storage direction, the 20 published `liu-outputs-*.npz` copies
(about 6.6 MB each) are not stored in Git or an external large-file backend.
Their byte-identical originals remain under the local ignored runtime tree at
`artifacts/runs/common_encoding_state_v1/liu-evaluation/`; the tracked shard
JSON files retain each array path, byte count, and SHA-256. Consequently the
registered summaries and decision are reviewable from a fresh checkout, while
full array-level `verify-record` requires regenerating those deterministic
runtime caches.
