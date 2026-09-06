# Fresh global-decay replication publication verification

The independent read-only reconstruction at source commit
`1467e4e4b18cee7b9b1ab2e078a011890b79905b` passed:

- command: `direnv exec . python -m fsrl.infra.formal_runtime global-decay-replication verify-record`
- registered outcome: `replicated_global_decay`
- fits reconstructed: 6
- prospective cohorts reconstructed per fit: 400
- maximum float32-versus-float64 recurrence error: `1.8462153903442413e-06`
- exact decision reconstructed: true
- exact canonical report bytes reconstructed: true
- codebook remained `[-1, -1/3, 1/3, 1]`

The first publication attempt is preserved separately. It failed only because
mapping insertion order differed from sorted-key JSON order during exact report
reconstruction. The prospective publication repair canonicalizes mapping-valued
table cells; it does not rewrite the frozen result, thresholds, parameters,
cohorts, scientific decision, or claim boundary.
