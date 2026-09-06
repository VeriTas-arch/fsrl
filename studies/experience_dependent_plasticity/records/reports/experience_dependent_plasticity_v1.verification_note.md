# Experience-dependent plasticity publication verification

The independent read-only audit at source commit
`a5f2a999c9794560f239b80946e58a62923dda1f` passed:

- command: `direnv exec . python -m tools.provenance.verify_experience_dependent_plasticity_v1`
- registered outcome: `global_schedule_preferred`
- fits reconstructed: 6
- prospective cohorts reconstructed per fit: 400
- maximum float32-versus-float64 recurrence error: `1.6211010542832582e-06`
- exact report bytes reconstructed: true
- codebook remained `[-1, -1/3, 1/3, 1]`

The original maintained verifier passed input, artifact, recurrence, cohort-point,
summary and decision checks, then failed its final report comparison because
`publish()` rendered dictionaries before sorted-key JSON serialization. The
independent audit reconstructs the original insertion order before applying the
same exact byte comparison. It does not rewrite the frozen result, report,
thresholds, parameters or scientific outcome.
