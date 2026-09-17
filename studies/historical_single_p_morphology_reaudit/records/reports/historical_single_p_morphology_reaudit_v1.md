# Current-standard morphology re-audit of historical alpha single-P

Registered outcome: `no_current_standard_precedent`.

Across all 18 clean-trained acute-noisy Ae units, 0 met the later constrained morphology standard, 0 passed all nine qualitative rows, and 18 met the paired A0-to-Ae error-inflation definition.

## Family-specific results

| historical family | competent | binding | all nine | constrained | error inflation | seeds with constrained panel | replicated precedent |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_single_p | 9/9 | 9/9 | 0/9 | 0/9 | 9/9 | 0/3 | false |
| local_memory_removal | 9/9 | 9/9 | 0/9 | 0/9 | 9/9 | 0/3 | false |

| historical family | failed qualitative rows across Ae units | A0 sampled bimodal mean | Ae minus A0 mean |
|---|---|---:|---:|
| clean_single_p | `{'difficult_pair_bimodality': 9}` | 8.222 | 3.667 |
| local_memory_removal | `{'difficult_pair_bimodality': 9}` | 7.667 | 3.444 |

| historical family | Ae stage distribution | mean latent bimodal pairs | mean sampled bimodal pairs | mean top-5 strong-error share |
|---|---|---:|---:|---:|
| clean_single_p | `{'latent_shape': 8, 'weak_margin': 1}` | 10.889 | 11.889 | 0.625 |
| local_memory_removal | `{'latent_shape': 6, 'weak_margin': 3}` | 10.444 | 11.111 | 0.654 |

## Interpretation boundary

The historical five-core and compensation findings remain exactly as registered. This audit asks the narrower retrospective question of whether those archived success cells also met a later, stricter pair-morphology standard; it does not rewrite the original gates.

All 36 mandatory A0/Ae units were retained. Full and global sampled behavior was deterministically replayed, and archived fields, probabilities, Hodge potentials, and decomposition identities passed the frozen tolerances.

No training, checkpoint loading, model inference, rescaling, extra choices, participant pooling, or selection was performed. The protocol stops here and does not authorize another alpha recipe or initialization factorial.
