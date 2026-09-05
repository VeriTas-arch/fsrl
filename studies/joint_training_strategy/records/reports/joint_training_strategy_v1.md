# Matched staged versus single-stage joint training

Registered outcome: **`competent_but_not_noninferior`**.

## Question and frozen comparison

Can one final query objective replace sequential global/local fitting at the same generic episode budget, while preserving competence, causal organization, and the frozen Liu behavior map?
Both conditions use the same imposed P/L structure, final evidence admission, paired initialization and task stream. Each sees 48,000 training episodes. Staged fitting updates the backbone 1,000 times and gain 500 times; joint fitting updates both 1,500 times. This is not an order-only or equal-FLOPs experiment.

Protocol SHA-256: `af6fe7ccd0785cf5ec09437cd958f5652a36596066aec28a77793dad28133ea2`. Implementation witness: `837f350ce1fbd52c37e89dc5abe22f2e68cb5e46`.
All six final artifacts were jointly locked and pushed before any new evaluation. Bootstrap is within each network (10,000 draws), with no participant pooling or network-population bootstrap. Probability endpoints average the two orientation-specific sigmoids; decision ties count as one half.

## Per-seed primary results

| Seed | Staged competence | Joint competence | Paired NI | Staged mechanism | Joint mechanism | Joint behavior | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2108 | PASS | PASS | FAIL | FAIL | FAIL | PASS | `competent_but_not_noninferior` |
| 2109 | PASS | PASS | PASS | FAIL | FAIL | FAIL | `alternative_computational_solution` |
| 2110 | PASS | PASS | PASS | FAIL | FAIL | PASS | `alternative_computational_solution` |

## Seed 2108

### Paired correct-probability noninferiority

| Endpoint | Joint minus staged | 95% lower | 95% upper | N | Pass (LB ≥ -0.02) |
| --- | --- | --- | --- | --- | --- |
| generic/learned | 0.00481 | 0.00247 | 0.00707 | 256 | PASS |
| generic/nonlearned | 0.01393 | 0.01161 | 0.01625 | 256 | PASS |
| liu/learned | -0.00338 | -0.01066 | 0.00335 | 77 | PASS |
| liu/nonlearned | 0.00982 | 0.00416 | 0.01551 | 77 | PASS |
| liu/retained | -0.00190 | -0.00905 | 0.00422 | 77 | PASS |
| liu/omitted | -0.00313 | -0.02701 | 0.01913 | 69 | FAIL |

### matched_staged: causal links

| Link / endpoint | Bound value | Registered criterion | Pass |
| --- | --- | --- | --- |
| direct_local_fidelity/intact_minus_local_off_omitted | 0.04013 | lower >= 0.01 | PASS |
| direct_local_fidelity/intact_minus_local_off_retained | 0.00448 | lower >= 0.01 | FAIL |
| global_necessity/P_off_nonlearned | 0.43973 | upper <= 0.55 | PASS |
| global_necessity/intact_minus_P_off_nonlearned | 0.35461 | lower >= 0.1 | PASS |
| local_only_partition/P_off_learned | 0.73529 | lower > 0.5 | PASS |
| local_only_partition/P_off_nonlearned | 0.43973 | upper <= 0.55 | PASS |
| local_only_partition/local_remote_minus_quarter_combined | -0.05292 | upper <= 0.0 | PASS |
| query_evidence_specificity/intact_minus_evidence_shuffle_learned | 0.02209 | lower >= 0.01 | PASS |
| query_evidence_specificity/intact_minus_query_shuffle_learned | 0.02070 | lower >= 0.01 | PASS |
| remote_reassembly/global_remote_absolute | 0.33696 | lower > 0.01 | PASS |
| remote_reassembly/global_third_party_relational | 0.21073 | lower > 0.05 | PASS |

### matched_staged: all nine behavior rows

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | FAIL |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | PASS |
| nonlearned_accuracy | PASS | PASS |
| self_consistent_vs_inconsistent_errors | PASS | FAIL |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Own-global legacy qualification (secondary): PASS.

### joint: causal links

| Link / endpoint | Bound value | Registered criterion | Pass |
| --- | --- | --- | --- |
| direct_local_fidelity/intact_minus_local_off_omitted | 0.03867 | lower >= 0.01 | PASS |
| direct_local_fidelity/intact_minus_local_off_retained | 0.00411 | lower >= 0.01 | FAIL |
| global_necessity/P_off_nonlearned | 0.43865 | upper <= 0.55 | PASS |
| global_necessity/intact_minus_P_off_nonlearned | 0.36320 | lower >= 0.1 | PASS |
| local_only_partition/P_off_learned | 0.74181 | lower > 0.5 | PASS |
| local_only_partition/P_off_nonlearned | 0.43865 | upper <= 0.55 | PASS |
| local_only_partition/local_remote_minus_quarter_combined | -0.06043 | upper <= 0.0 | PASS |
| query_evidence_specificity/intact_minus_evidence_shuffle_learned | 0.02152 | lower >= 0.01 | PASS |
| query_evidence_specificity/intact_minus_query_shuffle_learned | 0.01746 | lower >= 0.01 | PASS |
| remote_reassembly/global_remote_absolute | 0.36380 | lower > 0.01 | PASS |
| remote_reassembly/global_third_party_relational | 0.25839 | lower > 0.05 | PASS |

### joint: all nine behavior rows

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | PASS |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | PASS |
| nonlearned_accuracy | PASS | PASS |
| self_consistent_vs_inconsistent_errors | PASS | FAIL |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Own-global legacy qualification (secondary): PASS.

Joint/staged cost ratios: `{'warm_training_seconds': 1.2020436789913458, 'peak_allocated_bytes': 1.0005614053855458}`.

## Seed 2109

### Paired correct-probability noninferiority

| Endpoint | Joint minus staged | 95% lower | 95% upper | N | Pass (LB ≥ -0.02) |
| --- | --- | --- | --- | --- | --- |
| generic/learned | 0.00869 | 0.00643 | 0.01094 | 256 | PASS |
| generic/nonlearned | 0.01469 | 0.01266 | 0.01681 | 256 | PASS |
| liu/learned | 0.00142 | -0.00438 | 0.00698 | 77 | PASS |
| liu/nonlearned | 0.01182 | 0.00755 | 0.01602 | 77 | PASS |
| liu/retained | 0.00066 | -0.00262 | 0.00376 | 77 | PASS |
| liu/omitted | 0.00512 | -0.01381 | 0.02342 | 69 | PASS |

### matched_staged: causal links

| Link / endpoint | Bound value | Registered criterion | Pass |
| --- | --- | --- | --- |
| direct_local_fidelity/intact_minus_local_off_omitted | 0.05661 | lower >= 0.01 | PASS |
| direct_local_fidelity/intact_minus_local_off_retained | 0.00493 | lower >= 0.01 | FAIL |
| global_necessity/P_off_nonlearned | 0.44753 | upper <= 0.55 | PASS |
| global_necessity/intact_minus_P_off_nonlearned | 0.34195 | lower >= 0.1 | PASS |
| local_only_partition/P_off_learned | 0.75214 | lower > 0.5 | PASS |
| local_only_partition/P_off_nonlearned | 0.44753 | upper <= 0.55 | PASS |
| local_only_partition/local_remote_minus_quarter_combined | -0.05305 | upper <= 0.0 | PASS |
| query_evidence_specificity/intact_minus_evidence_shuffle_learned | 0.02563 | lower >= 0.01 | PASS |
| query_evidence_specificity/intact_minus_query_shuffle_learned | 0.02412 | lower >= 0.01 | PASS |
| remote_reassembly/global_remote_absolute | 0.33191 | lower > 0.01 | PASS |
| remote_reassembly/global_third_party_relational | 0.20229 | lower > 0.05 | PASS |

### matched_staged: all nine behavior rows

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | PASS |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | FAIL |
| nonlearned_accuracy | PASS | FAIL |
| self_consistent_vs_inconsistent_errors | PASS | FAIL |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Own-global legacy qualification (secondary): PASS.

### joint: causal links

| Link / endpoint | Bound value | Registered criterion | Pass |
| --- | --- | --- | --- |
| direct_local_fidelity/intact_minus_local_off_omitted | 0.04010 | lower >= 0.01 | PASS |
| direct_local_fidelity/intact_minus_local_off_retained | 0.00348 | lower >= 0.01 | FAIL |
| global_necessity/P_off_nonlearned | 0.44367 | upper <= 0.55 | PASS |
| global_necessity/intact_minus_P_off_nonlearned | 0.35524 | lower >= 0.1 | PASS |
| local_only_partition/P_off_learned | 0.77333 | lower > 0.5 | PASS |
| local_only_partition/P_off_nonlearned | 0.44367 | upper <= 0.55 | PASS |
| local_only_partition/local_remote_minus_quarter_combined | -0.05968 | upper <= 0.0 | PASS |
| query_evidence_specificity/intact_minus_evidence_shuffle_learned | 0.02237 | lower >= 0.01 | PASS |
| query_evidence_specificity/intact_minus_query_shuffle_learned | 0.01852 | lower >= 0.01 | PASS |
| remote_reassembly/global_remote_absolute | 0.35946 | lower > 0.01 | PASS |
| remote_reassembly/global_third_party_relational | 0.23438 | lower > 0.05 | PASS |

### joint: all nine behavior rows

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | PASS |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | FAIL |
| nonlearned_accuracy | PASS | PASS |
| self_consistent_vs_inconsistent_errors | PASS | PASS |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Own-global legacy qualification (secondary): PASS.

Joint/staged cost ratios: `{'warm_training_seconds': 1.1964728288330735, 'peak_allocated_bytes': 1.0000037199466603}`.

## Seed 2110

### Paired correct-probability noninferiority

| Endpoint | Joint minus staged | 95% lower | 95% upper | N | Pass (LB ≥ -0.02) |
| --- | --- | --- | --- | --- | --- |
| generic/learned | 0.00606 | 0.00435 | 0.00781 | 256 | PASS |
| generic/nonlearned | 0.01012 | 0.00835 | 0.01191 | 256 | PASS |
| liu/learned | 0.00466 | 0.00055 | 0.00887 | 77 | PASS |
| liu/nonlearned | 0.00920 | 0.00540 | 0.01320 | 77 | PASS |
| liu/retained | 0.00252 | -0.00052 | 0.00572 | 77 | PASS |
| liu/omitted | 0.01329 | -0.00082 | 0.02849 | 69 | PASS |

### matched_staged: causal links

| Link / endpoint | Bound value | Registered criterion | Pass |
| --- | --- | --- | --- |
| direct_local_fidelity/intact_minus_local_off_omitted | 0.04989 | lower >= 0.01 | PASS |
| direct_local_fidelity/intact_minus_local_off_retained | 0.00592 | lower >= 0.01 | FAIL |
| global_necessity/P_off_nonlearned | 0.46551 | upper <= 0.55 | PASS |
| global_necessity/intact_minus_P_off_nonlearned | 0.31977 | lower >= 0.1 | PASS |
| local_only_partition/P_off_learned | 0.71055 | lower > 0.5 | PASS |
| local_only_partition/P_off_nonlearned | 0.46551 | upper <= 0.55 | PASS |
| local_only_partition/local_remote_minus_quarter_combined | -0.05366 | upper <= 0.0 | PASS |
| query_evidence_specificity/intact_minus_evidence_shuffle_learned | 0.02211 | lower >= 0.01 | PASS |
| query_evidence_specificity/intact_minus_query_shuffle_learned | 0.02030 | lower >= 0.01 | PASS |
| remote_reassembly/global_remote_absolute | 0.32395 | lower > 0.01 | PASS |
| remote_reassembly/global_third_party_relational | 0.19577 | lower > 0.05 | PASS |

### matched_staged: all nine behavior rows

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | PASS |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | PASS |
| nonlearned_accuracy | PASS | FAIL |
| self_consistent_vs_inconsistent_errors | PASS | PASS |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Own-global legacy qualification (secondary): PASS.

### joint: causal links

| Link / endpoint | Bound value | Registered criterion | Pass |
| --- | --- | --- | --- |
| direct_local_fidelity/intact_minus_local_off_omitted | 0.04413 | lower >= 0.01 | PASS |
| direct_local_fidelity/intact_minus_local_off_retained | 0.00486 | lower >= 0.01 | FAIL |
| global_necessity/P_off_nonlearned | 0.45889 | upper <= 0.55 | PASS |
| global_necessity/intact_minus_P_off_nonlearned | 0.33496 | lower >= 0.1 | PASS |
| local_only_partition/P_off_learned | 0.72845 | lower > 0.5 | PASS |
| local_only_partition/P_off_nonlearned | 0.45889 | upper <= 0.55 | PASS |
| local_only_partition/local_remote_minus_quarter_combined | -0.05882 | upper <= 0.0 | PASS |
| query_evidence_specificity/intact_minus_evidence_shuffle_learned | 0.02126 | lower >= 0.01 | PASS |
| query_evidence_specificity/intact_minus_query_shuffle_learned | 0.01743 | lower >= 0.01 | PASS |
| remote_reassembly/global_remote_absolute | 0.34620 | lower > 0.01 | PASS |
| remote_reassembly/global_third_party_relational | 0.22107 | lower > 0.05 | PASS |

### joint: all nine behavior rows

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | PASS |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | PASS |
| nonlearned_accuracy | PASS | PASS |
| self_consistent_vs_inconsistent_errors | PASS | PASS |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Own-global legacy qualification (secondary): PASS.

Joint/staged cost ratios: `{'warm_training_seconds': 1.20897683407754, 'peak_allocated_bytes': 1.0000037353393425}`.
## Supported, rejected, and unidentified claims

The tables retain every registered link independently. A passed competence or noninferiority comparison does not by itself establish preserved mechanism; preserved mechanism does not substitute for the nine-row behavior map. No successful network repairs a failed mandatory network.
Single-stage preservation, if all registered gates pass, establishes recipe-level feasibility under imposed structural priors and can motivate independent confirmation. It does not establish minimal architecture, autonomous emergence of two memories, human neural implementation, or a population-level network effect.
If a stronger conjunction fails, the passing links remain positive evidence at their registered scope. The outcome labels distinguish recipe insufficiency, comparator insufficiency, noninferiority failure, an alternative computation, and incomplete behavior preservation.

## Costs, calibration, and next step

Measured joint efficiency advantage in every pair: **FAIL**. Stage count alone is not a compute claim. Compilation/warmup, measured training, peak allocated/reserved memory, and parameter/update counts are retained in the numerical records.
Liu temperature 0.25 is inherited historical human-informed calibration; it was not refitted. Human interval membership remains descriptive, not model-human equivalence. Old distance, serial-position, and self-inconsistency mismatches were not required to persist.
After this complete registered comparison, freeze the result and report positive, negative, and unresolved links. No automatic extra seeds, learning-rate sweep, gain/temperature calibration, auxiliary losses, alternative readouts, input redesign, or new task expansion follows from failure. A new scientific candidate requires a separately authorized and prospectively registered question.

The JSON result retains exact estimates, denominators, bootstrap bounds, human-map metrics, secondary qualification/projection diagnostics, and provenance. Registered typed NPZ files preserve raw oriented margins, controls, LOO arrays, evidence/routing, and generic inputs; sampled behavior is available through its verified runtime manifest.
