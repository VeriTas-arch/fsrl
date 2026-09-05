# Minimal metric-error relational learner

Registered outcome: **`score_only_sufficient`**.

Independent fixed-recipe test: normalized online score learning with an optional unchanged conjunctive local trace. This does not repair the old joint-training outcome or reconstruct P trajectories.
Three paired training streams (2111/2112/2113), identical scalar initialization, 48,000 episodes per model. All six artifacts locked before evaluation. Participant bootstrap is separate within each fit, never pooled across training streams.

## Per-run behavior and competence

| Stream/recipe | Competent | Nine-row adequacy | Learned scalars | Persistent state entries | Warm training seconds |
| --- | --- | --- | --- | --- | --- |
| 2111/score_trace | PASS | PASS | 3 | 240 | 20.93905 |
| 2111/score_only | PASS | PASS | 2 | 15 | 19.72877 |
| 2112/score_trace | PASS | PASS | 3 | 240 | 22.60608 |
| 2112/score_only | PASS | PASS | 2 | 15 | 22.01057 |
| 2113/score_trace | PASS | PASS | 3 | 240 | 22.09815 |
| 2113/score_only | PASS | PASS | 2 | 15 | 21.40387 |

## Training stream 2111: `score_only_sufficient`

Score-trace minus independently fitted score-only correct probability:

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| generic/learned | 0.01609 | 0.01494 | 0.01726 | 256 |
| generic/nonlearned | -0.00516 | -0.00593 | -0.00441 | 256 |
| liu/learned | 0.00686 | 0.00408 | 0.01003 | 77 |
| liu/nonlearned | -0.01001 | -0.01341 | -0.00691 | 77 |
| liu/retained | 0.00177 | 0.00061 | 0.00328 | 77 |
| liu/omitted | 0.01759 | 0.00788 | 0.02707 | 69 |

Acute local use (separate from independent fitting):

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| intact_minus_local_off_retained | 0.00183 | 0.00067 | 0.00333 | 77 |
| intact_minus_local_off_omitted | 0.01762 | 0.00792 | 0.02710 | 69 |
| intact_minus_query_shuffle_learned | 0.00897 | 0.00391 | 0.01434 | 77 |
| intact_minus_evidence_shuffle_learned | 0.01226 | 0.00778 | 0.01712 | 77 |

| Local-support requirement | Lower bound | Threshold | Pass |
| --- | --- | --- | --- |
| omitted_contribution | 0.00792 | > 0.0 | PASS |
| query_specificity | 0.00391 | > 0.0 | PASS |
| evidence_specificity | 0.00778 | > 0.0 | PASS |
| between_recipe | 0.00788 | > 0.0 | PASS |
| retained_preservation | 0.00061 | >= -0.02 | PASS |

## Training stream 2112: `score_only_sufficient`

Score-trace minus independently fitted score-only correct probability:

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| generic/learned | 0.01507 | 0.01401 | 0.01614 | 256 |
| generic/nonlearned | -0.00482 | -0.00553 | -0.00411 | 256 |
| liu/learned | 0.00647 | 0.00386 | 0.00946 | 77 |
| liu/nonlearned | -0.00918 | -0.01229 | -0.00618 | 77 |
| liu/retained | 0.00169 | 0.00062 | 0.00304 | 77 |
| liu/omitted | 0.01648 | 0.00762 | 0.02553 | 69 |

Acute local use (separate from independent fitting):

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| intact_minus_local_off_retained | 0.00175 | 0.00068 | 0.00310 | 77 |
| intact_minus_local_off_omitted | 0.01651 | 0.00765 | 0.02553 | 69 |
| intact_minus_query_shuffle_learned | 0.00825 | 0.00353 | 0.01350 | 77 |
| intact_minus_evidence_shuffle_learned | 0.01129 | 0.00725 | 0.01586 | 77 |

| Local-support requirement | Lower bound | Threshold | Pass |
| --- | --- | --- | --- |
| omitted_contribution | 0.00765 | > 0.0 | PASS |
| query_specificity | 0.00353 | > 0.0 | PASS |
| evidence_specificity | 0.00725 | > 0.0 | PASS |
| between_recipe | 0.00762 | > 0.0 | PASS |
| retained_preservation | 0.00062 | >= -0.02 | PASS |

## Training stream 2113: `score_only_sufficient`

Score-trace minus independently fitted score-only correct probability:

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| generic/learned | 0.01503 | 0.01398 | 0.01612 | 256 |
| generic/nonlearned | -0.00479 | -0.00552 | -0.00407 | 256 |
| liu/learned | 0.00644 | 0.00385 | 0.00933 | 77 |
| liu/nonlearned | -0.00916 | -0.01224 | -0.00614 | 77 |
| liu/retained | 0.00168 | 0.00060 | 0.00305 | 77 |
| liu/omitted | 0.01642 | 0.00755 | 0.02563 | 69 |

Acute local use (separate from independent fitting):

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| intact_minus_local_off_retained | 0.00174 | 0.00066 | 0.00310 | 77 |
| intact_minus_local_off_omitted | 0.01645 | 0.00760 | 0.02564 | 69 |
| intact_minus_query_shuffle_learned | 0.00822 | 0.00350 | 0.01316 | 77 |
| intact_minus_evidence_shuffle_learned | 0.01125 | 0.00719 | 0.01570 | 77 |

| Local-support requirement | Lower bound | Threshold | Pass |
| --- | --- | --- | --- |
| omitted_contribution | 0.00760 | > 0.0 | PASS |
| query_specificity | 0.00350 | > 0.0 | PASS |
| evidence_specificity | 0.00719 | > 0.0 | PASS |
| between_recipe | 0.00755 | > 0.0 | PASS |
| retained_preservation | 0.00060 | >= -0.02 | PASS |

## 2111/score_trace: complete behavior map

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | FAIL |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | FAIL |
| nonlearned_accuracy | PASS | FAIL |
| self_consistent_vs_inconsistent_errors | PASS | FAIL |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Constrained parameters: `{'eta': 0.9886998534202576, 'gamma_G': 7.14223575592041, 'gamma_L': 0.24688567221164703}`.
Inference seconds per episode (warm compiled batch): `4.802831084249759e-06`.
Float64 analytic bridge versus float32 rollout maximum absolute error: `1.2110494607142641e-06`.

History effect magnitudes:

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| all_direct_sensitivity | 0.93018 | 0.88218 | 0.97679 | 77 |
| all_history_sensitivity | 0.80097 | 0.75704 | 0.84404 | 77 |
| all_sensitivity | 0.30628 | 0.28948 | 0.32256 | 77 |
| remote_direct_sensitivity | 0.48758 | 0.45887 | 0.51658 | 77 |
| remote_history_sensitivity | 0.50628 | 0.47525 | 0.53753 | 77 |
| remote_sensitivity | 0.22510 | 0.20999 | 0.24005 | 77 |

## 2111/score_only: complete behavior map

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | FAIL |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | FAIL |
| nonlearned_accuracy | PASS | FAIL |
| self_consistent_vs_inconsistent_errors | PASS | FAIL |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Constrained parameters: `{'eta': 0.9884949326515198, 'gamma_G': 7.179845333099365, 'gamma_L': 0.0}`.
Inference seconds per episode (warm compiled batch): `1.1259351642318553e-06`.
Float64 analytic bridge versus float32 rollout maximum absolute error: `1.1209138071066604e-06`.

History effect magnitudes:

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| all_direct_sensitivity | 0.93489 | 0.88665 | 0.98173 | 77 |
| all_history_sensitivity | 0.80498 | 0.76083 | 0.84827 | 77 |
| all_sensitivity | 0.30784 | 0.29095 | 0.32420 | 77 |
| remote_direct_sensitivity | 0.49005 | 0.46119 | 0.51920 | 77 |
| remote_history_sensitivity | 0.50882 | 0.47763 | 0.54023 | 77 |
| remote_sensitivity | 0.22622 | 0.21105 | 0.24126 | 77 |

## 2112/score_trace: complete behavior map

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | FAIL |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | FAIL |
| nonlearned_accuracy | PASS | FAIL |
| self_consistent_vs_inconsistent_errors | PASS | FAIL |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Constrained parameters: `{'eta': 0.9889477491378784, 'gamma_G': 7.123929977416992, 'gamma_L': 0.22932417690753937}`.
Inference seconds per episode (warm compiled batch): `3.409402614290064e-06`.
Float64 analytic bridge versus float32 rollout maximum absolute error: `1.066931000437421e-06`.

History effect magnitudes:

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| all_direct_sensitivity | 0.92803 | 0.87966 | 0.97430 | 77 |
| all_history_sensitivity | 0.79916 | 0.75484 | 0.84201 | 77 |
| all_sensitivity | 0.30557 | 0.28848 | 0.32204 | 77 |
| remote_direct_sensitivity | 0.48645 | 0.45696 | 0.51495 | 77 |
| remote_history_sensitivity | 0.50513 | 0.47288 | 0.53633 | 77 |
| remote_sensitivity | 0.22458 | 0.20917 | 0.23988 | 77 |

## 2112/score_only: complete behavior map

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | FAIL |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | FAIL |
| nonlearned_accuracy | PASS | FAIL |
| self_consistent_vs_inconsistent_errors | PASS | FAIL |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Constrained parameters: `{'eta': 0.9887556433677673, 'gamma_G': 7.163000106811523, 'gamma_L': 0.0}`.
Inference seconds per episode (warm compiled batch): `1.1169740192398623e-06`.
Float64 analytic bridge versus float32 rollout maximum absolute error: `1.329694239160517e-06`.

History effect magnitudes:

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| all_direct_sensitivity | 0.93294 | 0.88431 | 0.97946 | 77 |
| all_history_sensitivity | 0.80335 | 0.75879 | 0.84642 | 77 |
| all_sensitivity | 0.30719 | 0.29001 | 0.32375 | 77 |
| remote_direct_sensitivity | 0.48903 | 0.45938 | 0.51768 | 77 |
| remote_history_sensitivity | 0.50779 | 0.47536 | 0.53914 | 77 |
| remote_sensitivity | 0.22576 | 0.21027 | 0.24114 | 77 |

## 2113/score_trace: complete behavior map

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | FAIL |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | FAIL |
| nonlearned_accuracy | PASS | FAIL |
| self_consistent_vs_inconsistent_errors | PASS | FAIL |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Constrained parameters: `{'eta': 0.9887816309928894, 'gamma_G': 7.160130500793457, 'gamma_L': 0.23005518317222595}`.
Inference seconds per episode (warm compiled batch): `3.3471298664082566e-06`.
Float64 analytic bridge versus float32 rollout maximum absolute error: `1.3109290009083452e-06`.

History effect magnitudes:

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| all_direct_sensitivity | 0.93259 | 0.88414 | 0.98039 | 77 |
| all_history_sensitivity | 0.80306 | 0.75844 | 0.84723 | 77 |
| all_sensitivity | 0.30707 | 0.29016 | 0.32385 | 77 |
| remote_direct_sensitivity | 0.48884 | 0.46025 | 0.51818 | 77 |
| remote_history_sensitivity | 0.50760 | 0.47610 | 0.53947 | 77 |
| remote_sensitivity | 0.22568 | 0.21046 | 0.24091 | 77 |

## 2113/score_only: complete behavior map

| Row | Qualitative | Frozen quantitative classifier |
| --- | --- | --- |
| difficult_pair_bimodality | PASS | PASS |
| hodge_reconstructed_subjective_ranking | PASS | FAIL |
| inter_subject_ranking_diversity | PASS | PASS |
| learned_accuracy | PASS | FAIL |
| nonlearned_accuracy | PASS | FAIL |
| self_consistent_vs_inconsistent_errors | PASS | FAIL |
| serial_position_effect | PASS | FAIL |
| stable_within_subject_errors | PASS | PASS |
| symbolic_distance_effect | PASS | FAIL |

Constrained parameters: `{'eta': 0.9885913729667664, 'gamma_G': 7.197957515716553, 'gamma_L': 0.0}`.
Inference seconds per episode (warm compiled batch): `1.0443635529826406e-06`.
Float64 analytic bridge versus float32 rollout maximum absolute error: `1.3458932865972884e-06`.

History effect magnitudes:

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| all_direct_sensitivity | 0.93734 | 0.88864 | 0.98538 | 77 |
| all_history_sensitivity | 0.80711 | 0.76227 | 0.85150 | 77 |
| all_sensitivity | 0.30864 | 0.29164 | 0.32550 | 77 |
| remote_direct_sensitivity | 0.49133 | 0.46260 | 0.52082 | 77 |
| remote_history_sensitivity | 0.51016 | 0.47850 | 0.54219 | 77 |
| remote_sensitivity | 0.22682 | 0.21153 | 0.24212 | 77 |

## Interpretation boundaries

Global additivity is imposed, not an emergent Hodge result. The exact derivative decomposes fixed-encoding influence into direct cue overlap and future-update context; a nonzero history component is not by itself correct semantic inference. Raw signed sensitivities and their exact margin bridge are preserved in typed NPZ.
The score-only baseline can win: it is not required to preserve an unnecessary local branch. Conversely, an acute L-off effect is not evidence that independently optimized score-only cannot solve the task. Missing behavior under a fixed recipe does not establish universal architectural impossibility.
All nine qualitative rows determine adequacy; all nine quantitative classifications are descriptive historical interval membership, not model-human equivalence. The historical temperature 0.25 and encoding distribution are inherited assumptions, not newly fitted parameters.
The JSON retains complete per-participant endpoints, denominators/exclusions, headroom, geometry, all local controls, paired uncertainty and historical RNN behavior/costs. Headroom is reported without claiming it explains an observed effect. Different training seeds and historical timing runs prevent paired RNN-effect or direct speed-benchmark claims.

## Frozen next step

After all six fixed training/evaluation runs, freeze and report. No extra seed, auxiliary objective, gain/temperature calibration, projection, alternative cue geometry, longer training or architecture patch. Only a separately authorized prospective question can follow. Failure localizes what this fixed recipe does not explain, not universal impossibility.
