# Finite-time opponent score circuit

Registered outcome: **`conditional_circuit_sufficiency`**.

Three exposed, frozen score-only fits (2111/2112/2113), not new human samples. No training, calibration or added memory trace. All saved generic and Liu inputs were locked together before evaluation.

Protocol: `db7145093501ba2051b11e0628182cd19525de72`; implementation: `418843beeb842cf2224070ec3611058dabf91462`; evaluation-lock commit: `bb0c382dc6bb1a8c3ceb25b663e734b3411b818b`.

The candidate directly integrates 30 bounded nonnegative efficacies and six effective opponent compartment/error states. Updates read centered presynaptic activity and dynamically generated compartment mismatch. Normalization is a shunting-like error-state dynamic driven by pooled binary-contrast activity. Baseline subtraction, opponent teaching, external stable admission and neutral task initialization are explicit assumptions.

Time units are dimensionless. A rate-level realization is not a complete conductance/spike model, an anatomical attribution, or evidence for a human neural mechanism. Historical quantitative failures are not repaired by this test.

Independent verification: `True`, 798 estimates reconstructed; maximum error 5.55e-16.

All paired intervals use 10,000 resamples, separately within each fit/domain. Undefined group subjects are excluded before resampling and their indices are retained in JSON. No pooling across fits.

## Fit 2111

Parameters: `{'eta': 0.9884949326515198, 'gamma_G': 7.179845333099365, 'gamma_L': 0.0}`.

Decision: `{'checks': {'behavior_preservation': True, 'correspondence': True, 'integrity': True, 'no_write_controls': True, 'physical_constraints': True, 'robustness': True}, 'outcome': 'conditional_circuit_sufficiency'}`.

| Cell | Max state error | Max margin error | Bound hits | Minimum efficacy |
| --- | ---: | ---: | ---: | ---: |
| fast/4096 | 0.0000360 | 0.0011123 | 0 | 0.8973027 |
| fast/8192 | 0.0000360 | 0.0011123 | 0 | 0.8973027 |
| mismatch_clamp | 0.2053947 | 10.0888496 | 0 | 1.0000000 |
| primary/4096 | 0.0000718 | 0.0022132 | 0 | 0.8973027 |
| primary/8192 | 0.0000718 | 0.0022132 | 0 | 0.8973027 |
| slow/4096 | 0.0001426 | 0.0043802 | 0 | 0.8973027 |
| slow/8192 | 0.0001426 | 0.0043802 | 0 | 0.8973027 |
| teacher_off | 0.2053947 | 10.0888496 | 0 | 1.0000000 |
| teaching_shuffle | 0.4177924 | 18.4812941 | 0 | 0.8765944 |

### Numerical and query checks

`{'affine_max_error': 2.1671553440683056e-13, 'query_errors': {'0.05/False': 2.801829879217621e-10, '0.05/True': 2.801829879217621e-10, '0.1/False': 0.0, '0.1/True': 0.0, '0.2/False': 0.0, '0.2/True': 0.0}, 'query_no_write': True}`

Step refinement: `{'fast': {'generic_28': {'margin': 8.419931418757187e-13, 'state': 1.554312234475219e-14}, 'generic_32': {'margin': 9.521272659185342e-13, 'state': 1.6653345369377348e-14}, 'generic_36': {'margin': 9.281464485866309e-13, 'state': 1.9095836023552692e-14}, 'generic_40': {'margin': 1.0551559626037488e-12, 'state': 2.0761170560490427e-14}, 'liu': {'margin': 1.0520473381347983e-12, 'state': 1.6542323066914832e-14}}, 'primary': {'generic_28': {'margin': 1.141309269314661e-12, 'state': 1.865174681370263e-14}, 'generic_32': {'margin': 1.013855666087693e-12, 'state': 2.2537527399890678e-14}, 'generic_36': {'margin': 1.0396128402589966e-12, 'state': 1.709743457922741e-14}, 'generic_40': {'margin': 1.007194327939942e-12, 'state': 1.8096635301390052e-14}, 'liu': {'margin': 9.15267861500979e-13, 'state': 1.509903313490213e-14}}, 'slow': {'generic_28': {'margin': 8.162359677044151e-13, 'state': 1.4765966227514582e-14}, 'generic_32': {'margin': 1.0902390101819037e-12, 'state': 1.5987211554602254e-14}, 'generic_36': {'margin': 9.43689570931383e-13, 'state': 1.7319479184152442e-14}, 'generic_40': {'margin': 9.245937349078304e-13, 'state': 1.687538997430238e-14}, 'liu': {'margin': 1.2119194536808209e-12, 'state': 1.8540724511240114e-14}}}`

### Sampled behavior: original definitions

| Row | Parent qualitative | Circuit qualitative | Parent quantitative | Circuit quantitative |
| --- | --- | --- | --- | --- |
| difficult_pair_bimodality | True | True | True | True |
| hodge_reconstructed_subjective_ranking | True | True | False | False |
| inter_subject_ranking_diversity | True | True | True | True |
| learned_accuracy | True | True | False | False |
| nonlearned_accuracy | True | True | False | False |
| self_consistent_vs_inconsistent_errors | True | True | False | False |
| serial_position_effect | True | True | False | False |
| stable_within_subject_errors | True | True | True | True |
| symbolic_distance_effect | True | True | False | False |

### fast/4096: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9296580 | 0.9518015 | 256 |
| generic/exact_decision/nonlearned | 0.8686510 | 0.8544755 | 0.8826621 | 256 |
| generic/probability/learned | 0.8520773 | 0.8428619 | 0.8608135 | 256 |
| generic/probability/nonlearned | 0.7908113 | 0.7797393 | 0.8013117 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8804348 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8882189 | 0.9290353 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465936 | 0.9283803 | 0.9631904 | 77 |
| liu/probability/nonlearned | 0.8840145 | 0.8614495 | 0.9056464 | 77 |
| liu/probability/omitted | 0.8095440 | 0.7426681 | 0.8721780 | 69 |
| liu/probability/overall | 0.9018943 | 0.8823924 | 0.9207131 | 77 |
| liu/probability/retained | 0.9936381 | 0.9891586 | 0.9968369 | 77 |

### fast/4096: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0001028 | 0.0000000 | 0.0003084 | 256 |
| generic/probability/learned | -0.0000033 | -0.0000046 | -0.0000021 | 256 |
| generic/probability/nonlearned | 0.0000016 | 0.0000005 | 0.0000027 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000013 | -0.0000017 | 0.0000048 | 77 |
| liu/probability/nonlearned | 0.0000053 | 0.0000017 | 0.0000088 | 77 |
| liu/probability/omitted | 0.0000094 | -0.0000004 | 0.0000206 | 69 |
| liu/probability/overall | 0.0000042 | 0.0000016 | 0.0000068 | 77 |
| liu/probability/retained | -0.0000007 | -0.0000025 | 0.0000010 | 77 |

### fast/8192: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9296580 | 0.9518015 | 256 |
| generic/exact_decision/nonlearned | 0.8686561 | 0.8545525 | 0.8826211 | 256 |
| generic/probability/learned | 0.8520773 | 0.8428619 | 0.8608135 | 256 |
| generic/probability/nonlearned | 0.7908113 | 0.7797393 | 0.8013117 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8804348 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8882189 | 0.9290353 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465936 | 0.9283803 | 0.9631904 | 77 |
| liu/probability/nonlearned | 0.8840145 | 0.8614495 | 0.9056464 | 77 |
| liu/probability/omitted | 0.8095440 | 0.7426681 | 0.8721780 | 69 |
| liu/probability/overall | 0.9018943 | 0.8823924 | 0.9207131 | 77 |
| liu/probability/retained | 0.9936381 | 0.9891586 | 0.9968369 | 77 |

### fast/8192: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0001079 | -0.0002930 | 0.0006168 | 256 |
| generic/probability/learned | -0.0000033 | -0.0000046 | -0.0000021 | 256 |
| generic/probability/nonlearned | 0.0000016 | 0.0000005 | 0.0000027 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000013 | -0.0000017 | 0.0000048 | 77 |
| liu/probability/nonlearned | 0.0000053 | 0.0000017 | 0.0000088 | 77 |
| liu/probability/omitted | 0.0000094 | -0.0000004 | 0.0000206 | 69 |
| liu/probability/overall | 0.0000042 | 0.0000016 | 0.0000068 | 77 |
| liu/probability/retained | -0.0000007 | -0.0000025 | 0.0000010 | 77 |

### mismatch_clamp: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| liu/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/exact_decision/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/probability/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |

### mismatch_clamp: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | -0.4411350 | -0.4518015 | -0.4296580 | 256 |
| generic/exact_decision/nonlearned | -0.3685482 | -0.3826048 | -0.3543502 | 256 |
| generic/probability/learned | -0.3520806 | -0.3608168 | -0.3428641 | 256 |
| generic/probability/nonlearned | -0.2908097 | -0.3013103 | -0.2797374 | 256 |
| liu/exact_decision/learned | -0.4496753 | -0.4675325 | -0.4301948 | 77 |
| liu/exact_decision/nonlearned | -0.3928571 | -0.4162338 | -0.3688312 | 77 |
| liu/exact_decision/omitted | -0.3140097 | -0.3804348 | -0.2415459 | 69 |
| liu/exact_decision/overall | -0.4090909 | -0.4290353 | -0.3882189 | 77 |
| liu/exact_decision/retained | -0.4978355 | -0.5000000 | -0.4935065 | 77 |
| liu/probability/learned | -0.4465923 | -0.4631912 | -0.4283783 | 77 |
| liu/probability/nonlearned | -0.3840092 | -0.4056420 | -0.3614415 | 77 |
| liu/probability/omitted | -0.3095346 | -0.3721761 | -0.2426524 | 69 |
| liu/probability/overall | -0.4018901 | -0.4207094 | -0.3823864 | 77 |
| liu/probability/retained | -0.4936388 | -0.4968371 | -0.4891588 | 77 |

### primary/4096: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9296580 | 0.9518015 | 256 |
| generic/exact_decision/nonlearned | 0.8683528 | 0.8541317 | 0.8824009 | 256 |
| generic/probability/learned | 0.8520739 | 0.8428597 | 0.8608101 | 256 |
| generic/probability/nonlearned | 0.7908129 | 0.7797413 | 0.8013130 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8804348 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8882189 | 0.9290353 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465949 | 0.9283821 | 0.9631918 | 77 |
| liu/probability/nonlearned | 0.8840198 | 0.8614574 | 0.9056507 | 77 |
| liu/probability/omitted | 0.8095533 | 0.7426836 | 0.8721798 | 69 |
| liu/probability/overall | 0.9018984 | 0.8823983 | 0.9207168 | 77 |
| liu/probability/retained | 0.9936374 | 0.9891582 | 0.9968366 | 77 |

### primary/4096: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | -0.0001953 | -0.0005859 | 0.0000000 | 256 |
| generic/probability/learned | -0.0000067 | -0.0000091 | -0.0000042 | 256 |
| generic/probability/nonlearned | 0.0000032 | 0.0000010 | 0.0000054 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000026 | -0.0000035 | 0.0000095 | 77 |
| liu/probability/nonlearned | 0.0000106 | 0.0000034 | 0.0000175 | 77 |
| liu/probability/omitted | 0.0000187 | -0.0000008 | 0.0000410 | 69 |
| liu/probability/overall | 0.0000083 | 0.0000032 | 0.0000135 | 77 |
| liu/probability/retained | -0.0000014 | -0.0000050 | 0.0000020 | 77 |

### primary/8192: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9296580 | 0.9518015 | 256 |
| generic/exact_decision/nonlearned | 0.8685482 | 0.8543502 | 0.8826048 | 256 |
| generic/probability/learned | 0.8520739 | 0.8428597 | 0.8608101 | 256 |
| generic/probability/nonlearned | 0.7908129 | 0.7797413 | 0.8013130 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8804348 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8882189 | 0.9290353 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465949 | 0.9283821 | 0.9631918 | 77 |
| liu/probability/nonlearned | 0.8840198 | 0.8614574 | 0.9056507 | 77 |
| liu/probability/omitted | 0.8095533 | 0.7426836 | 0.8721798 | 69 |
| liu/probability/overall | 0.9018984 | 0.8823983 | 0.9207168 | 77 |
| liu/probability/retained | 0.9936374 | 0.9891582 | 0.9968366 | 77 |

### primary/8192: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/probability/learned | -0.0000067 | -0.0000091 | -0.0000042 | 256 |
| generic/probability/nonlearned | 0.0000032 | 0.0000010 | 0.0000054 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000026 | -0.0000035 | 0.0000095 | 77 |
| liu/probability/nonlearned | 0.0000106 | 0.0000034 | 0.0000175 | 77 |
| liu/probability/omitted | 0.0000187 | -0.0000008 | 0.0000410 | 69 |
| liu/probability/overall | 0.0000083 | 0.0000032 | 0.0000135 | 77 |
| liu/probability/retained | -0.0000014 | -0.0000050 | 0.0000020 | 77 |

### slow/4096: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9296580 | 0.9518015 | 256 |
| generic/exact_decision/nonlearned | 0.8685584 | 0.8544230 | 0.8825236 | 256 |
| generic/probability/learned | 0.8520672 | 0.8428553 | 0.8608035 | 256 |
| generic/probability/nonlearned | 0.7908160 | 0.7797450 | 0.8013156 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8804348 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8882189 | 0.9290353 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465974 | 0.9283858 | 0.9631902 | 77 |
| liu/probability/nonlearned | 0.8840303 | 0.8614728 | 0.9056592 | 77 |
| liu/probability/omitted | 0.8095716 | 0.7427140 | 0.8721835 | 69 |
| liu/probability/overall | 0.9019066 | 0.8824100 | 0.9207243 | 77 |
| liu/probability/retained | 0.9936359 | 0.9891575 | 0.9968361 | 77 |

### slow/4096: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000103 | -0.0005859 | 0.0006168 | 256 |
| generic/probability/learned | -0.0000134 | -0.0000182 | -0.0000085 | 256 |
| generic/probability/nonlearned | 0.0000063 | 0.0000020 | 0.0000107 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000051 | -0.0000070 | 0.0000187 | 77 |
| liu/probability/nonlearned | 0.0000211 | 0.0000069 | 0.0000349 | 77 |
| liu/probability/omitted | 0.0000370 | -0.0000016 | 0.0000814 | 69 |
| liu/probability/overall | 0.0000165 | 0.0000063 | 0.0000269 | 77 |
| liu/probability/retained | -0.0000029 | -0.0000099 | 0.0000040 | 77 |

### slow/8192: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9296580 | 0.9518015 | 256 |
| generic/exact_decision/nonlearned | 0.8685584 | 0.8544230 | 0.8825236 | 256 |
| generic/probability/learned | 0.8520672 | 0.8428553 | 0.8608035 | 256 |
| generic/probability/nonlearned | 0.7908160 | 0.7797450 | 0.8013156 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8804348 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8882189 | 0.9290353 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465974 | 0.9283858 | 0.9631902 | 77 |
| liu/probability/nonlearned | 0.8840303 | 0.8614728 | 0.9056592 | 77 |
| liu/probability/omitted | 0.8095716 | 0.7427140 | 0.8721835 | 69 |
| liu/probability/overall | 0.9019066 | 0.8824100 | 0.9207243 | 77 |
| liu/probability/retained | 0.9936359 | 0.9891575 | 0.9968361 | 77 |

### slow/8192: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000103 | -0.0005859 | 0.0006168 | 256 |
| generic/probability/learned | -0.0000134 | -0.0000182 | -0.0000085 | 256 |
| generic/probability/nonlearned | 0.0000063 | 0.0000020 | 0.0000107 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000051 | -0.0000070 | 0.0000187 | 77 |
| liu/probability/nonlearned | 0.0000211 | 0.0000069 | 0.0000349 | 77 |
| liu/probability/omitted | 0.0000370 | -0.0000016 | 0.0000814 | 69 |
| liu/probability/overall | 0.0000165 | 0.0000063 | 0.0000269 | 77 |
| liu/probability/retained | -0.0000029 | -0.0000099 | 0.0000040 | 77 |

### teacher_off: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| liu/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/exact_decision/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/probability/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |

### teacher_off: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | -0.4411350 | -0.4518015 | -0.4296580 | 256 |
| generic/exact_decision/nonlearned | -0.3685482 | -0.3826048 | -0.3543502 | 256 |
| generic/probability/learned | -0.3520806 | -0.3608168 | -0.3428641 | 256 |
| generic/probability/nonlearned | -0.2908097 | -0.3013103 | -0.2797374 | 256 |
| liu/exact_decision/learned | -0.4496753 | -0.4675325 | -0.4301948 | 77 |
| liu/exact_decision/nonlearned | -0.3928571 | -0.4162338 | -0.3688312 | 77 |
| liu/exact_decision/omitted | -0.3140097 | -0.3804348 | -0.2415459 | 69 |
| liu/exact_decision/overall | -0.4090909 | -0.4290353 | -0.3882189 | 77 |
| liu/exact_decision/retained | -0.4978355 | -0.5000000 | -0.4935065 | 77 |
| liu/probability/learned | -0.4465923 | -0.4631912 | -0.4283783 | 77 |
| liu/probability/nonlearned | -0.3840092 | -0.4056420 | -0.3614415 | 77 |
| liu/probability/omitted | -0.3095346 | -0.3721761 | -0.2426524 | 69 |
| liu/probability/overall | -0.4018901 | -0.4207094 | -0.3823864 | 77 |
| liu/probability/retained | -0.4936388 | -0.4968371 | -0.4891588 | 77 |

### teaching_shuffle: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.5038551 | 0.4792504 | 0.5285071 | 256 |
| generic/exact_decision/nonlearned | 0.5008849 | 0.4827055 | 0.5190221 | 256 |
| generic/probability/learned | 0.5004817 | 0.4808862 | 0.5198201 | 256 |
| generic/probability/nonlearned | 0.5032346 | 0.4883959 | 0.5180889 | 256 |
| liu/exact_decision/learned | 0.4967532 | 0.4480519 | 0.5454545 | 77 |
| liu/exact_decision/nonlearned | 0.5032468 | 0.4694805 | 0.5370130 | 77 |
| liu/exact_decision/omitted | 0.5429952 | 0.4541063 | 0.6347886 | 69 |
| liu/exact_decision/overall | 0.5013915 | 0.4666048 | 0.5352505 | 77 |
| liu/exact_decision/retained | 0.4806122 | 0.4285092 | 0.5328850 | 77 |
| liu/probability/learned | 0.4982961 | 0.4504105 | 0.5455844 | 77 |
| liu/probability/nonlearned | 0.5026590 | 0.4697526 | 0.5352152 | 77 |
| liu/probability/omitted | 0.5367020 | 0.4525650 | 0.6229929 | 69 |
| liu/probability/overall | 0.5014124 | 0.4677275 | 0.5341946 | 77 |
| liu/probability/retained | 0.4854411 | 0.4345919 | 0.5366526 | 77 |

### teaching_shuffle: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | -0.4372799 | -0.4643185 | -0.4100415 | 256 |
| generic/exact_decision/nonlearned | -0.3676632 | -0.3908226 | -0.3446733 | 256 |
| generic/probability/learned | -0.3515989 | -0.3729795 | -0.3302854 | 256 |
| generic/probability/nonlearned | -0.2875751 | -0.3059420 | -0.2690076 | 256 |
| liu/exact_decision/learned | -0.4529221 | -0.5064935 | -0.3993506 | 77 |
| liu/exact_decision/nonlearned | -0.3896104 | -0.4272727 | -0.3525974 | 77 |
| liu/exact_decision/omitted | -0.2710145 | -0.3913043 | -0.1458937 | 69 |
| liu/exact_decision/overall | -0.4076994 | -0.4466605 | -0.3696660 | 77 |
| liu/exact_decision/retained | -0.5172233 | -0.5691713 | -0.4653042 | 77 |
| liu/probability/learned | -0.4482962 | -0.5002537 | -0.3971260 | 77 |
| liu/probability/nonlearned | -0.3813503 | -0.4180577 | -0.3462476 | 77 |
| liu/probability/omitted | -0.2728326 | -0.3856290 | -0.1546867 | 69 |
| liu/probability/overall | -0.4004777 | -0.4381754 | -0.3636597 | 77 |
| liu/probability/retained | -0.5081977 | -0.5588566 | -0.4570297 | 77 |

## Fit 2112

Parameters: `{'eta': 0.9887556433677673, 'gamma_G': 7.163000106811523, 'gamma_L': 0.0}`.

Decision: `{'checks': {'behavior_preservation': True, 'correspondence': True, 'integrity': True, 'no_write_controls': True, 'physical_constraints': True, 'robustness': True}, 'outcome': 'conditional_circuit_sufficiency'}`.

| Cell | Max state error | Max margin error | Bound hits | Minimum efficacy |
| --- | ---: | ---: | ---: | ---: |
| fast/4096 | 0.0000356 | 0.0010935 | 0 | 0.8972919 |
| fast/8192 | 0.0000356 | 0.0010935 | 0 | 0.8972919 |
| mismatch_clamp | 0.2054163 | 10.0652214 | 0 | 1.0000000 |
| primary/4096 | 0.0000710 | 0.0021756 | 0 | 0.8972918 |
| primary/8192 | 0.0000710 | 0.0021756 | 0 | 0.8972918 |
| slow/4096 | 0.0001410 | 0.0043048 | 0 | 0.8972917 |
| slow/8192 | 0.0001410 | 0.0043048 | 0 | 0.8972917 |
| teacher_off | 0.2054163 | 10.0652214 | 0 | 1.0000000 |
| teaching_shuffle | 0.4178602 | 18.4409233 | 0 | 0.8765806 |

### Numerical and query checks

`{'affine_max_error': 2.007283228522283e-13, 'query_errors': {'0.05/False': 2.79525735891184e-10, '0.05/True': 2.79525735891184e-10, '0.1/False': 0.0, '0.1/True': 0.0, '0.2/False': 0.0, '0.2/True': 0.0}, 'query_no_write': True}`

Step refinement: `{'fast': {'generic_28': {'margin': 9.798828415341632e-13, 'state': 1.5765166949677223e-14}, 'generic_32': {'margin': 1.1388667786604856e-12, 'state': 1.6764367671839864e-14}, 'generic_36': {'margin': 8.779643678735738e-13, 'state': 1.8096635301390052e-14}, 'generic_40': {'margin': 9.539036227579345e-13, 'state': 1.7208456881689926e-14}, 'liu': {'margin': 1.0178524689763435e-12, 'state': 1.84297022087776e-14}}, 'primary': {'generic_28': {'margin': 1.240785252321075e-12, 'state': 1.5432100042289676e-14}, 'generic_32': {'margin': 1.1168843627729075e-12, 'state': 1.587618925213974e-14}, 'generic_36': {'margin': 1.0387246618392965e-12, 'state': 1.8984813721090177e-14}, 'generic_40': {'margin': 1.2958523143424827e-12, 'state': 1.7985612998927536e-14}, 'liu': {'margin': 9.257039579324555e-13, 'state': 1.8318679906315083e-14}}, 'slow': {'generic_28': {'margin': 9.543477119677846e-13, 'state': 1.4876988529977098e-14}, 'generic_32': {'margin': 9.547918011776346e-13, 'state': 1.6764367671839864e-14}, 'generic_36': {'margin': 1.3997691894473974e-12, 'state': 1.765254609153999e-14}, 'generic_40': {'margin': 1.1832756996454918e-12, 'state': 1.8984813721090177e-14}, 'liu': {'margin': 1.170619157164765e-12, 'state': 1.9095836023552692e-14}}}`

### Sampled behavior: original definitions

| Row | Parent qualitative | Circuit qualitative | Parent quantitative | Circuit quantitative |
| --- | --- | --- | --- | --- |
| difficult_pair_bimodality | True | True | True | True |
| hodge_reconstructed_subjective_ranking | True | True | False | False |
| inter_subject_ranking_diversity | True | True | True | True |
| learned_accuracy | True | True | False | False |
| nonlearned_accuracy | True | True | False | False |
| self_consistent_vs_inconsistent_errors | True | True | False | False |
| serial_position_effect | True | True | False | False |
| stable_within_subject_errors | True | True | True | True |
| symbolic_distance_effect | True | True | False | False |

### fast/4096: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9297117 | 0.9520120 | 256 |
| generic/exact_decision/nonlearned | 0.8685533 | 0.8540045 | 0.8825085 | 256 |
| generic/probability/learned | 0.8518090 | 0.8427192 | 0.8606844 | 256 |
| generic/probability/nonlearned | 0.7905907 | 0.7794706 | 0.8014638 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9285714 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8681818 | 0.9168831 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7427536 | 0.8792271 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9294991 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465730 | 0.9278291 | 0.9628041 | 77 |
| liu/probability/nonlearned | 0.8839837 | 0.8606282 | 0.9063517 | 77 |
| liu/probability/omitted | 0.8095370 | 0.7413121 | 0.8709841 | 69 |
| liu/probability/overall | 0.9018663 | 0.8818744 | 0.9209992 | 77 |
| liu/probability/retained | 0.9936131 | 0.9891473 | 0.9968080 | 77 |

### fast/4096: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000051 | -0.0002930 | 0.0003084 | 256 |
| generic/probability/learned | -0.0000033 | -0.0000045 | -0.0000021 | 256 |
| generic/probability/nonlearned | 0.0000016 | 0.0000006 | 0.0000027 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000013 | -0.0000018 | 0.0000046 | 77 |
| liu/probability/nonlearned | 0.0000052 | 0.0000018 | 0.0000087 | 77 |
| liu/probability/omitted | 0.0000093 | -0.0000003 | 0.0000201 | 69 |
| liu/probability/overall | 0.0000041 | 0.0000016 | 0.0000068 | 77 |
| liu/probability/retained | -0.0000007 | -0.0000025 | 0.0000010 | 77 |

### fast/8192: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9297117 | 0.9520120 | 256 |
| generic/exact_decision/nonlearned | 0.8683528 | 0.8537758 | 0.8823310 | 256 |
| generic/probability/learned | 0.8518090 | 0.8427192 | 0.8606844 | 256 |
| generic/probability/nonlearned | 0.7905907 | 0.7794706 | 0.8014638 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9285714 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8681818 | 0.9168831 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7427536 | 0.8792271 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9294991 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465730 | 0.9278291 | 0.9628041 | 77 |
| liu/probability/nonlearned | 0.8839837 | 0.8606282 | 0.9063517 | 77 |
| liu/probability/omitted | 0.8095370 | 0.7413121 | 0.8709841 | 69 |
| liu/probability/overall | 0.9018663 | 0.8818744 | 0.9209992 | 77 |
| liu/probability/retained | 0.9936131 | 0.9891473 | 0.9968080 | 77 |

### fast/8192: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | -0.0001953 | -0.0005859 | 0.0000000 | 256 |
| generic/probability/learned | -0.0000033 | -0.0000045 | -0.0000021 | 256 |
| generic/probability/nonlearned | 0.0000016 | 0.0000006 | 0.0000027 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000013 | -0.0000018 | 0.0000046 | 77 |
| liu/probability/nonlearned | 0.0000052 | 0.0000018 | 0.0000087 | 77 |
| liu/probability/omitted | 0.0000093 | -0.0000003 | 0.0000201 | 69 |
| liu/probability/overall | 0.0000041 | 0.0000016 | 0.0000068 | 77 |
| liu/probability/retained | -0.0000007 | -0.0000025 | 0.0000010 | 77 |

### mismatch_clamp: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| liu/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/exact_decision/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/probability/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |

### mismatch_clamp: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | -0.4411350 | -0.4520120 | -0.4297117 | 256 |
| generic/exact_decision/nonlearned | -0.3685482 | -0.3825263 | -0.3539583 | 256 |
| generic/probability/learned | -0.3518122 | -0.3606879 | -0.3427226 | 256 |
| generic/probability/nonlearned | -0.2905891 | -0.3014616 | -0.2794694 | 256 |
| liu/exact_decision/learned | -0.4496753 | -0.4675325 | -0.4285714 | 77 |
| liu/exact_decision/nonlearned | -0.3928571 | -0.4168831 | -0.3681818 | 77 |
| liu/exact_decision/omitted | -0.3140097 | -0.3792271 | -0.2427536 | 69 |
| liu/exact_decision/overall | -0.4090909 | -0.4294991 | -0.3877551 | 77 |
| liu/exact_decision/retained | -0.4978355 | -0.5000000 | -0.4935065 | 77 |
| liu/probability/learned | -0.4465717 | -0.4628053 | -0.4278268 | 77 |
| liu/probability/nonlearned | -0.3839784 | -0.4063470 | -0.3606205 | 77 |
| liu/probability/omitted | -0.3095277 | -0.3709803 | -0.2413000 | 69 |
| liu/probability/overall | -0.4018622 | -0.4209941 | -0.3818683 | 77 |
| liu/probability/retained | -0.4936138 | -0.4968084 | -0.4891486 | 77 |

### primary/4096: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9297117 | 0.9520120 | 256 |
| generic/exact_decision/nonlearned | 0.8685482 | 0.8539583 | 0.8825263 | 256 |
| generic/probability/learned | 0.8518057 | 0.8427158 | 0.8606808 | 256 |
| generic/probability/nonlearned | 0.7905923 | 0.7794719 | 0.8014660 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9285714 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8681818 | 0.9168831 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7427536 | 0.8792271 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9294991 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465742 | 0.9278314 | 0.9628029 | 77 |
| liu/probability/nonlearned | 0.8839889 | 0.8606359 | 0.9063563 | 77 |
| liu/probability/omitted | 0.8095462 | 0.7413183 | 0.8709878 | 69 |
| liu/probability/overall | 0.9018704 | 0.8818805 | 0.9210043 | 77 |
| liu/probability/retained | 0.9936124 | 0.9891460 | 0.9968084 | 77 |

### primary/4096: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/probability/learned | -0.0000066 | -0.0000090 | -0.0000041 | 256 |
| generic/probability/nonlearned | 0.0000032 | 0.0000011 | 0.0000053 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000025 | -0.0000035 | 0.0000092 | 77 |
| liu/probability/nonlearned | 0.0000104 | 0.0000036 | 0.0000174 | 77 |
| liu/probability/omitted | 0.0000185 | -0.0000005 | 0.0000401 | 69 |
| liu/probability/overall | 0.0000082 | 0.0000032 | 0.0000135 | 77 |
| liu/probability/retained | -0.0000014 | -0.0000051 | 0.0000020 | 77 |

### primary/8192: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9297117 | 0.9520120 | 256 |
| generic/exact_decision/nonlearned | 0.8684556 | 0.8539245 | 0.8823744 | 256 |
| generic/probability/learned | 0.8518057 | 0.8427158 | 0.8606808 | 256 |
| generic/probability/nonlearned | 0.7905923 | 0.7794719 | 0.8014660 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9285714 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8681818 | 0.9168831 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7427536 | 0.8792271 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9294991 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465742 | 0.9278314 | 0.9628029 | 77 |
| liu/probability/nonlearned | 0.8839889 | 0.8606359 | 0.9063563 | 77 |
| liu/probability/omitted | 0.8095462 | 0.7413183 | 0.8709878 | 69 |
| liu/probability/overall | 0.9018704 | 0.8818805 | 0.9210043 | 77 |
| liu/probability/retained | 0.9936124 | 0.9891460 | 0.9968084 | 77 |

### primary/8192: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | -0.0000925 | -0.0005859 | 0.0003084 | 256 |
| generic/probability/learned | -0.0000066 | -0.0000090 | -0.0000041 | 256 |
| generic/probability/nonlearned | 0.0000032 | 0.0000011 | 0.0000053 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000025 | -0.0000035 | 0.0000092 | 77 |
| liu/probability/nonlearned | 0.0000104 | 0.0000036 | 0.0000174 | 77 |
| liu/probability/omitted | 0.0000185 | -0.0000005 | 0.0000401 | 69 |
| liu/probability/overall | 0.0000082 | 0.0000032 | 0.0000135 | 77 |
| liu/probability/retained | -0.0000014 | -0.0000051 | 0.0000020 | 77 |

### slow/4096: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9297117 | 0.9520120 | 256 |
| generic/exact_decision/nonlearned | 0.8685584 | 0.8540315 | 0.8824588 | 256 |
| generic/probability/learned | 0.8517991 | 0.8427089 | 0.8606740 | 256 |
| generic/probability/nonlearned | 0.7905954 | 0.7794744 | 0.8014703 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9285714 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8681818 | 0.9168831 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7427536 | 0.8792271 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9294991 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465767 | 0.9278360 | 0.9628004 | 77 |
| liu/probability/nonlearned | 0.8839992 | 0.8606511 | 0.9063655 | 77 |
| liu/probability/omitted | 0.8095643 | 0.7413309 | 0.8709950 | 69 |
| liu/probability/overall | 0.9018785 | 0.8818926 | 0.9210145 | 77 |
| liu/probability/retained | 0.9936109 | 0.9891436 | 0.9968092 | 77 |

### slow/4096: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000103 | -0.0005859 | 0.0006168 | 256 |
| generic/probability/learned | -0.0000131 | -0.0000179 | -0.0000083 | 256 |
| generic/probability/nonlearned | 0.0000063 | 0.0000021 | 0.0000105 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000050 | -0.0000070 | 0.0000183 | 77 |
| liu/probability/nonlearned | 0.0000208 | 0.0000072 | 0.0000347 | 77 |
| liu/probability/omitted | 0.0000365 | -0.0000011 | 0.0000795 | 69 |
| liu/probability/overall | 0.0000163 | 0.0000063 | 0.0000269 | 77 |
| liu/probability/retained | -0.0000028 | -0.0000101 | 0.0000038 | 77 |

### slow/8192: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9297117 | 0.9520120 | 256 |
| generic/exact_decision/nonlearned | 0.8685584 | 0.8540315 | 0.8824588 | 256 |
| generic/probability/learned | 0.8517991 | 0.8427089 | 0.8606740 | 256 |
| generic/probability/nonlearned | 0.7905954 | 0.7794744 | 0.8014703 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9285714 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8681818 | 0.9168831 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7427536 | 0.8792271 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9294991 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9465767 | 0.9278360 | 0.9628004 | 77 |
| liu/probability/nonlearned | 0.8839992 | 0.8606511 | 0.9063655 | 77 |
| liu/probability/omitted | 0.8095643 | 0.7413309 | 0.8709950 | 69 |
| liu/probability/overall | 0.9018785 | 0.8818926 | 0.9210145 | 77 |
| liu/probability/retained | 0.9936109 | 0.9891436 | 0.9968092 | 77 |

### slow/8192: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000103 | -0.0005859 | 0.0006168 | 256 |
| generic/probability/learned | -0.0000131 | -0.0000179 | -0.0000083 | 256 |
| generic/probability/nonlearned | 0.0000063 | 0.0000021 | 0.0000105 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000050 | -0.0000070 | 0.0000183 | 77 |
| liu/probability/nonlearned | 0.0000208 | 0.0000072 | 0.0000347 | 77 |
| liu/probability/omitted | 0.0000365 | -0.0000011 | 0.0000795 | 69 |
| liu/probability/overall | 0.0000163 | 0.0000063 | 0.0000269 | 77 |
| liu/probability/retained | -0.0000028 | -0.0000101 | 0.0000038 | 77 |

### teacher_off: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| liu/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/exact_decision/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/probability/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |

### teacher_off: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | -0.4411350 | -0.4520120 | -0.4297117 | 256 |
| generic/exact_decision/nonlearned | -0.3685482 | -0.3825263 | -0.3539583 | 256 |
| generic/probability/learned | -0.3518122 | -0.3606879 | -0.3427226 | 256 |
| generic/probability/nonlearned | -0.2905891 | -0.3014616 | -0.2794694 | 256 |
| liu/exact_decision/learned | -0.4496753 | -0.4675325 | -0.4285714 | 77 |
| liu/exact_decision/nonlearned | -0.3928571 | -0.4168831 | -0.3681818 | 77 |
| liu/exact_decision/omitted | -0.3140097 | -0.3792271 | -0.2427536 | 69 |
| liu/exact_decision/overall | -0.4090909 | -0.4294991 | -0.3877551 | 77 |
| liu/exact_decision/retained | -0.4978355 | -0.5000000 | -0.4935065 | 77 |
| liu/probability/learned | -0.4465717 | -0.4628053 | -0.4278268 | 77 |
| liu/probability/nonlearned | -0.3839784 | -0.4063470 | -0.3606205 | 77 |
| liu/probability/omitted | -0.3095277 | -0.3709803 | -0.2413000 | 69 |
| liu/probability/overall | -0.4018622 | -0.4209941 | -0.3818683 | 77 |
| liu/probability/retained | -0.4936138 | -0.4968084 | -0.4891486 | 77 |

### teaching_shuffle: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.5038551 | 0.4802899 | 0.5285575 | 256 |
| generic/exact_decision/nonlearned | 0.5011996 | 0.4829655 | 0.5191020 | 256 |
| generic/probability/learned | 0.5004762 | 0.4816843 | 0.5199494 | 256 |
| generic/probability/nonlearned | 0.5032371 | 0.4885286 | 0.5178109 | 256 |
| liu/exact_decision/learned | 0.4967532 | 0.4480519 | 0.5454545 | 77 |
| liu/exact_decision/nonlearned | 0.5032468 | 0.4688312 | 0.5376623 | 77 |
| liu/exact_decision/omitted | 0.5429952 | 0.4512077 | 0.6323671 | 69 |
| liu/exact_decision/overall | 0.5013915 | 0.4670686 | 0.5366419 | 77 |
| liu/exact_decision/retained | 0.4806122 | 0.4281076 | 0.5325309 | 77 |
| liu/probability/learned | 0.4983042 | 0.4516997 | 0.5452347 | 77 |
| liu/probability/nonlearned | 0.5026521 | 0.4695476 | 0.5359945 | 77 |
| liu/probability/omitted | 0.5367033 | 0.4511009 | 0.6203707 | 69 |
| liu/probability/overall | 0.5014098 | 0.4681379 | 0.5359628 | 77 |
| liu/probability/retained | 0.4854519 | 0.4347207 | 0.5363049 | 77 |

### teaching_shuffle: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | -0.4372799 | -0.4630306 | -0.4103291 | 256 |
| generic/exact_decision/nonlearned | -0.3673485 | -0.3905118 | -0.3436265 | 256 |
| generic/probability/learned | -0.3513360 | -0.3723123 | -0.3296637 | 256 |
| generic/probability/nonlearned | -0.2873520 | -0.3061119 | -0.2687604 | 256 |
| liu/exact_decision/learned | -0.4529221 | -0.5064935 | -0.3993506 | 77 |
| liu/exact_decision/nonlearned | -0.3896104 | -0.4279221 | -0.3519481 | 77 |
| liu/exact_decision/omitted | -0.2710145 | -0.3917874 | -0.1487923 | 69 |
| liu/exact_decision/overall | -0.4076994 | -0.4457328 | -0.3692022 | 77 |
| liu/exact_decision/retained | -0.5172233 | -0.5695737 | -0.4655535 | 77 |
| liu/probability/learned | -0.4482675 | -0.5000984 | -0.3967983 | 77 |
| liu/probability/nonlearned | -0.3813264 | -0.4178366 | -0.3448640 | 77 |
| liu/probability/omitted | -0.2728245 | -0.3877276 | -0.1573501 | 69 |
| liu/probability/overall | -0.4004524 | -0.4372549 | -0.3630918 | 77 |
| liu/probability/retained | -0.5081619 | -0.5588102 | -0.4575142 | 77 |

## Fit 2113

Parameters: `{'eta': 0.9885913729667664, 'gamma_G': 7.197957515716553, 'gamma_L': 0.0}`.

Decision: `{'checks': {'behavior_preservation': True, 'correspondence': True, 'integrity': True, 'no_write_controls': True, 'physical_constraints': True, 'robustness': True}, 'outcome': 'conditional_circuit_sufficiency'}`.

| Cell | Max state error | Max margin error | Bound hits | Minimum efficacy |
| --- | ---: | ---: | ---: | ---: |
| fast/4096 | 0.0000359 | 0.0011092 | 0 | 0.8972987 |
| fast/8192 | 0.0000359 | 0.0011092 | 0 | 0.8972987 |
| mismatch_clamp | 0.2054027 | 10.1143158 | 0 | 1.0000000 |
| primary/4096 | 0.0000715 | 0.0022068 | 0 | 0.8972987 |
| primary/8192 | 0.0000715 | 0.0022068 | 0 | 0.8972987 |
| slow/4096 | 0.0001420 | 0.0043671 | 0 | 0.8972986 |
| slow/8192 | 0.0001420 | 0.0043671 | 0 | 0.8972986 |
| teacher_off | 0.2054027 | 10.1143158 | 0 | 1.0000000 |
| teaching_shuffle | 0.4178175 | 18.5290272 | 0 | 0.8765893 |

### Numerical and query checks

`{'affine_max_error': 2.284838984678572e-13, 'query_errors': {'0.05/False': 2.808917543006828e-10, '0.05/True': 2.808917543006828e-10, '0.1/False': 0.0, '0.1/True': 0.0, '0.2/False': 0.0, '0.2/True': 0.0}, 'query_no_write': True}`

Step refinement: `{'fast': {'generic_28': {'margin': 9.36584143573782e-13, 'state': 1.5210055437364645e-14}, 'generic_32': {'margin': 9.590106486712102e-13, 'state': 1.6431300764452317e-14}, 'generic_36': {'margin': 1.2210232824827472e-12, 'state': 2.0317081350640365e-14}, 'generic_40': {'margin': 1.3660184094987926e-12, 'state': 1.965094753586527e-14}, 'liu': {'margin': 9.461320615855584e-13, 'state': 1.6986412276764895e-14}}, 'primary': {'generic_28': {'margin': 1.0100809078039674e-12, 'state': 1.6986412276764895e-14}, 'generic_32': {'margin': 1.1572964808692632e-12, 'state': 1.84297022087776e-14}, 'generic_36': {'margin': 1.305622276959184e-12, 'state': 1.765254609153999e-14}, 'generic_40': {'margin': 1.227462576025573e-12, 'state': 1.6431300764452317e-14}, 'liu': {'margin': 9.940936962493652e-13, 'state': 1.787459069646502e-14}}, 'slow': {'generic_28': {'margin': 1.077360423096252e-12, 'state': 1.587618925213974e-14}, 'generic_32': {'margin': 1.042277375518097e-12, 'state': 1.8984813721090177e-14}, 'generic_36': {'margin': 1.0389467064442215e-12, 'state': 1.7985612998927536e-14}, 'generic_40': {'margin': 1.0595968547022494e-12, 'state': 1.6986412276764895e-14}, 'liu': {'margin': 8.72635297355373e-13, 'state': 1.7985612998927536e-14}}}`

### Sampled behavior: original definitions

| Row | Parent qualitative | Circuit qualitative | Parent quantitative | Circuit quantitative |
| --- | --- | --- | --- | --- |
| difficult_pair_bimodality | True | True | True | True |
| hodge_reconstructed_subjective_ranking | True | True | False | False |
| inter_subject_ranking_diversity | True | True | True | True |
| learned_accuracy | True | True | False | False |
| nonlearned_accuracy | True | True | False | False |
| self_consistent_vs_inconsistent_errors | True | True | False | False |
| serial_position_effect | True | True | False | False |
| stable_within_subject_errors | True | True | True | True |
| symbolic_distance_effect | True | True | False | False |

### fast/4096: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9299383 | 0.9517144 | 256 |
| generic/exact_decision/nonlearned | 0.8684505 | 0.8539028 | 0.8824704 | 256 |
| generic/probability/learned | 0.8523696 | 0.8434255 | 0.8610322 | 256 |
| generic/probability/nonlearned | 0.7910620 | 0.7801171 | 0.8018787 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8828502 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9285714 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9466178 | 0.9283448 | 0.9634541 | 77 |
| liu/probability/nonlearned | 0.8840579 | 0.8612975 | 0.9055368 | 77 |
| liu/probability/omitted | 0.8095640 | 0.7414084 | 0.8721312 | 69 |
| liu/probability/overall | 0.9019322 | 0.8818664 | 0.9205146 | 77 |
| liu/probability/retained | 0.9936641 | 0.9891749 | 0.9968196 | 77 |

### fast/4096: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | -0.0001079 | -0.0006168 | 0.0002930 | 256 |
| generic/probability/learned | -0.0000033 | -0.0000045 | -0.0000021 | 256 |
| generic/probability/nonlearned | 0.0000016 | 0.0000005 | 0.0000027 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000013 | -0.0000018 | 0.0000047 | 77 |
| liu/probability/nonlearned | 0.0000053 | 0.0000019 | 0.0000088 | 77 |
| liu/probability/omitted | 0.0000094 | -0.0000004 | 0.0000208 | 69 |
| liu/probability/overall | 0.0000041 | 0.0000016 | 0.0000068 | 77 |
| liu/probability/retained | -0.0000007 | -0.0000026 | 0.0000010 | 77 |

### fast/8192: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9299383 | 0.9517144 | 256 |
| generic/exact_decision/nonlearned | 0.8685533 | 0.8540413 | 0.8825795 | 256 |
| generic/probability/learned | 0.8523696 | 0.8434255 | 0.8610322 | 256 |
| generic/probability/nonlearned | 0.7910620 | 0.7801171 | 0.8018787 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8828502 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9285714 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9466178 | 0.9283448 | 0.9634541 | 77 |
| liu/probability/nonlearned | 0.8840579 | 0.8612975 | 0.9055368 | 77 |
| liu/probability/omitted | 0.8095640 | 0.7414084 | 0.8721312 | 69 |
| liu/probability/overall | 0.9019322 | 0.8818664 | 0.9205146 | 77 |
| liu/probability/retained | 0.9936641 | 0.9891749 | 0.9968196 | 77 |

### fast/8192: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | -0.0000051 | -0.0003084 | 0.0002930 | 256 |
| generic/probability/learned | -0.0000033 | -0.0000045 | -0.0000021 | 256 |
| generic/probability/nonlearned | 0.0000016 | 0.0000005 | 0.0000027 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000013 | -0.0000018 | 0.0000047 | 77 |
| liu/probability/nonlearned | 0.0000053 | 0.0000019 | 0.0000088 | 77 |
| liu/probability/omitted | 0.0000094 | -0.0000004 | 0.0000208 | 69 |
| liu/probability/overall | 0.0000041 | 0.0000016 | 0.0000068 | 77 |
| liu/probability/retained | -0.0000007 | -0.0000026 | 0.0000010 | 77 |

### mismatch_clamp: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| liu/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/exact_decision/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/probability/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |

### mismatch_clamp: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | -0.4411350 | -0.4517144 | -0.4299383 | 256 |
| generic/exact_decision/nonlearned | -0.3685584 | -0.3826119 | -0.3540937 | 256 |
| generic/probability/learned | -0.3523729 | -0.3610359 | -0.3434289 | 256 |
| generic/probability/nonlearned | -0.2910604 | -0.3018777 | -0.2801148 | 256 |
| liu/exact_decision/learned | -0.4496753 | -0.4675325 | -0.4301948 | 77 |
| liu/exact_decision/nonlearned | -0.3928571 | -0.4162338 | -0.3688312 | 77 |
| liu/exact_decision/omitted | -0.3140097 | -0.3828502 | -0.2415459 | 69 |
| liu/exact_decision/overall | -0.4090909 | -0.4285714 | -0.3877551 | 77 |
| liu/exact_decision/retained | -0.4978355 | -0.5000000 | -0.4935065 | 77 |
| liu/probability/learned | -0.4466165 | -0.4634523 | -0.4283424 | 77 |
| liu/probability/nonlearned | -0.3840526 | -0.4055302 | -0.3612920 | 77 |
| liu/probability/omitted | -0.3095547 | -0.3721268 | -0.2413920 | 69 |
| liu/probability/overall | -0.4019280 | -0.4205109 | -0.3818609 | 77 |
| liu/probability/retained | -0.4936648 | -0.4968201 | -0.4891753 | 77 |

### primary/4096: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9299383 | 0.9517144 | 256 |
| generic/exact_decision/nonlearned | 0.8685584 | 0.8540937 | 0.8826119 | 256 |
| generic/probability/learned | 0.8523663 | 0.8434221 | 0.8610285 | 256 |
| generic/probability/nonlearned | 0.7910636 | 0.7801195 | 0.8018797 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8828502 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9285714 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9466191 | 0.9283472 | 0.9634560 | 77 |
| liu/probability/nonlearned | 0.8840632 | 0.8613030 | 0.9055434 | 77 |
| liu/probability/omitted | 0.8095733 | 0.7414245 | 0.8721355 | 69 |
| liu/probability/overall | 0.9019363 | 0.8818718 | 0.9205183 | 77 |
| liu/probability/retained | 0.9936634 | 0.9891754 | 0.9968190 | 77 |

### primary/4096: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/probability/learned | -0.0000066 | -0.0000090 | -0.0000042 | 256 |
| generic/probability/nonlearned | 0.0000032 | 0.0000010 | 0.0000054 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000026 | -0.0000036 | 0.0000094 | 77 |
| liu/probability/nonlearned | 0.0000105 | 0.0000038 | 0.0000176 | 77 |
| liu/probability/omitted | 0.0000186 | -0.0000007 | 0.0000415 | 69 |
| liu/probability/overall | 0.0000083 | 0.0000033 | 0.0000136 | 77 |
| liu/probability/retained | -0.0000014 | -0.0000051 | 0.0000019 | 77 |

### primary/8192: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9299383 | 0.9517144 | 256 |
| generic/exact_decision/nonlearned | 0.8685533 | 0.8540413 | 0.8825795 | 256 |
| generic/probability/learned | 0.8523663 | 0.8434221 | 0.8610285 | 256 |
| generic/probability/nonlearned | 0.7910636 | 0.7801195 | 0.8018797 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8828502 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9285714 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9466191 | 0.9283472 | 0.9634560 | 77 |
| liu/probability/nonlearned | 0.8840632 | 0.8613030 | 0.9055434 | 77 |
| liu/probability/omitted | 0.8095733 | 0.7414245 | 0.8721355 | 69 |
| liu/probability/overall | 0.9019363 | 0.8818718 | 0.9205183 | 77 |
| liu/probability/retained | 0.9936634 | 0.9891754 | 0.9968190 | 77 |

### primary/8192: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | -0.0000051 | -0.0003084 | 0.0002930 | 256 |
| generic/probability/learned | -0.0000066 | -0.0000090 | -0.0000042 | 256 |
| generic/probability/nonlearned | 0.0000032 | 0.0000010 | 0.0000054 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000026 | -0.0000036 | 0.0000094 | 77 |
| liu/probability/nonlearned | 0.0000105 | 0.0000038 | 0.0000176 | 77 |
| liu/probability/omitted | 0.0000186 | -0.0000007 | 0.0000415 | 69 |
| liu/probability/overall | 0.0000083 | 0.0000033 | 0.0000136 | 77 |
| liu/probability/retained | -0.0000014 | -0.0000051 | 0.0000019 | 77 |

### slow/4096: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9299383 | 0.9517144 | 256 |
| generic/exact_decision/nonlearned | 0.8685584 | 0.8540937 | 0.8826119 | 256 |
| generic/probability/learned | 0.8523597 | 0.8434152 | 0.8610212 | 256 |
| generic/probability/nonlearned | 0.7910668 | 0.7801241 | 0.8018817 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8828502 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9285714 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9466216 | 0.9283519 | 0.9634595 | 77 |
| liu/probability/nonlearned | 0.8840736 | 0.8613139 | 0.9055566 | 77 |
| liu/probability/omitted | 0.8095915 | 0.7414562 | 0.8721442 | 69 |
| liu/probability/overall | 0.9019445 | 0.8818826 | 0.9205257 | 77 |
| liu/probability/retained | 0.9936620 | 0.9891704 | 0.9968175 | 77 |

### slow/4096: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/probability/learned | -0.0000132 | -0.0000180 | -0.0000085 | 256 |
| generic/probability/nonlearned | 0.0000064 | 0.0000020 | 0.0000106 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000050 | -0.0000072 | 0.0000186 | 77 |
| liu/probability/nonlearned | 0.0000210 | 0.0000076 | 0.0000351 | 77 |
| liu/probability/omitted | 0.0000369 | -0.0000014 | 0.0000822 | 69 |
| liu/probability/overall | 0.0000164 | 0.0000065 | 0.0000270 | 77 |
| liu/probability/retained | -0.0000029 | -0.0000102 | 0.0000038 | 77 |

### slow/8192: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.9411350 | 0.9299383 | 0.9517144 | 256 |
| generic/exact_decision/nonlearned | 0.8685584 | 0.8540937 | 0.8826119 | 256 |
| generic/probability/learned | 0.8523597 | 0.8434152 | 0.8610212 | 256 |
| generic/probability/nonlearned | 0.7910668 | 0.7801241 | 0.8018817 | 256 |
| liu/exact_decision/learned | 0.9496753 | 0.9301948 | 0.9675325 | 77 |
| liu/exact_decision/nonlearned | 0.8928571 | 0.8688312 | 0.9162338 | 77 |
| liu/exact_decision/omitted | 0.8140097 | 0.7415459 | 0.8828502 | 69 |
| liu/exact_decision/overall | 0.9090909 | 0.8877551 | 0.9285714 | 77 |
| liu/exact_decision/retained | 0.9978355 | 0.9935065 | 1.0000000 | 77 |
| liu/probability/learned | 0.9466216 | 0.9283519 | 0.9634595 | 77 |
| liu/probability/nonlearned | 0.8840736 | 0.8613139 | 0.9055566 | 77 |
| liu/probability/omitted | 0.8095915 | 0.7414562 | 0.8721442 | 69 |
| liu/probability/overall | 0.9019445 | 0.8818826 | 0.9205257 | 77 |
| liu/probability/retained | 0.9936620 | 0.9891704 | 0.9968175 | 77 |

### slow/8192: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 256 |
| generic/probability/learned | -0.0000132 | -0.0000180 | -0.0000085 | 256 |
| generic/probability/nonlearned | 0.0000064 | 0.0000020 | 0.0000106 | 256 |
| liu/exact_decision/learned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/nonlearned | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/omitted | 0.0000000 | 0.0000000 | 0.0000000 | 69 |
| liu/exact_decision/overall | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/exact_decision/retained | 0.0000000 | 0.0000000 | 0.0000000 | 77 |
| liu/probability/learned | 0.0000050 | -0.0000072 | 0.0000186 | 77 |
| liu/probability/nonlearned | 0.0000210 | 0.0000076 | 0.0000351 | 77 |
| liu/probability/omitted | 0.0000369 | -0.0000014 | 0.0000822 | 69 |
| liu/probability/overall | 0.0000164 | 0.0000065 | 0.0000270 | 77 |
| liu/probability/retained | -0.0000029 | -0.0000102 | 0.0000038 | 77 |

### teacher_off: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| generic/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 256 |
| liu/exact_decision/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/exact_decision/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/exact_decision/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/learned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/nonlearned | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/omitted | 0.5000000 | 0.5000000 | 0.5000000 | 69 |
| liu/probability/overall | 0.5000000 | 0.5000000 | 0.5000000 | 77 |
| liu/probability/retained | 0.5000000 | 0.5000000 | 0.5000000 | 77 |

### teacher_off: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | -0.4411350 | -0.4517144 | -0.4299383 | 256 |
| generic/exact_decision/nonlearned | -0.3685584 | -0.3826119 | -0.3540937 | 256 |
| generic/probability/learned | -0.3523729 | -0.3610359 | -0.3434289 | 256 |
| generic/probability/nonlearned | -0.2910604 | -0.3018777 | -0.2801148 | 256 |
| liu/exact_decision/learned | -0.4496753 | -0.4675325 | -0.4301948 | 77 |
| liu/exact_decision/nonlearned | -0.3928571 | -0.4162338 | -0.3688312 | 77 |
| liu/exact_decision/omitted | -0.3140097 | -0.3828502 | -0.2415459 | 69 |
| liu/exact_decision/overall | -0.4090909 | -0.4285714 | -0.3877551 | 77 |
| liu/exact_decision/retained | -0.4978355 | -0.5000000 | -0.4935065 | 77 |
| liu/probability/learned | -0.4466165 | -0.4634523 | -0.4283424 | 77 |
| liu/probability/nonlearned | -0.3840526 | -0.4055302 | -0.3612920 | 77 |
| liu/probability/omitted | -0.3095547 | -0.3721268 | -0.2413920 | 69 |
| liu/probability/overall | -0.4019280 | -0.4205109 | -0.3818609 | 77 |
| liu/probability/retained | -0.4936648 | -0.4968201 | -0.4891753 | 77 |

### teaching_shuffle: absolute endpoints

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | 0.5038551 | 0.4794892 | 0.5282414 | 256 |
| generic/exact_decision/nonlearned | 0.5014052 | 0.4833227 | 0.5196046 | 256 |
| generic/probability/learned | 0.5004832 | 0.4808280 | 0.5196410 | 256 |
| generic/probability/nonlearned | 0.5032345 | 0.4885841 | 0.5181702 | 256 |
| liu/exact_decision/learned | 0.4967532 | 0.4480519 | 0.5454545 | 77 |
| liu/exact_decision/nonlearned | 0.5032468 | 0.4694805 | 0.5376623 | 77 |
| liu/exact_decision/omitted | 0.5429952 | 0.4528986 | 0.6335870 | 69 |
| liu/exact_decision/overall | 0.5013915 | 0.4670686 | 0.5366419 | 77 |
| liu/exact_decision/retained | 0.4806122 | 0.4290816 | 0.5327930 | 77 |
| liu/probability/learned | 0.4982995 | 0.4511905 | 0.5456611 | 77 |
| liu/probability/nonlearned | 0.5026563 | 0.4696600 | 0.5361442 | 77 |
| liu/probability/omitted | 0.5367367 | 0.4507157 | 0.6221676 | 69 |
| liu/probability/overall | 0.5014115 | 0.4678087 | 0.5355110 | 77 |
| liu/probability/retained | 0.4854330 | 0.4346035 | 0.5362930 | 77 |

### teaching_shuffle: paired circuit minus original

| Endpoint | Mean | 95% lower | 95% upper | N |
| --- | ---: | ---: | ---: | ---: |
| generic/exact_decision/learned | -0.4372799 | -0.4641319 | -0.4102344 | 256 |
| generic/exact_decision/nonlearned | -0.3671532 | -0.3906959 | -0.3439107 | 256 |
| generic/probability/learned | -0.3518897 | -0.3736279 | -0.3307489 | 256 |
| generic/probability/nonlearned | -0.2878259 | -0.3062072 | -0.2692581 | 256 |
| liu/exact_decision/learned | -0.4529221 | -0.5081169 | -0.3993506 | 77 |
| liu/exact_decision/nonlearned | -0.3896104 | -0.4285714 | -0.3519481 | 77 |
| liu/exact_decision/omitted | -0.2710145 | -0.3937319 | -0.1473430 | 69 |
| liu/exact_decision/overall | -0.4076994 | -0.4466605 | -0.3687384 | 77 |
| liu/exact_decision/retained | -0.5172233 | -0.5683990 | -0.4651036 | 77 |
| liu/probability/learned | -0.4483170 | -0.5000298 | -0.3962711 | 77 |
| liu/probability/nonlearned | -0.3813964 | -0.4184556 | -0.3446508 | 77 |
| liu/probability/omitted | -0.2728180 | -0.3883147 | -0.1562661 | 69 |
| liu/probability/overall | -0.4005166 | -0.4384280 | -0.3631135 | 77 |
| liu/probability/retained | -0.5082318 | -0.5587670 | -0.4571403 | 77 |

## Preserved boundaries

- The circuit tests conditional rate-level implementability. Centered presynaptic activity, opposite teaching signs, fixed baseline cancellation, pooled gain, stable evidence admission and neutral episode initialization remain explicit assumptions, not validated cellular mechanisms.
- Stable z is supplied by the original external encoder; its persistent relation-specific implementation is not explained by these 36 states. No new gate, noise model, eligibility mechanism, BTSP or dopamine interpretation is introduced.
- The exact sign wiring and nonnegative efficacy implementation do not establish a full conductance-based or spiking realization, a brain-region attribution, or a human mechanism.
- No training, eta/gamma/temperature calibration, new seed search, altered task evidence, feedback during query, human fitting, or main-model promotion. The source diagnosis and all historical studies remain frozen.
- The new worktree is detached from the locked dev parent; scoped validated commits are pushed to origin/dev without a new remote branch. The original worktree files remain untouched.

## Stop rule

Execute the complete fixed matrix once after qualification and lock, report all seeds/scales/controls, then stop. A valid failure does not authorize changing the parameters, bounds, admission, readout, or the current candidate. Any successor biological constraint requires a new prospective question.
