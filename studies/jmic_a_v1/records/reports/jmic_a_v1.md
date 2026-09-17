# JMIC-A v1: analytic qualification and frozen predictions

## Outcome

- Implementation: `qualified`
- Identification: `qualified_for_prediction_registration`
- Human-model status: not evaluated; no participant responses were read.
- Neural status: not evaluated; no JMIC-P model was trained.

The result qualifies or limits one fixed continuous Gaussian inference family. It is not a human rescue result and does not authorize model promotion.

## Exact structural predictions

- Cycle uncertainty classes: 4.
- A/B magnitude placement changes posterior means but not posterior covariance.
- Trialwise marginal and persistent-continuous readouts have identical pair marginals; their maximum recorded discrepancy is `0`.
- V1 defines no repetition-count learning law.

## Identification across all fixed cells

| sigma/tau0 | theta/tau0 | Jacobian condition | SBC | large recovery |
|---:|---:|---:|:---:|:---:|
| 0.250000 | 0.125000 | 4.234273 | pass | pass |
| 0.250000 | 0.250000 | 3.312339 | pass | pass |
| 0.250000 | 0.500000 | 7.367953 | pass | pass |
| 0.500000 | 0.125000 | 6.681564 | pass | pass |
| 0.500000 | 0.250000 | 3.157774 | pass | pass |
| 0.500000 | 0.500000 | 2.934050 | pass | pass |
| 1.000000 | 0.125000 | 9.494705 | pass | pass |
| 1.000000 | 0.250000 | 4.940645 | pass | pass |
| 1.000000 | 0.500000 | 3.098076 | pass | pass |

## Frozen A/B predictions

### `persistent_continuous`

| estimand | direction | range |
|---|---|---:|
| beta_conf | positive_in_all_cells | [0.031732, 0.068219] |
| beta_learned | positive_in_all_cells | [0.007903, 0.049818] |
| delta_flip | positive_in_all_cells | [0.330239, 0.855106] |

### `persistent_equal_spacing`

| estimand | direction | range |
|---|---|---:|
| beta_conf | positive_in_all_cells | [0.026856, 0.066345] |
| beta_learned | positive_in_all_cells | [0.005576, 0.047118] |
| delta_flip | positive_in_all_cells | [0.313804, 0.877241] |

### `posterior_mean`

| estimand | direction | range |
|---|---|---:|
| beta_conf | positive_in_all_cells | [0.007821, 0.063419] |
| beta_learned | positive_in_all_cells | [0.002909, 0.042600] |
| delta_flip | positive_in_all_cells | [0.456505, 0.964237] |

### `trialwise_marginal`

| estimand | direction | range |
|---|---|---:|
| beta_conf | positive_in_all_cells | [0.031732, 0.068219] |
| beta_learned | positive_in_all_cells | [0.007903, 0.049818] |
| delta_flip | positive_in_all_cells | [0.330239, 0.855106] |

## Descriptive historical morphology compatibility

These rows use fixed qualitative definitions without human calibration intervals or an evidence-binding intervention. They did not select parameters.

| readout | all-nine cells | constrained-without-binding cells |
|---|---:|---:|
| posterior_mean | 0/9 | 0/9 |
| trialwise_marginal | 0/9 | 0/9 |
| persistent_continuous | 1/9 | 1/9 |
| persistent_equal_spacing | 2/9 | 2/9 |

## Claim boundary

This is a theory-qualification and prediction-registration study. It uses no neural network, reads no Liu participant response file, performs no human-phenotype parameter fit, authorizes no human collection, and cannot promote JMIC-A or JMIC-P as a human or biological mechanism.

The next scientific step is prediction registration against independent behavioral conditions or an explicitly computational neural-implementation study. Synthetic calibration alone does not start JMIC-P.
