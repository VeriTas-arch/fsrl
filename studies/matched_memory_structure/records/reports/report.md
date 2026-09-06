# Information-matched RNN memory structure

Registered development outcome: **both_fail_core**.

Three paired initializations; every candidate jointly trained for 48,000 generic episodes with the same two effective-evidence inputs. Human interval inclusion is descriptive. No seeds or model outcomes were pooled.

| Seed | Condition | N | Learned probability | Nonlearned probability | Omitted probability | Core |
|---|---|---|---|---|---|---|
| 2129 | dual | 10 | 0.9930 | 0.9328 | 0.9914 | False |
| 2129 | dual | 6 | 0.9957 | 0.9860 | 0.9988 | False |
| 2129 | dual | 8 | 0.9954 | 0.9636 | 0.9884 | False |
| 2129 | single | 10 | 0.9931 | 0.9363 | 0.9910 | False |
| 2129 | single | 6 | 0.9963 | 0.9865 | 0.9987 | False |
| 2129 | single | 8 | 0.9953 | 0.9633 | 0.9890 | False |
| 2130 | dual | 10 | 0.9942 | 0.9463 | 0.9965 | False |
| 2130 | dual | 6 | 0.9978 | 0.9855 | 0.9976 | False |
| 2130 | dual | 8 | 0.9966 | 0.9565 | 0.9939 | False |
| 2130 | single | 10 | 0.9941 | 0.9461 | 0.9965 | False |
| 2130 | single | 6 | 0.9976 | 0.9856 | 0.9973 | False |
| 2130 | single | 8 | 0.9966 | 0.9570 | 0.9941 | False |
| 2131 | dual | 10 | 0.9941 | 0.9385 | 0.9961 | False |
| 2131 | dual | 6 | 0.9920 | 0.9877 | 0.9853 | False |
| 2131 | dual | 8 | 0.9943 | 0.9581 | 0.9951 | False |
| 2131 | single | 10 | 0.9938 | 0.9425 | 0.9960 | False |
| 2131 | single | 6 | 0.9915 | 0.9870 | 0.9814 | False |
| 2131 | single | 8 | 0.9939 | 0.9595 | 0.9940 | False |

## Paired N=8 differences

Single minus dual, except explicitly named dual-benefit contrasts; 95% participant bootstrap intervals within each network.

- 2129 learned: -0.000132 [-0.001162, 0.000904].
- 2129 nonlearned: -0.000312 [-0.002300, 0.001609].
- 2129 omitted: 0.000611 [-0.001587, 0.002780].
- 2129 dual_omitted_benefit: -0.000611 [-0.002780, 0.001587].
- 2129 dual_omitted_specificity: -0.001196 [-0.004448, 0.002127].
- 2130 learned: -0.000015 [-0.000099, 0.000100].
- 2130 nonlearned: 0.000434 [-0.000271, 0.001176].
- 2130 omitted: 0.000155 [-0.000105, 0.000571].
- 2130 dual_omitted_benefit: -0.000155 [-0.000571, 0.000105].
- 2130 dual_omitted_specificity: 0.000215 [-0.000657, 0.001043].
- 2131 learned: -0.000378 [-0.001356, 0.000524].
- 2131 nonlearned: 0.001411 [-0.000945, 0.003842].
- 2131 omitted: -0.001102 [-0.004585, 0.001224].
- 2131 dual_omitted_benefit: 0.001102 [-0.001224, 0.004585].
- 2131 dual_omitted_specificity: 0.001826 [-0.001373, 0.005882].

## Boundaries

- Tests architecture plus its prescribed optimization at H=200 and 48000 generic episodes; not universal necessity of separate memory or optimization-independent minimality.
- Both new RNNs receive s_G and s_L. The historical differential-admission joint checkpoint is context only, not a matched causal control for input routing.
- Effective evidence channels encode a model hypothesis; p/z are not extra observed labels. No ranks, query answers, omission labels or human-summary targets enter recurrent inputs.
- Three paired training seeds, not six independent draws. Virtual participants remain nested within each network. This is development on project-exposed Liu data, not new human confirmation.
- Historical outcomes and thresholds remain unchanged. No tuning, new seeds, temperature fitting, codebook or decay additions, automatic main-model promotion.

N=6/10 are frozen-parameter transport under covarying support/query counts. Their descriptive core flags do not establish size-invariant binary stable-error prevalence. The canonical JSON retains all gates, effects, nine-row classifications, costs and artifact identities.
