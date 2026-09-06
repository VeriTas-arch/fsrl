# Weak-evidence routing: paired development result

Supported recovery: 3/3 paired initializations. Conditional transport: completed_all_six_models.

The task data are project-exposed. Intervals describe participants within each network; networks are not pooled. The two arms retain the same local evidence and jointly train all slow parameters.

| Seed | Route | Learned probability | Nonlearned probability | Behavioral tau [95% CI] | Analysis N | Core |
| --- | --- | --- | --- | --- | --- | --- |
| 2132 | shared | 0.98172 | 0.93283 | 0.78196 [0.74857, 0.82050] | 51 | False |
| 2132 | isolated | 0.91295 | 0.79883 | 0.49446 [0.44993, 0.55567] | 73 | True |
| 2133 | shared | 0.98446 | 0.92653 | 0.75607 [0.73117, 0.79070] | 49 | True |
| 2133 | isolated | 0.90687 | 0.80378 | 0.47504 [0.42747, 0.53626] | 70 | True |
| 2134 | shared | 0.98469 | 0.92706 | 0.77006 [0.74538, 0.80417] | 53 | False |
| 2134 | isolated | 0.91313 | 0.79185 | 0.47945 [0.42971, 0.54769] | 71 | True |

All paired differences are isolated minus shared.

| Seed | Conditional tau difference | All-subject internal-order tau difference | Outcome |
| --- | --- | --- | --- |
| 2132 | -0.28751 [-0.34405, -0.22528] | -0.31242 [-0.36608, -0.25016] | supported_recovery |
| 2133 | -0.28104 [-0.33830, -0.22013] | -0.32892 [-0.38004, -0.26823] | supported_recovery |
| 2134 | -0.29061 [-0.34182, -0.22881] | -0.32033 [-0.37292, -0.26237] | supported_recovery |

## Accuracy and internal-error differences

- 2132 learned: -0.06877 [-0.08718, -0.05036].
- 2132 nonlearned: -0.13400 [-0.15931, -0.10990].
- 2132 omitted: -0.16244 [-0.20766, -0.12008].
- 2132 internal_stable_error_prevalence: 0.32468 [0.18182, 0.46753].
- 2132 internal_stable_error_density: 0.08163 [0.06354, 0.10019].
- 2133 learned: -0.07760 [-0.09903, -0.05675].
- 2133 nonlearned: -0.12275 [-0.14575, -0.10116].
- 2133 omitted: -0.16170 [-0.20639, -0.11679].
- 2133 internal_stable_error_prevalence: 0.41558 [0.31169, 0.51948].
- 2133 internal_stable_error_density: 0.08256 [0.06494, 0.10020].
- 2134 learned: -0.07156 [-0.09229, -0.05258].
- 2134 nonlearned: -0.13520 [-0.15995, -0.11354].
- 2134 omitted: -0.16393 [-0.20784, -0.12039].
- 2134 internal_stable_error_prevalence: 0.33766 [0.23377, 0.45455].
- 2134 internal_stable_error_density: 0.08163 [0.06494, 0.10065].

## Conditional transport

- 2132 shared N=10: nonlearned 0.89828 [0.88261, 0.91396]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': True, 'organized_errors': True, 'stable_errors': True}.
- 2132 shared N=6: nonlearned 0.98492 [0.97616, 0.99207]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': False, 'organized_errors': True, 'stable_errors': True}.
- 2132 isolated N=10: nonlearned 0.77462 [0.75133, 0.79788]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': True, 'organized_errors': True, 'stable_errors': True}.
- 2132 isolated N=6: nonlearned 0.86386 [0.82416, 0.90095]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': True, 'organized_errors': True, 'stable_errors': True}.
- 2133 shared N=10: nonlearned 0.88600 [0.86889, 0.90134]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': True, 'organized_errors': True, 'stable_errors': True}.
- 2133 shared N=6: nonlearned 0.98375 [0.97596, 0.99031]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': False, 'organized_errors': True, 'stable_errors': True}.
- 2133 isolated N=10: nonlearned 0.77279 [0.74892, 0.79583]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': True, 'organized_errors': True, 'stable_errors': True}.
- 2133 isolated N=6: nonlearned 0.85926 [0.82359, 0.89309]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': True, 'organized_errors': True, 'stable_errors': True}.
- 2134 shared N=10: nonlearned 0.88212 [0.86409, 0.89778]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': True, 'organized_errors': True, 'stable_errors': True}.
- 2134 shared N=6: nonlearned 0.98521 [0.97787, 0.99182]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': False, 'organized_errors': True, 'stable_errors': True}.
- 2134 isolated N=10: nonlearned 0.76165 [0.73813, 0.78515]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': True, 'organized_errors': True, 'stable_errors': True}.
- 2134 isolated N=6: nonlearned 0.85632 [0.81710, 0.89303]; competence=True; core flags={'coherence': True, 'competence': True, 'individual_diversity': True, 'organized_errors': True, 'stable_errors': True}.

## Claim boundaries

- Development on existing Liu task knowledge; three paired fits, one fixed virtual cohort per size; no human confirmation or main-model promotion.
- Identifies routing plus adaptation under a fixed joint learning recipe, not acute routing, necessity of L, or emergence of a learned gate.
- Shared and isolated are both dual-state models. Retained evidence is duplicated identically; omitted global evidence differs, local evidence does not.
- Old source and results stay frozen; no temperature, noise, human fitting, replacement seeds or selected checkpoints.

The canonical JSON also reports inclusion changes, undefined bootstrap draws, internal confidence-defined errors, local-off/fast-weight/evidence interventions, remote source-removal effects and all legacy qualitative rows. N6/N10 change support and query counts together with item count.
