# Claim-relative main-model evaluation

Registered outcome: `retrospective_core_behavior_supported`. This is a retrospective audit of exposed frozen simulations, not confirmation under the new stability threshold.

The unchanged legacy 9/9 metric remains `full_quantitative_fidelity`. The added `core_mechanism_adequacy` label separates central qualitative phenomena and task-difficulty guardrails from strict calibration of every summary statistic.

| Fit | Core (retrospective) | Legacy pilot | Joint qualitative stability | Mean learned | Mean nonlearned |
| --- | --- | --- | --- | --- | --- |
| 2114 | True | 9/9 qualitative; 8/9 quantitative | {'cohorts': 400, 'lower': 0.9514345181143852, 'rate': 0.9725, 'successes': 389, 'upper': 0.9845763637397046} | 0.9003603896103896 {'lower': 0.8990860186688311, 'upper': 0.9016099837662336} | 0.8333241883116884 {'lower': 0.8318660673701297, 'upper': 0.8347532508116884} |
| 2115 | True | 9/9 qualitative; 8/9 quantitative | {'cohorts': 400, 'lower': 0.9514345181143852, 'rate': 0.9725, 'successes': 389, 'upper': 0.9845763637397046} | 0.9004334415584415 {'lower': 0.8991562500000002, 'upper': 0.901639620535714} | 0.8333798701298701 {'lower': 0.8319420292207792, 'upper': 0.8348001988636361} |
| 2116 | True | 9/9 qualitative; 8/9 quantitative | {'cohorts': 400, 'lower': 0.9514345181143852, 'rate': 0.9725, 'successes': 389, 'upper': 0.9845763637397046} | 0.9004517045454545 {'lower': 0.8992037337662336, 'upper': 0.9016973315746756} | 0.8333464285714285 {'lower': 0.8318937987012988, 'upper': 0.8347655884740259} |

## Internal-to-sampled localization

Cross-fit classification: `direction_unresolved`.

| Fit | Internal strict correct | Sampled correct (all 77) | Loss flow | Rescue flow | Net sampling shift | Rank-composition TV |
| --- | --- | --- | --- | --- | --- | --- |
| 2114 | {'mean': 0.03370129870129881, 'interval': {'lower': 0.03168749999999997, 'upper': 0.03574675324675323}} | {'mean': 0.03334415584415596, 'interval': {'lower': 0.03133116883116882, 'upper': 0.035357142857142844}} | {'mean': 0.006623376623376617, 'interval': {'lower': 0.005779220779220777, 'upper': 0.0075324675324675286}} | {'mean': 0.006266233766233761, 'interval': {'lower': 0.005422077922077922, 'upper': 0.00714285714285714}} | {'mean': -0.00035714285714285714, 'interval': {'lower': -0.0015909090909090912, 'upper': 0.0008766233766233767}} | {'mean': 0.11867844041922991, 'interval': {'lower': 0.11615053870471627, 'upper': 0.12116186360788328}} |
| 2115 | {'mean': 0.03396103896103907, 'interval': {'lower': 0.03194805194805193, 'upper': 0.03607142857142857}} | {'mean': 0.033733766233766345, 'interval': {'lower': 0.03172077922077921, 'upper': 0.03577922077922078}} | {'mean': 0.0064285714285714215, 'interval': {'lower': 0.005551948051948051, 'upper': 0.007337662337662334}} | {'mean': 0.006201298701298696, 'interval': {'lower': 0.005357142857142857, 'upper': 0.0070779220779220746}} | {'mean': -0.0002272727272727271, 'interval': {'lower': -0.0014935064935064934, 'upper': 0.0010389610389610385}} | {'mean': 0.1182840168603327, 'interval': {'lower': 0.11581302688539528, 'upper': 0.12070401785714276}} |
| 2116 | {'mean': 0.03392857142857154, 'interval': {'lower': 0.031882305194805165, 'upper': 0.036006493506493494}} | {'mean': 0.034318181818181935, 'interval': {'lower': 0.03227191558441556, 'upper': 0.036363636363636355}} | {'mean': 0.006071428571428564, 'interval': {'lower': 0.005227272727272727, 'upper': 0.00691558441558441}} | {'mean': 0.0064610389610389556, 'interval': {'lower': 0.005584415584415583, 'upper': 0.007370129870129864}} | {'mean': 0.0003896103896103896, 'interval': {'lower': -0.0008766233766233768, 'upper': 0.0016558441558441562}} | {'mean': 0.11769833105491001, 'interval': {'lower': 0.11515843401116421, 'upper': 0.12018504699248113}} |

Transition cells and inversion bins are stored in the frozen JSON/NPZ result. Internal strict correctness is an unobserved model state and is not compared with a human latent-state target. Ranking composition uses the unchanged eligible-subject denominator; transition flows use all 77 simulated subjects.

## Consequence for the next single-stage model

The next primary candidate remains the prospectively specified relation-specific experience-dependent plasticity rule. Its purpose is narrow: reduce late-code domination and improve deterministic internal ordering while keeping the Resampled codebook, stable admission, 15-dimensional state, query form and single-stage objective fixed. Because the rule changes both mean shrinkage and variance propagation, those effects must be reported separately.

A paired fixed-eta baseline, a nonbalanced schedule control, one to three fresh development seeds and internal-order improvement are required before any replication. The observed sampling contribution remains a separate policy-expression boundary; the plasticity candidate is not allowed to claim it away. No training is authorized by this audit.

The current exposed recipe is not promoted: it lacks prospective validation under the new threshold, unchanged fresh training replication and selected-parameter biological-boundary verification, and it does not satisfy the preserved full 9/9 quantitative label.
