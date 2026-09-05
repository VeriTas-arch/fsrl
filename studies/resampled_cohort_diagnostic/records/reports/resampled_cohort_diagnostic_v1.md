# Fixed Resampled cohort diagnostic

Diagnostic outcome: `sustained_above_reference`. All 400 independent 77-person cohorts per frozen fit are included; no parameter adaptation, participant pooling or parent-outcome revision.

| Fit | Mean distance slope | Whole-cohort 95% CI | Classification | All-nine joint cohort pass rate |
| --- | --- | --- | --- | --- |
| 2114 | 0.045213412183055086 | {'lower': 0.04495061183077129, 'upper': 0.045483793786995325} | sustained_above_reference | {'successes': 8, 'cohorts': 400, 'rate': 0.02, 'lower': 0.010168386791664645, 'upper': 0.03896341576926} |

## 2114: all original continuous endpoints

| Endpoint | Mean | 95% CI | Original reference | Classification | Undefined cohorts |
| --- | --- | --- | --- | --- | --- |
| learned_accuracy | 0.9003603896103896 | {'lower': 0.8990860186688311, 'upper': 0.9016099837662336} | {'lower': 0.8931818181818183, 'upper': 0.933766233766234} | mean_within_reference | 0 |
| nonlearned_accuracy | 0.8333241883116884 | {'lower': 0.8318660673701297, 'upper': 0.8347532508116884} | {'lower': 0.8033100649350651, 'upper': 0.8501964285714286} | mean_within_reference | 0 |
| symbolic_distance_effect | 0.045213412183055086 | {'lower': 0.04495061183077129, 'upper': 0.045483793786995325} | {'lower': 0.03475439526459938, 'upper': 0.04489879848043117} | sustained_above_reference | 0 |
| serial_position_effect | 0.07459601113172541 | {'lower': 0.07323514996907853, 'upper': 0.07601103509585651} | {'lower': 0.05373840445269019, 'upper': 0.11070037105751393} | mean_within_reference | 0 |
| stable_within_subject_errors | 0.9392680871920445 | {'lower': 0.9365050145148869, 'upper': 0.9420154568629542} | {'lower': 0.8405797101449275, 'upper': 0.9714285714285714} | mean_within_reference | 0 |
| self_consistent_incorrect | 0.9497498689906584 | {'lower': 0.9472522376965142, 'upper': 0.95224454488494} | {'lower': 0.7402597402597403, 'upper': 0.9090909090909091} | sustained_above_reference | 0 |
| self_inconsistent | 0.01684600706311233 | {'lower': 0.015420448564593287, 'upper': 0.01831386933242195} | {'lower': 0.012987012987012988, 'upper': 0.12987012987012986} | mean_within_reference | 0 |
| correct_ranker | 0.033404123946229214 | {'lower': 0.03143788932558669, 'upper': 0.035448341592617894} | {'lower': 0.03896103896103896, 'upper': 0.16883116883116883} | sustained_below_reference | 0 |
| inter_subject_ranking_diversity | 0.5637267890173642 | {'lower': 0.5602842392308984, 'upper': 0.5671367660614304} | {'lower': 0.5071839605407381, 'upper': 0.6124101814638899} | mean_within_reference | 0 |

| Original behavior row | Qualitative pass rate | Quantitative pass rate |
| --- | --- | --- |
| difficult_pair_bimodality | {'successes': 389, 'cohorts': 400, 'rate': 0.9725, 'lower': 0.9514345181143852, 'upper': 0.9845763637397046} | {'successes': 389, 'cohorts': 400, 'rate': 0.9725, 'lower': 0.9514345181143852, 'upper': 0.9845763637397046} |
| hodge_reconstructed_subjective_ranking | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 185, 'cohorts': 400, 'rate': 0.4625, 'lower': 0.41422725327343163, 'upper': 0.5114861688016406} |
| inter_subject_ranking_diversity | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 344, 'cohorts': 400, 'rate': 0.86, 'lower': 0.8225607694538968, 'upper': 0.8905903786254097} |
| learned_accuracy | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 289, 'cohorts': 400, 'rate': 0.7225, 'lower': 0.6766612653345575, 'upper': 0.7641057636866807} |
| nonlearned_accuracy | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 337, 'cohorts': 400, 'rate': 0.8425, 'lower': 0.8035652075632727, 'upper': 0.8749188708177342} |
| self_consistent_vs_inconsistent_errors | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 41, 'cohorts': 400, 'rate': 0.1025, 'lower': 0.07645885298219338, 'upper': 0.1361034210135723} |
| serial_position_effect | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 368, 'cohorts': 400, 'rate': 0.92, 'lower': 0.8892454231693245, 'upper': 0.9427642495898665} |
| stable_within_subject_errors | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 325, 'cohorts': 400, 'rate': 0.8125, 'lower': 0.7713439706023679, 'upper': 0.8477108454386968} |
| symbolic_distance_effect | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 184, 'cohorts': 400, 'rate': 0.46, 'lower': 0.4117698068715022, 'upper': 0.5089911766752415} |
| 2115 | 0.045199495317607606 | {'lower': 0.04492802660349858, 'upper': 0.04546443493793624} | sustained_above_reference | {'successes': 8, 'cohorts': 400, 'rate': 0.02, 'lower': 0.010168386791664645, 'upper': 0.03896341576926} |

## 2115: all original continuous endpoints

| Endpoint | Mean | 95% CI | Original reference | Classification | Undefined cohorts |
| --- | --- | --- | --- | --- | --- |
| learned_accuracy | 0.9004334415584415 | {'lower': 0.8991562500000002, 'upper': 0.901639620535714} | {'lower': 0.8931818181818183, 'upper': 0.933766233766234} | mean_within_reference | 0 |
| nonlearned_accuracy | 0.8333798701298701 | {'lower': 0.8319420292207792, 'upper': 0.8348001988636361} | {'lower': 0.8033100649350651, 'upper': 0.8501964285714286} | mean_within_reference | 0 |
| symbolic_distance_effect | 0.045199495317607606 | {'lower': 0.04492802660349858, 'upper': 0.04546443493793624} | {'lower': 0.03475439526459938, 'upper': 0.04489879848043117} | sustained_above_reference | 0 |
| serial_position_effect | 0.07466141001855288 | {'lower': 0.07325895562770562, 'upper': 0.0760357683982684} | {'lower': 0.05373840445269019, 'upper': 0.11070037105751393} | mean_within_reference | 0 |
| stable_within_subject_errors | 0.9390738371238424 | {'lower': 0.9363302109462915, 'upper': 0.9418552863026319} | {'lower': 0.8405797101449275, 'upper': 0.9714285714285714} | mean_within_reference | 0 |
| self_consistent_incorrect | 0.9493554454317612 | {'lower': 0.9467508158179541, 'upper': 0.9517834142173619} | {'lower': 0.7402597402597403, 'upper': 0.9090909090909091} | sustained_above_reference | 0 |
| self_inconsistent | 0.01684904306220096 | {'lower': 0.015419606402369543, 'upper': 0.018321365487582578} | {'lower': 0.012987012987012988, 'upper': 0.12987012987012986} | mean_within_reference | 0 |
| correct_ranker | 0.03379551150603782 | {'lower': 0.03180418631806787, 'upper': 0.03587503061631349} | {'lower': 0.03896103896103896, 'upper': 0.16883116883116883} | sustained_below_reference | 0 |
| inter_subject_ranking_diversity | 0.5639206112799573 | {'lower': 0.5603973524473981, 'upper': 0.5673741650847323} | {'lower': 0.5071839605407381, 'upper': 0.6124101814638899} | mean_within_reference | 0 |

| Original behavior row | Qualitative pass rate | Quantitative pass rate |
| --- | --- | --- |
| difficult_pair_bimodality | {'successes': 389, 'cohorts': 400, 'rate': 0.9725, 'lower': 0.9514345181143852, 'upper': 0.9845763637397046} | {'successes': 389, 'cohorts': 400, 'rate': 0.9725, 'lower': 0.9514345181143852, 'upper': 0.9845763637397046} |
| hodge_reconstructed_subjective_ranking | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 190, 'cohorts': 400, 'rate': 0.475, 'lower': 0.4265327259812066, 'upper': 0.5239428887355082} |
| inter_subject_ranking_diversity | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 346, 'cohorts': 400, 'rate': 0.865, 'lower': 0.8280190157519688, 'upper': 0.8950370093839947} |
| learned_accuracy | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 291, 'cohorts': 400, 'rate': 0.7275, 'lower': 0.6818568086225227, 'upper': 0.7688150974553725} |
| nonlearned_accuracy | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 337, 'cohorts': 400, 'rate': 0.8425, 'lower': 0.8035652075632727, 'upper': 0.8749188708177342} |
| self_consistent_vs_inconsistent_errors | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 36, 'cohorts': 400, 'rate': 0.09, 'lower': 0.06571729376939447, 'upper': 0.12208278758472865} |
| serial_position_effect | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 369, 'cohorts': 400, 'rate': 0.9225, 'lower': 0.892095045964364, 'upper': 0.9448670653231555} |
| stable_within_subject_errors | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 328, 'cohorts': 400, 'rate': 0.82, 'lower': 0.7793624469518832, 'upper': 0.8545496846741669} |
| symbolic_distance_effect | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 188, 'cohorts': 400, 'rate': 0.47, 'lower': 0.4216068935658794, 'upper': 0.5189638440941783} |
| 2116 | 0.04519622316459054 | {'lower': 0.04492887125629477, 'upper': 0.04546741803052394} | sustained_above_reference | {'successes': 11, 'cohorts': 400, 'rate': 0.0275, 'lower': 0.015423636260295411, 'upper': 0.048565481885614784} |

## 2116: all original continuous endpoints

| Endpoint | Mean | 95% CI | Original reference | Classification | Undefined cohorts |
| --- | --- | --- | --- | --- | --- |
| learned_accuracy | 0.9004517045454545 | {'lower': 0.8992037337662336, 'upper': 0.9016973315746756} | {'lower': 0.8931818181818183, 'upper': 0.933766233766234} | mean_within_reference | 0 |
| nonlearned_accuracy | 0.8333464285714285 | {'lower': 0.8318937987012988, 'upper': 0.8347655884740259} | {'lower': 0.8033100649350651, 'upper': 0.8501964285714286} | mean_within_reference | 0 |
| symbolic_distance_effect | 0.04519622316459054 | {'lower': 0.04492887125629477, 'upper': 0.04546741803052394} | {'lower': 0.03475439526459938, 'upper': 0.04489879848043117} | sustained_above_reference | 0 |
| serial_position_effect | 0.07478153988868275 | {'lower': 0.0733824752628324, 'upper': 0.07617998995052566} | {'lower': 0.05373840445269019, 'upper': 0.11070037105751393} | mean_within_reference | 0 |
| stable_within_subject_errors | 0.9376346302563117 | {'lower': 0.9348205025721494, 'upper': 0.94035911238435} | {'lower': 0.8405797101449275, 'upper': 0.9714285714285714} | mean_within_reference | 0 |
| self_consistent_incorrect | 0.9488022271588059 | {'lower': 0.9461974202551838, 'upper': 0.951345552232855} | {'lower': 0.7402597402597403, 'upper': 0.9090909090909091} | sustained_above_reference | 0 |
| self_inconsistent | 0.016816136933242198 | {'lower': 0.015326465168603315, 'upper': 0.018335679824561385} | {'lower': 0.012987012987012988, 'upper': 0.12987012987012986} | mean_within_reference | 0 |
| correct_ranker | 0.03438163590795171 | {'lower': 0.03226380838459784, 'upper': 0.03649999131351104} | {'lower': 0.03896103896103896, 'upper': 0.16883116883116883} | sustained_below_reference | 0 |
| inter_subject_ranking_diversity | 0.5640670194170734 | {'lower': 0.5606538223662333, 'upper': 0.5674395693885365} | {'lower': 0.5071839605407381, 'upper': 0.6124101814638899} | mean_within_reference | 0 |

| Original behavior row | Qualitative pass rate | Quantitative pass rate |
| --- | --- | --- |
| difficult_pair_bimodality | {'successes': 389, 'cohorts': 400, 'rate': 0.9725, 'lower': 0.9514345181143852, 'upper': 0.9845763637397046} | {'successes': 389, 'cohorts': 400, 'rate': 0.9725, 'lower': 0.9514345181143852, 'upper': 0.9845763637397046} |
| hodge_reconstructed_subjective_ranking | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 198, 'cohorts': 400, 'rate': 0.495, 'lower': 0.44628448078239025, 'upper': 0.5438106421609528} |
| inter_subject_ranking_diversity | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 342, 'cohorts': 400, 'rate': 0.855, 'lower': 0.8171167606818923, 'upper': 0.8861295103407572} |
| learned_accuracy | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 291, 'cohorts': 400, 'rate': 0.7275, 'lower': 0.6818568086225227, 'upper': 0.7688150974553725} |
| nonlearned_accuracy | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 337, 'cohorts': 400, 'rate': 0.8425, 'lower': 0.8035652075632727, 'upper': 0.8749188708177342} |
| self_consistent_vs_inconsistent_errors | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 35, 'cohorts': 400, 'rate': 0.0875, 'lower': 0.06358691021150324, 'upper': 0.11926073261429136} |
| serial_position_effect | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 369, 'cohorts': 400, 'rate': 0.9225, 'lower': 0.892095045964364, 'upper': 0.9448670653231555} |
| stable_within_subject_errors | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 340, 'cohorts': 400, 'rate': 0.85, 'lower': 0.8116863892693679, 'upper': 0.8816550046966246} |
| symbolic_distance_effect | {'successes': 400, 'cohorts': 400, 'rate': 1.0, 'lower': 0.9904877056657034, 'upper': 1.0} | {'successes': 185, 'cohorts': 400, 'rate': 0.4625, 'lower': 0.41422725327343163, 'upper': 0.5114861688016406} |

Pointwise simulation uncertainty, not new human equivalence intervals. The input evidence, temperature and human reference intervals retain their historical exposure. All six morphology counts per cohort and the joint bimodal/unimodal/low-accuracy distribution are retained in linked records.

This is a diagnostic of the failed pilot, not a new admission test. The original partial_behavioral_reproduction outcome remains unchanged; no main model is promoted.

After all 400 cohorts for all three fits, publish and stop. No extra cohorts, retuning, new codebook, metric/readout shopping, new architecture or biological-realization run as repair. Any successor requires a new question and authorization.
