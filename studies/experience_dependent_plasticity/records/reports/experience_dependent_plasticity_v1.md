# Experience-dependent score plasticity pilot

Registered outcome: `global_schedule_preferred`. Selected scheduler: `global`.

All six paired 2117--2119 fits and all 400 prospective 77-subject cohorts per fit are included. Participants are analyzed within each fit; no network or participant pooling is used.

| Seed | eta fixed | eta0 adaptive | gamma fixed | gamma adaptive | Internal correct delta | Inversion delta | Rank-TV delta | Adaptive core | Preservation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2117 | 0.5424730777740479 | 0.9802668690681458 | 5.968433380126953 | 6.2126383781433105 | {'mean': 0.018084415584415585, 'interval': {'lower': 0.015324675324675322, 'upper': 0.020844155844155835}} | {'mean': -0.15227272727272728, 'interval': {'lower': -0.1655194805194805, 'upper': -0.1393498376623377}} | {'mean': -0.015231061991588309, 'interval': {'lower': -0.01848185442106495, 'upper': -0.011994622740050372}} | True | True |

## Seed 2117: continuous profile

| Endpoint | Fixed mean (95% CI) | Adaptive mean (95% CI) | Human reference | Adaptive classification |
| --- | --- | --- | --- | --- |
| learned_accuracy | 0.900895698051948 {'lower': 0.8996960024350651, 'upper': 0.902144906655844} | 0.9098413149350649 {'lower': 0.9087267349837665, 'upper': 0.9110058035714286} | {'lower': 0.8931818181818183, 'upper': 0.933766233766234} | mean_within_reference |
| nonlearned_accuracy | 0.8329612012987013 {'lower': 0.8315257792207791, 'upper': 0.8344222605519479} | 0.8369698051948052 {'lower': 0.8355532426948048, 'upper': 0.8384328003246753} | {'lower': 0.8033100649350651, 'upper': 0.8501964285714286} | mean_within_reference |
| symbolic_distance_effect | 0.045499639985864516 {'lower': 0.045221082830859655, 'upper': 0.04576681699465504} | 0.04400293643431403 {'lower': 0.04372544029949645, 'upper': 0.04426917928372652} | {'lower': 0.03475439526459938, 'upper': 0.04489879848043117} | mean_within_reference |
| serial_position_effect | 0.07623036487322202 {'lower': 0.07476374845392703, 'upper': 0.07771306431663574} | 0.0719534632034632 {'lower': 0.0705101654298083, 'upper': 0.07339862399505255} | {'lower': 0.05373840445269019, 'upper': 0.11070037105751393} | mean_within_reference |
| stable_within_subject_errors | 0.9405591095808247 {'lower': 0.937867814327869, 'upper': 0.9431926889804313} | 0.9291008221694141 {'lower': 0.9258996180268873, 'upper': 0.9322826925937977} | {'lower': 0.8405797101449275, 'upper': 0.9714285714285714} | mean_within_reference |
| self_consistent_incorrect | 0.9495333333333333 {'lower': 0.9469279971234908, 'upper': 0.9520795938710415} | 0.9340694449576027 {'lower': 0.9311500076204023, 'upper': 0.9369672925980493} | {'lower': 0.7402597402597403, 'upper': 0.9090909090909091} | sustained_above_reference |
| self_inconsistent | 0.017594030530872638 {'lower': 0.01616332222601959, 'upper': 0.01908927830940988} | 0.015015806561859196 {'lower': 0.013652159518113464, 'upper': 0.016411493933697875} | {'lower': 0.012987012987012988, 'upper': 0.12987012987012986} | mean_within_reference |
| correct_ranker | 0.03287263613579403 {'lower': 0.03075106245727955, 'upper': 0.03500520463089541} | 0.05091474848053796 {'lower': 0.04840272267739371, 'upper': 0.05352672682935838} | {'lower': 0.03896103896103896, 'upper': 0.16883116883116883} | mean_within_reference |
| inter_subject_ranking_diversity | 0.5641860357941831 {'lower': 0.5607175130031766, 'upper': 0.5676715479786361} | 0.5707920176730042 {'lower': 0.5674374486725285, 'upper': 0.5741537747601978} | {'lower': 0.5071839605407381, 'upper': 0.6124101814638899} | mean_within_reference |

Adaptive all-nine qualitative stability: {'successes': 393, 'cohorts': 400, 'rate': 0.9825, 'lower': 0.964322927074442, 'upper': 0.991497708892962}; quantitative stability: {'successes': 35, 'cohorts': 400, 'rate': 0.0875, 'lower': 0.06358691021150324, 'upper': 0.11926073261429136}; joint: {'successes': 35, 'cohorts': 400, 'rate': 0.0875, 'lower': 0.06358691021150324, 'upper': 0.11926073261429136}.
| 2118 | 0.5670230388641357 | 0.9801944494247437 | 5.967877388000488 | 6.199335098266602 | {'mean': 0.019090909090909092, 'interval': {'lower': 0.016330357142857136, 'upper': 0.02194805194805195}} | {'mean': -0.15896103896103897, 'interval': {'lower': -0.1722727272727273, 'upper': -0.14558441558441562}} | {'mean': -0.01702582731398521, 'interval': {'lower': -0.020374356941955628, 'upper': -0.013778559820928243}} | True | True |

## Seed 2118: continuous profile

| Endpoint | Fixed mean (95% CI) | Adaptive mean (95% CI) | Human reference | Adaptive classification |
| --- | --- | --- | --- | --- |
| learned_accuracy | 0.9006525974025975 {'lower': 0.899451674107143, 'upper': 0.9018539468344156} | 0.909820211038961 {'lower': 0.9086927556818181, 'upper': 0.9109400263798703} | {'lower': 0.8931818181818183, 'upper': 0.933766233766234} | mean_within_reference |
| nonlearned_accuracy | 0.8328612012987012 {'lower': 0.8314277353896102, 'upper': 0.8342680316558443} | 0.8369332792207791 {'lower': 0.8354798701298702, 'upper': 0.8383519561688311} | {'lower': 0.8033100649350651, 'upper': 0.8501964285714286} | mean_within_reference |
| symbolic_distance_effect | 0.045556023058574126 {'lower': 0.045286819451806744, 'upper': 0.04582316089097981} | 0.044009857319551246 {'lower': 0.04373347032644231, 'upper': 0.04427915888108494} | {'lower': 0.03475439526459938, 'upper': 0.04489879848043117} | mean_within_reference |
| serial_position_effect | 0.0760112863327149 {'lower': 0.0745537917439703, 'upper': 0.07743986162646878} | 0.07196474953617811 {'lower': 0.07053566790352504, 'upper': 0.07338950216450217} | {'lower': 0.05373840445269019, 'upper': 0.11070037105751393} | mean_within_reference |
| stable_within_subject_errors | 0.9422096611084031 {'lower': 0.9394755030126184, 'upper': 0.9449004882878894} | 0.9289305180480053 {'lower': 0.9257061940138394, 'upper': 0.9319831890474534} | {'lower': 0.8405797101449275, 'upper': 0.9714285714285714} | mean_within_reference |
| self_consistent_incorrect | 0.9511960925039871 {'lower': 0.9486383334757352, 'upper': 0.9536141139781273} | 0.9339395748277326 {'lower': 0.9309325551093645, 'upper': 0.9368006266049033} | {'lower': 0.7402597402597403, 'upper': 0.9090909090909091} | sustained_above_reference |
| self_inconsistent | 0.017232604237867398 {'lower': 0.015829664217361577, 'upper': 0.018658151913875588} | 0.015145676691729327 {'lower': 0.01378880083732057, 'upper': 0.016546073991797673} | {'lower': 0.012987012987012988, 'upper': 0.12987012987012986} | mean_within_reference |
| correct_ranker | 0.03157130325814537 {'lower': 0.029564286426292993, 'upper': 0.03364535628844838} | 0.05091474848053796 {'lower': 0.04837228985008589, 'upper': 0.053568451418780354} | {'lower': 0.03896103896103896, 'upper': 0.16883116883116883} | mean_within_reference |
| inter_subject_ranking_diversity | 0.5631591073898884 {'lower': 0.5597178569737296, 'upper': 0.5666235989128515} | 0.5707867201414466 {'lower': 0.5673864035384555, 'upper': 0.5741730854457033} | {'lower': 0.5071839605407381, 'upper': 0.6124101814638899} | mean_within_reference |

Adaptive all-nine qualitative stability: {'successes': 393, 'cohorts': 400, 'rate': 0.9825, 'lower': 0.964322927074442, 'upper': 0.991497708892962}; quantitative stability: {'successes': 36, 'cohorts': 400, 'rate': 0.09, 'lower': 0.06571729376939447, 'upper': 0.12208278758472865}; joint: {'successes': 36, 'cohorts': 400, 'rate': 0.09, 'lower': 0.06571729376939447, 'upper': 0.12208278758472865}.
| 2119 | 0.5635948777198792 | 0.9796604514122009 | 5.946534156799316 | 6.186568737030029 | {'mean': 0.01905844155844156, 'interval': {'lower': 0.016233766233766232, 'upper': 0.02188311688311688}} | {'mean': -0.1590909090909091, 'interval': {'lower': -0.1718847402597403, 'upper': -0.14603814935064938}} | {'mean': -0.016766514258619525, 'interval': {'lower': -0.01998675666438824, 'upper': -0.013477394830750094}} | True | True |

## Seed 2119: continuous profile

| Endpoint | Fixed mean (95% CI) | Adaptive mean (95% CI) | Human reference | Adaptive classification |
| --- | --- | --- | --- | --- |
| learned_accuracy | 0.9006351461038961 {'lower': 0.8994253246753249, 'upper': 0.9018344155844156} | 0.9097958603896104 {'lower': 0.908664752435065, 'upper': 0.9109188413149351} | {'lower': 0.8931818181818183, 'upper': 0.933766233766234} | mean_within_reference |
| nonlearned_accuracy | 0.8328258116883117 {'lower': 0.8314028652597403, 'upper': 0.8342560470779219} | 0.8369047077922079 {'lower': 0.8354616761363637, 'upper': 0.8383279707792207} | {'lower': 0.8033100649350651, 'upper': 0.8501964285714286} | mean_within_reference |
| symbolic_distance_effect | 0.04556136584503936 {'lower': 0.04528963104072801, 'upper': 0.0458347604967312} | 0.0440148301528404 {'lower': 0.04373845326442269, 'upper': 0.044290590075315876} | {'lower': 0.03475439526459938, 'upper': 0.04489879848043117} | mean_within_reference |
| serial_position_effect | 0.07606679035250465 {'lower': 0.07461779529993816, 'upper': 0.0775337082560297} | 0.07197773654916512 {'lower': 0.07055178184910325, 'upper': 0.07341265074211503} | {'lower': 0.05373840445269019, 'upper': 0.11070037105751393} | mean_within_reference |
| stable_within_subject_errors | 0.9416583803315968 {'lower': 0.9389017527888098, 'upper': 0.9443399334946334} | 0.9286957965257548 {'lower': 0.9256198753751851, 'upper': 0.9318303811049353} | {'lower': 0.8405797101449275, 'upper': 0.9714285714285714} | mean_within_reference |
| self_consistent_incorrect | 0.950967110959216 {'lower': 0.9483465402141723, 'upper': 0.9534834758771933} | 0.9340023738707949 {'lower': 0.931122105979014, 'upper': 0.9369099508022207} | {'lower': 0.7402597402597403, 'upper': 0.9090909090909091} | sustained_above_reference |
| self_inconsistent | 0.01716852358168148 {'lower': 0.01573308270676691, 'upper': 0.018636433270676683} | 0.015147385509227616 {'lower': 0.01381535158920027, 'upper': 0.01652051649008885} | {'lower': 0.012987012987012988, 'upper': 0.12987012987012986} | mean_within_reference |
| correct_ranker | 0.0318643654591023 {'lower': 0.029718902369560252, 'upper': 0.03400888699020276} | 0.05085024061997746 {'lower': 0.048252512134913446, 'upper': 0.05346547842271524} | {'lower': 0.03896103896103896, 'upper': 0.16883116883116883} | mean_within_reference |
| inter_subject_ranking_diversity | 0.5632807526601669 {'lower': 0.5598132258144541, 'upper': 0.5667032271715295} | 0.5708403618816368 {'lower': 0.5674350240368844, 'upper': 0.5741682999853791} | {'lower': 0.5071839605407381, 'upper': 0.6124101814638899} | mean_within_reference |

Adaptive all-nine qualitative stability: {'successes': 393, 'cohorts': 400, 'rate': 0.9825, 'lower': 0.964322927074442, 'upper': 0.991497708892962}; quantitative stability: {'successes': 35, 'cohorts': 400, 'rate': 0.0875, 'lower': 0.06358691021150324, 'upper': 0.11926073261429136}; joint: {'successes': 35, 'cohorts': 400, 'rate': 0.0875, 'lower': 0.06358691021150324, 'upper': 0.11926073261429136}.

## Mechanism and claim boundary

Generic late-sensitivity and scheduler decision: `True`; relation-specific support: `False`. Exact occurrence sensitivities, conditional mean margins and independently propagated quantization variances are retained in the generic-selection arrays.

The ranking-composition gate is a paired relative improvement in total variation, not equality of the complete human distribution. The unchanged nine-row profile remains fully reported. Internal strict order has no observed human latent-state counterpart.

This fixed pilot cannot promote a main model. Fresh unchanged training replication and a finite-time circuit check of the selected efficacy rule remain outside this authorization.

After this complete fixed pilot, do not tune the decay curve, codebook, temperature, loss, local branch, training length, checkpoints or seeds. A successor requires a separately frozen question and explicit authorization.
