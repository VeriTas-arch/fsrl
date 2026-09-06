# Fresh global-decay replication

Registered outcome: `replicated_global_decay`.

All six paired 2120--2122 fits and all 400 prospective 77-subject cohorts per fit are included. Participants are analyzed within each fit; no network or participant pooling is used.

| Seed | eta fixed | eta0 adaptive | gamma fixed | gamma adaptive | Internal correct delta | Inversion delta | Rank-TV delta | Adaptive core | Preservation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2120 | 0.5653976202011108 | 0.9798508286476135 | 5.986732482910156 | 6.222573757171631 | {'mean': 0.019448051948051947, 'interval': {'lower': 0.016915584415584415, 'upper': 0.02201298701298701}} | {'mean': -0.1589935064935065, 'interval': {'lower': -0.17250081168831166, 'upper': -0.1457792207792208}} | {'mean': -0.016464883800410123, 'interval': {'lower': -0.01945289345522899, 'upper': -0.013458101930963778}} | True | True |

## Seed 2120: continuous profile

| Endpoint | Fixed mean (95% CI) | Adaptive mean (95% CI) | Human reference | Adaptive classification |
| --- | --- | --- | --- | --- |
| learned_accuracy | 0.9011318993506494 {'lower': 0.8998436688311686, 'upper': 0.9023994216720778} | 0.9100365259740261 {'lower': 0.9087755580357142, 'upper': 0.9112820718344156} | {'lower': 0.8931818181818183, 'upper': 0.933766233766234} | mean_within_reference |
| nonlearned_accuracy | 0.8331056818181818 {'lower': 0.8316171996753245, 'upper': 0.8345724107142858} | 0.8376402597402598 {'lower': 0.8361641233766235, 'upper': 0.839127463474026} | {'lower': 0.8033100649350651, 'upper': 0.8501964285714286} | mean_within_reference |
| symbolic_distance_effect | 0.045544449598021064 {'lower': 0.04523937353675241, 'upper': 0.04583985500044177} | 0.04393076022616843 {'lower': 0.04362357184269816, 'upper': 0.044227057933563074} | {'lower': 0.03475439526459938, 'upper': 0.04489879848043117} | mean_within_reference |
| serial_position_effect | 0.07558750773036488 {'lower': 0.07414300788497216, 'upper': 0.07701253478664194} | 0.07135281385281386 {'lower': 0.06994418676561535, 'upper': 0.0727447704081633} | {'lower': 0.05373840445269019, 'upper': 0.11070037105751393} | mean_within_reference |
| stable_within_subject_errors | 0.9424605078034883 {'lower': 0.9397734894645193, 'upper': 0.9451057536994756} | 0.9283532259744289 {'lower': 0.9252337473360936, 'upper': 0.931364618628664} | {'lower': 0.8405797101449275, 'upper': 0.9714285714285714} | mean_within_reference |
| self_consistent_incorrect | 0.9529310036454771 {'lower': 0.9503177477785375, 'upper': 0.9554434538049672} | 0.9360688197767143 {'lower': 0.9332462040043291, 'upper': 0.9388905029619505} | {'lower': 0.7402597402597403, 'upper': 0.9090909090909091} | sustained_above_reference |
| self_inconsistent | 0.016613157894736844 {'lower': 0.015184531784005468, 'upper': 0.018108384740259737} | 0.015533162451583506 {'lower': 0.014165680963773068, 'upper': 0.016932288961038957} | {'lower': 0.012987012987012988, 'upper': 0.12987012987012986} | mean_within_reference |
| correct_ranker | 0.03045583845978583 {'lower': 0.028395346462747766, 'upper': 0.032567114519252656} | 0.04839801777170199 {'lower': 0.04590227828092958, 'upper': 0.05097321684894051} | {'lower': 0.03896103896103896, 'upper': 0.16883116883116883} | mean_within_reference |
| inter_subject_ranking_diversity | 0.5637045391160784 {'lower': 0.5601683020131316, 'upper': 0.5671866036941259} | 0.5724226566052046 {'lower': 0.5689322413401795, 'upper': 0.5758797203537036} | {'lower': 0.5071839605407381, 'upper': 0.6124101814638899} | mean_within_reference |

Adaptive all-nine qualitative stability: {'successes': 392, 'cohorts': 400, 'rate': 0.98, 'lower': 0.96103658423074, 'upper': 0.9898316132083353}; quantitative stability: {'successes': 24, 'cohorts': 400, 'rate': 0.06, 'lower': 0.040647970113724695, 'upper': 0.0877228489004562}; joint: {'successes': 24, 'cohorts': 400, 'rate': 0.06, 'lower': 0.040647970113724695, 'upper': 0.0877228489004562}.
| 2121 | 0.5360676646232605 | 0.9804381132125854 | 5.966856002807617 | 6.2237467765808105 | {'mean': 0.018441558441558446, 'interval': {'lower': 0.015909090909090907, 'upper': 0.020941558441558438}} | {'mean': -0.14935064935064934, 'interval': {'lower': -0.16233766233766234, 'upper': -0.1362337662337662}} | {'mean': -0.015067452722715884, 'interval': {'lower': -0.01795617324561404, 'upper': -0.012252194406470726}} | True | True |

## Seed 2121: continuous profile

| Endpoint | Fixed mean (95% CI) | Adaptive mean (95% CI) | Human reference | Adaptive classification |
| --- | --- | --- | --- | --- |
| learned_accuracy | 0.9014058441558441 {'lower': 0.9001622970779218, 'upper': 0.9027017248376622} | 0.9100507305194805 {'lower': 0.9088161221590909, 'upper': 0.9113023640422078} | {'lower': 0.8931818181818183, 'upper': 0.933766233766234} | mean_within_reference |
| nonlearned_accuracy | 0.8331511363636364 {'lower': 0.8316373295454544, 'upper': 0.8346633157467533} | 0.8376449675324676 {'lower': 0.8361379788961039, 'upper': 0.8391401055194806} | {'lower': 0.8033100649350651, 'upper': 0.8501964285714286} | mean_within_reference |
| symbolic_distance_effect | 0.045496914480077776 {'lower': 0.045196694468371795, 'upper': 0.04580403080550406} | 0.04392880775686903 {'lower': 0.04362879533306833, 'upper': 0.044229756024162944} | {'lower': 0.03475439526459938, 'upper': 0.04489879848043117} | mean_within_reference |
| serial_position_effect | 0.07590924551638838 {'lower': 0.07446160714285714, 'upper': 0.07732563002473718} | 0.07134075448361163 {'lower': 0.06991141001855289, 'upper': 0.07275061842918987} | {'lower': 0.05373840445269019, 'upper': 0.11070037105751393} | mean_within_reference |
| stable_within_subject_errors | 0.9402521334280166 {'lower': 0.9375532602026371, 'upper': 0.9429236457104646} | 0.9282123771955122 {'lower': 0.925119476486951, 'upper': 0.9312070586660641} | {'lower': 0.8405797101449275, 'upper': 0.9714285714285714} | mean_within_reference |
| self_consistent_incorrect | 0.951598507632718 {'lower': 0.9490019718329918, 'upper': 0.9540701339997726} | 0.9361662223741171 {'lower': 0.9333564778423331, 'upper': 0.9388615419229893} | {'lower': 0.7402597402597403, 'upper': 0.9090909090909091} | sustained_above_reference |
| self_inconsistent | 0.016774663932558668 {'lower': 0.015345283663704712, 'upper': 0.01820971904192298} | 0.015468227386648442 {'lower': 0.014105434751651857, 'upper': 0.016864747379813166} | {'lower': 0.012987012987012988, 'upper': 0.12987012987012986} | mean_within_reference |
| correct_ranker | 0.03162682843472318 {'lower': 0.029544612383230792, 'upper': 0.03377273966165412} | 0.04836555023923446 {'lower': 0.045950521189336964, 'upper': 0.05089715054682158} | {'lower': 0.03896103896103896, 'upper': 0.16883116883116883} | mean_within_reference |
| inter_subject_ranking_diversity | 0.5649170932202624 {'lower': 0.5613908209603077, 'upper': 0.568441733423817} | 0.5724287536772293 {'lower': 0.5689536101278394, 'upper': 0.5759537180779476} | {'lower': 0.5071839605407381, 'upper': 0.6124101814638899} | mean_within_reference |

Adaptive all-nine qualitative stability: {'successes': 392, 'cohorts': 400, 'rate': 0.98, 'lower': 0.96103658423074, 'upper': 0.9898316132083353}; quantitative stability: {'successes': 23, 'cohorts': 400, 'rate': 0.0575, 'lower': 0.038617542757733075, 'upper': 0.08480083772811935}; joint: {'successes': 23, 'cohorts': 400, 'rate': 0.0575, 'lower': 0.038617542757733075, 'upper': 0.08480083772811935}.
| 2122 | 0.5410366058349609 | 0.9800493717193604 | 5.974481582641602 | 6.20604944229126 | {'mean': 0.018246753246753247, 'interval': {'lower': 0.015714285714285715, 'upper': 0.02074675324675324}} | {'mean': -0.15038961038961038, 'interval': {'lower': -0.16344155844155847, 'upper': -0.1372727272727273}} | {'mean': -0.015067891319207112, 'interval': {'lower': -0.017993433299156987, 'upper': -0.01211929482797904}} | True | True |

## Seed 2122: continuous profile

| Endpoint | Fixed mean (95% CI) | Adaptive mean (95% CI) | Human reference | Adaptive classification |
| --- | --- | --- | --- | --- |
| learned_accuracy | 0.9013883928571428 {'lower': 0.9000997564935065, 'upper': 0.9026522017045452} | 0.9100137987012986 {'lower': 0.9087450588474024, 'upper': 0.9112536525974028} | {'lower': 0.8931818181818183, 'upper': 0.933766233766234} | mean_within_reference |
| nonlearned_accuracy | 0.8331568181818182 {'lower': 0.8316714204545455, 'upper': 0.8346287702922078} | 0.8376006493506494 {'lower': 0.8361210633116883, 'upper': 0.8390613798701299} | {'lower': 0.8033100649350651, 'upper': 0.8501964285714286} | mean_within_reference |
| symbolic_distance_effect | 0.04550048701298705 {'lower': 0.04520353849169541, 'upper': 0.04580315603189331} | 0.04393788872691938 {'lower': 0.043642416843360766, 'upper': 0.044247878373752145} | {'lower': 0.03475439526459938, 'upper': 0.04489879848043117} | mean_within_reference |
| serial_position_effect | 0.07583039579468151 {'lower': 0.07439046846011131, 'upper': 0.07726793831168832} | 0.07136301793444652 {'lower': 0.06992624458874461, 'upper': 0.0727889146567718} | {'lower': 0.05373840445269019, 'upper': 0.11070037105751393} | mean_within_reference |
| stable_within_subject_errors | 0.9410596522951409 {'lower': 0.9383760086453287, 'upper': 0.9436392300474942} | 0.9281759196074826 {'lower': 0.9250888965893344, 'upper': 0.9312504337397411} | {'lower': 0.8405797101449275, 'upper': 0.9714285714285714} | mean_within_reference |
| self_consistent_incorrect | 0.9515015436318066 {'lower': 0.9488354282011852, 'upper': 0.9541609864149011} | 0.936036352244247 {'lower': 0.9332501146331739, 'upper': 0.9388127090453406} | {'lower': 0.7402597402597403, 'upper': 0.9090909090909091} | sustained_above_reference |
| self_inconsistent | 0.01677422533606744 {'lower': 0.015320460668717246, 'upper': 0.01824306390977443} | 0.01556562998405104 {'lower': 0.014171651144907722, 'upper': 0.016975319406470714} | {'lower': 0.012987012987012988, 'upper': 0.12987012987012986} | mean_within_reference |
| correct_ranker | 0.03172423103212577 {'lower': 0.029517271445659587, 'upper': 0.03398256080542263} | 0.04839801777170199 {'lower': 0.04593818737183867, 'upper': 0.05096046479835953} | {'lower': 0.03896103896103896, 'upper': 0.16883116883116883} | mean_within_reference |
| inter_subject_ranking_diversity | 0.56454990946996 {'lower': 0.561048294402752, 'upper': 0.5679700932246518} | 0.5724065851310595 {'lower': 0.5689375569365305, 'upper': 0.575820799982944} | {'lower': 0.5071839605407381, 'upper': 0.6124101814638899} | mean_within_reference |

Adaptive all-nine qualitative stability: {'successes': 392, 'cohorts': 400, 'rate': 0.98, 'lower': 0.96103658423074, 'upper': 0.9898316132083353}; quantitative stability: {'successes': 26, 'cohorts': 400, 'rate': 0.065, 'lower': 0.04474048189620036, 'upper': 0.09353521417463759}; joint: {'successes': 26, 'cohorts': 400, 'rate': 0.065, 'lower': 0.04474048189620036, 'upper': 0.09353521417463759}.

## Mechanism and claim boundary

Generic competence and late-sensitivity confirmation: `True`. Exact occurrence sensitivities and mean/variance decomposition are retained in the locked generic arrays.

The primary claim is unchanged replication of a global blockwise diminishing-plasticity effect. Complete nine-row quantitative equality remains a mandatory report but is not a primary replication gate.

This result neither reopens relation familiarity nor promotes a main model. Common encoding state and finite-time circuit realization remain separate prospective questions.

After the fixed replication, do not tune or rerun the candidate and do not use these fresh results to choose a common-state encoder.
