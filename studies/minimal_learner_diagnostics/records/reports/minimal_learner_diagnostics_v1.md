# Frozen minimal learner: quantitative-source diagnosis

Three mandatory exposed training streams; no training, parameter selection, human fitting, or additional seeds. Registered outcome is diagnostic localization, not a main-model promotion or a quantitative-equivalence test.

RF/AF = retained/all-observed admission with the original finite update. RL/AL = retained/all-observed minimum-norm least-squares references. All cells use each score-only fit's unchanged gain and T=0.25. L cells use observed support constraints, never query labels; they are offline references, not candidate models or human posteriors.

Every estimate is a separate within-fit participant bootstrap (10,000 draws, 95% percentile). Missing group subjects are removed before resampling; JSON contains their exact indices. The direction labels require each of the three intervals to lie on the same side of zero; they are descriptive and not familywise confirmation. All registered contrasts are reported below.

Protocol witness: `d8bdc3af04cc31f3ef436d751ab823caa38dd593`. Implementation witness: `2e976484b3c31bdfcd39ff28e1b8ec36fabc0d6c`. Execution-lock witness: `2edeedf2017a4b9cfadda3016077285a14cc4d53`.
Independent validation passed: `True`.

## Cross-stream descriptive directions

| Domain | Contrast | Endpoint | Direction |
| --- | --- | --- | --- |
| between_recipe | acute_local | learned | consistently_positive |
| between_recipe | acute_local | nonlearned | consistently_negative |
| between_recipe | acute_local | omitted | consistently_positive |
| between_recipe | acute_local | overall | consistently_negative |
| between_recipe | acute_local | retained | consistently_positive |
| between_recipe | global_fit_difference | learned | consistently_negative |
| between_recipe | global_fit_difference | nonlearned | consistently_negative |
| between_recipe | global_fit_difference | omitted | mixed_or_uncertain |
| between_recipe | global_fit_difference | overall | consistently_negative |
| between_recipe | global_fit_difference | retained | consistently_negative |
| between_recipe | total_recipe_difference | learned | consistently_positive |
| between_recipe | total_recipe_difference | nonlearned | consistently_negative |
| between_recipe | total_recipe_difference | omitted | consistently_positive |
| between_recipe | total_recipe_difference | overall | consistently_negative |
| between_recipe | total_recipe_difference | retained | consistently_positive |
| global | admission_at_finite | exact/distance_slope | consistently_negative |
| global | admission_at_finite | exact/learned | consistently_positive |
| global | admission_at_finite | exact/nonlearned | consistently_positive |
| global | admission_at_finite | exact/omitted | consistently_positive |
| global | admission_at_finite | exact/overall | consistently_positive |
| global | admission_at_finite | exact/retained | mixed_or_uncertain |
| global | admission_at_finite | exact/serial_contrast | consistently_negative |
| global | admission_at_finite | latent/has_tied_pair | mixed_or_uncertain |
| global | admission_at_finite | latent/pair_discordance | consistently_negative |
| global | admission_at_finite | latent/strict_correct_order | consistently_positive |
| global | admission_at_finite | probability/distance_slope | consistently_negative |
| global | admission_at_finite | probability/learned | consistently_positive |
| global | admission_at_finite | probability/nonlearned | consistently_positive |
| global | admission_at_finite | probability/omitted | consistently_positive |
| global | admission_at_finite | probability/overall | consistently_positive |
| global | admission_at_finite | probability/retained | consistently_positive |
| global | admission_at_finite | probability/serial_contrast | consistently_negative |
| global | admission_at_finite | support/all_rmse | consistently_negative |
| global | admission_at_finite | support/retained_rmse | mixed_or_uncertain |
| global | admission_at_least_squares | exact/distance_slope | consistently_negative |
| global | admission_at_least_squares | exact/learned | consistently_positive |
| global | admission_at_least_squares | exact/nonlearned | consistently_positive |
| global | admission_at_least_squares | exact/omitted | consistently_positive |
| global | admission_at_least_squares | exact/overall | consistently_positive |
| global | admission_at_least_squares | exact/retained | mixed_or_uncertain |
| global | admission_at_least_squares | exact/serial_contrast | consistently_negative |
| global | admission_at_least_squares | latent/has_tied_pair | mixed_or_uncertain |
| global | admission_at_least_squares | latent/pair_discordance | consistently_negative |
| global | admission_at_least_squares | latent/strict_correct_order | consistently_positive |
| global | admission_at_least_squares | probability/distance_slope | consistently_negative |
| global | admission_at_least_squares | probability/learned | consistently_positive |
| global | admission_at_least_squares | probability/nonlearned | consistently_positive |
| global | admission_at_least_squares | probability/omitted | consistently_positive |
| global | admission_at_least_squares | probability/overall | consistently_positive |
| global | admission_at_least_squares | probability/retained | mixed_or_uncertain |
| global | admission_at_least_squares | probability/serial_contrast | consistently_negative |
| global | admission_at_least_squares | support/all_rmse | consistently_negative |
| global | admission_at_least_squares | support/retained_rmse | mixed_or_uncertain |
| global | integration_at_all | exact/distance_slope | consistently_negative |
| global | integration_at_all | exact/learned | mixed_or_uncertain |
| global | integration_at_all | exact/nonlearned | consistently_positive |
| global | integration_at_all | exact/omitted | mixed_or_uncertain |
| global | integration_at_all | exact/overall | consistently_positive |
| global | integration_at_all | exact/retained | mixed_or_uncertain |
| global | integration_at_all | exact/serial_contrast | consistently_negative |
| global | integration_at_all | latent/has_tied_pair | mixed_or_uncertain |
| global | integration_at_all | latent/pair_discordance | consistently_negative |
| global | integration_at_all | latent/strict_correct_order | consistently_positive |
| global | integration_at_all | probability/distance_slope | consistently_negative |
| global | integration_at_all | probability/learned | consistently_positive |
| global | integration_at_all | probability/nonlearned | consistently_positive |
| global | integration_at_all | probability/omitted | mixed_or_uncertain |
| global | integration_at_all | probability/overall | consistently_positive |
| global | integration_at_all | probability/retained | consistently_positive |
| global | integration_at_all | probability/serial_contrast | consistently_negative |
| global | integration_at_all | support/all_rmse | consistently_negative |
| global | integration_at_all | support/retained_rmse | consistently_negative |
| global | integration_at_retained | exact/distance_slope | consistently_negative |
| global | integration_at_retained | exact/learned | consistently_positive |
| global | integration_at_retained | exact/nonlearned | consistently_positive |
| global | integration_at_retained | exact/omitted | consistently_positive |
| global | integration_at_retained | exact/overall | consistently_positive |
| global | integration_at_retained | exact/retained | mixed_or_uncertain |
| global | integration_at_retained | exact/serial_contrast | consistently_negative |
| global | integration_at_retained | latent/has_tied_pair | mixed_or_uncertain |
| global | integration_at_retained | latent/pair_discordance | consistently_negative |
| global | integration_at_retained | latent/strict_correct_order | consistently_positive |
| global | integration_at_retained | probability/distance_slope | consistently_negative |
| global | integration_at_retained | probability/learned | consistently_positive |
| global | integration_at_retained | probability/nonlearned | consistently_positive |
| global | integration_at_retained | probability/omitted | consistently_positive |
| global | integration_at_retained | probability/overall | consistently_positive |
| global | integration_at_retained | probability/retained | consistently_positive |
| global | integration_at_retained | probability/serial_contrast | consistently_negative |
| global | integration_at_retained | support/all_rmse | consistently_negative |
| global | integration_at_retained | support/retained_rmse | consistently_negative |
| global | interaction | exact/distance_slope | mixed_or_uncertain |
| global | interaction | exact/learned | consistently_negative |
| global | interaction | exact/nonlearned | mixed_or_uncertain |
| global | interaction | exact/omitted | consistently_negative |
| global | interaction | exact/overall | mixed_or_uncertain |
| global | interaction | exact/retained | mixed_or_uncertain |
| global | interaction | exact/serial_contrast | mixed_or_uncertain |
| global | interaction | latent/has_tied_pair | mixed_or_uncertain |
| global | interaction | latent/pair_discordance | mixed_or_uncertain |
| global | interaction | latent/strict_correct_order | mixed_or_uncertain |
| global | interaction | probability/distance_slope | mixed_or_uncertain |
| global | interaction | probability/learned | consistently_negative |
| global | interaction | probability/nonlearned | mixed_or_uncertain |
| global | interaction | probability/omitted | consistently_negative |
| global | interaction | probability/overall | mixed_or_uncertain |
| global | interaction | probability/retained | consistently_negative |
| global | interaction | probability/serial_contrast | mixed_or_uncertain |
| global | interaction | support/all_rmse | consistently_negative |
| global | interaction | support/retained_rmse | mixed_or_uncertain |
| global | total | exact/distance_slope | consistently_negative |
| global | total | exact/learned | consistently_positive |
| global | total | exact/nonlearned | consistently_positive |
| global | total | exact/omitted | consistently_positive |
| global | total | exact/overall | consistently_positive |
| global | total | exact/retained | mixed_or_uncertain |
| global | total | exact/serial_contrast | consistently_negative |
| global | total | latent/has_tied_pair | mixed_or_uncertain |
| global | total | latent/pair_discordance | consistently_negative |
| global | total | latent/strict_correct_order | consistently_positive |
| global | total | probability/distance_slope | consistently_negative |
| global | total | probability/learned | consistently_positive |
| global | total | probability/nonlearned | consistently_positive |
| global | total | probability/omitted | consistently_positive |
| global | total | probability/overall | consistently_positive |
| global | total | probability/retained | consistently_positive |
| global | total | probability/serial_contrast | consistently_negative |
| global | total | support/all_rmse | consistently_negative |
| global | total | support/retained_rmse | consistently_negative |
| local | cross | learned | mixed_or_uncertain |
| local | cross | nonlearned | consistently_negative |
| local | cross | omitted | mixed_or_uncertain |
| local | cross | overall | consistently_negative |
| local | cross | retained | mixed_or_uncertain |
| local | full_minus_self_only | learned | mixed_or_uncertain |
| local | full_minus_self_only | nonlearned | consistently_negative |
| local | full_minus_self_only | omitted | mixed_or_uncertain |
| local | full_minus_self_only | overall | consistently_negative |
| local | full_minus_self_only | retained | mixed_or_uncertain |
| local | self | learned | consistently_positive |
| local | self | nonlearned | mixed_or_uncertain |
| local | self | omitted | consistently_positive |
| local | self | overall | consistently_positive |
| local | self | retained | consistently_positive |
| local | total | learned | consistently_positive |
| local | total | nonlearned | consistently_negative |
| local | total | omitted | consistently_positive |
| local | total | overall | consistently_negative |
| local | total | retained | consistently_positive |

## Training stream 2111

Frozen parameters: `{'score_only': {'eta': 0.9884949326515198, 'gamma_G': 7.179845333099365, 'gamma_L': 0.0}, 'score_trace': {'eta': 0.9886998534202576, 'gamma_G': 7.14223575592041, 'gamma_L': 0.24688567221164703}}`.

### Global reference: AF

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.012851 | 0.008582 | 0.017346 | 77 |
| exact/learned | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/nonlearned | 0.979221 | 0.971429 | 0.986364 | 77 |
| exact/omitted | 1.000000 | 1.000000 | 1.000000 | 69 |
| exact/overall | 0.985158 | 0.979592 | 0.990260 | 77 |
| exact/retained | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/serial_contrast | 0.015584 | 0.005195 | 0.026840 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.014842 | 0.009740 | 0.020408 | 77 |
| latent/strict_correct_order | 0.675325 | 0.571429 | 0.779221 | 77 |
| probability/distance_slope | 0.019121 | 0.015573 | 0.022795 | 77 |
| probability/learned | 0.996840 | 0.995746 | 0.997729 | 77 |
| probability/nonlearned | 0.968630 | 0.962347 | 0.974644 | 77 |
| probability/omitted | 0.997041 | 0.994901 | 0.998835 | 69 |
| probability/overall | 0.976690 | 0.972157 | 0.981023 | 77 |
| probability/retained | 0.996612 | 0.995229 | 0.997720 | 77 |
| probability/serial_contrast | 0.021603 | 0.013240 | 0.030318 | 77 |
| support/all_rmse | 0.049605 | 0.043683 | 0.055602 | 77 |
| support/retained_rmse | 0.048902 | 0.042484 | 0.055292 | 77 |

### Global reference: AL

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/learned | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/nonlearned | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/omitted | 1.000000 | 1.000000 | 1.000000 | 69 |
| exact/overall | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/retained | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/serial_contrast | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/strict_correct_order | 1.000000 | 1.000000 | 1.000000 | 77 |
| probability/distance_slope | 0.003089 | 0.003089 | 0.003089 | 77 |
| probability/learned | 0.997932 | 0.997932 | 0.997932 | 77 |
| probability/nonlearned | 0.995054 | 0.995054 | 0.995054 | 77 |
| probability/omitted | 0.998115 | 0.997119 | 0.998998 | 69 |
| probability/overall | 0.995876 | 0.995876 | 0.995876 | 77 |
| probability/retained | 0.997771 | 0.997509 | 0.998037 | 77 |
| probability/serial_contrast | 0.002186 | 0.002186 | 0.002186 | 77 |
| support/all_rmse | 0.000000 | 0.000000 | 0.000000 | 77 |
| support/retained_rmse | 0.000000 | 0.000000 | 0.000000 | 77 |

### Global reference: RF

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.038144 | 0.029834 | 0.046635 | 77 |
| exact/learned | 0.949675 | 0.930195 | 0.967532 | 77 |
| exact/nonlearned | 0.892857 | 0.868182 | 0.915584 | 77 |
| exact/omitted | 0.814010 | 0.740308 | 0.880435 | 69 |
| exact/overall | 0.909091 | 0.887755 | 0.928571 | 77 |
| exact/retained | 0.997835 | 0.993506 | 1.000000 | 77 |
| exact/serial_contrast | 0.049351 | 0.025108 | 0.073593 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.090909 | 0.071429 | 0.112245 | 77 |
| latent/strict_correct_order | 0.324675 | 0.220779 | 0.428571 | 77 |
| probability/distance_slope | 0.044564 | 0.037364 | 0.051937 | 77 |
| probability/learned | 0.946592 | 0.928474 | 0.962741 | 77 |
| probability/nonlearned | 0.884009 | 0.860549 | 0.905850 | 77 |
| probability/omitted | 0.809535 | 0.740069 | 0.871563 | 69 |
| probability/overall | 0.901890 | 0.881704 | 0.920520 | 77 |
| probability/retained | 0.993639 | 0.989243 | 0.996842 | 77 |
| probability/serial_contrast | 0.050891 | 0.028785 | 0.072598 | 77 |
| support/all_rmse | 0.152532 | 0.127430 | 0.179186 | 77 |
| support/retained_rmse | 0.044792 | 0.037981 | 0.052237 | 77 |

### Global reference: RL

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.024839 | 0.016938 | 0.033421 | 77 |
| exact/learned | 0.961039 | 0.943182 | 0.975649 | 77 |
| exact/nonlearned | 0.920130 | 0.895455 | 0.943506 | 77 |
| exact/omitted | 0.861111 | 0.798309 | 0.916667 | 69 |
| exact/overall | 0.931818 | 0.910946 | 0.951299 | 77 |
| exact/retained | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/serial_contrast | 0.032035 | 0.009524 | 0.054545 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.068182 | 0.048701 | 0.089054 | 77 |
| latent/strict_correct_order | 0.545455 | 0.428571 | 0.662338 | 77 |
| probability/distance_slope | 0.030181 | 0.022767 | 0.037982 | 77 |
| probability/learned | 0.956379 | 0.939721 | 0.970907 | 77 |
| probability/nonlearned | 0.912463 | 0.888753 | 0.934866 | 77 |
| probability/omitted | 0.848238 | 0.787169 | 0.902261 | 69 |
| probability/overall | 0.925011 | 0.904527 | 0.943721 | 77 |
| probability/retained | 0.997771 | 0.997509 | 0.998037 | 77 |
| probability/serial_contrast | 0.035062 | 0.013344 | 0.056526 | 77 |
| support/all_rmse | 0.118203 | 0.088963 | 0.149316 | 77 |
| support/retained_rmse | 0.000000 | 0.000000 | 0.000000 | 77 |

### Global contrast: admission_at_finite

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.025293 | -0.034148 | -0.016847 | 77 |
| exact/learned | 0.050325 | 0.032468 | 0.069805 | 77 |
| exact/nonlearned | 0.086364 | 0.063636 | 0.110390 | 77 |
| exact/omitted | 0.185990 | 0.119565 | 0.259692 | 69 |
| exact/overall | 0.076067 | 0.057050 | 0.096939 | 77 |
| exact/retained | 0.002165 | 0.000000 | 0.006494 | 77 |
| exact/serial_contrast | -0.033766 | -0.057143 | -0.009524 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.076067 | -0.096939 | -0.057050 | 77 |
| latent/strict_correct_order | 0.350649 | 0.233766 | 0.467532 | 77 |
| probability/distance_slope | -0.025444 | -0.032764 | -0.018389 | 77 |
| probability/learned | 0.050247 | 0.034308 | 0.068115 | 77 |
| probability/nonlearned | 0.084621 | 0.064091 | 0.106397 | 77 |
| probability/omitted | 0.187507 | 0.126208 | 0.256308 | 69 |
| probability/overall | 0.074800 | 0.057157 | 0.094006 | 77 |
| probability/retained | 0.002973 | 0.000227 | 0.006928 | 77 |
| probability/serial_contrast | -0.029288 | -0.049525 | -0.008113 | 77 |
| support/all_rmse | -0.102926 | -0.127896 | -0.079893 | 77 |
| support/retained_rmse | 0.004110 | -0.002763 | 0.011084 | 77 |

### Global contrast: admission_at_least_squares

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.024839 | -0.033421 | -0.016938 | 77 |
| exact/learned | 0.038961 | 0.024351 | 0.056818 | 77 |
| exact/nonlearned | 0.079870 | 0.056494 | 0.104545 | 77 |
| exact/omitted | 0.138889 | 0.083333 | 0.201691 | 69 |
| exact/overall | 0.068182 | 0.048701 | 0.089054 | 77 |
| exact/retained | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/serial_contrast | -0.032035 | -0.054545 | -0.009524 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.068182 | -0.089054 | -0.048701 | 77 |
| latent/strict_correct_order | 0.454545 | 0.337662 | 0.571429 | 77 |
| probability/distance_slope | -0.027093 | -0.034893 | -0.019679 | 77 |
| probability/learned | 0.041553 | 0.027025 | 0.058210 | 77 |
| probability/nonlearned | 0.082591 | 0.060188 | 0.106301 | 77 |
| probability/omitted | 0.149877 | 0.095889 | 0.210773 | 69 |
| probability/overall | 0.070866 | 0.052155 | 0.091349 | 77 |
| probability/retained | 0.000000 | -0.000000 | 0.000000 | 77 |
| probability/serial_contrast | -0.032876 | -0.054340 | -0.011158 | 77 |
| support/all_rmse | -0.118203 | -0.149316 | -0.088963 | 77 |
| support/retained_rmse | 0.000000 | -0.000000 | 0.000000 | 77 |

### Global contrast: integration_at_all

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.012851 | -0.017346 | -0.008582 | 77 |
| exact/learned | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/nonlearned | 0.020779 | 0.013636 | 0.028571 | 77 |
| exact/omitted | 0.000000 | 0.000000 | 0.000000 | 69 |
| exact/overall | 0.014842 | 0.009740 | 0.020408 | 77 |
| exact/retained | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/serial_contrast | -0.015584 | -0.026840 | -0.005195 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.014842 | -0.020408 | -0.009740 | 77 |
| latent/strict_correct_order | 0.324675 | 0.220779 | 0.428571 | 77 |
| probability/distance_slope | -0.016032 | -0.019706 | -0.012484 | 77 |
| probability/learned | 0.001092 | 0.000203 | 0.002186 | 77 |
| probability/nonlearned | 0.026424 | 0.020409 | 0.032707 | 77 |
| probability/omitted | 0.001073 | -0.000216 | 0.002693 | 69 |
| probability/overall | 0.019186 | 0.014853 | 0.023719 | 77 |
| probability/retained | 0.001159 | 0.000102 | 0.002519 | 77 |
| probability/serial_contrast | -0.019417 | -0.028132 | -0.011054 | 77 |
| support/all_rmse | -0.049605 | -0.055602 | -0.043683 | 77 |
| support/retained_rmse | -0.048902 | -0.055292 | -0.042484 | 77 |

### Global contrast: integration_at_retained

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.013305 | -0.019662 | -0.007491 | 77 |
| exact/learned | 0.011364 | 0.004870 | 0.019481 | 77 |
| exact/nonlearned | 0.027273 | 0.012987 | 0.043506 | 77 |
| exact/omitted | 0.047101 | 0.014493 | 0.090580 | 69 |
| exact/overall | 0.022727 | 0.012059 | 0.035714 | 77 |
| exact/retained | 0.002165 | 0.000000 | 0.006494 | 77 |
| exact/serial_contrast | -0.017316 | -0.029437 | -0.005195 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.022727 | -0.035714 | -0.012059 | 77 |
| latent/strict_correct_order | 0.220779 | 0.129870 | 0.311688 | 77 |
| probability/distance_slope | -0.014383 | -0.019243 | -0.010158 | 77 |
| probability/learned | 0.009787 | 0.004291 | 0.016412 | 77 |
| probability/nonlearned | 0.028454 | 0.017444 | 0.042156 | 77 |
| probability/omitted | 0.038704 | 0.010650 | 0.076719 | 69 |
| probability/overall | 0.023120 | 0.014283 | 0.034121 | 77 |
| probability/retained | 0.004132 | 0.000976 | 0.008511 | 77 |
| probability/serial_contrast | -0.015829 | -0.025199 | -0.006933 | 77 |
| support/all_rmse | -0.034328 | -0.046382 | -0.023600 | 77 |
| support/retained_rmse | -0.044792 | -0.052237 | -0.037981 | 77 |

### Global contrast: interaction

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.000454 | -0.006221 | 0.007311 | 77 |
| exact/learned | -0.011364 | -0.019481 | -0.004870 | 77 |
| exact/nonlearned | -0.006494 | -0.022078 | 0.007792 | 77 |
| exact/omitted | -0.047101 | -0.090580 | -0.014493 | 69 |
| exact/overall | -0.007885 | -0.020408 | 0.003247 | 77 |
| exact/retained | -0.002165 | -0.006494 | 0.000000 | 77 |
| exact/serial_contrast | 0.001732 | -0.012121 | 0.015584 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.007885 | -0.003247 | 0.020408 | 77 |
| latent/strict_correct_order | 0.103896 | -0.025974 | 0.233766 | 77 |
| probability/distance_slope | -0.001649 | -0.006080 | 0.002966 | 77 |
| probability/learned | -0.008694 | -0.015152 | -0.003361 | 77 |
| probability/nonlearned | -0.002030 | -0.014276 | 0.008557 | 77 |
| probability/omitted | -0.037630 | -0.075648 | -0.009815 | 69 |
| probability/overall | -0.003934 | -0.013898 | 0.004382 | 77 |
| probability/retained | -0.002973 | -0.006928 | -0.000227 | 77 |
| probability/serial_contrast | -0.003588 | -0.012813 | 0.005749 | 77 |
| support/all_rmse | -0.015277 | -0.026911 | -0.003365 | 77 |
| support/retained_rmse | -0.004110 | -0.011084 | 0.002763 | 77 |

### Global contrast: total

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.038144 | -0.046635 | -0.029834 | 77 |
| exact/learned | 0.050325 | 0.032468 | 0.069805 | 77 |
| exact/nonlearned | 0.107143 | 0.084416 | 0.131818 | 77 |
| exact/omitted | 0.185990 | 0.119565 | 0.259692 | 69 |
| exact/overall | 0.090909 | 0.071429 | 0.112245 | 77 |
| exact/retained | 0.002165 | 0.000000 | 0.006494 | 77 |
| exact/serial_contrast | -0.049351 | -0.073593 | -0.025108 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.090909 | -0.112245 | -0.071429 | 77 |
| latent/strict_correct_order | 0.675325 | 0.571429 | 0.779221 | 77 |
| probability/distance_slope | -0.041476 | -0.048848 | -0.034275 | 77 |
| probability/learned | 0.051340 | 0.035191 | 0.069458 | 77 |
| probability/nonlearned | 0.111045 | 0.089203 | 0.134505 | 77 |
| probability/omitted | 0.188580 | 0.126886 | 0.257695 | 69 |
| probability/overall | 0.093986 | 0.075356 | 0.114172 | 77 |
| probability/retained | 0.004132 | 0.000976 | 0.008511 | 77 |
| probability/serial_contrast | -0.048705 | -0.070412 | -0.026599 | 77 |
| support/all_rmse | -0.152532 | -0.179186 | -0.127430 | 77 |
| support/retained_rmse | -0.044792 | -0.052237 | -0.037981 | 77 |

### Readout accounting: correct_shortfall

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.008669 | -0.011962 | -0.005795 | 77 |
| nonlearned | -0.019753 | -0.022406 | -0.017168 | 77 |
| omitted | -0.022683 | -0.037994 | -0.010268 | 69 |
| overall | -0.016586 | -0.018708 | -0.014572 | 77 |
| retained | -0.004687 | -0.007328 | -0.002832 | 77 |

### Readout accounting: tie_fraction

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.000000 | 0.000000 | 0.000000 | 77 |
| nonlearned | 0.000000 | 0.000000 | 0.000000 | 77 |
| omitted | 0.000000 | 0.000000 | 0.000000 | 69 |
| overall | 0.000000 | 0.000000 | 0.000000 | 77 |
| retained | 0.000000 | 0.000000 | 0.000000 | 77 |

### Readout accounting: ties

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.000000 | 0.000000 | 0.000000 | 77 |
| nonlearned | 0.000000 | 0.000000 | 0.000000 | 77 |
| omitted | 0.000000 | 0.000000 | 0.000000 | 69 |
| overall | 0.000000 | 0.000000 | 0.000000 | 77 |
| retained | 0.000000 | 0.000000 | 0.000000 | 77 |

### Readout accounting: total

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.003083 | -0.007398 | 0.001355 | 77 |
| nonlearned | -0.008848 | -0.012791 | -0.004849 | 77 |
| omitted | -0.004475 | -0.022038 | 0.012260 | 69 |
| overall | -0.007201 | -0.010441 | -0.003933 | 77 |
| retained | -0.004197 | -0.007057 | -0.001943 | 77 |

### Readout accounting: wrong_rescue

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.005586 | 0.002535 | 0.009220 | 77 |
| nonlearned | 0.010906 | 0.007964 | 0.014027 | 77 |
| omitted | 0.018208 | 0.008454 | 0.030390 | 69 |
| overall | 0.009386 | 0.007001 | 0.012011 | 77 |
| retained | 0.000491 | 0.000000 | 0.001472 | 77 |

### Retained graph coverage (fixed connected/disconnected strata)

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| connected/RF/latent/strict_correct_order | 0.592593 | 0.407407 | 0.777778 | 27 |
| connected/RF/probability/distance_slope | 0.027065 | 0.018500 | 0.036806 | 27 |
| connected/RF/probability/learned | 0.989570 | 0.978841 | 0.996385 | 27 |
| connected/RF/probability/nonlearned | 0.950393 | 0.926721 | 0.969129 | 27 |
| connected/RF/probability/omitted | 0.920917 | 0.799325 | 0.996165 | 19 |
| connected/RF/probability/overall | 0.961586 | 0.943392 | 0.976116 | 27 |
| connected/RF/probability/retained | 0.996099 | 0.994042 | 0.997836 | 27 |
| connected/RF/probability/serial_contrast | 0.032182 | 0.017578 | 0.047335 | 27 |
| connected/RL/latent/strict_correct_order | 1.000000 | 1.000000 | 1.000000 | 27 |
| connected/RL/probability/distance_slope | 0.003089 | 0.003089 | 0.003089 | 27 |
| connected/RL/probability/learned | 0.997932 | 0.997932 | 0.997932 | 27 |
| connected/RL/probability/nonlearned | 0.995054 | 0.995054 | 0.995054 | 27 |
| connected/RL/probability/omitted | 0.997431 | 0.994863 | 0.999997 | 19 |
| connected/RL/probability/overall | 0.995876 | 0.995876 | 0.995876 | 27 |
| connected/RL/probability/retained | 0.997982 | 0.997746 | 0.998283 | 27 |
| connected/RL/probability/serial_contrast | 0.002186 | 0.002186 | 0.002186 | 27 |
| disconnected/RF/latent/strict_correct_order | 0.180000 | 0.080000 | 0.300000 | 50 |
| disconnected/RF/probability/distance_slope | 0.054014 | 0.045247 | 0.062705 | 50 |
| disconnected/RF/probability/learned | 0.923385 | 0.898720 | 0.946515 | 50 |
| disconnected/RF/probability/nonlearned | 0.848162 | 0.819719 | 0.875480 | 50 |
| disconnected/RF/probability/omitted | 0.767209 | 0.688976 | 0.841703 | 50 |
| disconnected/RF/probability/overall | 0.869654 | 0.844798 | 0.893537 | 50 |
| disconnected/RF/probability/retained | 0.992310 | 0.985664 | 0.997034 | 50 |
| disconnected/RF/probability/serial_contrast | 0.060993 | 0.027583 | 0.093651 | 50 |
| disconnected/RL/latent/strict_correct_order | 0.300000 | 0.180000 | 0.440000 | 50 |
| disconnected/RL/probability/distance_slope | 0.044812 | 0.035761 | 0.054146 | 50 |
| disconnected/RL/probability/learned | 0.933941 | 0.911071 | 0.954874 | 50 |
| disconnected/RL/probability/nonlearned | 0.867864 | 0.837998 | 0.895600 | 50 |
| disconnected/RL/probability/omitted | 0.791545 | 0.718818 | 0.860755 | 50 |
| disconnected/RL/probability/overall | 0.886743 | 0.861255 | 0.910405 | 50 |
| disconnected/RL/probability/retained | 0.997657 | 0.997290 | 0.998048 | 50 |
| disconnected/RL/probability/serial_contrast | 0.052815 | 0.020066 | 0.084845 | 50 |

### Frozen probability minus exact decision

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| distance_slope | 0.006421 | 0.003952 | 0.008972 | 77 |
| learned | -0.003083 | -0.007398 | 0.001355 | 77 |
| nonlearned | -0.008848 | -0.012791 | -0.004849 | 77 |
| omitted | -0.004475 | -0.022038 | 0.012260 | 69 |
| overall | -0.007201 | -0.010441 | -0.003933 | 77 |
| retained | -0.004197 | -0.007057 | -0.001943 | 77 |
| serial_contrast | 0.001540 | -0.004535 | 0.007627 | 77 |

### Local cells: G

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.946544 | 0.928434 | 0.962685 | 77 |
| nonlearned | 0.883930 | 0.860477 | 0.905758 | 77 |
| omitted | 0.809506 | 0.740028 | 0.871525 | 69 |
| overall | 0.901819 | 0.881639 | 0.920446 | 77 |
| retained | 0.993583 | 0.989193 | 0.996787 | 77 |

### Local cells: GC

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.945067 | 0.926501 | 0.961619 | 77 |
| nonlearned | 0.873994 | 0.849104 | 0.897286 | 77 |
| omitted | 0.802884 | 0.730803 | 0.866562 | 69 |
| overall | 0.894301 | 0.872940 | 0.913917 | 77 |
| retained | 0.992937 | 0.988694 | 0.996210 | 77 |

### Local cells: GS

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.955248 | 0.938846 | 0.969809 | 77 |
| nonlearned | 0.883930 | 0.860477 | 0.905758 | 77 |
| omitted | 0.835868 | 0.770196 | 0.894289 | 69 |
| overall | 0.904306 | 0.884689 | 0.922443 | 77 |
| retained | 0.995682 | 0.992070 | 0.998163 | 77 |

### Local cells: GSC

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.953455 | 0.936614 | 0.968394 | 77 |
| nonlearned | 0.873994 | 0.849104 | 0.897286 | 77 |
| omitted | 0.827124 | 0.759544 | 0.886862 | 69 |
| overall | 0.896697 | 0.875854 | 0.915783 | 77 |
| retained | 0.995410 | 0.992153 | 0.997817 | 77 |

### Local effects: cross

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.001635 | -0.003782 | 0.000405 | 77 |
| nonlearned | -0.009935 | -0.013240 | -0.006739 | 77 |
| omitted | -0.007683 | -0.019417 | 0.001675 | 69 |
| overall | -0.007564 | -0.010013 | -0.005170 | 77 |
| retained | -0.000459 | -0.001498 | 0.000514 | 77 |

### Local effects: full_minus_self_only

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.001793 | -0.004021 | 0.000287 | 77 |
| nonlearned | -0.009935 | -0.013240 | -0.006739 | 77 |
| omitted | -0.008744 | -0.020755 | 0.001155 | 69 |
| overall | -0.007609 | -0.010067 | -0.005217 | 77 |
| retained | -0.000272 | -0.001162 | 0.000579 | 77 |

### Local effects: self

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.008546 | 0.005679 | 0.011989 | 77 |
| nonlearned | 0.000000 | 0.000000 | 0.000000 | 77 |
| omitted | 0.025301 | 0.015751 | 0.036546 | 69 |
| overall | 0.002442 | 0.001623 | 0.003425 | 77 |
| retained | 0.002286 | 0.001491 | 0.003253 | 77 |

### Local effects: total

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.006911 | 0.004168 | 0.010080 | 77 |
| nonlearned | -0.009935 | -0.013240 | -0.006739 | 77 |
| omitted | 0.017618 | 0.008210 | 0.027606 | 69 |
| overall | -0.005122 | -0.007594 | -0.002754 | 77 |
| retained | 0.001827 | 0.000668 | 0.003301 | 77 |

### Local between_recipe: acute_local

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.006911 | 0.004168 | 0.010080 | 77 |
| nonlearned | -0.009935 | -0.013240 | -0.006739 | 77 |
| omitted | 0.017618 | 0.008210 | 0.027606 | 69 |
| overall | -0.005122 | -0.007594 | -0.002754 | 77 |
| retained | 0.001827 | 0.000668 | 0.003301 | 77 |

### Local between_recipe: global_fit_difference

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.000049 | -0.000067 | -0.000029 | 77 |
| nonlearned | -0.000080 | -0.000096 | -0.000063 | 77 |
| omitted | -0.000028 | -0.000089 | 0.000035 | 69 |
| overall | -0.000071 | -0.000084 | -0.000057 | 77 |
| retained | -0.000056 | -0.000068 | -0.000043 | 77 |

### Local between_recipe: total_recipe_difference

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.006863 | 0.004114 | 0.010032 | 77 |
| nonlearned | -0.010015 | -0.013317 | -0.006823 | 77 |
| omitted | 0.017589 | 0.008161 | 0.027609 | 69 |
| overall | -0.005193 | -0.007661 | -0.002825 | 77 |
| retained | 0.001771 | 0.000612 | 0.003248 | 77 |

### Original behavior anchors (unchanged)

| Recipe | Qualitative rows | Frozen quantitative rows |
| --- | --- | --- |
| score_only | 9/9 | 3/9 |
| score_trace | 9/9 | 3/9 |

## Training stream 2112

Frozen parameters: `{'score_only': {'eta': 0.9887556433677673, 'gamma_G': 7.163000106811523, 'gamma_L': 0.0}, 'score_trace': {'eta': 0.9889477491378784, 'gamma_G': 7.123929977416992, 'gamma_L': 0.22932417690753937}}`.

### Global reference: AF

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.012442 | 0.008174 | 0.017119 | 77 |
| exact/learned | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/nonlearned | 0.979870 | 0.972078 | 0.987013 | 77 |
| exact/omitted | 1.000000 | 1.000000 | 1.000000 | 69 |
| exact/overall | 0.985622 | 0.980056 | 0.990724 | 77 |
| exact/retained | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/serial_contrast | 0.016450 | 0.006061 | 0.027706 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.014378 | 0.009276 | 0.019944 | 77 |
| latent/strict_correct_order | 0.688312 | 0.584416 | 0.792208 | 77 |
| probability/distance_slope | 0.019146 | 0.015655 | 0.022796 | 77 |
| probability/learned | 0.996818 | 0.995746 | 0.997729 | 77 |
| probability/nonlearned | 0.968586 | 0.962276 | 0.974426 | 77 |
| probability/omitted | 0.997020 | 0.994790 | 0.998817 | 69 |
| probability/overall | 0.976652 | 0.972124 | 0.980865 | 77 |
| probability/retained | 0.996589 | 0.995232 | 0.997699 | 77 |
| probability/serial_contrast | 0.021617 | 0.013330 | 0.030513 | 77 |
| support/all_rmse | 0.049603 | 0.043688 | 0.055733 | 77 |
| support/retained_rmse | 0.048900 | 0.042561 | 0.055515 | 77 |

### Global reference: AL

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/learned | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/nonlearned | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/omitted | 1.000000 | 1.000000 | 1.000000 | 69 |
| exact/overall | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/retained | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/serial_contrast | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/strict_correct_order | 1.000000 | 1.000000 | 1.000000 | 77 |
| probability/distance_slope | 0.003118 | 0.003118 | 0.003118 | 77 |
| probability/learned | 0.997912 | 0.997912 | 0.997912 | 77 |
| probability/nonlearned | 0.995006 | 0.995006 | 0.995006 | 77 |
| probability/omitted | 0.998096 | 0.997094 | 0.998990 | 69 |
| probability/overall | 0.995836 | 0.995836 | 0.995836 | 77 |
| probability/retained | 0.997749 | 0.997487 | 0.998020 | 77 |
| probability/serial_contrast | 0.002207 | 0.002207 | 0.002207 | 77 |
| support/all_rmse | 0.000000 | 0.000000 | 0.000000 | 77 |
| support/retained_rmse | 0.000000 | 0.000000 | 0.000000 | 77 |

### Global reference: RF

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.038144 | 0.029879 | 0.046590 | 77 |
| exact/learned | 0.949675 | 0.930195 | 0.967532 | 77 |
| exact/nonlearned | 0.892857 | 0.868182 | 0.915601 | 77 |
| exact/omitted | 0.814010 | 0.741546 | 0.880435 | 69 |
| exact/overall | 0.909091 | 0.887755 | 0.928571 | 77 |
| exact/retained | 0.997835 | 0.993506 | 1.000000 | 77 |
| exact/serial_contrast | 0.049351 | 0.025108 | 0.073593 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.090909 | 0.071429 | 0.112245 | 77 |
| latent/strict_correct_order | 0.324675 | 0.220779 | 0.428571 | 77 |
| probability/distance_slope | 0.044580 | 0.037523 | 0.051787 | 77 |
| probability/learned | 0.946572 | 0.928284 | 0.962941 | 77 |
| probability/nonlearned | 0.883978 | 0.860784 | 0.905685 | 77 |
| probability/omitted | 0.809528 | 0.741192 | 0.871499 | 69 |
| probability/overall | 0.901862 | 0.881595 | 0.920478 | 77 |
| probability/retained | 0.993614 | 0.989159 | 0.996787 | 77 |
| probability/serial_contrast | 0.050898 | 0.028607 | 0.073067 | 77 |
| support/all_rmse | 0.152527 | 0.127280 | 0.179777 | 77 |
| support/retained_rmse | 0.044787 | 0.037944 | 0.052397 | 77 |

### Global reference: RL

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.024839 | 0.017074 | 0.033058 | 77 |
| exact/learned | 0.961039 | 0.943182 | 0.975649 | 77 |
| exact/nonlearned | 0.920130 | 0.894805 | 0.942857 | 77 |
| exact/omitted | 0.861111 | 0.800725 | 0.916667 | 69 |
| exact/overall | 0.931818 | 0.910482 | 0.951299 | 77 |
| exact/retained | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/serial_contrast | 0.032035 | 0.008658 | 0.054545 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.068182 | 0.048701 | 0.089518 | 77 |
| latent/strict_correct_order | 0.545455 | 0.441558 | 0.662338 | 77 |
| probability/distance_slope | 0.030202 | 0.022937 | 0.037858 | 77 |
| probability/learned | 0.956356 | 0.939410 | 0.971043 | 77 |
| probability/nonlearned | 0.912429 | 0.888472 | 0.934240 | 77 |
| probability/omitted | 0.848217 | 0.789622 | 0.901644 | 69 |
| probability/overall | 0.924980 | 0.904341 | 0.943603 | 77 |
| probability/retained | 0.997749 | 0.997487 | 0.998020 | 77 |
| probability/serial_contrast | 0.035076 | 0.013494 | 0.056801 | 77 |
| support/all_rmse | 0.118203 | 0.089447 | 0.150012 | 77 |
| support/retained_rmse | 0.000000 | 0.000000 | 0.000000 | 77 |

### Global contrast: admission_at_finite

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.025702 | -0.034238 | -0.017346 | 77 |
| exact/learned | 0.050325 | 0.032468 | 0.069805 | 77 |
| exact/nonlearned | 0.087013 | 0.064935 | 0.111039 | 77 |
| exact/omitted | 0.185990 | 0.119565 | 0.258454 | 69 |
| exact/overall | 0.076531 | 0.057978 | 0.097403 | 77 |
| exact/retained | 0.002165 | 0.000000 | 0.006494 | 77 |
| exact/serial_contrast | -0.032900 | -0.056299 | -0.008658 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.076531 | -0.097403 | -0.057978 | 77 |
| latent/strict_correct_order | 0.363636 | 0.246753 | 0.493506 | 77 |
| probability/distance_slope | -0.025434 | -0.032615 | -0.018455 | 77 |
| probability/learned | 0.050247 | 0.033966 | 0.068272 | 77 |
| probability/nonlearned | 0.084607 | 0.064360 | 0.106890 | 77 |
| probability/omitted | 0.187493 | 0.126179 | 0.255427 | 69 |
| probability/overall | 0.074790 | 0.057436 | 0.093970 | 77 |
| probability/retained | 0.002975 | 0.000256 | 0.006944 | 77 |
| probability/serial_contrast | -0.029281 | -0.050074 | -0.008627 | 77 |
| support/all_rmse | -0.102924 | -0.128301 | -0.079782 | 77 |
| support/retained_rmse | 0.004113 | -0.003037 | 0.011122 | 77 |

### Global contrast: admission_at_least_squares

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.024839 | -0.033058 | -0.017074 | 77 |
| exact/learned | 0.038961 | 0.024351 | 0.056818 | 77 |
| exact/nonlearned | 0.079870 | 0.057143 | 0.105195 | 77 |
| exact/omitted | 0.138889 | 0.083333 | 0.199275 | 69 |
| exact/overall | 0.068182 | 0.048701 | 0.089518 | 77 |
| exact/retained | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/serial_contrast | -0.032035 | -0.054545 | -0.008658 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.068182 | -0.089518 | -0.048701 | 77 |
| latent/strict_correct_order | 0.454545 | 0.337662 | 0.558442 | 77 |
| probability/distance_slope | -0.027083 | -0.034740 | -0.019819 | 77 |
| probability/learned | 0.041555 | 0.026869 | 0.058502 | 77 |
| probability/nonlearned | 0.082577 | 0.060766 | 0.106534 | 77 |
| probability/omitted | 0.149880 | 0.096567 | 0.208518 | 69 |
| probability/overall | 0.070857 | 0.052233 | 0.091495 | 77 |
| probability/retained | -0.000000 | -0.000000 | 0.000000 | 77 |
| probability/serial_contrast | -0.032869 | -0.054594 | -0.011287 | 77 |
| support/all_rmse | -0.118203 | -0.150012 | -0.089447 | 77 |
| support/retained_rmse | 0.000000 | -0.000000 | 0.000000 | 77 |

### Global contrast: integration_at_all

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.012442 | -0.017119 | -0.008174 | 77 |
| exact/learned | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/nonlearned | 0.020130 | 0.012987 | 0.027922 | 77 |
| exact/omitted | 0.000000 | 0.000000 | 0.000000 | 69 |
| exact/overall | 0.014378 | 0.009276 | 0.019944 | 77 |
| exact/retained | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/serial_contrast | -0.016450 | -0.027706 | -0.006061 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.014378 | -0.019944 | -0.009276 | 77 |
| latent/strict_correct_order | 0.311688 | 0.207792 | 0.415584 | 77 |
| probability/distance_slope | -0.016028 | -0.019678 | -0.012537 | 77 |
| probability/learned | 0.001094 | 0.000183 | 0.002166 | 77 |
| probability/nonlearned | 0.026420 | 0.020581 | 0.032730 | 77 |
| probability/omitted | 0.001076 | -0.000212 | 0.002723 | 69 |
| probability/overall | 0.019184 | 0.014971 | 0.023713 | 77 |
| probability/retained | 0.001160 | 0.000068 | 0.002475 | 77 |
| probability/serial_contrast | -0.019410 | -0.028306 | -0.011123 | 77 |
| support/all_rmse | -0.049603 | -0.055733 | -0.043688 | 77 |
| support/retained_rmse | -0.048900 | -0.055515 | -0.042561 | 77 |

### Global contrast: integration_at_retained

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.013305 | -0.019708 | -0.007583 | 77 |
| exact/learned | 0.011364 | 0.004870 | 0.019481 | 77 |
| exact/nonlearned | 0.027273 | 0.013636 | 0.042857 | 77 |
| exact/omitted | 0.047101 | 0.014493 | 0.090580 | 69 |
| exact/overall | 0.022727 | 0.012059 | 0.035250 | 77 |
| exact/retained | 0.002165 | 0.000000 | 0.006494 | 77 |
| exact/serial_contrast | -0.017316 | -0.030303 | -0.005195 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.022727 | -0.035250 | -0.012059 | 77 |
| latent/strict_correct_order | 0.220779 | 0.129870 | 0.311688 | 77 |
| probability/distance_slope | -0.014378 | -0.019146 | -0.010012 | 77 |
| probability/learned | 0.009785 | 0.004329 | 0.016306 | 77 |
| probability/nonlearned | 0.028450 | 0.017248 | 0.042126 | 77 |
| probability/omitted | 0.038689 | 0.010848 | 0.077429 | 69 |
| probability/overall | 0.023117 | 0.014190 | 0.033968 | 77 |
| probability/retained | 0.004135 | 0.000995 | 0.008546 | 77 |
| probability/serial_contrast | -0.015822 | -0.025049 | -0.006783 | 77 |
| support/all_rmse | -0.034324 | -0.046549 | -0.023555 | 77 |
| support/retained_rmse | -0.044787 | -0.052397 | -0.037944 | 77 |

### Global contrast: interaction

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.000863 | -0.005722 | 0.007629 | 77 |
| exact/learned | -0.011364 | -0.019481 | -0.004870 | 77 |
| exact/nonlearned | -0.007143 | -0.022727 | 0.007143 | 77 |
| exact/omitted | -0.047101 | -0.090580 | -0.014493 | 69 |
| exact/overall | -0.008349 | -0.020872 | 0.002783 | 77 |
| exact/retained | -0.002165 | -0.006494 | 0.000000 | 77 |
| exact/serial_contrast | 0.000866 | -0.013853 | 0.014719 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.008349 | -0.002783 | 0.020872 | 77 |
| latent/strict_correct_order | 0.090909 | -0.038961 | 0.220779 | 77 |
| probability/distance_slope | -0.001650 | -0.006136 | 0.002902 | 77 |
| probability/learned | -0.008691 | -0.014980 | -0.003464 | 77 |
| probability/nonlearned | -0.002030 | -0.014237 | 0.008633 | 77 |
| probability/omitted | -0.037613 | -0.076179 | -0.010011 | 69 |
| probability/overall | -0.003933 | -0.013752 | 0.004597 | 77 |
| probability/retained | -0.002975 | -0.006944 | -0.000256 | 77 |
| probability/serial_contrast | -0.003588 | -0.012793 | 0.005643 | 77 |
| support/all_rmse | -0.015279 | -0.026941 | -0.003389 | 77 |
| support/retained_rmse | -0.004113 | -0.011122 | 0.003037 | 77 |

### Global contrast: total

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.038144 | -0.046590 | -0.029879 | 77 |
| exact/learned | 0.050325 | 0.032468 | 0.069805 | 77 |
| exact/nonlearned | 0.107143 | 0.084399 | 0.131818 | 77 |
| exact/omitted | 0.185990 | 0.119565 | 0.258454 | 69 |
| exact/overall | 0.090909 | 0.071429 | 0.112245 | 77 |
| exact/retained | 0.002165 | 0.000000 | 0.006494 | 77 |
| exact/serial_contrast | -0.049351 | -0.073593 | -0.025108 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.090909 | -0.112245 | -0.071429 | 77 |
| latent/strict_correct_order | 0.675325 | 0.571429 | 0.779221 | 77 |
| probability/distance_slope | -0.041461 | -0.048669 | -0.034405 | 77 |
| probability/learned | 0.051340 | 0.034971 | 0.069628 | 77 |
| probability/nonlearned | 0.111028 | 0.089321 | 0.134222 | 77 |
| probability/omitted | 0.188569 | 0.126678 | 0.256383 | 69 |
| probability/overall | 0.093974 | 0.075358 | 0.114241 | 77 |
| probability/retained | 0.004135 | 0.000995 | 0.008546 | 77 |
| probability/serial_contrast | -0.048691 | -0.070860 | -0.026400 | 77 |
| support/all_rmse | -0.152527 | -0.179777 | -0.127280 | 77 |
| support/retained_rmse | -0.044787 | -0.052397 | -0.037944 | 77 |

### Readout accounting: correct_shortfall

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.008701 | -0.011998 | -0.005855 | 77 |
| nonlearned | -0.019808 | -0.022514 | -0.017238 | 77 |
| omitted | -0.022733 | -0.038411 | -0.010556 | 69 |
| overall | -0.016635 | -0.018795 | -0.014602 | 77 |
| retained | -0.004713 | -0.007391 | -0.002875 | 77 |

### Readout accounting: tie_fraction

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.000000 | 0.000000 | 0.000000 | 77 |
| nonlearned | 0.000000 | 0.000000 | 0.000000 | 77 |
| omitted | 0.000000 | 0.000000 | 0.000000 | 69 |
| overall | 0.000000 | 0.000000 | 0.000000 | 77 |
| retained | 0.000000 | 0.000000 | 0.000000 | 77 |

### Readout accounting: ties

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.000000 | 0.000000 | 0.000000 | 77 |
| nonlearned | 0.000000 | 0.000000 | 0.000000 | 77 |
| omitted | 0.000000 | 0.000000 | 0.000000 | 69 |
| overall | 0.000000 | 0.000000 | 0.000000 | 77 |
| retained | 0.000000 | 0.000000 | 0.000000 | 77 |

### Readout accounting: total

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.003104 | -0.007458 | 0.001356 | 77 |
| nonlearned | -0.008879 | -0.012689 | -0.004918 | 77 |
| omitted | -0.004482 | -0.022560 | 0.011993 | 69 |
| overall | -0.007229 | -0.010409 | -0.003988 | 77 |
| retained | -0.004222 | -0.007045 | -0.002013 | 77 |

### Readout accounting: wrong_rescue

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.005597 | 0.002491 | 0.009278 | 77 |
| nonlearned | 0.010930 | 0.008008 | 0.014108 | 77 |
| omitted | 0.018251 | 0.008246 | 0.030088 | 69 |
| overall | 0.009406 | 0.006982 | 0.012021 | 77 |
| retained | 0.000492 | 0.000000 | 0.001475 | 77 |

### Retained graph coverage (fixed connected/disconnected strata)

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| connected/RF/latent/strict_correct_order | 0.592593 | 0.407407 | 0.777778 | 27 |
| connected/RF/probability/distance_slope | 0.027085 | 0.018603 | 0.037021 | 27 |
| connected/RF/probability/learned | 0.989541 | 0.979153 | 0.996320 | 27 |
| connected/RF/probability/nonlearned | 0.950358 | 0.926072 | 0.969082 | 27 |
| connected/RF/probability/omitted | 0.920864 | 0.797831 | 0.995963 | 19 |
| connected/RF/probability/overall | 0.961553 | 0.943238 | 0.976003 | 27 |
| connected/RF/probability/retained | 0.996073 | 0.994012 | 0.997822 | 27 |
| connected/RF/probability/serial_contrast | 0.032188 | 0.017137 | 0.047639 | 27 |
| connected/RL/latent/strict_correct_order | 1.000000 | 1.000000 | 1.000000 | 27 |
| connected/RL/probability/distance_slope | 0.003118 | 0.003118 | 0.003118 | 27 |
| connected/RL/probability/learned | 0.997912 | 0.997912 | 0.997912 | 27 |
| connected/RL/probability/nonlearned | 0.995006 | 0.995006 | 0.995006 | 27 |
| connected/RL/probability/omitted | 0.997406 | 0.994815 | 0.999997 | 19 |
| connected/RL/probability/overall | 0.995836 | 0.995836 | 0.995836 | 27 |
| connected/RL/probability/retained | 0.997963 | 0.997713 | 0.998255 | 27 |
| connected/RL/probability/serial_contrast | 0.002207 | 0.002207 | 0.002207 | 27 |
| disconnected/RF/latent/strict_correct_order | 0.180000 | 0.080000 | 0.300000 | 50 |
| disconnected/RF/probability/distance_slope | 0.054027 | 0.045199 | 0.062872 | 50 |
| disconnected/RF/probability/learned | 0.923368 | 0.898181 | 0.946324 | 50 |
| disconnected/RF/probability/nonlearned | 0.848134 | 0.819061 | 0.874919 | 50 |
| disconnected/RF/probability/omitted | 0.767220 | 0.688573 | 0.840958 | 50 |
| disconnected/RF/probability/overall | 0.869629 | 0.844697 | 0.892710 | 50 |
| disconnected/RF/probability/retained | 0.992286 | 0.985609 | 0.997033 | 50 |
| disconnected/RF/probability/serial_contrast | 0.061002 | 0.028522 | 0.093323 | 50 |
| disconnected/RL/latent/strict_correct_order | 0.300000 | 0.180000 | 0.420000 | 50 |
| disconnected/RL/probability/distance_slope | 0.044827 | 0.035814 | 0.054319 | 50 |
| disconnected/RL/probability/learned | 0.933917 | 0.910682 | 0.955060 | 50 |
| disconnected/RL/probability/nonlearned | 0.867837 | 0.838020 | 0.894687 | 50 |
| disconnected/RL/probability/omitted | 0.791525 | 0.717271 | 0.860343 | 50 |
| disconnected/RL/probability/overall | 0.886717 | 0.860996 | 0.909431 | 50 |
| disconnected/RL/probability/retained | 0.997634 | 0.997257 | 0.998026 | 50 |
| disconnected/RL/probability/serial_contrast | 0.052825 | 0.020523 | 0.084998 | 50 |

### Frozen probability minus exact decision

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| distance_slope | 0.006436 | 0.003965 | 0.008968 | 77 |
| learned | -0.003104 | -0.007458 | 0.001356 | 77 |
| nonlearned | -0.008879 | -0.012689 | -0.004918 | 77 |
| omitted | -0.004482 | -0.022560 | 0.011993 | 69 |
| overall | -0.007229 | -0.010409 | -0.003988 | 77 |
| retained | -0.004222 | -0.007045 | -0.002013 | 77 |
| serial_contrast | 0.001547 | -0.004457 | 0.007817 | 77 |

### Local cells: G

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.946521 | 0.928248 | 0.962881 | 77 |
| nonlearned | 0.883895 | 0.860706 | 0.905593 | 77 |
| omitted | 0.809497 | 0.741176 | 0.871429 | 69 |
| overall | 0.901788 | 0.881527 | 0.920405 | 77 |
| retained | 0.993556 | 0.989099 | 0.996729 | 77 |

### Local cells: GC

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.945177 | 0.926718 | 0.961805 | 77 |
| nonlearned | 0.874796 | 0.850314 | 0.897597 | 77 |
| omitted | 0.803263 | 0.734342 | 0.868052 | 69 |
| overall | 0.894905 | 0.873833 | 0.914422 | 77 |
| retained | 0.993011 | 0.988735 | 0.996216 | 77 |

### Local cells: GS

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.954690 | 0.937888 | 0.969557 | 77 |
| nonlearned | 0.883895 | 0.860706 | 0.905593 | 77 |
| omitted | 0.834157 | 0.769106 | 0.893045 | 69 |
| overall | 0.904122 | 0.884441 | 0.922365 | 77 |
| retained | 0.995542 | 0.991811 | 0.998051 | 77 |

### Local cells: GSC

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.953038 | 0.936064 | 0.968121 | 77 |
| nonlearned | 0.874796 | 0.850314 | 0.897597 | 77 |
| omitted | 0.826011 | 0.759541 | 0.886536 | 69 |
| overall | 0.897151 | 0.876408 | 0.916246 | 77 |
| retained | 0.995306 | 0.991982 | 0.997739 | 77 |

### Local effects: cross

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.001497 | -0.003436 | 0.000412 | 77 |
| nonlearned | -0.009099 | -0.012174 | -0.006119 | 77 |
| omitted | -0.007190 | -0.018177 | 0.001495 | 69 |
| overall | -0.006927 | -0.009164 | -0.004699 | 77 |
| retained | -0.000390 | -0.001332 | 0.000496 | 77 |

### Local effects: full_minus_self_only

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.001651 | -0.003683 | 0.000282 | 77 |
| nonlearned | -0.009099 | -0.012174 | -0.006119 | 77 |
| omitted | -0.008146 | -0.019424 | 0.000841 | 69 |
| overall | -0.006971 | -0.009226 | -0.004750 | 77 |
| retained | -0.000236 | -0.001039 | 0.000563 | 77 |

### Local effects: self

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.008015 | 0.005311 | 0.011140 | 77 |
| nonlearned | 0.000000 | 0.000000 | 0.000000 | 77 |
| omitted | 0.023704 | 0.014867 | 0.034178 | 69 |
| overall | 0.002290 | 0.001517 | 0.003183 | 77 |
| retained | 0.002141 | 0.001406 | 0.003053 | 77 |

### Local effects: total

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.006517 | 0.003935 | 0.009463 | 77 |
| nonlearned | -0.009099 | -0.012174 | -0.006119 | 77 |
| omitted | 0.016514 | 0.007820 | 0.025569 | 69 |
| overall | -0.004637 | -0.006902 | -0.002410 | 77 |
| retained | 0.001750 | 0.000673 | 0.003137 | 77 |

### Local between_recipe: acute_local

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.006517 | 0.003935 | 0.009463 | 77 |
| nonlearned | -0.009099 | -0.012174 | -0.006119 | 77 |
| omitted | 0.016514 | 0.007820 | 0.025569 | 69 |
| overall | -0.004637 | -0.006902 | -0.002410 | 77 |
| retained | 0.001750 | 0.000673 | 0.003137 | 77 |

### Local between_recipe: global_fit_difference

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.000051 | -0.000070 | -0.000031 | 77 |
| nonlearned | -0.000084 | -0.000101 | -0.000067 | 77 |
| omitted | -0.000030 | -0.000095 | 0.000036 | 69 |
| overall | -0.000075 | -0.000088 | -0.000061 | 77 |
| retained | -0.000058 | -0.000071 | -0.000045 | 77 |

### Local between_recipe: total_recipe_difference

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.006466 | 0.003886 | 0.009415 | 77 |
| nonlearned | -0.009183 | -0.012253 | -0.006203 | 77 |
| omitted | 0.016484 | 0.007782 | 0.025552 | 69 |
| overall | -0.004711 | -0.006979 | -0.002482 | 77 |
| retained | 0.001692 | 0.000614 | 0.003076 | 77 |

### Original behavior anchors (unchanged)

| Recipe | Qualitative rows | Frozen quantitative rows |
| --- | --- | --- |
| score_only | 9/9 | 3/9 |
| score_trace | 9/9 | 3/9 |

## Training stream 2113

Frozen parameters: `{'score_only': {'eta': 0.9885913729667664, 'gamma_G': 7.197957515716553, 'gamma_L': 0.0}, 'score_trace': {'eta': 0.9887816309928894, 'gamma_G': 7.160130500793457, 'gamma_L': 0.23005518317222595}}`.

### Global reference: AF

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.012851 | 0.008582 | 0.017755 | 77 |
| exact/learned | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/nonlearned | 0.979221 | 0.971429 | 0.986364 | 77 |
| exact/omitted | 1.000000 | 1.000000 | 1.000000 | 69 |
| exact/overall | 0.985158 | 0.979592 | 0.990260 | 77 |
| exact/retained | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/serial_contrast | 0.015584 | 0.005195 | 0.026840 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.014842 | 0.009740 | 0.020408 | 77 |
| latent/strict_correct_order | 0.675325 | 0.558442 | 0.779221 | 77 |
| probability/distance_slope | 0.019084 | 0.015557 | 0.022989 | 77 |
| probability/learned | 0.996864 | 0.995747 | 0.997741 | 77 |
| probability/nonlearned | 0.968694 | 0.962107 | 0.974634 | 77 |
| probability/omitted | 0.997066 | 0.994924 | 0.998833 | 69 |
| probability/overall | 0.976743 | 0.971992 | 0.981034 | 77 |
| probability/retained | 0.996637 | 0.995198 | 0.997745 | 77 |
| probability/serial_contrast | 0.021573 | 0.013550 | 0.030552 | 77 |
| support/all_rmse | 0.049604 | 0.043836 | 0.055822 | 77 |
| support/retained_rmse | 0.048902 | 0.042749 | 0.055701 | 77 |

### Global reference: AL

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/learned | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/nonlearned | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/omitted | 1.000000 | 1.000000 | 1.000000 | 69 |
| exact/overall | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/retained | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/serial_contrast | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/strict_correct_order | 1.000000 | 1.000000 | 1.000000 | 77 |
| probability/distance_slope | 0.003057 | 0.003057 | 0.003057 | 77 |
| probability/learned | 0.997953 | 0.997953 | 0.997953 | 77 |
| probability/nonlearned | 0.995105 | 0.995105 | 0.995105 | 77 |
| probability/omitted | 0.998134 | 0.997152 | 0.999003 | 69 |
| probability/overall | 0.995919 | 0.995919 | 0.995919 | 77 |
| probability/retained | 0.997794 | 0.997537 | 0.998059 | 77 |
| probability/serial_contrast | 0.002164 | 0.002164 | 0.002164 | 77 |
| support/all_rmse | 0.000000 | 0.000000 | 0.000000 | 77 |
| support/retained_rmse | 0.000000 | 0.000000 | 0.000000 | 77 |

### Global reference: RF

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.038144 | 0.030015 | 0.046590 | 77 |
| exact/learned | 0.949675 | 0.930195 | 0.967532 | 77 |
| exact/nonlearned | 0.892857 | 0.868182 | 0.915584 | 77 |
| exact/omitted | 0.814010 | 0.740338 | 0.879227 | 69 |
| exact/overall | 0.909091 | 0.887744 | 0.928571 | 77 |
| exact/retained | 0.997835 | 0.993506 | 1.000000 | 77 |
| exact/serial_contrast | 0.049351 | 0.025108 | 0.073593 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.090909 | 0.071429 | 0.112256 | 77 |
| latent/strict_correct_order | 0.324675 | 0.220779 | 0.428571 | 77 |
| probability/distance_slope | 0.044543 | 0.037478 | 0.051814 | 77 |
| probability/learned | 0.946617 | 0.928116 | 0.962926 | 77 |
| probability/nonlearned | 0.884053 | 0.860971 | 0.905414 | 77 |
| probability/omitted | 0.809555 | 0.741971 | 0.869917 | 69 |
| probability/overall | 0.901928 | 0.881832 | 0.920307 | 77 |
| probability/retained | 0.993665 | 0.989218 | 0.996810 | 77 |
| probability/serial_contrast | 0.050875 | 0.028889 | 0.072994 | 77 |
| support/all_rmse | 0.152530 | 0.127473 | 0.179317 | 77 |
| support/retained_rmse | 0.044790 | 0.037990 | 0.052442 | 77 |

### Global reference: RL

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.024839 | 0.017119 | 0.033149 | 77 |
| exact/learned | 0.961039 | 0.944765 | 0.975649 | 77 |
| exact/nonlearned | 0.920130 | 0.895455 | 0.943506 | 77 |
| exact/omitted | 0.861111 | 0.798309 | 0.915459 | 69 |
| exact/overall | 0.931818 | 0.911410 | 0.951299 | 77 |
| exact/retained | 1.000000 | 1.000000 | 1.000000 | 77 |
| exact/serial_contrast | 0.032035 | 0.009524 | 0.054545 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.068182 | 0.048701 | 0.088590 | 77 |
| latent/strict_correct_order | 0.545455 | 0.428571 | 0.662338 | 77 |
| probability/distance_slope | 0.030160 | 0.022897 | 0.037852 | 77 |
| probability/learned | 0.956403 | 0.939586 | 0.971136 | 77 |
| probability/nonlearned | 0.912500 | 0.889103 | 0.934684 | 77 |
| probability/omitted | 0.848261 | 0.788712 | 0.900772 | 69 |
| probability/overall | 0.925044 | 0.904975 | 0.944101 | 77 |
| probability/retained | 0.997794 | 0.997537 | 0.998059 | 77 |
| probability/serial_contrast | 0.035047 | 0.013840 | 0.056876 | 77 |
| support/all_rmse | 0.118203 | 0.088827 | 0.149791 | 77 |
| support/retained_rmse | 0.000000 | 0.000000 | 0.000000 | 77 |

### Global contrast: admission_at_finite

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.025293 | -0.034148 | -0.016801 | 77 |
| exact/learned | 0.050325 | 0.032468 | 0.069805 | 77 |
| exact/nonlearned | 0.086364 | 0.064286 | 0.110390 | 77 |
| exact/omitted | 0.185990 | 0.120773 | 0.259662 | 69 |
| exact/overall | 0.076067 | 0.057050 | 0.096939 | 77 |
| exact/retained | 0.002165 | 0.000000 | 0.006494 | 77 |
| exact/serial_contrast | -0.033766 | -0.057143 | -0.009524 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.076067 | -0.096939 | -0.057050 | 77 |
| latent/strict_correct_order | 0.350649 | 0.233766 | 0.467532 | 77 |
| probability/distance_slope | -0.025460 | -0.032738 | -0.018442 | 77 |
| probability/learned | 0.050247 | 0.034063 | 0.068615 | 77 |
| probability/nonlearned | 0.084642 | 0.064484 | 0.106251 | 77 |
| probability/omitted | 0.187511 | 0.127595 | 0.254637 | 69 |
| probability/overall | 0.074815 | 0.057295 | 0.093939 | 77 |
| probability/retained | 0.002973 | 0.000248 | 0.006912 | 77 |
| probability/serial_contrast | -0.029302 | -0.049804 | -0.008412 | 77 |
| support/all_rmse | -0.102925 | -0.127909 | -0.079693 | 77 |
| support/retained_rmse | 0.004112 | -0.002796 | 0.011164 | 77 |

### Global contrast: admission_at_least_squares

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.024839 | -0.033149 | -0.017119 | 77 |
| exact/learned | 0.038961 | 0.024351 | 0.055235 | 77 |
| exact/nonlearned | 0.079870 | 0.056494 | 0.104545 | 77 |
| exact/omitted | 0.138889 | 0.084541 | 0.201691 | 69 |
| exact/overall | 0.068182 | 0.048701 | 0.088590 | 77 |
| exact/retained | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/serial_contrast | -0.032035 | -0.054545 | -0.009524 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.068182 | -0.088590 | -0.048701 | 77 |
| latent/strict_correct_order | 0.454545 | 0.337662 | 0.571429 | 77 |
| probability/distance_slope | -0.027103 | -0.034794 | -0.019840 | 77 |
| probability/learned | 0.041550 | 0.026818 | 0.058367 | 77 |
| probability/nonlearned | 0.082605 | 0.060420 | 0.106002 | 77 |
| probability/omitted | 0.149873 | 0.097501 | 0.209163 | 69 |
| probability/overall | 0.070875 | 0.051818 | 0.090944 | 77 |
| probability/retained | 0.000000 | -0.000000 | 0.000000 | 77 |
| probability/serial_contrast | -0.032883 | -0.054713 | -0.011676 | 77 |
| support/all_rmse | -0.118203 | -0.149791 | -0.088827 | 77 |
| support/retained_rmse | 0.000000 | -0.000000 | 0.000000 | 77 |

### Global contrast: integration_at_all

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.012851 | -0.017755 | -0.008582 | 77 |
| exact/learned | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/nonlearned | 0.020779 | 0.013636 | 0.028571 | 77 |
| exact/omitted | 0.000000 | 0.000000 | 0.000000 | 69 |
| exact/overall | 0.014842 | 0.009740 | 0.020408 | 77 |
| exact/retained | 0.000000 | 0.000000 | 0.000000 | 77 |
| exact/serial_contrast | -0.015584 | -0.026840 | -0.005195 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.014842 | -0.020408 | -0.009740 | 77 |
| latent/strict_correct_order | 0.324675 | 0.220779 | 0.441558 | 77 |
| probability/distance_slope | -0.016026 | -0.019932 | -0.012500 | 77 |
| probability/learned | 0.001090 | 0.000213 | 0.002206 | 77 |
| probability/nonlearned | 0.026410 | 0.020471 | 0.032997 | 77 |
| probability/omitted | 0.001068 | -0.000202 | 0.002662 | 69 |
| probability/overall | 0.019176 | 0.014885 | 0.023926 | 77 |
| probability/retained | 0.001156 | 0.000091 | 0.002560 | 77 |
| probability/serial_contrast | -0.019409 | -0.028388 | -0.011386 | 77 |
| support/all_rmse | -0.049604 | -0.055822 | -0.043836 | 77 |
| support/retained_rmse | -0.048902 | -0.055701 | -0.042749 | 77 |

### Global contrast: integration_at_retained

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.013305 | -0.019844 | -0.007583 | 77 |
| exact/learned | 0.011364 | 0.004870 | 0.019481 | 77 |
| exact/nonlearned | 0.027273 | 0.012987 | 0.043506 | 77 |
| exact/omitted | 0.047101 | 0.014493 | 0.090580 | 69 |
| exact/overall | 0.022727 | 0.011596 | 0.035714 | 77 |
| exact/retained | 0.002165 | 0.000000 | 0.006494 | 77 |
| exact/serial_contrast | -0.017316 | -0.030303 | -0.005195 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.022727 | -0.035714 | -0.011596 | 77 |
| latent/strict_correct_order | 0.220779 | 0.129870 | 0.311688 | 77 |
| probability/distance_slope | -0.014384 | -0.019395 | -0.010146 | 77 |
| probability/learned | 0.009787 | 0.004302 | 0.016390 | 77 |
| probability/nonlearned | 0.028447 | 0.017433 | 0.042161 | 77 |
| probability/omitted | 0.038707 | 0.010708 | 0.076610 | 69 |
| probability/overall | 0.023115 | 0.014221 | 0.034066 | 77 |
| probability/retained | 0.004129 | 0.001008 | 0.008558 | 77 |
| probability/serial_contrast | -0.015828 | -0.025495 | -0.006831 | 77 |
| support/all_rmse | -0.034327 | -0.046761 | -0.023478 | 77 |
| support/retained_rmse | -0.044790 | -0.052442 | -0.037990 | 77 |

### Global contrast: interaction

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | 0.000454 | -0.006266 | 0.007221 | 77 |
| exact/learned | -0.011364 | -0.019481 | -0.004870 | 77 |
| exact/nonlearned | -0.006494 | -0.021429 | 0.008442 | 77 |
| exact/omitted | -0.047101 | -0.090580 | -0.014493 | 69 |
| exact/overall | -0.007885 | -0.019944 | 0.003711 | 77 |
| exact/retained | -0.002165 | -0.006494 | 0.000000 | 77 |
| exact/serial_contrast | 0.001732 | -0.012121 | 0.015584 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | 0.007885 | -0.003711 | 0.019944 | 77 |
| latent/strict_correct_order | 0.103896 | -0.025974 | 0.233766 | 77 |
| probability/distance_slope | -0.001643 | -0.006235 | 0.003076 | 77 |
| probability/learned | -0.008697 | -0.015017 | -0.003400 | 77 |
| probability/nonlearned | -0.002037 | -0.014001 | 0.008592 | 77 |
| probability/omitted | -0.037638 | -0.075601 | -0.009687 | 69 |
| probability/overall | -0.003940 | -0.013770 | 0.004555 | 77 |
| probability/retained | -0.002973 | -0.006912 | -0.000248 | 77 |
| probability/serial_contrast | -0.003582 | -0.012971 | 0.005512 | 77 |
| support/all_rmse | -0.015278 | -0.027196 | -0.003264 | 77 |
| support/retained_rmse | -0.004112 | -0.011164 | 0.002796 | 77 |

### Global contrast: total

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| exact/distance_slope | -0.038144 | -0.046590 | -0.030015 | 77 |
| exact/learned | 0.050325 | 0.032468 | 0.069805 | 77 |
| exact/nonlearned | 0.107143 | 0.084416 | 0.131818 | 77 |
| exact/omitted | 0.185990 | 0.120773 | 0.259662 | 69 |
| exact/overall | 0.090909 | 0.071429 | 0.112256 | 77 |
| exact/retained | 0.002165 | 0.000000 | 0.006494 | 77 |
| exact/serial_contrast | -0.049351 | -0.073593 | -0.025108 | 77 |
| latent/has_tied_pair | 0.000000 | 0.000000 | 0.000000 | 77 |
| latent/pair_discordance | -0.090909 | -0.112256 | -0.071429 | 77 |
| latent/strict_correct_order | 0.675325 | 0.571429 | 0.779221 | 77 |
| probability/distance_slope | -0.041486 | -0.048757 | -0.034420 | 77 |
| probability/learned | 0.051337 | 0.035027 | 0.069837 | 77 |
| probability/nonlearned | 0.111052 | 0.089691 | 0.134134 | 77 |
| probability/omitted | 0.188580 | 0.128567 | 0.256090 | 69 |
| probability/overall | 0.093990 | 0.075612 | 0.114086 | 77 |
| probability/retained | 0.004129 | 0.001008 | 0.008558 | 77 |
| probability/serial_contrast | -0.048711 | -0.070830 | -0.026725 | 77 |
| support/all_rmse | -0.152530 | -0.179317 | -0.127473 | 77 |
| support/retained_rmse | -0.044790 | -0.052442 | -0.037990 | 77 |

### Readout accounting: correct_shortfall

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.008633 | -0.011845 | -0.005863 | 77 |
| nonlearned | -0.019686 | -0.022373 | -0.017080 | 77 |
| omitted | -0.022623 | -0.038373 | -0.010078 | 69 |
| overall | -0.016528 | -0.018666 | -0.014442 | 77 |
| retained | -0.004660 | -0.007343 | -0.002804 | 77 |

### Readout accounting: tie_fraction

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.000000 | 0.000000 | 0.000000 | 77 |
| nonlearned | 0.000000 | 0.000000 | 0.000000 | 77 |
| omitted | 0.000000 | 0.000000 | 0.000000 | 69 |
| overall | 0.000000 | 0.000000 | 0.000000 | 77 |
| retained | 0.000000 | 0.000000 | 0.000000 | 77 |

### Readout accounting: ties

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.000000 | 0.000000 | 0.000000 | 77 |
| nonlearned | 0.000000 | 0.000000 | 0.000000 | 77 |
| omitted | 0.000000 | 0.000000 | 0.000000 | 69 |
| overall | 0.000000 | 0.000000 | 0.000000 | 77 |
| retained | 0.000000 | 0.000000 | 0.000000 | 77 |

### Readout accounting: total

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.003059 | -0.007484 | 0.001331 | 77 |
| nonlearned | -0.008805 | -0.012666 | -0.004883 | 77 |
| omitted | -0.004455 | -0.022734 | 0.012303 | 69 |
| overall | -0.007163 | -0.010396 | -0.003842 | 77 |
| retained | -0.004171 | -0.006947 | -0.001901 | 77 |

### Readout accounting: wrong_rescue

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.005574 | 0.002465 | 0.009320 | 77 |
| nonlearned | 0.010882 | 0.007984 | 0.014021 | 77 |
| omitted | 0.018168 | 0.008384 | 0.029928 | 69 |
| overall | 0.009365 | 0.006969 | 0.011949 | 77 |
| retained | 0.000489 | 0.000000 | 0.001468 | 77 |

### Retained graph coverage (fixed connected/disconnected strata)

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| connected/RF/latent/strict_correct_order | 0.592593 | 0.407407 | 0.777778 | 27 |
| connected/RF/probability/distance_slope | 0.027037 | 0.018455 | 0.036867 | 27 |
| connected/RF/probability/learned | 0.989601 | 0.978824 | 0.996407 | 27 |
| connected/RF/probability/nonlearned | 0.950445 | 0.926706 | 0.969242 | 27 |
| connected/RF/probability/omitted | 0.920983 | 0.799359 | 0.995998 | 19 |
| connected/RF/probability/overall | 0.961632 | 0.943431 | 0.976175 | 27 |
| connected/RF/probability/retained | 0.996128 | 0.994117 | 0.997833 | 27 |
| connected/RF/probability/serial_contrast | 0.032164 | 0.016978 | 0.047738 | 27 |
| connected/RL/latent/strict_correct_order | 1.000000 | 1.000000 | 1.000000 | 27 |
| connected/RL/probability/distance_slope | 0.003057 | 0.003057 | 0.003057 | 27 |
| connected/RL/probability/learned | 0.997953 | 0.997953 | 0.997953 | 27 |
| connected/RL/probability/nonlearned | 0.995105 | 0.995105 | 0.995105 | 27 |
| connected/RL/probability/omitted | 0.997457 | 0.994915 | 0.999997 | 19 |
| connected/RL/probability/overall | 0.995919 | 0.995919 | 0.995919 | 27 |
| connected/RL/probability/retained | 0.998003 | 0.997769 | 0.998280 | 27 |
| connected/RL/probability/serial_contrast | 0.002164 | 0.002164 | 0.002164 | 27 |
| disconnected/RF/latent/strict_correct_order | 0.180000 | 0.080000 | 0.300000 | 50 |
| disconnected/RF/probability/distance_slope | 0.053997 | 0.045038 | 0.062683 | 50 |
| disconnected/RF/probability/learned | 0.923405 | 0.898654 | 0.946411 | 50 |
| disconnected/RF/probability/nonlearned | 0.848201 | 0.819886 | 0.875627 | 50 |
| disconnected/RF/probability/omitted | 0.767212 | 0.687942 | 0.841201 | 50 |
| disconnected/RF/probability/overall | 0.869688 | 0.844754 | 0.893403 | 50 |
| disconnected/RF/probability/retained | 0.992335 | 0.985614 | 0.997092 | 50 |
| disconnected/RF/probability/serial_contrast | 0.060978 | 0.027585 | 0.093120 | 50 |
| disconnected/RL/latent/strict_correct_order | 0.300000 | 0.180000 | 0.420000 | 50 |
| disconnected/RL/probability/distance_slope | 0.044795 | 0.035429 | 0.054202 | 50 |
| disconnected/RL/probability/learned | 0.933966 | 0.911079 | 0.955198 | 50 |
| disconnected/RL/probability/nonlearned | 0.867893 | 0.838710 | 0.896618 | 50 |
| disconnected/RL/probability/omitted | 0.791567 | 0.717446 | 0.860098 | 50 |
| disconnected/RL/probability/overall | 0.886771 | 0.861375 | 0.911041 | 50 |
| disconnected/RL/probability/retained | 0.997681 | 0.997317 | 0.998070 | 50 |
| disconnected/RL/probability/serial_contrast | 0.052804 | 0.019946 | 0.085213 | 50 |

### Frozen probability minus exact decision

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| distance_slope | 0.006400 | 0.003917 | 0.008889 | 77 |
| learned | -0.003059 | -0.007484 | 0.001331 | 77 |
| nonlearned | -0.008805 | -0.012666 | -0.004883 | 77 |
| omitted | -0.004455 | -0.022734 | 0.012303 | 69 |
| overall | -0.007163 | -0.010396 | -0.003842 | 77 |
| retained | -0.004171 | -0.006947 | -0.001901 | 77 |
| serial_contrast | 0.001524 | -0.004497 | 0.007598 | 77 |

### Local cells: G

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.946568 | 0.928085 | 0.962859 | 77 |
| nonlearned | 0.883973 | 0.860903 | 0.905327 | 77 |
| omitted | 0.809526 | 0.741964 | 0.869898 | 69 |
| overall | 0.901857 | 0.881768 | 0.920239 | 77 |
| retained | 0.993610 | 0.989165 | 0.996758 | 77 |

### Local cells: GC

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.945231 | 0.926336 | 0.961758 | 77 |
| nonlearned | 0.874894 | 0.850518 | 0.897680 | 77 |
| omitted | 0.803285 | 0.733646 | 0.865460 | 69 |
| overall | 0.894990 | 0.873832 | 0.914371 | 77 |
| retained | 0.993071 | 0.988852 | 0.996270 | 77 |

### Local cells: GS

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.954709 | 0.937855 | 0.969646 | 77 |
| nonlearned | 0.883973 | 0.860903 | 0.905327 | 77 |
| omitted | 0.834131 | 0.769458 | 0.890617 | 69 |
| overall | 0.904183 | 0.884660 | 0.922236 | 77 |
| retained | 0.995578 | 0.991906 | 0.998070 | 77 |

### Local cells: GSC

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.953061 | 0.935884 | 0.968208 | 77 |
| nonlearned | 0.874894 | 0.850518 | 0.897680 | 77 |
| omitted | 0.825974 | 0.759607 | 0.884074 | 69 |
| overall | 0.897227 | 0.876671 | 0.916107 | 77 |
| retained | 0.995347 | 0.992041 | 0.997768 | 77 |

### Local effects: cross

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.001492 | -0.003465 | 0.000421 | 77 |
| nonlearned | -0.009079 | -0.012178 | -0.006055 | 77 |
| omitted | -0.007199 | -0.018234 | 0.001547 | 69 |
| overall | -0.006911 | -0.009199 | -0.004686 | 77 |
| retained | -0.000385 | -0.001379 | 0.000479 | 77 |

### Local effects: full_minus_self_only

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.001648 | -0.003701 | 0.000319 | 77 |
| nonlearned | -0.009079 | -0.012178 | -0.006055 | 77 |
| omitted | -0.008158 | -0.019403 | 0.000920 | 69 |
| overall | -0.006956 | -0.009236 | -0.004729 | 77 |
| retained | -0.000231 | -0.001076 | 0.000534 | 77 |

### Local effects: self

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.007985 | 0.005319 | 0.011227 | 77 |
| nonlearned | 0.000000 | 0.000000 | 0.000000 | 77 |
| omitted | 0.023647 | 0.014689 | 0.034202 | 69 |
| overall | 0.002282 | 0.001520 | 0.003208 | 77 |
| retained | 0.002122 | 0.001400 | 0.003032 | 77 |

### Local effects: total

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.006493 | 0.003919 | 0.009443 | 77 |
| nonlearned | -0.009079 | -0.012178 | -0.006055 | 77 |
| omitted | 0.016448 | 0.007524 | 0.025513 | 69 |
| overall | -0.004630 | -0.006933 | -0.002377 | 77 |
| retained | 0.001737 | 0.000652 | 0.003068 | 77 |

### Local between_recipe: acute_local

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.006493 | 0.003919 | 0.009443 | 77 |
| nonlearned | -0.009079 | -0.012178 | -0.006055 | 77 |
| omitted | 0.016448 | 0.007524 | 0.025513 | 69 |
| overall | -0.004630 | -0.006933 | -0.002377 | 77 |
| retained | 0.001737 | 0.000652 | 0.003068 | 77 |

### Local between_recipe: global_fit_difference

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | -0.000048 | -0.000067 | -0.000030 | 77 |
| nonlearned | -0.000080 | -0.000096 | -0.000063 | 77 |
| omitted | -0.000029 | -0.000090 | 0.000035 | 69 |
| overall | -0.000071 | -0.000084 | -0.000057 | 77 |
| retained | -0.000055 | -0.000068 | -0.000043 | 77 |

### Local between_recipe: total_recipe_difference

| Estimand | Mean | 95% lower | 95% upper | N |
| --- | --- | --- | --- | --- |
| learned | 0.006445 | 0.003867 | 0.009399 | 77 |
| nonlearned | -0.009159 | -0.012258 | -0.006142 | 77 |
| omitted | 0.016419 | 0.007478 | 0.025482 | 69 |
| overall | -0.004701 | -0.007005 | -0.002446 | 77 |
| retained | 0.001682 | 0.000592 | 0.003019 | 77 |

### Original behavior anchors (unchanged)

| Recipe | Qualitative rows | Frozen quantitative rows |
| --- | --- | --- |
| score_only | 9/9 | 3/9 |
| score_trace | 9/9 | 3/9 |

## Interpretation boundaries

Strict latent correct order uses all 77 subjects and a fixed raw-score tie tolerance. It is not the old sampled-choice/Hodge classification or its eligible-subject cohort. Serial contrast uses the two endpoints versus six interior positions; probability profiles are expected-choice diagnostics, not fresh sampled behavioral classification.
Positive gain cannot change latent ordering, but can change sampling. No optimal readout or encoding-precision parameter is identified here. Reference cells need not be closer to humans when their task accuracy increases.
Local self/cross terms use relation identity only for offline attribution. They do not authorize a learned-query flag or self-only model. The two-order sigmoid allocation is an exact response attribution, not an independent neural circuit. Between-recipe effects separately retain changes in fitted global parameters.
No new noise family, training, calibration or additional evaluation is admitted after these fixed analyses. The frozen parent evidence and closed family outcomes remain unchanged.
