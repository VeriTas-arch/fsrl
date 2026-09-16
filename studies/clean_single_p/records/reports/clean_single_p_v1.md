# Clean single-P direct-training result

Registered outcome: `time_removal_failure`.

| seed | control valid | candidate competent/bound | Ce noninferior | Ae/Ce core | compensation | outcome |
|---:|:---:|:---:|:---:|:---:|:---:|---|
| 3011 | PASS | PASS | FAIL | PASS | PASS | `time_removal_failure` |
| 3012 | PASS | PASS | PASS | PASS | PASS | `clean_single_p_admitted` |
| 3013 | PASS | PASS | PASS | PASS | PASS | `clean_single_p_admitted` |

Seed 3011 failed the registered Ce noninferiority rule, whose lower confidence
bound had to be at least -0.02. The no-time-minus-control estimates were:

| Ce endpoint | point | 95% CI | decision |
|---|---:|---:|:---:|
| generic learned | -0.02537 | [-0.02747, -0.02325] | FAIL |
| generic nonlearned | -0.02822 | [-0.03043, -0.02616] | FAIL |
| Liu learned | -0.00921 | [-0.01288, -0.00560] | PASS |
| Liu nonlearned | -0.00966 | [-0.01433, -0.00533] | PASS |
| Liu omitted | -0.01397 | [-0.02186, -0.00648] | FAIL |

The candidate has one 32-channel task input, one affine modulation scalar, one binary margin, one 40,000-scalar P state, no local store, no normalized-time parameter, no value head, and no initial blank rollout. Both conditions retained H=200, four support microsteps, two query microsteps, 1,500 updates and the registered P penalty.

Thus single-P competence and the registered core/compensation phenotype are
supported in all three seeds, while removing normalized time with this exact
training recipe is not all-seed reliable. The result does not show that time is
a cognitive variable or that an L store is structurally necessary.

This development result does not establish universal minimality or biological implementation. It does not authorize width/timestep compression, post-result repair, main-model promotion, or checkpoint deletion.
