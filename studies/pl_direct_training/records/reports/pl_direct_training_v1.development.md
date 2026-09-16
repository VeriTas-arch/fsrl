# Direct P/L training: development

Registered outcome: **training_parameterization_failure**.

Both conditions were freshly initialized from paired shadow-legacy draws and trained for 1,500 joint updates. The control retains separate normalized time; the candidate has no time parameter or time argument. Both retain four support steps, two query steps, a 40,000-scalar P state, and a 105-scalar packed L state.

| Seed | Condition | Competence | P/L organization | Behavior | Paired noninferiority | Local gain |
| --- | --- | --- | --- | --- | --- | --- |
| 3001 | no_time_candidate | PASS | FAIL | FAIL | PASS | 0.277187 |
| 3001 | time_retained_control | PASS | FAIL | FAIL | PASS | 0.275804 |
| 3002 | no_time_candidate | PASS | FAIL | FAIL | PASS | 0.286078 |
| 3002 | time_retained_control | PASS | FAIL | FAIL | PASS | 0.284342 |
| 3003 | no_time_candidate | PASS | FAIL | FAIL | PASS | 0.280445 |
| 3003 | time_retained_control | PASS | FAIL | FAIL | PASS | 0.280233 |

## Boundaries

Every gate is evaluated within seed and condition. Participant uncertainty is not network-population uncertainty, and no successful network repairs a failed mandatory network. Human intervals and temperature are inherited descriptive references rather than newly fitted targets.

Next step: Stop this fixed recipe without tuning and retain the complete negative or mixed outcome.

The canonical JSON retains endpoint estimates, bootstrap bounds, behavior flags, stream fingerprints, tensor identities, optimizer counters, and runtime costs. Dense arrays and checkpoints remain hash-locked runtime artifacts.
