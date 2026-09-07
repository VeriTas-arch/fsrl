# Joint revision learning: fixed-budget closure

The separately frozen comparison completed six from-scratch networks (three plastic/GRU pairs), 3,000 Adam steps per network, and all 156 evaluation units. Each pair had identical complete training-stream digests. All locked-source, input, parameter and finite-output checks passed; revision/outlier margins were exactly equal through the indistinguishable prefix. Native network computation used CUDA float32; sign-based revision losses followed the frozen implementation's float64 target arithmetic, while generic integer targets retained float32 loss arithmetic. There was no checkpoint selection, added training or threshold change.

The authority is [result.json](result.json), with [protocol](../benchmarks/protocol.json), [source/input lock](../benchmarks/source_lock.json), and [model lock](../benchmarks/model_lock.json). Evaluation markers reference all raw margins and physical inputs. Trained models have 88,605 parameters and 40,000 persistent scalars for the plastic RNN, versus 144,402 parameters and 200 persistent scalars for the GRU. This is an information/exposure-matched comparison, not a state-capacity match.

## Positive results

All three joint-trained plastic networks retained generic ranking competence: mean correct probabilities were 0.859 [0.853,0.864], 0.860 [0.854,0.865], and 0.829 [0.823,0.835]. Each network passed stable competence, remote revision and outlier recovery in both chain histories. Revision-specific remote CE improvements were 0.462–0.498 across these six network/history cells. After a transient outlier, the remote probability displacement declined from about 0.092–0.109 initially to 0.051–0.061 at the final prefix; recovery was partial, not complete.

All three plastic networks also passed competence, revision and recovery on both star histories, despite star being excluded from the joint training stream. This is development transport within the prespecified task family, not independent confirmation of a general algorithm.

## Failed stronger requirements

None of the three plastic networks passed either history-dependent crossover or preservation on the chain task. All six AD-change intervals included zero. For example, seed2711 produced H0 +0.0266 [−0.0419,0.0998] and H1 +0.0194 [−0.0490,0.0905]; there is no supported positive/negative directional crossover. In contrast, BC changes were strongly positive in all six cells, with means 2.591–2.788. A clear local reaction therefore did not establish the predicted immediate remote response.

Preservation costs were 0.066–0.106 CE; every lower95 bound exceeded the prospective 0.05 tolerance. Thus this successor's preservation failure is supported beyond merely uncertain admission. Star crossover and preservation requirements also failed for every plastic network. These outcomes do not establish an identically zero remote effect or an architecture-wide impossibility.

The three GRUs remained near chance on generic ranking (0.50002,0.50005,0.49963; all intervals included .5) and failed revision-task competence. Their nominal preservation and tiny outlier-recovery flags cannot be interpreted as knowledge maintenance or meaningful revision: initial outlier effects were only about 0.00013–0.00017. This is a failed learning recipe for the comparator, not evidence that nonplastic recurrent networks cannot represent the task. The full-covariance control in the [frozen diagnosis](../../../relational_revision/records/results/interpretation.md) remains the competence-qualified constructive reference.

## Theory and program decision

The supported computation is learnable relational prediction, adaptation to accumulating changed evidence, and partial recovery from an isolated observation error. The stronger proposal—that this recipe learns the dependency information needed to choose opposite immediate remote update directions while protecting unaffected relations—remains unsupported. Adding these task episodes once to the original training mixture was insufficient under this fixed budget. New and old studies use different cohorts and training distributions; the similar qualitative pattern is not a registered paired estimate of the effect of joint training.

The prospective program therefore closes this joint candidate. Directional intervention and a subsequent external dependency-topology prediction were conditional on a qualified candidate, and were not launched. No new architecture, longer budget, seed replacement, human fitting or main-model promotion follows automatically. The new weights have not separately established retention of Liu individualization or A/C compensation; generic competence cannot substitute for those claims or inherit them from old checkpoints.

For the manuscript, relational knowledge revision remains a discriminating research question, not a completed headline result. A future authorized diagnostic could distinguish absent dependency information from dependency information that the update does not use; it should be frozen independently and should not treat a successful post-hoc probe as proof that the native learner performs that computation. Preserve the current positive assembly/observation evidence and both revision failures without extending their claim boundaries.
