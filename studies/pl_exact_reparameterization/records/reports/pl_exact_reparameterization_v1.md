# Exact P/L structural reparameterization

The repaired mapping passed the registered checkpoint-preserving qualification for
seeds 2104 and 2105 on both CPU and CUDA. It exposes 32 task channels plus a separate
retained time drive, a single public binary margin, and a 105-scalar packed local
trace. Fresh models contain no legacy compatibility buffers. Converted historical
checkpoints carry 802 non-trainable scalars solely to preserve the original float32
input-projection and two-logit subtraction order.

The qualification covered the Liu 32-support-trial horizon and the generic
40-support-trial upper horizon. Hidden state, eligibility, P, modulation, and global
margin matched exactly. The overall maximum absolute error was
`9.5367431640625e-7`, below the registered `1e-6` limit, and all 5,376 intact, P-off,
and L-off categorical comparisons matched.

Timestep accounting is part of the result. A generic training episode contains
112--160 support microsteps plus 56 query microsteps, for 168--216 active microsteps
and 56--80 effective P writes. A literal Liu-v2 presentation contains 688 active
microsteps. The two old initial blank steps are excluded only because they leave P
exactly unchanged and their transient h/E states are discarded.

The first qualification attempt remains registered as a negative result: folding the
constant input and output heads directly changed float32 accumulation order and failed
the state and margin thresholds despite zero decision mismatches. The repaired result
does not establish that time can be removed or that this clean architecture is directly
trainable. Those are separate prospective stages.
