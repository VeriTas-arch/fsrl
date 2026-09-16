# Exact P/L reparameterization: qualification attempt 1

The first source-locked mapping failed its registered numerical gate. Across CPU and
CUDA, seeds 2104 and 2105, the Liu 32-support-trial panel and the generic
40-support-trial panel, all 5,376 intact, P-off, and L-off categorical comparisons
were identical. Parameter mapping, the two-step blank P identity, and the packed local
state also stayed within their gates.

The mapping nevertheless failed. Folding the active constant input into the bias and
computing a single margin changed float32 accumulation order. The maximum terminal-P
error was `9.5367431640625e-6`, and the maximum final-margin error was
`1.430511474609375e-6`; both are above the registered `1e-6` limit. A zero decision
mismatch does not supersede that failure.

This result rejects the first numerical implementation, not the real-arithmetic
factorization and not the hypothesis that a retrained no-time model can work. The next
attempt must keep the same threshold and add a narrowly scoped, non-trainable legacy
arithmetic adapter for converted checkpoints. No no-time training is authorized by
this result.
