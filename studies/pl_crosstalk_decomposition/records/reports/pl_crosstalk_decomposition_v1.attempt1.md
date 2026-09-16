# Exact P/L retained cross-talk decomposition: attempt 1

Attempt 1 is non-interpretable and carries no scientific component conclusion.

All locked input-identity checks passed, but two implementation checks failed.
The source reconstruction used a per-trial dot-product order rather than the
registered trace-then-read order, producing a maximum raw-margin error of
`1.6987323760986328e-6` against the frozen `1e-6` tolerance. Separately, the
canonical-summary checker compared numeric values by JSON insertion order and
therefore reported a spurious error near `77`.

The parent result is unchanged: seed 3006 remains a formal retained-fidelity
failure and the all-seed outcome remains `competent_alternative_organization`.
No training, threshold change, transport, or scientific interpretation followed
from this attempt. The append-only repair contract permits only the two stated
implementation corrections before a fresh source lock and attempt 2.
