# Frozen mechanism confirmation protocol

`benchmarks/mechanism_confirmation_v1.json` is the prospective formal contract
for the computational mechanism developed on frozen pilot seeds 1901 and 1902.
It was frozen only after the preregistered history state-by-factor closure met
its stop rule, and before any formal-seed artifact was accessed. It is separate
from and does not modify `benchmarks/confirmation_v1.json`.

## Formal population and execution boundary

The population is the ten independently trained networks already declared by
`confirmation_v1`: seeds 2001 through 2010, using only their step-1000
checkpoints. Every seed is mandatory. Training, checkpoint selection, seed or
subject filtering, temperature refitting, architecture changes, and
result-conditioned threshold changes are forbidden.

Neural replay uses GPU. Lightweight validation, exact enumeration, and
bootstrap summaries use CPU. A formal runner may orchestrate the source-locked
assembly, factor-swap, and history-closure computations for formal artifacts,
but it may not change their equations, donor assignments, factor/cell
definitions, null permutations, targets, or tolerances. Such a change requires
a new contract version before formal access.

The source-locked execution layer limits PyTorch intra-op and inter-op work to
one CPU thread, performs one contiguous input transfer per trial, and trains
with `torch.compile(net, fullgraph=True)` in the default mode. NumPy/BLAS thread
settings remain unchanged to preserve the frozen analysis reductions. Each
formal artifact records the compiler configuration, runtime environment, and
execution-source hashes; aggregation rejects mixed execution versions.

The formal inferential unit is the independently trained network. Each
registered subject-level contrast is first averaged within a seed over the
fixed 77-subject cohort, then the ten seed means are bootstrapped 10,000 times
with seed `20260826`. Except for the prospectively fixed DA direction threshold,
a positive link requires the 95% network-bootstrap lower bound above zero.

## Competence gates

All ten networks must pass the existing `qualification_v2` rules, including
intact performance and the write-off, alpha-zero, reset, shuffle, and order
invariance controls. All replay identities, endpoints, source branches,
batched readouts, and stable-omitted zero controls must reproduce within
`3.814697265625e-6`. A failed gate is reported without filtering and makes its
dependent link non-interpretable for the joint mechanism decision.

## Primary supported chain

The contract tests seven development-supported links:

1. retained evidence has immediate causal effects on remote pairs and
   relation-LOO interventions redistribute third-party relations;
2. eligibility transfers donor relation direction under matched recipient DA,
   baseline, alpha, and effective norm;
3. high natural DA increases write and policy magnitude while preserving
   direction, prospectively defined as a mean cosine lower bound above `0.99`;
4. actual alpha places writes in higher-gain recurrent directions than 32 fixed
   norm-matched permutation nulls;
5. accumulated history positively changes baseline-dependent recurrent
   expression of an exposure-4 factor;
6. baseline and factor history have a positive exposure-4 interaction;
7. the terminal neural potential aligns more closely with the exact posterior
   expected-rank/Hodge potential than with a MAP-order potential.

The complete chain passes only if every competence gate and all seven links
pass. Linkwise outcomes are always reported separately. A failed link revises
that link; it does not erase independent positives or shrink the project-level
question.

## Preserved negative evidence

The unresolved total factor-generation main effect is not a formal primary.
Neither are correctness-aligned remote effects, trialwise exact posterior
innovation, or the development-seed-dependent alignment generation effect.
They remain mandatory diagnostics. The original behavioral confirmation,
including the excessive symbolic-distance slope seen in the pilots, also
retains its old frozen rule and is reported in parallel rather than being
retrofitted into this mechanism contract.

No formal seed should be trained, loaded, evaluated, or listed until this
contract is committed and pushed. Once formal access begins, this file is
immutable; a scientifically motivated revision requires
`mechanism_confirmation_v2`.

## Source-locked execution

Validate the frozen sources and reproduce all three development components
through the formal adapter before accessing formal artifacts:

```bash
direnv exec . python -m fsrl.formal_runtime mechanism validate
direnv exec . python -m fsrl.formal_runtime mechanism validate-development
```

Then process seeds 2001 through 2010 serially. For each seed, the old workflow
trains and evaluates the frozen `confirmation_v1`; the mechanism workflow then
registers the checkpoint, behavior, qualification, adapters, and three raw
component artifacts by SHA-256:

```bash
direnv exec . python -m fsrl.formal_runtime confirmation run-seed --seed 2001
direnv exec . python -m fsrl.formal_runtime mechanism run-seed --seed 2001
```

Do not inspect or aggregate scientific outcomes between seeds. After all ten
seed-level artifacts exist, aggregate both contracts without optional
omissions:

```bash
direnv exec . python -m fsrl.formal_runtime confirmation aggregate \
  --output results/confirmation_v1.json
direnv exec . python -m fsrl.formal_runtime mechanism aggregate
```

The mechanism aggregate first reduces within each network, then applies the
registered 10,000-sample bootstrap across the ten network-seed means. It reports
all linkwise outcomes even when the complete-chain conjunction fails.
