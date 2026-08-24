# Frozen v2.4 behavioral reproduction map

## Outcome

The read-only map classifies six of nine registered Liu phenomena as
`reproduced`, three as
`qualitatively_reproduced_quantitatively_mismatched`, and none as
`not_reproduced`. Every judgment passes separately in fresh v2.4 networks 2104
and 2105. Virtual participants and networks are never pooled.

This is a model-phenomenon result. It is not evidence that the internal model
mechanism is implemented by human brains, and it is not a model-selection or
calibration objective.

| Phenomenon | Human reference | Seed 2104 | Seed 2105 | Status |
|---|---:|---:|---:|---|
| Learned accuracy | 0.91396, 95% CI [0.89318, 0.93377] | 0.92516 | 0.93149 | reproduced |
| Nonlearned accuracy | 0.82773, 95% CI [0.80331, 0.85020] | 0.80357 | 0.80526 | reproduced |
| Symbolic-distance slope | 0.03983, 95% CI [0.03475, 0.04490] | 0.04949 | 0.04954 | qualitative, quantitatively too steep |
| Serial-position endpoint contrast | 0.08219, 95% CI [0.05374, 0.11070] | 0.05269 | 0.05139 | qualitative, quantitatively too weak |
| Pair Beta classes | 15 bimodal, 13 high-accuracy | 18 / 10 | 15 / 13 | reproduced |
| At least one 80%-stable error | 0.91304, 95% CI [0.84058, 0.97143] | 0.94366 | 0.94521 | reproduced |
| Self-consistent / self-inconsistent | 64 / 5 of 77 | 59 / 12 | 65 / 8 | qualitative; 2104 overproduces inconsistency |
| Correct Hodge ranking | 8/77, 95% CI [0.03896, 0.16883] | 6/77 | 4/77 | reproduced |
| Mean inter-subject Hodge-rank tau | 0.55365, 95% CI [0.50718, 0.61241] | 0.53515 | 0.53550 | reproduced |

## Frozen execution

The scientific contract was committed and pushed before implementation. The
implementation and its source hashes were then committed and pushed before the
write-once result was created. Execution read the registered human trials,
human benchmark, and the existing `dual_access_matched` fields in the frozen
v2.4 confirmation result. It did not load a checkpoint, invoke the neural
evaluator, train or adapt a parameter, change temperature, or resample a model
choice.

The serial-position row was the only registered phenomenon not already
explicitly summarized in the v2.4 result. Before computation, it was defined
as follows. Pair accuracy is averaged over the seven pairs incident on each
true rank position. The qualitative rule requires both endpoint positions to
exceed the mean of the six interior positions. The quantitative estimand is

\[
C_{\mathrm{end}}
=
\frac{a_1+a_8}{2}
-
\frac{1}{6}\sum_{r=2}^{7}a_r.
\]

The human 95% interval was obtained by participant bootstrap from the frozen
77-person trial files. Model profiles were reassembled only from each frozen
network's 28 pair means. Because the v2.4 result does not retain participant by
pair matrices, no model participant-level serial interval was invented and no
checkpoint replay was allowed.

The full human overall position profile from low to high is

\[
(0.89963, 0.87978, 0.85195, 0.81763,
  0.81299, 0.81058, 0.81800, 0.92839).
\]

The two model profiles are

\[
(0.87978, 0.87199, 0.81800, 0.79926,
  0.79814, 0.80353, 0.85993, 0.87588)
\]

and

\[
(0.87792, 0.86660, 0.82171, 0.80297,
  0.80631, 0.80408, 0.86920, 0.88182).
\]

Both networks reproduce the endpoint direction, but their endpoint contrasts
fall just below the frozen human interval.

## What is reproduced

The frozen v2.4 model jointly produces:

- above-chance direct learned fidelity and nonlearned transitive inference at
  human-range cohort means;
- pairwise polarization into bimodal versus high-accuracy regimes;
- highly stable within-subject errors;
- a strong predominance of coherent but incorrect global rankings;
- complete Hodge-reconstructed subjective orders with a human-range minority
  of correct rankers; and
- human-range inter-subject ranking diversity.

Together these support the model-level claim that sparse evidence can yield
stable, coherent, individualized global relational structures while a
separable local path preserves direct experience.

## Negative constraints

The three mismatches must be retained together.

1. Symbolic-distance dependence is reliably positive but too steep in both
   networks.
2. Serial-position dependence has the correct endpoint shape but a smaller
   endpoint contrast than humans in both networks.
3. Self-consistent errors dominate in both networks, but seed 2104 produces
   `12/77` self-inconsistent subjects, or 0.15584, above the frozen human 95%
   upper bound of 0.12987. Seed 2105 is calibrated on this row.

The first two mismatches have opposite amplitude directions. They therefore do
not support a single temperature, output gain, or `P_T`-amplitude correction.
The registered no-tuning rule remains binding. The appropriate next step is to
compress the already confirmed `P_T` and `L_T` causal chains into a main
computational mechanism and reduced algorithm, while carrying these behavioral
shape constraints forward as unresolved outputs rather than intervention
targets.

## Provenance

- Contract: `benchmarks/model_behavior_reproduction_map_v1.json`
- Implementation lock: `benchmarks/model_behavior_reproduction_map_v1.lock.json`
- Result: `results/model_behavior_reproduction_map_v1.json`
- Frozen model source: `results/dual_evidence_access_confirmation_v2_4.json`
- Frozen human source: `benchmarks/liu_human_exact_v1.json` and its registered
  preregistered/replication trial files
