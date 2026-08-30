# Prospective training

This directory owns maintained meta-training and checkpoint-writing code. The
commands below are for a prospectively authorized development run; they do not
reopen the frozen model program or replace a registered historical execution.

## Run the current execution profile

```bash
direnv exec . python -m fsrl.training \
  --output-dir artifacts/runs/relational_model/seed-1 \
  --seed 1 \
  --device cuda \
  --optimized-execution
```

The current CUDA profile compiles complete recurrent trial sequences with
`fullgraph=True` and defaults to `mode="reduce-overhead"`. It marks one explicit
CUDA Graph iteration boundary per outer step and records the effective compile
mode, device, CUDA capability, matrix precision, determinism flags, and PyTorch
and BLAS thread limits.

`--optimized-execution` is a compatibility alias for
`--execution-schema current`. Use `--execution-schema historical` only when a
registered replay contract explicitly requires the stepwise implementation.
`--compile-mode default` retains the non-CUDA-Graph prospective profile. The
`max-autotune` modes remain explicit opt-ins because kernel selection can alter
floating-point reduction order and therefore the optimizer trajectory.

The output directory must not already exist. Training creates `run.json`
before execution, marks the run `complete` or `failed`, records owned-file
hashes, and writes the maintained checkpoint as `net.pth`. See the
[artifact contract](../../artifacts/README.md) for output ownership and
promotion rules.

## Engineering benchmark

The optimizer-step benchmark reports throughput, diagnostic synchronization,
and peak CUDA memory for the prospective path:

```bash
direnv exec . python -m fsrl.training.performance \
  --warmups 2 --repeats 5 \
  --output /tmp/fsrl-training-hot-path-benchmark.json
```

This is an engineering diagnostic, not scientific evidence or a portable
performance threshold. Variable edge counts can introduce compilation startup
costs into individual samples; report aligned duration and edge-count samples,
the aggregate, and the median rather than dropping slow startup observations.

Evaluate a completed checkpoint through the
[maintained evaluation interface](../evaluation/README.md).
