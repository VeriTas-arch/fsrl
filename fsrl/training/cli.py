"""Command-line boundary for prospective sparse-ranking training."""

from __future__ import annotations

import argparse
from pathlib import Path

from fsrl.infra.runtime import (
    DEFAULT_COMPILED_PROFILE,
    SUPPORTED_TORCH_COMPILE_MODES,
    ExecutionProfile,
    default_device,
)

from .backbone import OPTIMIZED_TRAINING_PROFILE, MetaTrainConfig, train_meta_model


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Meta-train a plastic RNN on generic sparse ranking graphs."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--cue-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--execution-schema",
        choices=("current", "historical"),
        default="current",
        help=(
            "current uses the versioned sequence execution; historical preserves "
            "the registered stepwise implementation"
        ),
    )
    parser.add_argument(
        "--optimized-execution",
        action="store_true",
        help=("compatibility alias for --execution-schema current"),
    )
    parser.add_argument(
        "--compile-mode",
        choices=SUPPORTED_TORCH_COMPILE_MODES,
        help=(
            "override the torch.compile mode; optimized CUDA execution defaults "
            "to reduce-overhead"
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=default_device())
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--subject-encoding",
        choices=["stable_attenuation", "stable_omission"],
        default="stable_omission",
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    training_config = MetaTrainConfig(
        seed=parsed.seed,
        outer_steps=parsed.outer_steps,
        batch_size=parsed.batch_size,
        hidden_size=parsed.hidden_size,
        cue_size=parsed.cue_size,
        learning_rate=parsed.learning_rate,
        save_every=parsed.save_every,
        subject_encoding_mode=parsed.subject_encoding,
    )
    optimized_execution = (
        parsed.execution_schema == "current" or parsed.optimized_execution
    )
    compile_model = parsed.compile_model or (
        optimized_execution and parsed.device == "cuda"
    )
    compile_mode = parsed.compile_mode or (
        OPTIMIZED_TRAINING_PROFILE.compile_mode
        if optimized_execution and parsed.device == "cuda"
        else DEFAULT_COMPILED_PROFILE.compile_mode
    )
    profile = ExecutionProfile(
        device=parsed.device,
        cpu_threads=parsed.cpu_threads,
        blas_threads=parsed.blas_threads,
        compile=compile_model,
        compile_mode=compile_mode,
        require_cuda=parsed.device == "cuda",
    )
    train_meta_model(
        training_config,
        parsed.output_dir,
        compile_model=compile_model,
        optimized_execution=optimized_execution,
        execution_profile=profile,
    )
