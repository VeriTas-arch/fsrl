"""Command-line boundary for maintained frozen causal evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from fsrl.infra.provenance import write_json_exclusive

from .causal_suite import DEFAULT_PROTOCOL_ID, run_causal_suite
from .contracts import FrozenEvaluationBackend


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the registered fast-weight causal qualification suite."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cue-seed", type=int, default=1)
    parser.add_argument(
        "--cue-mode", choices=["shared", "permuted_shared"], default="permuted_shared"
    )
    parser.add_argument(
        "--subject-encoding",
        choices=[
            "none",
            "stable_bottleneck",
            "stable_omission",
            "presentationwise_omission",
            "blockwise_omission",
            "uniform_no_bottleneck",
        ],
        default="stable_omission",
    )
    parser.add_argument("--subject-encoding-seed", type=int, default=300)
    parser.add_argument("--support-seed", type=int, default=100)
    parser.add_argument("--order-seed", type=int, default=200)
    parser.add_argument("--order-schedules", type=int, default=8)
    parser.add_argument(
        "--protocol",
        type=Path,
        help=f"default: registered protocol {DEFAULT_PROTOCOL_ID}",
    )
    parser.add_argument(
        "--evaluation-backend",
        choices=[backend.value for backend in FrozenEvaluationBackend],
        default=FrozenEvaluationBackend.BATCHED_SEQUENCE.value,
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_causal_suite(
        parsed.checkpoint,
        batch_size=parsed.batch_size,
        cue_seed=parsed.cue_seed,
        support_seed=parsed.support_seed,
        order_seed=parsed.order_seed,
        order_schedules=parsed.order_schedules,
        cue_mode=parsed.cue_mode,
        subject_encoding_mode=parsed.subject_encoding,
        subject_encoding_seed=parsed.subject_encoding_seed,
        protocol_path=parsed.protocol,
        evaluation_backend=parsed.evaluation_backend,
    )
    write_json_exclusive(parsed.output, result)
