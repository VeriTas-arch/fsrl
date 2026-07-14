import argparse
from pathlib import Path

import numpy as np

from .config import TrainConfig
from .train import set_seed, train


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Readable training script for the plastic RNN transitive-inference task."
    )
    parser.add_argument("--nbiter", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=101)
    parser.add_argument("--output-dir", default=str(ROOT_DIR))
    parser.add_argument(
        "--trace-steps",
        action="store_true",
        help="Print per-step debugging details on summary episodes.",
    )
    return parser.parse_args(args)


def main(args=None):
    parsed_args = parse_args(args)
    config = TrainConfig(
        rngseed=parsed_args.seed,
        bs=parsed_args.batch_size,
        nbiter=parsed_args.nbiter,
        save_every=parsed_args.save_every,
        pe=parsed_args.print_every,
    )
    np.set_printoptions(precision=5)
    set_seed(config.rngseed)
    train(config, Path(parsed_args.output_dir), trace_steps=parsed_args.trace_steps)
