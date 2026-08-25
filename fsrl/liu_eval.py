"""Compatibility entry point for the registered relational-ranking evaluator."""

from .evaluation import frozen_fast_weight as _frozen_fast_weight
from .evaluation.frozen_fast_weight import *

main = _frozen_fast_weight.main


if __name__ == "__main__":
    main()
