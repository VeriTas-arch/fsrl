"""Compatibility entry point for maintained sparse-ranking backbone training."""

from .training import backbone as _backbone
from .training.backbone import *

main = _backbone.main


if __name__ == "__main__":
    main()
