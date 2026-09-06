"""Bounded CUDA with the reduction-fusion workaround qualified for this study."""

from fsrl.experiments.training_strategy.execution import (
    configure_execution as configure_parent,
)


def configure_execution():
    from torch._inductor import config

    # The installed compiler asserts on mixed-order reductions in cost backward.
    # Keep the mathematical graph and fullgraph compilation; disable this fusion.
    config.triton.mix_order_reduction = False
    snapshot = configure_parent()
    snapshot["triton_mix_order_reduction"] = config.triton.mix_order_reduction
    return snapshot
