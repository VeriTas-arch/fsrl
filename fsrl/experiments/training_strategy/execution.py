"""Bounded CUDA execution shared by the prospective runner and parity audit."""

from __future__ import annotations

import os

import torch

from fsrl.infra.formal_runtime import require_formal_runtime
from fsrl.infra.runtime import ExecutionProfile, runtime_snapshot

PROFILE = ExecutionProfile(compile_fullgraph=True, compile_mode="default")


def configure_execution() -> dict:
    require_formal_runtime()
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    from torch._inductor import config

    config.compile_threads = 1
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = runtime_snapshot(PROFILE)
    snapshot["compiler_threads"] = config.compile_threads
    snapshot["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    return snapshot
