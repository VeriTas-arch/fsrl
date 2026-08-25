"""Explicit execution profiles for training and registered evaluation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, cast

from threadpoolctl import threadpool_info, threadpool_limits

_ACTIVE_BLAS_LIMITER = None
_BLAS_ENVIRONMENT_VARIABLES = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class ExecutionProfile:
    """Runtime choices that can affect performance or numerical provenance."""

    device: str = "cuda"
    cpu_threads: int = 1
    blas_threads: int = 1
    compile: bool = True
    compile_backend: str = "inductor"
    compile_fullgraph: bool = True
    compile_mode: str = "default"
    require_cuda: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_COMPILED_PROFILE = ExecutionProfile()
CPU_TEST_PROFILE = ExecutionProfile(device="cpu", compile=False, require_cuda=False)


def default_device() -> str:
    """Return the best visible torch device without fixing it at import time."""

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_runtime(profile: ExecutionProfile) -> dict[str, Any]:
    """Apply one prospective runtime profile and return its observed snapshot."""

    if profile.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if profile.blas_threads < 1:
        raise ValueError("blas_threads must be positive")

    os.environ["OMP_NUM_THREADS"] = str(profile.cpu_threads)
    for name in _BLAS_ENVIRONMENT_VARIABLES:
        os.environ[name] = str(profile.blas_threads)

    import numpy as np
    import torch

    if profile.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("this execution profile requires a visible CUDA device")
    if profile.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    if profile.device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {profile.device}")

    torch.set_num_threads(profile.cpu_threads)
    if torch.get_num_interop_threads() != profile.cpu_threads:
        torch.set_num_interop_threads(profile.cpu_threads)

    # Load NumPy's BLAS before applying a process-global limit. Environment
    # variables above cover SciPy or another BLAS library loaded later.
    np.dot(np.ones((1, 1)), np.ones((1, 1)))
    global _ACTIVE_BLAS_LIMITER
    if _ACTIVE_BLAS_LIMITER is not None:
        _ACTIVE_BLAS_LIMITER.restore_original_limits()
    _ACTIVE_BLAS_LIMITER = threadpool_limits(
        limits=profile.blas_threads, user_api="blas"
    )
    return runtime_snapshot(profile)


def runtime_snapshot(profile: ExecutionProfile) -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "execution_schema_version": 2,
        "profile": profile.to_dict(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_name": (torch.cuda.get_device_name(0) if cuda_available else None),
        "device_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "blas_thread_limit": profile.blas_threads,
        "blas_threadpools": [
            {
                key: pool.get(key)
                for key in (
                    "architecture",
                    "internal_api",
                    "num_threads",
                    "prefix",
                    "threading_layer",
                    "user_api",
                    "version",
                )
            }
            for pool in threadpool_info()
            if pool.get("user_api") == "blas"
        ],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def compile_module[Compiled](module: Compiled, profile: ExecutionProfile) -> Compiled:
    """Compile a module using the exact profile recorded in run provenance."""

    if not profile.compile:
        return module
    import torch

    compiler = cast(Any, torch.compile)
    return cast(
        Compiled,
        compiler(
            module,
            backend=profile.compile_backend,
            fullgraph=profile.compile_fullgraph,
            mode=profile.compile_mode,
        ),
    )
