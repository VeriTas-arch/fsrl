"""Explicit execution profiles for training and registered evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionProfile:
    """Runtime choices that can affect performance or numerical provenance."""

    device: str = "cuda"
    cpu_threads: int = 1
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

    import torch

    if profile.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if profile.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("this execution profile requires a visible CUDA device")
    if profile.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    if profile.device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {profile.device}")

    torch.set_num_threads(profile.cpu_threads)
    if torch.get_num_interop_threads() != profile.cpu_threads:
        torch.set_num_interop_threads(profile.cpu_threads)
    return runtime_snapshot(profile)


def runtime_snapshot(profile: ExecutionProfile) -> dict[str, Any]:
    import torch

    return {
        "profile": profile.to_dict(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def compile_module(module, profile: ExecutionProfile):
    """Compile a module using the exact profile recorded in run provenance."""

    if not profile.compile:
        return module
    import torch

    return torch.compile(
        module,
        backend=profile.compile_backend,
        fullgraph=profile.compile_fullgraph,
        mode=profile.compile_mode,
    )
