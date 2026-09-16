"""Exact eager float32 runtime settings for the direct arithmetic authority."""

from __future__ import annotations

import torch

from fsrl.infra.formal_runtime import require_formal_runtime


def configure_authority_runtime() -> dict:
    require_formal_runtime()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("direct authority forbids autocast")
    snapshot = {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "autocast_enabled": torch.is_autocast_enabled(),
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "execution_mode": "eager",
        "dtype": "float32",
    }
    required = {
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "float32_matmul_precision": "highest",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
        "autocast_enabled": False,
        "cuda_autocast_enabled": False,
        "execution_mode": "eager",
        "dtype": "float32",
    }
    if any(snapshot[name] != value for name, value in required.items()):
        raise RuntimeError("direct-authority runtime settings differ")
    return snapshot


__all__ = ["configure_authority_runtime"]
