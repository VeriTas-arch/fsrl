"""Resource-bounded entry point for registered formal workflows."""

from __future__ import annotations

import os
import sys
from importlib import import_module

CPU_THREAD_LIMIT = 1
ACTIVE_ENVIRONMENT_VARIABLE = "FSRL_FORMAL_RUNTIME_ACTIVE"
WORKFLOW_MODULES = {
    "confirmation": "fsrl.confirmation",
    "mechanism": "fsrl.mechanism_confirmation",
    "global-policy-allocation-audit": "fsrl.global_policy_allocation_audit",
    "global-policy-amplitude-provenance": "fsrl.global_policy_amplitude_provenance",
    "global-policy-comparator-adequacy": "fsrl.global_policy_comparator_adequacy",
    "global-policy-field-fingerprint-replication": (
        "fsrl.global_policy_field_fingerprint_replication"
    ),
    "global-policy-field-reassembly": "fsrl.global_policy_field_reassembly",
    "global-policy-slope-localization": "fsrl.global_policy_slope_localization",
    "human-metric-constructive-comparator": (
        "fsrl.human_metric_constructive_comparator"
    ),
    "liu-support-topology-transport": "fsrl.support_topology_transport",
}


def configure_formal_runtime() -> dict:
    """Bound PyTorch CPU work without changing NumPy/BLAS reductions."""

    import torch

    torch.set_num_threads(CPU_THREAD_LIMIT)
    if torch.get_num_interop_threads() != CPU_THREAD_LIMIT:
        torch.set_num_interop_threads(CPU_THREAD_LIMIT)
    os.environ[ACTIVE_ENVIRONMENT_VARIABLE] = "1"
    return formal_runtime_snapshot()


def formal_runtime_snapshot() -> dict:
    import torch

    return {
        "active": os.environ.get(ACTIVE_ENVIRONMENT_VARIABLE) == "1",
        "cpu_thread_limit": CPU_THREAD_LIMIT,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def require_formal_runtime() -> dict:
    snapshot = formal_runtime_snapshot()
    if not (
        snapshot["active"]
        and snapshot["torch_intraop_threads"] == CPU_THREAD_LIMIT
        and snapshot["torch_interop_threads"] == CPU_THREAD_LIMIT
        and snapshot["cuda_available"]
    ):
        raise RuntimeError(
            "formal execution requires the GPU and the bounded runtime entry point: "
            "python -m fsrl.formal_runtime <registered-workflow> ..."
        )
    return snapshot


def main(args=None) -> int:
    arguments = list(sys.argv[1:] if args is None else args)
    if not arguments or arguments[0] not in WORKFLOW_MODULES:
        raise ValueError("first argument must select a registered formal workflow")
    workflow = arguments.pop(0)
    configure_formal_runtime()
    workflow_main = import_module(WORKFLOW_MODULES[workflow]).main
    return workflow_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
