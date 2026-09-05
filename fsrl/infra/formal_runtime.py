"""Resource-bounded entry point for registered formal workflows."""

from __future__ import annotations

import os
import sys
from importlib import import_module

from fsrl.infra.runtime import (
    ExecutionProfile,
    configure_runtime,
    default_device,
    runtime_snapshot,
)

CPU_THREAD_LIMIT = 1
BLAS_THREAD_LIMIT = 1
ACTIVE_ENVIRONMENT_VARIABLE = "FSRL_FORMAL_RUNTIME_ACTIVE"
WORKFLOW_MODULES = {
    "confirmation": "fsrl.experiments.confirmation.behavioral",
    "mechanism": "fsrl.experiments.confirmation.mechanism",
    "joint-training-strategy": "fsrl.experiments.training_strategy.__main__",
    "minimal-relational-learner": "fsrl.experiments.minimal_learner.__main__",
    "quantized-relational-learner": "fsrl.experiments.quantized_learner.__main__",
    "global-policy-allocation-audit": "fsrl.experiments.global_policy.allocation_audit",
    "global-policy-amplitude-provenance": "fsrl.experiments.global_policy.amplitude_provenance",
    "global-policy-comparator-adequacy": "fsrl.experiments.global_policy.comparator_adequacy",
    "global-policy-field-fingerprint-replication": (
        "fsrl.experiments.global_policy.field_replication"
    ),
    "global-policy-field-reassembly": "fsrl.experiments.global_policy.field_reassembly",
    "global-policy-slope-localization": "fsrl.experiments.global_policy.slope_localization",
    "human-metric-constructive-comparator": (
        "fsrl.experiments.human.constructive_comparator"
    ),
    "liu-evidence-sparsity-transport": "fsrl.experiments.transport.evidence_sparsity",
    "liu-item-count-transport": "fsrl.experiments.transport.item_count",
    "liu-presentation-order-transport": "fsrl.experiments.transport.presentation_order",
    "liu-support-topology-transport": "fsrl.experiments.transport.topology",
}


def _formal_profile() -> ExecutionProfile:
    return ExecutionProfile(
        device=default_device(),
        cpu_threads=CPU_THREAD_LIMIT,
        blas_threads=BLAS_THREAD_LIMIT,
        compile=False,
        require_cuda=False,
    )


def configure_formal_runtime() -> dict:
    """Bound PyTorch and BLAS work before importing a formal workflow."""

    configure_runtime(_formal_profile())
    os.environ[ACTIVE_ENVIRONMENT_VARIABLE] = "1"
    return formal_runtime_snapshot()


def configure_formal_cuda_runtime() -> dict:
    """Configure the formal runtime and require CUDA for a direct workflow call."""

    snapshot = configure_formal_runtime()
    if not snapshot["cuda_available"] or snapshot["device"] != "cuda":
        raise RuntimeError("formal execution requires a visible CUDA GPU")
    return snapshot


def formal_runtime_snapshot() -> dict:
    profile = _formal_profile()
    snapshot = runtime_snapshot(profile)
    return {
        "execution_schema_version": snapshot["execution_schema_version"],
        "active": os.environ.get(ACTIVE_ENVIRONMENT_VARIABLE) == "1",
        "cpu_thread_limit": CPU_THREAD_LIMIT,
        "blas_thread_limit": BLAS_THREAD_LIMIT,
        "blas_threadpools": snapshot["blas_threadpools"],
        "torch_intraop_threads": snapshot["torch_intraop_threads"],
        "torch_interop_threads": snapshot["torch_interop_threads"],
        "torch_version": snapshot["torch_version"],
        "cuda_version": snapshot["cuda_version"],
        "cuda_available": snapshot["cuda_available"],
        "device": profile.device,
        "device_name": snapshot["device_name"],
        "device_capability": snapshot["device_capability"],
        "float32_matmul_precision": snapshot["float32_matmul_precision"],
        "deterministic_algorithms": snapshot["deterministic_algorithms"],
        "cudnn_benchmark": snapshot["cudnn_benchmark"],
        "cudnn_deterministic": snapshot["cudnn_deterministic"],
        "cudnn_allow_tf32": snapshot["cudnn_allow_tf32"],
    }


def require_formal_runtime() -> dict:
    snapshot = formal_runtime_snapshot()
    if not (
        snapshot["active"]
        and snapshot["torch_intraop_threads"] == CPU_THREAD_LIMIT
        and snapshot["torch_interop_threads"] == CPU_THREAD_LIMIT
        and snapshot["blas_thread_limit"] == BLAS_THREAD_LIMIT
        and snapshot["blas_threadpools"]
        and all(
            pool["num_threads"] == BLAS_THREAD_LIMIT
            for pool in snapshot["blas_threadpools"]
        )
        and snapshot["cuda_available"]
    ):
        raise RuntimeError(
            "formal execution requires the GPU and the bounded runtime entry point: "
            "python -m fsrl.infra.formal_runtime <registered-workflow> ..."
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
