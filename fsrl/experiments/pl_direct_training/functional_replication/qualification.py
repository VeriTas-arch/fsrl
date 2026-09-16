"""Pre-seed algebra, estimator, routing, horizon, and CUDA qualification."""

from __future__ import annotations

import numpy as np
import torch

from fsrl.analysis.statistics import bootstrap_counts
from fsrl.core.local_trace import PackedConjunctiveLocalTrace
from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    _paired as historical_paired,
)
from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    _probability_metrics as historical_probability_metrics,
)
from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    blockwise_derangements,
)
from fsrl.experiments.local_fidelity.trace_pilot import shuffled_pair_indices
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from ..qualification import cpu_qualification, cuda_qualification
from .estimands import paired_summary, probability_metrics
from .evaluation import probability_endpoint_parity_error
from .liu import FunctionalLiuEvaluator, admitted_local_evidence
from .locks import QUALIFICATION_ATTEMPT, implementation_sources, qualification_path
from .protocol import (
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    candidate_specification,
)

QUALIFICATION_SEED = 930001


def _check(error: float, tolerance: float, **values) -> dict:
    return {
        **values,
        "max_abs_error": error,
        "tolerance": tolerance,
        "passed": bool(np.isfinite(error) and error <= tolerance),
    }


def access_algebra_qualification() -> dict:
    rows = (
        (0.7, 1.0, 0.2),
        (-0.4, 0.0, 0.3),
        (0.9, 0.0, 0.8),
    )
    errors = []
    for signed, retained, probability in rows:
        shared = admitted_local_evidence(signed, retained, probability, "shared")
        dual = admitted_local_evidence(signed, retained, probability, "dual")
        errors.append(abs(shared - signed * retained))
        errors.append(abs(dual - signed * (retained + (1.0 - retained) * probability)))
        if retained == 1.0:
            errors.append(abs(dual - shared))
        else:
            errors.append(abs(dual - signed * probability))
    return _check(max(errors), 1e-12, cases=len(rows))


def presentation_invariance_qualification() -> dict:
    torch.manual_seed(QUALIFICATION_SEED)
    local = PackedConjunctiveLocalTrace(15, initial_gain=0.1, device="cpu")
    cues = torch.randn(12, 30)
    values = torch.linspace(-1.0, 1.0, 12)
    reversed_cues = torch.cat((cues[:, 15:], cues[:, :15]), dim=1)
    forward_state = local.write(local.initial_state(12), cues, values)
    reversed_state = local.write(local.initial_state(12), reversed_cues, -values)
    query = torch.randn(12, 30)
    reversed_query = torch.cat((query[:, 15:], query[:, :15]), dim=1)
    forward_read = local.read(forward_state, query)[0]
    reversed_read = local.read(forward_state, reversed_query)[0]
    error = max(
        float((forward_state - reversed_state).abs().max()),
        float((forward_read + reversed_read).abs().max()),
    )
    return _check(error, 1e-7)


def historical_estimand_parity_qualification() -> dict:
    rng = np.random.default_rng(QUALIFICATION_SEED)
    probabilities = rng.uniform(0.05, 0.95, size=(7, 8, 2))
    retention = np.asarray(
        [
            [(subject + relation) % 3 != 0 for relation in range(8)]
            for subject in range(7)
        ]
    )
    counts = bootstrap_counts(rng, 200, 7)
    current_probability = probability_metrics(probabilities, retention, counts, 0.95)
    historical_probability = historical_probability_metrics(
        probabilities, retention.T, counts, 0.95
    )
    first = rng.normal(size=7)
    second = rng.normal(size=7)
    current_paired = paired_summary(first, second, counts, 0.95)
    historical_difference = historical_paired(first, second, counts, 0.95)
    passed = (
        current_probability == historical_probability
        and current_paired == historical_difference
    )
    return {
        "probability_exact_match": current_probability == historical_probability,
        "paired_exact_match": current_paired == historical_difference,
        "passed": passed,
    }


def nullable_endpoint_parity_qualification() -> dict:
    probabilities = {
        "dual_access": {
            "raw_subject_level": {
                "retained": [None, 0.75, 0.60],
                "omitted": [0.55, None, 0.70],
            }
        }
    }
    endpoints = {
        "dual_access": {
            "probability": {
                "retained": np.asarray([np.nan, 0.75, 0.60]),
                "omitted": np.asarray([0.55, np.nan, 0.70]),
            }
        }
    }
    error = probability_endpoint_parity_error(probabilities, endpoints)
    return _check(error, 0.0, matched_missingness=True)


def routing_integrity_qualification() -> dict:
    subjects, blocks, block_size = 5, 4, 8
    maps = blockwise_derangements(subjects, blocks, block_size, QUALIFICATION_SEED)
    values = np.arange(subjects * blocks * block_size, dtype=np.float32).reshape(
        subjects, blocks * block_size
    )
    routed = FunctionalLiuEvaluator.route_evidence(values, maps)
    multiset_error = 0.0
    for subject in range(subjects):
        for block in range(blocks):
            start = block * block_size
            stop = start + block_size
            multiset_error = max(
                multiset_error,
                float(
                    np.max(
                        np.abs(
                            np.sort(values[subject, start:stop])
                            - np.sort(routed[subject, start:stop])
                        )
                    )
                ),
            )
    query_maps = shuffled_pair_indices(subjects, 8, QUALIFICATION_SEED + 1)
    evidence_deranged = bool(np.all(maps != np.arange(block_size)[None, None, :]))
    query_deranged = bool(
        np.all(query_maps[:, 0::2] // 2 != np.arange(query_maps.shape[1] // 2)[None, :])
    )
    return {
        "evidence_maps_are_derangements": evidence_deranged,
        "query_maps_are_derangements": query_deranged,
        "evidence_multiset_max_abs_error": multiset_error,
        "passed": evidence_deranged and query_deranged and multiset_error == 0.0,
    }


def cpu_functional_qualification() -> dict:
    checks = cpu_qualification(candidate_specification())
    checks.update(
        {
            "access_algebra": access_algebra_qualification(),
            "presentation_invariance": presentation_invariance_qualification(),
            "historical_estimand_parity": historical_estimand_parity_qualification(),
            "nullable_endpoint_parity": nullable_endpoint_parity_qualification(),
            "routing_integrity": routing_integrity_qualification(),
        }
    )
    return checks


def run_qualification() -> dict:
    specification = candidate_specification()
    directory = qualification_path().parent
    with ProspectiveRun.start(
        directory,
        workflow_id="pl_functional_replication_v1",
        execution_id=(
            f"qualification-{QUALIFICATION_SEED}-attempt{QUALIFICATION_ATTEMPT}"
        ),
        producer={
            "module": __name__,
            "protocol_sha256": PROTOCOL_SHA256,
            "repair_sha256": REPAIR_SHA256,
        },
        resolved_config={"seed": QUALIFICATION_SEED, "scientific_seed": False},
    ):
        checks = cpu_functional_qualification()
        cuda_checks, runtime = cuda_qualification(specification)
        checks.update(cuda_checks)
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "repair_sha256": REPAIR_SHA256,
            "seed": QUALIFICATION_SEED,
            "attempt": QUALIFICATION_ATTEMPT,
            "liu_evaluated": False,
            "runtime": runtime,
            "checks": checks,
            "sources": implementation_sources(),
            "passed": all(row["passed"] for row in checks.values()),
        }
        write_json_exclusive(directory / "qualification.json", result)
    return result
