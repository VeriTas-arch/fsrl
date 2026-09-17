"""Pre-outcome synthetic qualification for the historical morphology re-audit."""

from __future__ import annotations

import inspect
from itertools import permutations

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.experiments.minimal_single_p_pair_morphology.analysis import _replay_route
from fsrl.experiments.minimal_single_p_pair_morphology.methods import (
    hodge_components,
    panel_stage,
    sample_pair_accuracies,
)
from fsrl.infra.provenance import write_json_exclusive
from fsrl.tasks.protocol import RankingProtocol

from .decisions import constrained_unit, error_inflation, family_replicated, outcome
from .locks import reference
from .protocol import QUALIFICATION, REPAIR, REPAIR_SHA256, register


def _protocol() -> RankingProtocol:
    return RankingProtocol(
        protocol_id="synthetic-historical-morphology",
        item_labels=("A", "B", "C"),
        true_order_high_to_low=(0, 1, 2),
        support_pairs_higher_lower=((0, 1), (1, 2)),
        support_blocks=1,
        query_blocks=10,
        human_targets={},
    )


def _choice_check() -> bool:
    protocol = _protocol()
    logits = {
        pair: (100.0 if pair[0] < pair[1] else -100.0)
        for pair in permutations(range(3), 2)
    }
    accuracy = sample_pair_accuracies(
        protocol, (logits, logits), seed=17, temperature=0.25
    )
    return np.array_equal(accuracy, np.ones((2, 3)))


def _hodge_check() -> bool:
    geometry = build_complete_graph_geometry(_protocol())
    fields = np.asarray([[1.0, -0.5, 0.25], [-0.2, 0.8, 1.1]])
    gradient, residual = hodge_components(fields, geometry)
    return bool(
        np.max(np.abs(fields - gradient - residual)) <= 1e-12
        and np.max(np.abs(np.sum(gradient * residual, axis=1))) <= 1e-12
    )


def _decision_check() -> bool:
    units = [
        {"seed": seed, "constrained_morphology": seed != 2}
        for seed in (1, 2, 3)
    ]
    units.append({"seed": 2, "constrained_morphology": True})
    return all(
        (
            constrained_unit(
                competent=True,
                binding=True,
                nine=True,
                latent_bimodal=15,
                sampled_bimodal=15,
                bad_pair=False,
            ),
            not constrained_unit(
                competent=True,
                binding=True,
                nine=True,
                latent_bimodal=14,
                sampled_bimodal=15,
                bad_pair=False,
            ),
            error_inflation(
                sampled_bimodal=16,
                control_sampled_bimodal=15,
                nine=False,
                bad_pair=False,
            ),
            not error_inflation(
                sampled_bimodal=16,
                control_sampled_bimodal=15,
                nine=True,
                bad_pair=False,
            ),
            family_replicated(units, [1, 2, 3]),
            outcome(0, 0) == "no_current_standard_precedent",
            outcome(1, 0) == "isolated_current_standard_precedent",
            outcome(3, 1) == "single_family_current_standard_precedent",
            outcome(6, 2) == "cross_family_current_standard_precedent",
        )
    )


def _no_torch_import() -> bool:
    from . import analysis, decisions, locks, reporting

    sources = "\n".join(
        inspect.getsource(module) for module in (analysis, decisions, locks, reporting)
    )
    return "import torch" not in sources and "torch.load" not in sources


def run_qualification() -> dict:
    from .locks import sources

    stages = {
        "direction_absent": panel_stage(14, 14, 14, 14),
        "weak_margin": panel_stage(15, 14, 14, 14),
        "latent_shape": panel_stage(15, 15, 14, 14),
        "finite_sampling": panel_stage(15, 15, 15, 14),
        "already_latent_and_sampled": panel_stage(15, 15, 15, 15),
    }
    checks = {
        "deterministic_choice_replay": _choice_check(),
        "replay_function_available": callable(_replay_route),
        "hodge_reconstruction_and_orthogonality": _hodge_check(),
        "all_stage_branches": stages == {name: name for name in stages},
        "all_decision_branches_and_inclusive_thresholds": _decision_check(),
        "analysis_has_no_torch_import": _no_torch_import(),
    }
    payload = {
        "schema_version": 1,
        "repair": reference(REPAIR),
        "checks": checks,
        "sources": sources(),
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise RuntimeError(f"historical morphology qualification failed: {checks}")
    if payload["repair"]["sha256"] != REPAIR_SHA256:
        raise RuntimeError("historical morphology repair identity differs")
    write_json_exclusive(QUALIFICATION, payload)
    register(finding="Synthetic qualification passed; source/input locking pending.")
    return payload


__all__ = ["run_qualification"]
