"""Pre-outcome synthetic qualification for pair-morphology attribution."""

from __future__ import annotations

import inspect
from itertools import permutations

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.infra.provenance import write_json_exclusive
from fsrl.tasks.protocol import RankingProtocol

from .methods import (
    hodge_components,
    interface_checks,
    panel_stage,
    sample_pair_accuracies,
    study_outcome,
)
from .protocol import QUALIFICATION, register


def _protocol() -> RankingProtocol:
    return RankingProtocol(
        protocol_id="synthetic-pair-morphology",
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
    first = sample_pair_accuracies(
        protocol, (logits, logits), seed=17, temperature=0.25
    )
    second = sample_pair_accuracies(
        protocol, (logits, logits), seed=17, temperature=0.25
    )
    return np.array_equal(first, np.ones((2, 3))) and np.array_equal(first, second)


def _hodge_check() -> bool:
    geometry = build_complete_graph_geometry(_protocol())
    fields = np.asarray([[1.0, -0.5, 0.25], [-0.2, 0.8, 1.1]])
    gradient, residual = hodge_components(fields, geometry)
    inner = np.sum(gradient * residual, axis=1)
    return bool(
        np.max(np.abs(fields - gradient - residual)) <= 1e-12
        and np.max(np.abs(inner)) <= 1e-12
    )


def _interface_check() -> bool:
    codes = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    support_pairs = np.asarray([[[0, 1], [0, 1]], [[1, 2], [1, 2]]])
    query_pairs = np.asarray([[0, 1], [1, 0]])
    m2 = {
        "support_pairs": support_pairs,
        "query_pairs": query_pairs,
        "item_codes": codes,
        "signed_magnitudes": np.ones((2, 2)),
        "trial_retention": np.ones((2, 2)),
        "probabilities": np.ones((2, 2)),
        "local_evidence": np.ones((2, 2)),
    }
    subject_pairs = support_pairs.transpose(1, 0, 2)
    support_cues = np.concatenate(
        (
            codes[np.arange(2)[:, None], subject_pairs[..., 0]],
            codes[np.arange(2)[:, None], subject_pairs[..., 1]],
        ),
        axis=-1,
    ).transpose(1, 0, 2)
    repeated_query = np.broadcast_to(query_pairs[None], (2, 2, 2))
    query_cues = np.concatenate(
        (
            codes[np.arange(2)[:, None], repeated_query[..., 0]],
            codes[np.arange(2)[:, None], repeated_query[..., 1]],
        ),
        axis=-1,
    )
    score = {
        "support_pairs": subject_pairs,
        "query_pairs": repeated_query,
        "codes": codes.copy(),
        "support_cues": support_cues,
        "query_cues": query_cues,
        "signed": np.ones((2, 2)),
        "retention": np.ones((2, 2)),
        "probabilities": np.ones((2, 2)),
        "local_evidence": np.ones((2, 2)),
    }
    exact = interface_checks(m2, score)
    score["local_evidence"] = np.zeros((2, 2))
    changed = interface_checks(m2, score)
    return all(exact.values()) and not changed["realized_evidence_equal"]


def _no_torch_import() -> bool:
    from . import analysis, locks, methods, reporting

    sources = "\n".join(
        inspect.getsource(module) for module in (analysis, locks, methods, reporting)
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
    expected = {name: name for name in stages}
    checks = {
        "deterministic_choice_replay": _choice_check(),
        "hodge_reconstruction_and_orthogonality": _hodge_check(),
        "interface_exact_and_mismatch": _interface_check(),
        "all_stage_branches": stages == expected,
        "study_majority": study_outcome({"weak_margin": 40}) == "weak_margin",
        "study_mixed": study_outcome({"weak_margin": 39, "latent_shape": 21})
        == "mixed_or_unidentified",
        "analysis_has_no_torch_import": _no_torch_import(),
    }
    payload = {
        "schema_version": 1,
        "checks": checks,
        "sources": sources(),
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise RuntimeError(f"pair-morphology qualification failed: {checks}")
    write_json_exclusive(QUALIFICATION, payload)
    register(finding="Synthetic qualification passed; source/input locking pending.")
    return payload


__all__ = ["run_qualification"]
