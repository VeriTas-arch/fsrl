"""Pre-outcome qualification, including complete parent-stream replay."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.clean_single_p.batches import prepare_single_p
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive

from .algebra import recurrence_and_kernel
from .data import visible_batch
from .decisions import generic_category, generic_panel_passed, liu_evidence_binding
from .locks import PARENT_PROMOTION_LOCK, sources
from .model import QOnlyScore, physical_parameters
from .protocol import (
    PAIR_TABLE,
    PARAMETERS,
    PROTOCOL_SHA256,
    QUALIFICATION,
    REPAIR2_QUALIFICATION,
    REPAIR3,
    REPAIR3_QUALIFICATION,
    REPAIR_QUALIFICATION,
    REPORT,
    RESULT,
    specification,
)


def _fixture() -> tuple[torch.Tensor, ...]:
    rng = np.random.default_rng(910101)
    support = rng.normal(size=(7, 5, 30)).astype(np.float32)
    q = rng.normal(size=(7, 5)).astype(np.float32)
    query = rng.normal(size=(5, 9, 30)).astype(np.float32)
    targets = rng.integers(0, 2, size=(5, 9), dtype=np.int64)
    return tuple(
        torch.as_tensor(value, device="cuda") for value in (support, q, query, targets)
    )


def _model_checks() -> dict:
    support, q, query, targets = _fixture()
    eager = QOnlyScore(device="cuda")
    compiled = QOnlyScore(device="cuda")
    compiled.load_state_dict(eager.state_dict())
    runner = torch.compile(compiled, backend="inductor", fullgraph=True, mode="default")
    eager_margin, eager_w = eager(support, q, query)
    compiled_margin, compiled_w = runner(support, q, query)
    torch.testing.assert_close(eager_margin, compiled_margin, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(eager_w, compiled_w, atol=1e-5, rtol=1e-4)
    eager_loss = F.softplus(-(2 * targets - 1) * eager_margin).mean()
    compiled_loss = F.softplus(-(2 * targets - 1) * compiled_margin).mean()
    eager_loss.backward()
    compiled_loss.backward()
    for (_, left), (_, right) in zip(
        eager.named_parameters(), compiled.named_parameters(), strict=True
    ):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-4)
    left_opt = torch.optim.Adam(eager.parameters(), lr=0.01)
    right_opt = torch.optim.Adam(compiled.parameters(), lr=0.01)
    torch.nn.utils.clip_grad_norm_(eager.parameters(), 2.0)
    torch.nn.utils.clip_grad_norm_(compiled.parameters(), 2.0)
    left_opt.step()
    right_opt.step()
    for (_, left), (_, right) in zip(
        eager.named_parameters(), compiled.named_parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)

    numpy_margin, kernel, reconstructed = recurrence_and_kernel(
        support.cpu().numpy(),
        q.cpu().numpy(),
        query.cpu().numpy(),
        eta=0.5,
        gamma=1.0,
        epsilon=1e-8,
    )
    np.testing.assert_allclose(numpy_margin, reconstructed, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        eager_margin.detach().cpu().numpy(), numpy_margin, atol=1e-5, rtol=1e-5
    )

    one_cue = torch.zeros((2, 1, 30), device="cuda")
    one_cue[:, 0, 0] = 1
    one_cue[:, 0, 15] = -1
    zero_q = torch.tensor([[1.0], [0.0]], device="cuda")
    _, corrected = QOnlyScore(device="cuda")(
        one_cue, zero_q, one_cue[:1].transpose(0, 1)
    )
    if not 0.0 < float(corrected[0, 0].detach()) < 0.5:
        raise RuntimeError("q=0 did not correct the existing prediction toward zero")
    reversed_query = torch.cat((query[..., 15:], query[..., :15]), dim=-1)
    current_margin = eager(support, q, query)[0]
    reverse_margin = eager(support, q, reversed_query)[0]
    torch.testing.assert_close(reverse_margin, -current_margin, atol=1e-6, rtol=1e-6)
    if sum(value.numel() for value in eager.parameters()) != 2:
        raise RuntimeError("aligned comparator parameter count differs")
    return {
        "eager_compiled_margin": True,
        "eager_compiled_state": True,
        "gradient_and_adam": True,
        "float64_recurrence_max_abs": float(
            np.max(np.abs(eager_margin.detach().cpu().numpy() - numpy_margin))
        ),
        "float64_kernel_max_abs": float(np.max(np.abs(numpy_margin - reconstructed))),
        "kernel_shape": list(kernel.shape),
        "q_zero_correction": True,
        "orientation_reversal": True,
        "parameters": physical_parameters(eager),
    }


def _visibility_checks() -> dict:
    rng = np.random.default_rng(910102)
    arrays = {
        "support_inputs": rng.normal(size=(8, 4, 3, 38)).astype(np.float32),
        "query_inputs": rng.normal(size=(2, 15, 38)).astype(np.float32),
        "targets": rng.integers(0, 2, size=15, dtype=np.int64),
        "retention": rng.integers(0, 2, size=(3, 8), dtype=np.int8),
        "probabilities": rng.random((8, 3)),
        "support_pairs": rng.integers(0, 8, size=(8, 3, 2)),
    }
    first = visible_batch(arrays)
    changed = {key: value.copy() for key, value in arrays.items()}
    for key in ("retention", "probabilities", "support_pairs"):
        changed[key][...] = 0
    second = visible_batch(changed)
    for name in ("support_cues", "realized_q", "query_cues", "targets"):
        if not np.array_equal(getattr(first, name), getattr(second, name)):
            raise RuntimeError(f"forbidden metadata changed visible {name}")
    return {
        "visible_fields": ["support_cues", "realized_q", "query_cues"],
        "outer_loss_only": ["targets"],
        "metadata_blind": True,
    }


def _replay_streams() -> dict:
    parent = load_json(PARENT_PROMOTION_LOCK)
    task_spec = parent["task"]
    results = {}
    for seed in specification()["design"]["training_streams"]:
        path = (
            PARENT_PROMOTION_LOCK.parents[4]
            / f"artifacts/runs/minimal_single_p_promotion_v1/training/{seed}/train_log.jsonl"
        )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 1500:
            raise RuntimeError(f"parent training log length differs: {seed}")
        task = make_task_generator({"task": task_spec})
        rng = np.random.default_rng(151000 + seed)
        digest = hashlib.sha256()
        for step, row in enumerate(rows):
            episodes = sample_episodes(task, rng, 32, validation=False)
            batch = prepare_single_p(
                episodes,
                "clean",
                observation_seed=910000000 + seed * 10000 + step,
            )
            fingerprint = batch.fingerprint()
            if fingerprint != row["batch_fingerprint"]:
                raise RuntimeError(f"parent replay differs: {seed}/{step}")
            digest.update(bytes.fromhex(fingerprint))
            visible_batch(batch.arrays)
        if digest.hexdigest() != rows[-1]["stream_fingerprint"]:
            raise RuntimeError(f"parent cumulative stream differs: {seed}")
        results[str(seed)] = {
            "steps": 1500,
            "stream_fingerprint": digest.hexdigest(),
            "all_batch_fingerprints_equal": True,
        }
        print(f"qualified parent stream {seed}", flush=True)
    return results


def qualify() -> dict:
    if QUALIFICATION.exists():
        raise RuntimeError("qualification record already exists")
    if (
        generic_category(0) != "noncompetent"
        or generic_category(2) != "panel_variable"
        or generic_category(3) != "stable_competent"
    ):
        raise RuntimeError("generic decision fixtures failed")
    model = _model_checks()
    visibility = _visibility_checks()
    replay = _replay_streams()
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "passed": True,
        "sources": sources(),
        "model": model,
        "visibility": visibility,
        "stream_replay": replay,
        "decisions": True,
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(QUALIFICATION, payload)
    return {
        "passed": True,
        "qualified_streams": len(replay),
        "source_files": len(payload["sources"]),
    }


def qualify_repair() -> dict:
    if REPAIR_QUALIFICATION.exists():
        raise RuntimeError("repair qualification record already exists")

    def estimate(lower):
        return {"bootstrap": {"lower": lower}}

    passing = {
        "competence": {"learned": estimate(0.6), "nonlearned": estimate(0.6)},
        "state_dependence": {
            "learned": estimate(0.1),
            "nonlearned": estimate(0.1),
        },
        "evidence_binding": {
            "learned": estimate(0.1),
            "nonlearned": estimate(0.1),
        },
    }
    failing = {
        **passing,
        "evidence_binding": {
            "learned": estimate(0.0),
            "nonlearned": estimate(0.1),
        },
    }
    if not generic_panel_passed(passing) or generic_panel_passed(failing):
        raise RuntimeError("repaired generic schema decision fixtures failed")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "passed": True,
        "sources": sources(),
        "repair_scope": "estimate.bootstrap.lower schema access only",
        "passing_fixture": True,
        "threshold_failure_fixture": True,
        "parent_stream_replay_reused": True,
        "model_lock_unchanged": True,
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(REPAIR_QUALIFICATION, payload)
    return {"passed": True, "source_files": len(payload["sources"])}


def qualify_repair2() -> dict:
    if REPAIR2_QUALIFICATION.exists():
        raise RuntimeError("repair2 qualification record already exists")

    def result(lower):
        return {
            "liu": {
                "effects": {
                    "intact_minus_evidence_shuffle_learned": {
                        "bootstrap": {"lower": lower}
                    }
                }
            }
        }

    if not liu_evidence_binding(result(0.1)) or liu_evidence_binding(result(0.0)):
        raise RuntimeError("repaired Liu binding schema fixtures failed")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "passed": True,
        "sources": sources(),
        "repair_scope": "effect.bootstrap.lower schema access only",
        "passing_fixture": True,
        "threshold_failure_fixture": True,
        "parent_stream_replay_reused": True,
        "generic_result_unchanged": True,
        "model_lock_unchanged": True,
        "generic_outcomes_exposed": True,
        "liu_outcomes_exposed": False,
    }
    write_json_exclusive(REPAIR2_QUALIFICATION, payload)
    return {"passed": True, "source_files": len(payload["sources"])}


def qualify_repair3() -> dict:
    if REPAIR3_QUALIFICATION.exists():
        raise RuntimeError("repair3 qualification record already exists")
    contract = load_json(REPAIR3)
    observed = {
        "result": file_sha256(RESULT),
        "pairs": file_sha256(PAIR_TABLE),
        "parameters": file_sha256(PARAMETERS),
        "report": file_sha256(REPORT),
    }
    if observed != contract["frozen_artifact_sha256"]:
        raise RuntimeError("repair3 changed a frozen scientific artifact")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "passed": True,
        "sources": sources(),
        "repair_scope": "post-result engineering decomposition and test rename only",
        "frozen_artifact_sha256": observed,
        "scientific_evaluation_rerun": False,
        "scientific_artifacts_unchanged": True,
        "model_lock_unchanged": True,
        "scientific_outcomes_exposed": True,
    }
    write_json_exclusive(REPAIR3_QUALIFICATION, payload)
    return {"passed": True, "source_files": len(payload["sources"])}


__all__ = ["qualify", "qualify_repair", "qualify_repair2", "qualify_repair3"]
