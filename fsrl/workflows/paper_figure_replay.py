"""Read-only model replay used to materialize paper-figure source cells."""

from __future__ import annotations

import hashlib
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import load_ranking_protocol
from fsrl.workflows.paper_figure_contract import (
    MODEL_RESULT_PATH,
    PROTOCOL_PATH,
    REPLAY_CSV_PATH,
    REPLAY_MANIFEST_PATH,
    SPECIFICATION_PATH,
    _write_csv,
    _write_json,
    validate_specification,
)

ROOT = REPO_ROOT


def _sample_pair_accuracies(protocol, subject_logits, *, seed: int, temperature: float):
    pairs = tuple(combinations(range(protocol.n_items), 2))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    output = np.zeros((len(subject_logits), len(pairs)), dtype=np.float64)
    for subject_index, logits in enumerate(subject_logits):
        schedule_rng = np.random.default_rng(seed + 2 * subject_index)
        choice_rng = np.random.default_rng(seed + 2 * subject_index + 1)
        correct = np.zeros(len(pairs), dtype=np.float64)
        total = np.zeros(len(pairs), dtype=np.float64)
        for trial in protocol.query_schedule(schedule_rng):
            oriented = (trial.left_item, trial.right_item)
            probability_left = 1.0 / (
                1.0 + np.exp(-float(logits[oriented]) / temperature)
            )
            choose_left = bool(choice_rng.random() < probability_left)
            pair = tuple(sorted(oriented))
            index = pair_index[pair]
            correct[index] += float(choose_left == bool(trial.correct_action))
            total[index] += 1.0
        output[subject_index] = correct / total
    return output


def _strip_bootstrap(result: dict) -> dict:
    clean = dict(result)
    clean.pop("participant_bootstrap", None)
    return clean


def replay_model_subject_pairs(
    csv_path: Path = REPLAY_CSV_PATH,
    manifest_path: Path = REPLAY_MANIFEST_PATH,
) -> dict:
    """Replay only the matched v2.4 query field and export sampled pair accuracy."""

    validation = validate_specification()
    specification = validation["specification"]

    import torch

    import fsrl.experiments.local_fidelity.evidence_access_pilot as dual_access
    from fsrl.analysis.policy import bundle_logits
    from fsrl.evaluation.frozen_fast_weight import (
        FastWeightIntervention,
        FrozenFastWeightEvaluator,
        load_frozen_retro_checkpoint,
    )
    from fsrl.evaluation.relational_query import readout_relational_query_bundle
    from fsrl.experiments.local_fidelity.evidence_access_confirmation import (
        DEFAULT_ARTIFACT_LOCK_PATH,
        DEFAULT_IMPLEMENTATION_LOCK_PATH,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_SPECIFICATION_PATH,
        validate_artifacts,
        validate_sources,
    )
    from fsrl.experiments.local_fidelity.trace_pilot import (
        create_local_trace,
    )
    from fsrl.experiments.local_fidelity.trace_replication import (
        seed_paths,
        seed_specification,
    )
    from fsrl.infra.formal_runtime import configure_formal_cuda_runtime
    from fsrl.tasks.protocol import ordered_pairs

    runtime = configure_formal_cuda_runtime()
    source_validation = validate_sources()
    frozen_specification = load_json(DEFAULT_SPECIFICATION_PATH)
    artifact_validation = validate_artifacts(
        frozen_specification,
        DEFAULT_SPECIFICATION_PATH,
        DEFAULT_IMPLEMENTATION_LOCK_PATH,
        DEFAULT_ARTIFACT_LOCK_PATH,
        DEFAULT_OUTPUT_ROOT,
    )
    frozen_result = load_json(MODEL_RESULT_PATH)
    evaluation = frozen_specification["liu_evaluation"]
    protocol = load_ranking_protocol(PROTOCOL_PATH)
    pairs = tuple(combinations(range(protocol.n_items), 2))
    labels = protocol.item_labels
    rank = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    rows = []
    seed_checks = {}
    for seed in specification["model"]["network_seeds"]:
        paths = seed_paths(DEFAULT_OUTPUT_ROOT, int(seed))
        gain = load_json(paths["gain"])
        backbone, model_config, checkpoint = load_frozen_retro_checkpoint(
            paths["checkpoint"], int(evaluation["subjects"])
        )
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        local = create_local_trace(
            seed_specification(frozen_specification, int(seed)), model_config.cs
        )
        with torch.no_grad():
            local.raw_gain.fill_(float(gain["raw_lambda_L"]))
        evaluator = FrozenFastWeightEvaluator(
            backbone,
            model_config,
            protocol,
            cue_seed=int(evaluation["cue_seed"]),
            support_seed=int(evaluation["support_seed"]),
            cue_mode=str(evaluation["cue_mode"]),
            subject_encoding_mode=str(evaluation["subject_encoding_mode"]),
            subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
        )
        schedules = tuple(
            ordered_pairs(protocol.n_items) for _ in range(model_config.bs)
        )
        fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        trace = dual_access.build_access_trace(evaluator, local, dual_access=True)
        bundle = readout_relational_query_bundle(
            evaluator,
            local,
            fast_weights,
            trace.state,
            schedules,
            local_off=False,
            global_off=False,
            shuffled_indices=None,
        )
        subject_logits = bundle_logits(bundle, schedules)
        replayed = analyze_sampled_query_policy(
            protocol,
            subject_logits,
            seed=int(evaluation["choice_seed"]),
            temperature=float(evaluation["temperature"]),
        )
        frozen = frozen_result["seeds"][str(seed)]["behavior"][
            specification["model"]["condition"]
        ]
        if replayed != _strip_bootstrap(frozen):
            raise RuntimeError(f"seed {seed} sampled behavior does not replay exactly")
        matrix = _sample_pair_accuracies(
            protocol,
            subject_logits,
            seed=int(evaluation["choice_seed"]),
            temperature=float(evaluation["temperature"]),
        )
        stored_means = np.asarray(
            [row["mean_accuracy_all"] for row in replayed["pairs"]],
            dtype=np.float64,
        )
        mean_error = float(np.max(np.abs(np.mean(matrix, axis=0) - stored_means)))
        if mean_error > 1e-12:
            raise RuntimeError(f"seed {seed} pair means fail replay: {mean_error}")
        for subject, values in enumerate(matrix):
            for index, pair in enumerate(pairs):
                rows.append(
                    {
                        "network_seed": seed,
                        "subject": subject,
                        "pair_index": index,
                        "item_1": labels[pair[0]],
                        "item_2": labels[pair[1]],
                        "learned": pair in protocol.learned_pairs,
                        "symbolic_distance": abs(rank[pair[0]] - rank[pair[1]]),
                        "pair_accuracy": values[index],
                    }
                )
        seed_checks[str(seed)] = {
            "checkpoint_sha256": checkpoint.sha256,
            "gain_sha256": file_sha256(paths["gain"]),
            "subjects": int(matrix.shape[0]),
            "pairs": int(matrix.shape[1]),
            "stored_behavior_exact_match": True,
            "pair_mean_max_abs_error": mean_error,
        }

    fieldnames = [
        "network_seed",
        "subject",
        "pair_index",
        "item_1",
        "item_2",
        "learned",
        "symbolic_distance",
        "pair_accuracy",
    ]
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / csv_path.name
        _write_csv(candidate, fieldnames, rows)
        candidate_bytes = candidate.read_bytes()
    if csv_path.exists() and csv_path.read_bytes() != candidate_bytes:
        raise RuntimeError("existing pair replay differs; refusing to overwrite")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(candidate_bytes)
    manifest = {
        "schema_version": 1,
        "replay_id": "published-behavior-alignment-pair-replay-v1",
        "execution": "minimal read-only dual_access_matched query replay",
        "scientific_estimand": "none; exports the sampled subject-by-pair cells used by already-frozen behavioral summaries",
        "figure_specification_sha256": file_sha256(SPECIFICATION_PATH),
        "model_result_sha256": file_sha256(MODEL_RESULT_PATH),
        "source_validation_passed": bool(source_validation["passed"]),
        "artifact_validation_passed": bool(artifact_validation["passed"]),
        "runtime": {
            name: runtime[name]
            for name in (
                "device",
                "device_name",
                "torch_version",
                "cuda_version",
                "torch_intraop_threads",
                "torch_interop_threads",
            )
        },
        "network_pooling": "not_performed",
        "seeds": seed_checks,
        "output": {
            "path": str(csv_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "rows": len(rows),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest
