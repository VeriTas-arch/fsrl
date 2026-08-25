"""Render paper-aligned human/model behavioral figures from frozen evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

from fsrl.infra.provenance import file_sha256, load_json

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from fsrl.analysis.behavioral import analyze_sampled_query_policy, kendall_tau_positions
from fsrl.experiments.human.benchmark import (
    DEFAULT_FIGURE2D_PATH,
    DEFAULT_FIGURE3B_PATH,
    DEFAULT_PREREGISTERED_PATH,
    DEFAULT_REPLICATION_PATH,
    SOURCE_FILES,
    load_human_cohort,
    load_published_figure_checks,
)
from fsrl.infra.study_registry import ROOT, resolve_record
from fsrl.tasks.registered_protocol import load_ranking_protocol

SUITE_ROOT = ROOT / "synthesis" / "figures" / "paper_alignment"
SPECIFICATION_PATH = SUITE_ROOT / "figure_spec.json"
REPLAY_CSV_PATH = SUITE_ROOT / "source" / "model_subject_pair_accuracy.csv"
REPLAY_MANIFEST_PATH = (
    SUITE_ROOT / "source" / "model_subject_pair_accuracy.manifest.json"
)
MODEL_RESULT_PATH = resolve_record(
    "results/dual_evidence_access_confirmation_v2_4.json"
)
HUMAN_BENCHMARK_PATH = resolve_record("benchmarks/liu_human_exact_v1.json")
PROTOCOL_PATH = resolve_record("benchmarks/liu_v2.json")

DATASET_ORDER = ("human", "2104", "2105")
DATASET_LABELS = {
    "human": "Human",
    "2104": "Network 2104",
    "2105": "Network 2105",
}
DATASET_COLORS = {
    "human": "#333333",
    "2104": "#2878B5",
    "2105": "#D95F02",
}
PAIR_CLASS_ORDER = (
    "high_accuracy",
    "bimodal",
    "ordinary_unimodal",
    "low_accuracy",
    "boundary",
    "not_fit",
)
PAIR_CLASS_COLORS = {
    "high_accuracy": "#A6761D",
    "bimodal": "#1B9E77",
    "ordinary_unimodal": "#BDBDBD",
    "low_accuracy": "#D95F02",
    "boundary": "#7570B3",
    "not_fit": "#F2F2F2",
}
ITEM_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
)


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    subjects: list[dict]
    pair_accuracy: np.ndarray
    pair_rows: list[dict]

    @property
    def eligible(self) -> np.ndarray:
        return np.asarray(
            [subject["overall_accuracy"] >= 0.5 for subject in self.subjects]
        )

    @property
    def analysis(self) -> np.ndarray:
        return self.eligible & np.asarray(
            [subject["ranking_class"] != "correct" for subject in self.subjects]
        )


def _repo_path(value: str) -> Path:
    return ROOT / value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _format_cell(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".12g")
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return value


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: _format_cell(row.get(name, "")) for name in fieldnames}
            )


def validate_specification(path: Path = SPECIFICATION_PATH) -> dict:
    specification = load_json(path)
    if specification.get("schema_version") != 1:
        raise RuntimeError("paper-alignment specification schema must be 1")
    if tuple(specification["model"]["network_seeds"]) != (2104, 2105):
        raise RuntimeError("paper alignment requires the two frozen networks")
    registrations = {
        "paper": specification["paper"],
        "human_benchmark": specification["human"]["benchmark"],
        "model_result": specification["model"]["result"],
        "model_specification": specification["model"]["specification"],
        "model_artifact_lock": specification["model"]["artifact_lock"],
    }
    checks = []
    for name, registration in registrations.items():
        path_value = registration.get("path", registration.get("local_pdf"))
        source_path = _repo_path(path_value)
        observed = file_sha256(source_path)
        expected = registration["sha256"]
        checks.append(
            {
                "name": name,
                "path": path_value,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"paper-alignment source validation failed: {checks}")
    return {"specification": specification, "checks": checks, "passed": True}


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
    from fsrl.evaluation.frozen_fast_weight import (
        FastWeightIntervention,
        FrozenFastWeightEvaluator,
        load_retro_checkpoint,
    )
    from fsrl.experiments.local_fidelity.curvature_gate_pilot import (
        bundle_logits,
        configure_runtime,
    )
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
        query_pass,
    )
    from fsrl.experiments.local_fidelity.trace_replication import (
        seed_paths,
        seed_specification,
    )
    from fsrl.tasks.protocol import ordered_pairs

    runtime = configure_runtime()
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
        backbone, model_config, checkpoint = load_retro_checkpoint(
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
        bundle = query_pass(
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


def _load_human_subjects(protocol) -> tuple[list[dict], dict, dict]:
    preregistered = load_human_cohort(
        DEFAULT_PREREGISTERED_PATH,
        "preregistered",
        protocol,
        expected_sha256=SOURCE_FILES["preregistered"]["sha256"],
    )
    replication = load_human_cohort(
        DEFAULT_REPLICATION_PATH,
        "replication",
        protocol,
        expected_sha256=SOURCE_FILES["replication"]["sha256"],
    )
    published = load_published_figure_checks(
        DEFAULT_FIGURE2D_PATH, DEFAULT_FIGURE3B_PATH, protocol
    )
    subjects = preregistered + replication
    for combined_id, (subject, ranking_class) in enumerate(
        zip(subjects, published["ranking_classes_by_combined_id"], strict=True),
        start=1,
    ):
        subject["combined_id"] = combined_id
        subject["ranking_class_trial_majority"] = subject["ranking_class"]
        subject["ranking_class"] = ranking_class
    return subjects, published, load_json(HUMAN_BENCHMARK_PATH)


def _load_replay_matrix(
    seed: int, pairs: tuple[tuple[int, int], ...], labels
) -> np.ndarray:
    manifest = load_json(REPLAY_MANIFEST_PATH)
    if file_sha256(REPLAY_CSV_PATH) != manifest["output"]["sha256"]:
        raise RuntimeError("pair replay CSV hash mismatch")
    label_index = {label: index for index, label in enumerate(labels)}
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    matrix = np.full((77, len(pairs)), np.nan, dtype=np.float64)
    with REPLAY_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["network_seed"]) != seed:
                continue
            pair = (label_index[row["item_1"]], label_index[row["item_2"]])
            matrix[int(row["subject"]), pair_index[pair]] = float(row["pair_accuracy"])
    if np.any(~np.isfinite(matrix)):
        raise RuntimeError(f"seed {seed} pair replay is incomplete")
    return matrix


def load_datasets() -> tuple[object, dict[str, Dataset], dict]:
    validate_specification()
    protocol = load_ranking_protocol(PROTOCOL_PATH)
    pairs = tuple(combinations(range(protocol.n_items), 2))
    human_subjects, published, human_benchmark = _load_human_subjects(protocol)
    human_matrix = np.asarray(
        [subject["pair_accuracy"] for subject in human_subjects], dtype=np.float64
    )
    human_rows = human_benchmark["combined"]["pairs"]
    result = load_json(MODEL_RESULT_PATH)
    datasets = {"human": Dataset("human", human_subjects, human_matrix, human_rows)}
    for seed in (2104, 2105):
        behavior = result["seeds"][str(seed)]["behavior"]["dual_access_matched"]
        matrix = _load_replay_matrix(seed, pairs, protocol.item_labels)
        if (
            float(
                np.max(
                    np.abs(
                        np.mean(matrix, axis=0)
                        - np.asarray(
                            [row["mean_accuracy_all"] for row in behavior["pairs"]]
                        )
                    )
                )
            )
            > 1e-12
        ):
            raise RuntimeError(
                f"seed {seed} replay no longer matches frozen pair means"
            )
        datasets[str(seed)] = Dataset(
            str(seed), behavior["subjects"], matrix, behavior["pairs"]
        )
    if tuple(datasets) != DATASET_ORDER:
        raise RuntimeError("dataset ordering changed")
    return protocol, datasets, published


def _bootstrap_rows(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(values), np.full(len(values), 1.0 / len(values)), size=samples
    )
    draws = counts @ values / len(values)
    return (
        np.mean(values, axis=0),
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
    )


def _profiles(dataset: Dataset, protocol) -> dict[str, np.ndarray]:
    pairs = tuple(combinations(range(protocol.n_items), 2))
    rank = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    learned = np.asarray([pair in protocol.learned_pairs for pair in pairs])
    eligible_matrix = dataset.pair_accuracy[dataset.eligible]
    group = np.column_stack(
        (
            np.mean(eligible_matrix[:, learned], axis=1),
            np.mean(eligible_matrix[:, ~learned], axis=1),
        )
    )
    serial = np.column_stack(
        [
            np.mean(
                eligible_matrix[
                    :, [index for index, pair in enumerate(pairs) if item in pair]
                ],
                axis=1,
            )
            for item in reversed(protocol.true_order_high_to_low)
        ]
    )
    distance = np.column_stack(
        [
            np.mean(
                eligible_matrix[
                    :,
                    [
                        index
                        for index, pair in enumerate(pairs)
                        if abs(rank[pair[0]] - rank[pair[1]]) == value
                    ],
                ],
                axis=1,
            )
            for value in range(1, protocol.n_items)
        ]
    )
    return {"group": group, "serial": serial, "distance": distance}


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "legend.frameon": False,
            "svg.hashsalt": "published-behavior-alignment-v1",
        }
    )


def _save_figure(fig, directory: Path, figure_id: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("svg", "pdf", "png"):
        path = directory / f"{figure_id}.{suffix}"
        if suffix == "svg":
            metadata = {"Date": None, "Creator": "fsrl.paper_figure_alignment"}
        elif suffix == "pdf":
            metadata = {
                "CreationDate": None,
                "ModDate": None,
                "Creator": "fsrl.paper_figure_alignment",
            }
        else:
            metadata = {"Software": "fsrl.paper_figure_alignment"}
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata)
        if suffix == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n",
                encoding="utf-8",
            )
        outputs.append(path)
    plt.close(fig)
    return outputs


def _matrix(values: np.ndarray, n_items: int) -> np.ma.MaskedArray:
    output = np.full((n_items, n_items), np.nan, dtype=np.float64)
    for value, (first, second) in zip(
        values, combinations(range(n_items), 2), strict=True
    ):
        output[second, first] = value
    return np.ma.masked_invalid(output)


def ranking_positions(order: list[int]) -> np.ndarray:
    positions = np.empty(len(order), dtype=np.int64)
    for position, item in enumerate(order):
        positions[item] = position
    return positions


def _pairwise_tau(dataset: Dataset) -> np.ndarray:
    positions = [
        ranking_positions(subject["subjective_order_high_to_low"])
        for subject, include in zip(dataset.subjects, dataset.analysis, strict=True)
        if include
    ]
    return np.asarray(
        [
            kendall_tau_positions(positions[first], positions[second])
            for first, second in combinations(range(len(positions)), 2)
        ],
        dtype=np.float64,
    )


def _subject_id(dataset: Dataset, index: int) -> int:
    subject = dataset.subjects[index]
    return int(subject.get("combined_id", subject.get("subject", index) + 1))


def _subject_tau_to_true(subject: dict, true_positions: np.ndarray) -> float:
    return kendall_tau_positions(
        ranking_positions(subject["subjective_order_high_to_low"]), true_positions
    )


def select_exemplar(dataset: Dataset, true_positions: np.ndarray) -> int:
    candidates = [
        index
        for index, subject in enumerate(dataset.subjects)
        if dataset.eligible[index]
        and subject["ranking_class"] == "self_consistent_incorrect"
    ]
    taus = np.asarray(
        [
            _subject_tau_to_true(dataset.subjects[index], true_positions)
            for index in candidates
        ]
    )
    median = float(np.median(taus))
    return min(
        candidates,
        key=lambda index: (
            abs(_subject_tau_to_true(dataset.subjects[index], true_positions) - median),
            _subject_id(dataset, index),
        ),
    )


def render_figure_01(
    output_root: Path, protocol, datasets: dict[str, Dataset], specification: dict
) -> dict:
    figure_id = "figure_01_group_behavior"
    directory = output_root / figure_id
    samples = int(specification["visual_uncertainty"]["samples"])
    base_seed = int(specification["visual_uncertainty"]["seed"])
    profiles = {
        dataset_id: _profiles(dataset, protocol)
        for dataset_id, dataset in datasets.items()
    }
    summaries = {}
    rows = []
    for dataset_offset, dataset_id in enumerate(DATASET_ORDER):
        summaries[dataset_id] = {}
        for panel_offset, name in enumerate(("group", "serial", "distance")):
            point, lower, upper = _bootstrap_rows(
                profiles[dataset_id][name],
                samples=samples,
                seed=base_seed + dataset_offset * 10 + panel_offset,
            )
            summaries[dataset_id][name] = (point, lower, upper)
            labels = {
                "group": ("Learned", "Nonlearned"),
                "serial": tuple(protocol.item_labels),
                "distance": tuple(str(value) for value in range(1, protocol.n_items)),
            }[name]
            for index, label in enumerate(labels):
                rows.append(
                    {
                        "panel": {"group": "1E", "serial": "1F", "distance": "1G"}[
                            name
                        ],
                        "dataset": dataset_id,
                        "network_seed": "" if dataset_id == "human" else dataset_id,
                        "x_index": index,
                        "x_label": label,
                        "value": point[index],
                        "lower": lower[index],
                        "upper": upper[index],
                        "participants": int(np.sum(datasets[dataset_id].eligible)),
                        "pooling": "within_dataset_only",
                    }
                )

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65))
    x = np.arange(2)
    width = 0.24
    for offset, dataset_id in enumerate(DATASET_ORDER):
        point, lower, upper = summaries[dataset_id]["group"]
        axes[0].bar(
            x + (offset - 1) * width,
            point,
            width,
            color=DATASET_COLORS[dataset_id],
            label=DATASET_LABELS[dataset_id],
            yerr=np.vstack((point - lower, upper - point)),
            capsize=2,
            linewidth=0,
        )
    axes[0].set_xticks(x, ("Learned", "Nonlearned"))
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("Choice accuracy")
    axes[0].set_title("1E  Direct and inferred accuracy")
    axes[0].legend(fontsize=8, loc="lower left")

    for axis, name, title, x_values, x_labels in (
        (
            axes[1],
            "serial",
            "1F  Serial-position effect",
            np.arange(8),
            protocol.item_labels,
        ),
        (
            axes[2],
            "distance",
            "1G  Symbolic-distance effect",
            np.arange(1, 8),
            tuple(str(v) for v in range(1, 8)),
        ),
    ):
        for dataset_id in DATASET_ORDER:
            point, lower, upper = summaries[dataset_id][name]
            axis.plot(
                x_values,
                point,
                marker="o",
                markersize=3,
                linewidth=1.6,
                color=DATASET_COLORS[dataset_id],
                label=DATASET_LABELS[dataset_id],
            )
            axis.fill_between(
                x_values, lower, upper, color=DATASET_COLORS[dataset_id], alpha=0.12
            )
        axis.set_ylim(0.5, 1.0)
        axis.set_ylabel("Choice accuracy")
        axis.set_title(title)
        axis.set_xticks(x_values, x_labels)
    axes[1].set_xlabel("True rank position (low to high)")
    axes[2].set_xlabel("Symbolic distance")
    fig.suptitle(
        "Released human behavior and frozen v2.4 model counterparts",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Shading/error bars: 95% participant bootstrap within each dataset; networks are not pooled.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    outputs = _save_figure(fig, directory, figure_id)
    source_path = directory / "source_data.csv"
    _write_csv(
        source_path,
        [
            "panel",
            "dataset",
            "network_seed",
            "x_index",
            "x_label",
            "value",
            "lower",
            "upper",
            "participants",
            "pooling",
        ],
        rows,
    )
    return {"id": figure_id, "outputs": outputs + [source_path], "rows": len(rows)}


def render_figure_02(
    output_root: Path,
    protocol,
    datasets: dict[str, Dataset],
    published: dict,
    specification: dict,
) -> dict:
    figure_id = "figure_02_pair_structure"
    directory = output_root / figure_id
    pairs = tuple(combinations(range(protocol.n_items), 2))
    labels = protocol.item_labels
    exemplar_labels = specification["selection_rules"]["figure_2b_pair"]
    exemplar = (labels.index(exemplar_labels[0]), labels.index(exemplar_labels[1]))
    exemplar_index = pairs.index(exemplar)
    class_index = {name: index for index, name in enumerate(PAIR_CLASS_ORDER)}
    pair_rows = []
    stable_rows = []
    distribution_rows = []
    stable_summaries = {}
    samples = int(specification["visual_uncertainty"]["samples"])
    base_seed = int(specification["visual_uncertainty"]["seed"]) + 100
    for offset, dataset_id in enumerate(DATASET_ORDER):
        dataset = datasets[dataset_id]
        if dataset_id == "human":
            class_rows = published["beta_pairs"]
            means = np.mean(dataset.pair_accuracy[dataset.eligible], axis=0)
        else:
            class_rows = dataset.pair_rows
            means = np.asarray([row["mean_accuracy_all"] for row in class_rows])
        for pair_index, (pair, row, mean) in enumerate(
            zip(pairs, class_rows, means, strict=True)
        ):
            fit = row if dataset_id == "human" else row["beta_fit_analysis"]
            pair_rows.append(
                {
                    "dataset": dataset_id,
                    "network_seed": "" if dataset_id == "human" else dataset_id,
                    "pair_index": pair_index,
                    "item_1": labels[pair[0]],
                    "item_2": labels[pair[1]],
                    "learned": pair in protocol.learned_pairs,
                    "mean_accuracy": mean,
                    "beta_alpha": fit.get("alpha"),
                    "beta_beta": fit.get("beta"),
                    "beta_class": fit["class"],
                }
            )
        analysis_indices = np.flatnonzero(dataset.analysis)
        for index in analysis_indices:
            distribution_rows.append(
                {
                    "dataset": dataset_id,
                    "network_seed": "" if dataset_id == "human" else dataset_id,
                    "subject_id": _subject_id(dataset, int(index)),
                    "pair": "C-D",
                    "pair_accuracy": dataset.pair_accuracy[index, exemplar_index],
                }
            )
        stable_values = np.column_stack(
            [
                np.asarray(
                    [
                        dataset.subjects[index]["stable_error_pair_counts"][
                            str(threshold)
                        ]
                        > 0
                        for index in analysis_indices
                    ],
                    dtype=np.float64,
                )
                for threshold in (60, 70, 80, 90, 100)
            ]
        )
        point, lower, upper = _bootstrap_rows(
            stable_values, samples=samples, seed=base_seed + offset
        )
        stable_summaries[dataset_id] = (point, lower, upper)
        for index, threshold in enumerate((60, 70, 80, 90, 100)):
            stable_rows.append(
                {
                    "dataset": dataset_id,
                    "network_seed": "" if dataset_id == "human" else dataset_id,
                    "threshold_percent": threshold,
                    "proportion": point[index],
                    "lower": lower[index],
                    "upper": upper[index],
                    "analysis_subjects": len(analysis_indices),
                }
            )

    _style()
    fig = plt.figure(figsize=(12.2, 12.0))
    grid = fig.add_gridspec(
        4, 3, height_ratios=(1.0, 0.9, 1.0, 0.8), hspace=0.48, wspace=0.28
    )
    accuracy_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    distribution_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    class_axes = [fig.add_subplot(grid[2, index]) for index in range(3)]
    stable_axis = fig.add_subplot(grid[3, :])
    accuracy_image = None
    for column, dataset_id in enumerate(DATASET_ORDER):
        dataset_pair_rows = [row for row in pair_rows if row["dataset"] == dataset_id]
        accuracy = _matrix(
            np.asarray([row["mean_accuracy"] for row in dataset_pair_rows]), 8
        )
        accuracy_image = accuracy_axes[column].imshow(
            accuracy, vmin=0.5, vmax=1.0, cmap="Blues"
        )
        for pair in protocol.learned_pairs:
            first, second = sorted(pair)
            accuracy_axes[column].scatter(
                first,
                second,
                marker="^",
                s=24,
                facecolors="none",
                edgecolors="black",
                linewidths=0.7,
            )
        accuracy_axes[column].set_title(DATASET_LABELS[dataset_id])
        accuracy_axes[column].set_xticks(range(8), labels)
        accuracy_axes[column].set_yticks(range(8), labels)
        accuracy_axes[column].set_xlabel("Item 1")
        if column == 0:
            accuracy_axes[column].set_ylabel("2A  Pair accuracy\nItem 2")

        values = [
            row["pair_accuracy"]
            for row in distribution_rows
            if row["dataset"] == dataset_id
        ]
        distribution_axes[column].hist(
            values,
            bins=np.arange(-0.05, 1.051, 0.1),
            weights=np.full(len(values), 1.0 / len(values)),
            color=DATASET_COLORS[dataset_id],
            edgecolor="white",
            linewidth=0.5,
        )
        distribution_axes[column].set_xlim(-0.05, 1.05)
        distribution_axes[column].set_ylim(0, 0.55)
        distribution_axes[column].set_xlabel("C-D accuracy")
        distribution_axes[column].set_title(DATASET_LABELS[dataset_id])
        if column == 0:
            distribution_axes[column].set_ylabel("2B  Subject proportion")

        classes = np.asarray(
            [class_index[row["beta_class"]] for row in dataset_pair_rows]
        )
        class_matrix = _matrix(classes, 8)
        cmap = ListedColormap([PAIR_CLASS_COLORS[name] for name in PAIR_CLASS_ORDER])
        norm = BoundaryNorm(
            np.arange(-0.5, len(PAIR_CLASS_ORDER) + 0.5), len(PAIR_CLASS_ORDER)
        )
        class_axes[column].imshow(class_matrix, cmap=cmap, norm=norm)
        class_axes[column].set_title(DATASET_LABELS[dataset_id])
        class_axes[column].set_xticks(range(8), labels)
        class_axes[column].set_yticks(range(8), labels)
        class_axes[column].set_xlabel("Item 1")
        if column == 0:
            class_axes[column].set_ylabel("2D  Beta class\nItem 2")

    fig.colorbar(
        accuracy_image,
        ax=accuracy_axes,
        location="right",
        fraction=0.018,
        pad=0.02,
        label="Mean accuracy",
    )
    legend_classes = ("high_accuracy", "bimodal", "ordinary_unimodal", "low_accuracy")
    class_axes[2].legend(
        handles=[
            Patch(facecolor=PAIR_CLASS_COLORS[name], label=name.replace("_", " "))
            for name in legend_classes
        ],
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        fontsize=8,
    )
    thresholds = np.asarray((60, 70, 80, 90, 100))
    for dataset_id in DATASET_ORDER:
        point, lower, upper = stable_summaries[dataset_id]
        stable_axis.plot(
            thresholds,
            point,
            marker="o",
            color=DATASET_COLORS[dataset_id],
            label=DATASET_LABELS[dataset_id],
        )
        stable_axis.fill_between(
            thresholds, lower, upper, color=DATASET_COLORS[dataset_id], alpha=0.12
        )
    stable_axis.set_ylim(0, 1.02)
    stable_axis.set_xlabel("Error-consistency threshold (%)")
    stable_axis.set_ylabel("Subjects with at least one stable error")
    stable_axis.set_title("2E  Individual-level error consistency")
    stable_axis.legend(ncol=3, loc="upper right")
    fig.suptitle(
        "Pair-level structure: released human data and frozen model counterparts",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    outputs = _save_figure(fig, directory, figure_id)
    source_path = directory / "source_data.csv"
    fields = [
        "row_type",
        "dataset",
        "network_seed",
        "pair_index",
        "item_1",
        "item_2",
        "learned",
        "mean_accuracy",
        "beta_alpha",
        "beta_beta",
        "beta_class",
        "subject_id",
        "pair",
        "pair_accuracy",
        "threshold_percent",
        "proportion",
        "lower",
        "upper",
        "analysis_subjects",
    ]
    rows = (
        [{"row_type": "pair", **row} for row in pair_rows]
        + [{"row_type": "figure_2b_subject", **row} for row in distribution_rows]
        + [{"row_type": "stable_error", **row} for row in stable_rows]
    )
    _write_csv(source_path, fields, rows)
    return {"id": figure_id, "outputs": outputs + [source_path], "rows": len(rows)}


def render_figure_02h(
    output_root: Path, protocol, datasets: dict[str, Dataset]
) -> dict:
    figure_id = "figure_02h_error_fingerprints"
    directory = output_root / figure_id
    pairs = tuple(combinations(range(protocol.n_items), 2))
    pair_labels = [
        f"{protocol.item_labels[first]}{protocol.item_labels[second]}"
        for first, second in pairs
    ]
    rows = []
    _style()
    fig = plt.figure(figsize=(13.2, 6.2))
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.045), wspace=0.28)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    colorbar_axis = fig.add_subplot(grid[0, 3])
    image = None
    for axis, dataset_id in zip(axes, DATASET_ORDER, strict=True):
        dataset = datasets[dataset_id]
        indices = np.flatnonzero(dataset.analysis)
        accuracy = dataset.pair_accuracy[indices]
        error = 1.0 - accuracy
        displayed = accuracy <= 0.5
        image = axis.imshow(
            np.ma.masked_where(~displayed, error),
            aspect="auto",
            interpolation="nearest",
            cmap="Reds",
            vmin=0.5,
            vmax=1.0,
        )
        axis.set_title(f"{DATASET_LABELS[dataset_id]} (n={len(indices)})")
        axis.set_xticks(range(len(pairs)), pair_labels, rotation=90, fontsize=6)
        axis.set_xlabel("Tested pair")
        axis.set_ylabel("Analysis subject")
        for row_position, subject_index in enumerate(indices):
            for pair_index, pair in enumerate(pairs):
                rows.append(
                    {
                        "dataset": dataset_id,
                        "network_seed": "" if dataset_id == "human" else dataset_id,
                        "row_position": row_position,
                        "subject_id": _subject_id(dataset, int(subject_index)),
                        "item_1": protocol.item_labels[pair[0]],
                        "item_2": protocol.item_labels[pair[1]],
                        "pair_accuracy": accuracy[row_position, pair_index],
                        "error_proportion": error[row_position, pair_index],
                        "displayed_as_error": displayed[row_position, pair_index],
                    }
                )
    fig.colorbar(
        image,
        cax=colorbar_axis,
        label="Error proportion (correct pairs blank)",
    )
    fig.suptitle(
        "2H  Stable subject-specific local error fingerprints",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Layout-adapted subject-by-pair view of the paper's per-subject lower-triangle matrices.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.13, top=0.90)
    outputs = _save_figure(fig, directory, figure_id)
    source_path = directory / "source_data.csv"
    _write_csv(
        source_path,
        [
            "dataset",
            "network_seed",
            "row_position",
            "subject_id",
            "item_1",
            "item_2",
            "pair_accuracy",
            "error_proportion",
            "displayed_as_error",
        ],
        rows,
    )
    return {"id": figure_id, "outputs": outputs + [source_path], "rows": len(rows)}


def render_figure_03(output_root: Path, protocol, datasets: dict[str, Dataset]) -> dict:
    figure_id = "figure_03_global_rankings"
    directory = output_root / figure_id
    pairs = tuple(combinations(range(protocol.n_items), 2))
    class_names = ("correct", "self_consistent_incorrect", "self_inconsistent")
    class_labels = ("Correct", "Self-consistent\nincorrect", "Self-inconsistent")
    class_rows = []
    tau_rows = []
    exemplar_rows = []
    order_rows = []
    tau_values = {}
    exemplars = {}
    true_positions = ranking_positions(list(protocol.true_order_high_to_low))
    for dataset_id in DATASET_ORDER:
        dataset = datasets[dataset_id]
        counts = {
            name: sum(
                include and subject["ranking_class"] == name
                for subject, include in zip(
                    dataset.subjects, dataset.eligible, strict=True
                )
            )
            for name in class_names
        }
        for name in class_names:
            class_rows.append(
                {
                    "dataset": dataset_id,
                    "network_seed": "" if dataset_id == "human" else dataset_id,
                    "ranking_class": name,
                    "count": counts[name],
                    "eligible_subjects": int(np.sum(dataset.eligible)),
                }
            )
        tau_values[dataset_id] = _pairwise_tau(dataset)
        for value in tau_values[dataset_id]:
            tau_rows.append(
                {
                    "dataset": dataset_id,
                    "network_seed": "" if dataset_id == "human" else dataset_id,
                    "kendall_tau": value,
                }
            )
        exemplars[dataset_id] = select_exemplar(dataset, true_positions)
        exemplar_index = exemplars[dataset_id]
        for pair_index, pair in enumerate(pairs):
            exemplar_rows.append(
                {
                    "dataset": dataset_id,
                    "network_seed": "" if dataset_id == "human" else dataset_id,
                    "subject_id": _subject_id(dataset, exemplar_index),
                    "pair_index": pair_index,
                    "item_1": protocol.item_labels[pair[0]],
                    "item_2": protocol.item_labels[pair[1]],
                    "pair_accuracy": dataset.pair_accuracy[exemplar_index, pair_index],
                    "kendall_tau_to_true": _subject_tau_to_true(
                        dataset.subjects[exemplar_index], true_positions
                    ),
                }
            )
        for row_position, subject_index in enumerate(np.flatnonzero(dataset.analysis)):
            order = dataset.subjects[subject_index]["subjective_order_high_to_low"]
            for rank_position, item in enumerate(order):
                order_rows.append(
                    {
                        "dataset": dataset_id,
                        "network_seed": "" if dataset_id == "human" else dataset_id,
                        "row_position": row_position,
                        "subject_id": _subject_id(dataset, int(subject_index)),
                        "subjective_rank_high_to_low": rank_position + 1,
                        "item": protocol.item_labels[item],
                    }
                )

    _style()
    fig = plt.figure(figsize=(12.4, 12.2))
    grid = fig.add_gridspec(
        3, 3, height_ratios=(0.8, 1.0, 1.55), hspace=0.46, wspace=0.32
    )
    class_axis = fig.add_subplot(grid[0, 0])
    tau_axis = fig.add_subplot(grid[0, 1:])
    exemplar_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    order_axes = [fig.add_subplot(grid[2, index]) for index in range(3)]

    x = np.arange(len(class_names))
    width = 0.24
    for offset, dataset_id in enumerate(DATASET_ORDER):
        values = [
            next(
                row["count"]
                for row in class_rows
                if row["dataset"] == dataset_id and row["ranking_class"] == name
            )
            for name in class_names
        ]
        class_axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            color=DATASET_COLORS[dataset_id],
            label=DATASET_LABELS[dataset_id],
        )
    class_axis.set_xticks(x, class_labels, fontsize=7)
    class_axis.set_ylabel("Subjects")
    class_axis.set_title("3B  Reconstructed ranking classes")
    class_axis.legend(fontsize=7)

    violins = tau_axis.violinplot(
        [tau_values[name] for name in DATASET_ORDER],
        positions=(1, 2, 3),
        showmeans=True,
        showextrema=False,
    )
    for body, dataset_id in zip(violins["bodies"], DATASET_ORDER, strict=True):
        body.set_facecolor(DATASET_COLORS[dataset_id])
        body.set_edgecolor(DATASET_COLORS[dataset_id])
        body.set_alpha(0.55)
    violins["cmeans"].set_color("black")
    tau_axis.set_xticks((1, 2, 3), [DATASET_LABELS[name] for name in DATASET_ORDER])
    tau_axis.set_ylabel("Pairwise Kendall tau")
    tau_axis.set_ylim(-1.0, 1.0)
    tau_axis.set_title("3D  Inter-subject ranking similarity")

    exemplar_image = None
    for axis, dataset_id in zip(exemplar_axes, DATASET_ORDER, strict=True):
        dataset = datasets[dataset_id]
        subject_index = exemplars[dataset_id]
        exemplar_image = axis.imshow(
            _matrix(dataset.pair_accuracy[subject_index], 8),
            vmin=0,
            vmax=1,
            cmap="Blues",
        )
        order = dataset.subjects[subject_index]["subjective_order_high_to_low"]
        order_label = ">".join(protocol.item_labels[item] for item in order)
        axis.set_title(
            f"{DATASET_LABELS[dataset_id]} subject {_subject_id(dataset, subject_index)}\n{order_label}",
            fontsize=8,
        )
        axis.set_xticks(range(8), protocol.item_labels)
        axis.set_yticks(range(8), protocol.item_labels)
        axis.set_xlabel("Item 1")
        if dataset_id == "human":
            axis.set_ylabel("Item 2")
            axis.text(
                -0.18,
                1.12,
                "3C",
                transform=axis.transAxes,
                fontsize=11,
                fontweight="bold",
            )
    fig.colorbar(
        exemplar_image,
        ax=exemplar_axes,
        location="right",
        fraction=0.018,
        pad=0.02,
        label="Pair accuracy",
    )

    item_cmap = ListedColormap(ITEM_COLORS)
    item_norm = BoundaryNorm(np.arange(-0.5, 8.5), 8)
    order_image = None
    for axis, dataset_id in zip(order_axes, DATASET_ORDER, strict=True):
        dataset = datasets[dataset_id]
        indices = np.flatnonzero(dataset.analysis)
        matrix = np.asarray(
            [
                dataset.subjects[index]["subjective_order_high_to_low"]
                for index in indices
            ],
            dtype=np.int64,
        )
        order_image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=item_cmap,
            norm=item_norm,
        )
        axis.set_title(f"{DATASET_LABELS[dataset_id]} (n={len(indices)})")
        axis.set_xticks(range(8), range(1, 9))
        axis.set_xlabel("Subjective rank (high to low)")
        axis.set_ylabel("Analysis subject")
        if dataset_id == "human":
            axis.text(
                -0.18,
                1.04,
                "3E  All reconstructed orders",
                transform=axis.transAxes,
                va="bottom",
                ha="left",
                fontweight="bold",
            )
    colorbar = fig.colorbar(
        order_image,
        ax=order_axes,
        location="right",
        fraction=0.018,
        pad=0.02,
        ticks=range(8),
    )
    colorbar.ax.set_yticklabels(protocol.item_labels)
    colorbar.set_label("Item identity")
    fig.suptitle(
        "Coherent and individualized global rankings",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    outputs = _save_figure(fig, directory, figure_id)
    source_path = directory / "source_data.csv"
    fields = [
        "row_type",
        "dataset",
        "network_seed",
        "ranking_class",
        "count",
        "eligible_subjects",
        "kendall_tau",
        "subject_id",
        "pair_index",
        "item_1",
        "item_2",
        "pair_accuracy",
        "kendall_tau_to_true",
        "row_position",
        "subjective_rank_high_to_low",
        "item",
    ]
    rows = (
        [{"row_type": "ranking_class", **row} for row in class_rows]
        + [{"row_type": "pairwise_similarity", **row} for row in tau_rows]
        + [{"row_type": "exemplar_pair", **row} for row in exemplar_rows]
        + [{"row_type": "subjective_order", **row} for row in order_rows]
    )
    _write_csv(source_path, fields, rows)
    return {"id": figure_id, "outputs": outputs + [source_path], "rows": len(rows)}


def render_suite(output_root: Path = SUITE_ROOT) -> dict:
    validation = validate_specification()
    specification = validation["specification"]
    if not REPLAY_CSV_PATH.is_file() or not REPLAY_MANIFEST_PATH.is_file():
        raise RuntimeError("run the read-only pair replay before rendering")
    protocol, datasets, published = load_datasets()
    output_root.mkdir(parents=True, exist_ok=True)
    figures = [
        render_figure_01(output_root, protocol, datasets, specification),
        render_figure_02(output_root, protocol, datasets, published, specification),
        render_figure_02h(output_root, protocol, datasets),
        render_figure_03(output_root, protocol, datasets),
    ]
    manifest = {
        "schema_version": 1,
        "suite_id": specification["suite_id"],
        "status": specification["status"],
        "claim_boundary": specification["claim_boundary"],
        "figure_specification": {
            "path": str(SPECIFICATION_PATH.relative_to(ROOT)),
            "sha256": file_sha256(SPECIFICATION_PATH),
        },
        "pair_replay": {
            "path": str(REPLAY_MANIFEST_PATH.relative_to(ROOT)),
            "sha256": file_sha256(REPLAY_MANIFEST_PATH),
        },
        "generator": {
            "path": "fsrl/paper_figure_alignment.py",
            "sha256": "0c0058c7d7964a28013093167955643be48a5d7a65cd9575a3d557761b1c2445",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "network_pooling": "not_performed",
        "figures": [
            {
                "id": figure["id"],
                "source_rows": figure["rows"],
                "files": [
                    {
                        "path": str(path.relative_to(output_root)),
                        "sha256": file_sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in figure["outputs"]
                ],
            }
            for figure in figures
        ],
        "excluded_panels": specification["excluded_panels"],
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def check_suite(output_root: Path = SUITE_ROOT) -> dict:
    expected_files = {
        "manifest.json",
        *{
            f"{figure['id']}/{figure['id']}.{suffix}"
            for figure in load_json(SPECIFICATION_PATH)["figures"]
            for suffix in ("svg", "pdf", "png")
        },
        *{
            f"{figure['id']}/source_data.csv"
            for figure in load_json(SPECIFICATION_PATH)["figures"]
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        candidate_root = Path(directory) / "paper_alignment"
        render_suite(candidate_root)
        mismatches = []
        for relative in sorted(expected_files):
            committed = output_root / relative
            candidate = candidate_root / relative
            if not committed.is_file():
                mismatches.append({"path": relative, "reason": "missing"})
            elif committed.read_bytes() != candidate.read_bytes():
                mismatches.append({"path": relative, "reason": "content_differs"})
    return {
        "passed": not mismatches,
        "checked_files": len(expected_files),
        "mismatches": mismatches,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "render", "check"))
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    if parsed.stage == "replay":
        result = replay_model_subject_pairs()
    elif parsed.stage == "render":
        result = render_suite()
    else:
        result = check_suite()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
