"""Load provenance-locked human and model datasets for paper figures."""

from __future__ import annotations

import csv
from itertools import combinations

import numpy as np

from fsrl.experiments.human.benchmark import (
    DEFAULT_FIGURE2D_PATH,
    DEFAULT_FIGURE3B_PATH,
    DEFAULT_PREREGISTERED_PATH,
    DEFAULT_REPLICATION_PATH,
    LIU_DATASET_FILES,
    load_human_cohort,
    load_published_figure_checks,
)
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.tasks.protocol import load_ranking_protocol
from fsrl.workflows.paper_figure_contract import (
    DATASET_ORDER,
    HUMAN_BENCHMARK_PATH,
    MODEL_RESULT_PATH,
    PROTOCOL_PATH,
    REPLAY_CSV_PATH,
    REPLAY_MANIFEST_PATH,
    Dataset,
    validate_specification,
)


def _load_human_subjects(protocol) -> tuple[list[dict], dict, dict]:
    preregistered = load_human_cohort(
        DEFAULT_PREREGISTERED_PATH,
        "preregistered",
        protocol,
        expected_sha256=LIU_DATASET_FILES["preregistered"]["sha256"],
    )
    replication = load_human_cohort(
        DEFAULT_REPLICATION_PATH,
        "replication",
        protocol,
        expected_sha256=LIU_DATASET_FILES["replication"]["sha256"],
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
