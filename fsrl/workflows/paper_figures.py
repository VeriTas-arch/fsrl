"""Render paper-aligned human/model behavioral figures from frozen evidence."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import matplotlib

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import REPO_ROOT

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from fsrl.analysis.behavioral import kendall_tau_positions
from fsrl.analysis.geometry import rank_positions
from fsrl.workflows.paper_figure_contract import (
    DATASET_COLORS,
    DATASET_LABELS,
    DATASET_ORDER,
    ITEM_COLORS,
    PAIR_CLASS_COLORS,
    PAIR_CLASS_ORDER,
    REPLAY_CSV_PATH,
    REPLAY_MANIFEST_PATH,
    SPECIFICATION_PATH,
    SUITE_ROOT,
    Dataset,
    _write_csv,
    _write_json,
    validate_specification,
)
from fsrl.workflows.paper_figure_data import load_datasets
from fsrl.workflows.paper_figure_replay import replay_model_subject_pairs

ROOT = REPO_ROOT


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


def _pairwise_tau(dataset: Dataset) -> np.ndarray:
    positions = [
        rank_positions(subject["subjective_order_high_to_low"])
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
        rank_positions(subject["subjective_order_high_to_low"]), true_positions
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
            bins=np.arange(-0.05, 1.051, 0.1).tolist(),
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

    assert accuracy_image is not None
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
    assert image is not None
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
    true_positions = rank_positions(list(protocol.true_order_high_to_low))
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
    bodies = cast(list[Any], violins["bodies"])
    for body, dataset_id in zip(bodies, DATASET_ORDER, strict=True):
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
    assert exemplar_image is not None
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
        axis.set_xticks(range(8), [str(rank) for rank in range(1, 9)])
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
    assert order_image is not None
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
