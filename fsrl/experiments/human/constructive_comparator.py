"""Human-only metric-preserving constructive comparator derivation and test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import subprocess
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
from scipy import optimize

from fsrl.infra.formal_runtime import require_formal_runtime
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.infra.study_registry import (
    legacy_identifier,
    registered_file_sha256,
    resolve_record,
    resolve_registered_path,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import RankingProtocol, load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/human_metric_constructive_comparator_v1.json"
)
INITIAL_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/human_metric_constructive_comparator_v1.lock.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/human_metric_constructive_comparator_v1.repair1.lock.json"
)
DEFAULT_REPAIR_PATH = resolve_record(
    "benchmarks/human_metric_constructive_comparator_v1.repair1.json"
)
NONINTERPRETABLE_ATTEMPT_PATH = resolve_record(
    "results/human_metric_constructive_comparator_v1_attempt1_noninterpretable.json"
)
DEFAULT_PARAMETER_PATH = resolve_record(
    "benchmarks/human_metric_constructive_comparator_v1.parameters.json"
)
DEFAULT_PARAMETER_LOCK_PATH = resolve_record(
    "benchmarks/human_metric_constructive_comparator_v1.parameters.lock.json"
)
DEFAULT_RESULT_PATH = resolve_record(
    "results/human_metric_constructive_comparator_v1.json"
)

REQUIRED_COLUMNS = {
    "id",
    "trial",
    "block",
    "film_choose_index",
    "film_index_1",
    "film_index_2",
    "r_or_w",
}
IMPLEMENTATION_SOURCE_PATHS = {
    "runner": "fsrl/human_metric_constructive_comparator.py",
    "runner_tests": "tests/test_human_metric_constructive_comparator.py",
    "formal_runtime": "fsrl/formal_runtime.py",
    "formal_runtime_tests": "tests/test_formal_runtime.py",
}


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"git freeze verification failed: {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def require_pushed_freeze(paths: tuple[Path, ...]) -> dict:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    origin_dev = _git("rev-parse", "origin/dev")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if branch != "dev" or head != origin_dev or status:
        raise RuntimeError(
            "formal execution requires clean dev with HEAD equal to origin/dev"
        )
    tracked = []
    for path in paths:
        relative = path.resolve().relative_to(ROOT)
        _git("ls-files", "--error-unmatch", str(relative))
        _git("diff", "--quiet", "HEAD", "--", str(relative))
        tracked.append(str(relative))
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "worktree_clean": True,
        "tracked_freeze_files": tracked,
    }


def _canonical_paths(parsed: argparse.Namespace) -> None:
    expected = {
        "specification": DEFAULT_SPECIFICATION_PATH,
        "implementation_lock": DEFAULT_IMPLEMENTATION_LOCK_PATH,
        "parameters": DEFAULT_PARAMETER_PATH,
        "parameter_lock": DEFAULT_PARAMETER_LOCK_PATH,
    }
    for name, canonical in expected.items():
        if getattr(parsed, name).resolve() != canonical.resolve():
            raise RuntimeError(f"formal workflow requires canonical {name}")
    if parsed.phase == "derive":
        for path in (parsed.parameters, parsed.parameter_lock):
            if path.exists() or path.is_symlink():
                raise RuntimeError("derivation artifact already exists or is a symlink")
    else:
        result = parsed.result.resolve()
        if result != DEFAULT_RESULT_PATH.resolve():
            try:
                relative = result.relative_to(Path("/tmp").resolve())
            except ValueError as error:
                raise RuntimeError("formal replay result must be below /tmp") from error
            if not relative.parts:
                raise RuntimeError("formal replay result must be below /tmp")
        if parsed.result.exists() or parsed.result.is_symlink():
            raise RuntimeError("confirmation result already exists or is a symlink")


def validate_sources(
    specification_path: Path,
    implementation_lock_path: Path,
    *,
    phase: str,
) -> dict:
    specification = load_json(specification_path)
    lock = load_json(implementation_lock_path)
    repair = load_json(DEFAULT_REPAIR_PATH)
    expected_supersedes = {
        "path": legacy_identifier(INITIAL_IMPLEMENTATION_LOCK_PATH),
        "sha256": file_sha256(INITIAL_IMPLEMENTATION_LOCK_PATH),
    }
    expected_attempt = {
        "path": legacy_identifier(NONINTERPRETABLE_ATTEMPT_PATH),
        "sha256": file_sha256(NONINTERPRETABLE_ATTEMPT_PATH),
    }
    expected_repair = {
        "path": legacy_identifier(DEFAULT_REPAIR_PATH),
        "sha256": file_sha256(DEFAULT_REPAIR_PATH),
    }
    if not (
        lock.get("schema_version") == 1
        and lock.get("study_id") == specification.get("study_id")
        and lock.get("freeze_status")
        == "repair1_frozen_after_noninterpretable_attempt1_and_before_derivation_replay"
        and lock.get("specification_sha256") == file_sha256(specification_path)
        and lock.get("supersedes") == expected_supersedes
        and lock.get("noninterpretable_attempt") == expected_attempt
        and lock.get("repair") == expected_repair
        and set(lock.get("implementation_sources", {}))
        == set(IMPLEMENTATION_SOURCE_PATHS)
        and lock.get("registered_sources") == specification["registered_sources"]
        and repair.get("scientific_contract_changed") is False
    ):
        raise RuntimeError("metric-constructive implementation lock mismatch")

    checks = {}
    for name, relative in IMPLEMENTATION_SOURCE_PATHS.items():
        record = lock["implementation_sources"][name]
        path = resolve_registered_path(relative)
        checks[f"implementation:{name}"] = bool(
            record.get("path") == relative
            and record.get("sha256")
            == registered_file_sha256(relative, record["sha256"], resolved_path=path)
        )
    checks["repair:initial_implementation_lock"] = bool(
        expected_supersedes["sha256"] == file_sha256(INITIAL_IMPLEMENTATION_LOCK_PATH)
    )
    checks["repair:noninterpretable_attempt"] = bool(
        expected_attempt["sha256"] == file_sha256(NONINTERPRETABLE_ATTEMPT_PATH)
    )
    checks["repair:registration"] = bool(
        expected_repair["sha256"] == file_sha256(DEFAULT_REPAIR_PATH)
    )
    opened_trial_sources = []
    for name, record in specification["registered_sources"].items():
        if phase == "derive" and name == "confirmation_trials":
            checks[f"registered:{name}"] = "deferred_without_opening"
            continue
        path = resolve_registered_path(record["path"])
        checks[f"registered:{name}"] = bool(
            path.is_file()
            and registered_file_sha256(
                record["path"], record["sha256"], resolved_path=path
            )
            == record["sha256"]
        )
        if name in {"derivation_trials", "confirmation_trials"}:
            opened_trial_sources.append(name)
    passed = all(
        value is True or value == "deferred_without_opening"
        for value in checks.values()
    )
    expected_opened = (
        ["derivation_trials"]
        if phase == "derive"
        else [
            "derivation_trials",
            "confirmation_trials",
        ]
    )
    passed = passed and opened_trial_sources == expected_opened
    return {
        "passed": bool(passed),
        "phase": phase,
        "checks": checks,
        "opened_trial_sources": opened_trial_sources,
        "confirmation_trial_contents_opened": "confirmation_trials"
        in opened_trial_sources,
    }


def validate_parameter_lock(
    specification_path: Path,
    implementation_lock_path: Path,
    parameter_path: Path,
    parameter_lock_path: Path,
) -> tuple[dict, dict]:
    specification = load_json(specification_path)
    parameter_lock = load_json(parameter_lock_path)
    parameters = load_json(parameter_path)
    passed = bool(
        parameter_lock.get("schema_version") == 1
        and parameter_lock.get("study_id") == specification.get("study_id")
        and parameter_lock.get("freeze_status")
        == "derivation_frozen_before_confirmation"
        and parameter_lock.get("specification_sha256")
        == file_sha256(specification_path)
        and parameter_lock.get("implementation_lock_sha256")
        == file_sha256(implementation_lock_path)
        and parameter_lock.get("parameter_artifact")
        == {
            "path": str(parameter_path.relative_to(ROOT)),
            "sha256": file_sha256(parameter_path),
        }
        and parameters.get("study_id") == specification.get("study_id")
        and parameters.get("derivation_decision", {}).get("passed") is True
        and parameters.get("source_validation", {}).get("opened_trial_sources")
        == ["derivation_trials"]
        and parameters.get("source_validation", {}).get(
            "confirmation_trial_contents_opened"
        )
        is False
    )
    if not passed:
        raise RuntimeError("frozen derivation parameter artifact failed validation")
    return parameters, {
        "passed": True,
        "path": str(parameter_lock_path.relative_to(ROOT)),
        "sha256": file_sha256(parameter_lock_path),
        "parameter_artifact_sha256": file_sha256(parameter_path),
    }


def pair_metadata(protocol: RankingProtocol) -> dict:
    pairs = tuple(combinations(range(protocol.n_items), 2))
    positions = np.empty(protocol.n_items, dtype=np.int64)
    positions[np.asarray(protocol.true_order_high_to_low)] = np.arange(protocol.n_items)
    labels = tuple(
        f"{protocol.item_labels[first]}-{protocol.item_labels[second]}"
        for first, second in pairs
    )
    learned = np.asarray([pair in protocol.learned_pairs for pair in pairs])
    selected = ~learned
    distances = np.asarray(
        [abs(positions[first] - positions[second]) for first, second in pairs],
        dtype=np.float64,
    )
    selected_distances = distances[selected]
    centered = selected_distances - np.mean(selected_distances)
    denominator = float(centered @ centered)
    design = np.column_stack((np.ones(np.sum(selected)), selected_distances))
    residualizer = np.eye(np.sum(selected)) - design @ np.linalg.pinv(design)
    true_higher = np.asarray(
        [
            first if positions[first] < positions[second] else second
            for first, second in pairs
        ]
    )
    expected_labels = (
        "A-B",
        "A-C",
        "A-D",
        "A-E",
        "A-G",
        "B-D",
        "B-F",
        "B-G",
        "B-H",
        "C-D",
        "C-E",
        "C-F",
        "C-H",
        "D-E",
        "D-H",
        "E-F",
        "E-G",
        "F-G",
        "F-H",
        "G-H",
    )
    if (
        tuple(np.asarray(labels)[selected]) != expected_labels
        or not np.isclose(np.mean(selected_distances), 2.8)
        or not np.isclose(denominator, 57.2)
    ):
        raise RuntimeError("frozen 20-pair geometry changed")
    return {
        "pairs": pairs,
        "pair_labels": labels,
        "selected": selected,
        "selected_labels": expected_labels,
        "positions": positions,
        "true_higher": true_higher,
        "distances": distances,
        "selected_distances": selected_distances,
        "distance_weights": centered / denominator,
        "design": design,
        "residualizer": residualizer,
    }


def model_arrays(protocol: RankingProtocol, specification: dict) -> dict:
    metadata = pair_metadata(protocol)
    orders = np.asarray(list(permutations(range(protocol.n_items))), dtype=np.int16)
    positions = np.empty_like(orders)
    positions[np.arange(len(orders))[:, None], orders] = np.arange(
        protocol.n_items, dtype=np.int16
    )
    relation_labels = tuple(
        f"{protocol.item_labels[higher]}>{protocol.item_labels[lower]}"
        for higher, lower in protocol.support_pairs_higher_lower
    )
    expected_relations = tuple(
        specification["candidate_model"]["support_relation_order"]
    )
    if relation_labels != expected_relations:
        raise RuntimeError("support relation order changed")
    support_magnitudes = np.asarray(
        [
            (metadata["positions"][lower] - metadata["positions"][higher])
            / float(protocol.n_items - 1)
            for higher, lower in protocol.support_pairs_higher_lower
        ],
        dtype=np.float64,
    )
    relation_residual_sq = []
    for (higher, lower), magnitude in zip(
        protocol.support_pairs_higher_lower, support_magnitudes, strict=True
    ):
        predicted = (positions[:, lower] - positions[:, higher]) / float(
            protocol.n_items - 1
        )
        relation_residual_sq.append((predicted - magnitude) ** 2)
    relation_residual_sq = np.stack(relation_residual_sq, axis=1)
    mask_ids = np.arange(2 ** len(expected_relations), dtype=np.int64)
    masks = ((mask_ids[:, None] >> np.arange(len(expected_relations))) & 1).astype(
        np.float64
    )
    energies = masks @ relation_residual_sq.T
    order_correct = np.empty((len(orders), len(metadata["pairs"])), dtype=bool)
    for index, (first, second) in enumerate(metadata["pairs"]):
        higher = metadata["true_higher"][index]
        lower = second if higher == first else first
        order_correct[:, index] = positions[:, higher] < positions[:, lower]
    return {
        **metadata,
        "orders": orders,
        "order_positions": positions,
        "relation_labels": relation_labels,
        "support_magnitudes": support_magnitudes,
        "masks": masks,
        "mask_counts": np.sum(masks, axis=1),
        "relation_residual_sq": relation_residual_sq,
        "energies": energies,
        "order_correct": order_correct,
    }


def load_trial_cohort(
    path: Path,
    cohort: str,
    metadata: dict,
    *,
    expected_subjects: int,
) -> tuple[np.ndarray, list[str]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise RuntimeError(f"{cohort} human source columns changed")
        for row in reader:
            grouped.setdefault(int(row["id"]), []).append(row)
    if len(grouped) != expected_subjects:
        raise RuntimeError(f"{cohort} participant count changed")
    pair_to_index = {pair: index for index, pair in enumerate(metadata["pairs"])}
    arrays = []
    labels = []
    for source_id, rows in sorted(grouped.items()):
        matrix = np.full((10, len(pair_to_index)), np.nan, dtype=np.float64)
        identifiers = set()
        for row in rows:
            trial = int(row["trial"])
            block = int(row["block"])
            first_source = int(row["film_index_1"])
            second_source = int(row["film_index_2"])
            chosen_source = int(row["film_choose_index"])
            correct = int(row["r_or_w"])
            if not (
                1 <= block <= 10
                and 1 <= first_source <= 8
                and 1 <= second_source <= 8
                and first_source != second_source
                and chosen_source in {first_source, second_source}
                and correct in {0, 1}
                and correct == int(chosen_source == max(first_source, second_source))
            ):
                raise RuntimeError(f"invalid human trial for {cohort}:{source_id}")
            pair = tuple(sorted((first_source - 1, second_source - 1)))
            pair_index = pair_to_index[pair]
            if np.isfinite(matrix[block - 1, pair_index]):
                raise RuntimeError(f"duplicate pair-block for {cohort}:{source_id}")
            matrix[block - 1, pair_index] = float(correct)
            identifiers.add((block, trial))
        if (
            len(rows) != 280
            or len(identifiers) != 280
            or not np.all(np.isfinite(matrix))
        ):
            raise RuntimeError(f"incomplete trials for {cohort}:{source_id}")
        arrays.append(matrix)
        labels.append(f"{cohort}:{source_id}")
    return np.stack(arrays), labels


def torch_order_distribution(
    parameters: np.ndarray,
    arrays: dict,
    *,
    device: str,
):
    import torch

    rho, tau, _epsilon = parameters
    energies = torch.as_tensor(arrays["energies"], dtype=torch.float64, device=device)
    mask_counts = torch.as_tensor(
        arrays["mask_counts"], dtype=torch.float64, device=device
    )
    log_mask = mask_counts * torch.log(
        torch.as_tensor(rho, dtype=torch.float64, device=device)
    )
    log_mask = log_mask + (8.0 - mask_counts) * torch.log1p(
        -torch.as_tensor(rho, dtype=torch.float64, device=device)
    )
    log_conditional = torch.log_softmax(-energies / tau, dim=1)
    log_order = torch.logsumexp(log_mask[:, None] + log_conditional, dim=0)
    order = torch.exp(log_order)
    order = order / torch.sum(order)
    return order


def distribution_and_field(
    parameters: np.ndarray,
    arrays: dict,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch

    order = torch_order_distribution(parameters, arrays, device=device)
    correct = torch.as_tensor(
        arrays["order_correct"], dtype=torch.float64, device=device
    )
    epsilon = float(parameters[2])
    latent_correct = order @ correct
    field = epsilon + (1.0 - 2.0 * epsilon) * latent_correct
    order_np = order.detach().cpu().numpy()
    field_np = field.detach().cpu().numpy()
    identities = {
        "order_probability_sum_abs_error": float(abs(np.sum(order_np) - 1.0)),
        "order_probability_min": float(np.min(order_np)),
        "order_probability_max": float(np.max(order_np)),
        "field_min": float(np.min(field_np)),
        "field_max": float(np.max(field_np)),
    }
    return order_np, field_np, identities


def subject_log_likelihood_matrix(
    correct_counts: np.ndarray,
    order_correct: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    counts = np.asarray(correct_counts, dtype=np.float64)
    correct = np.asarray(order_correct, dtype=np.float64)
    log_good = math.log1p(-epsilon)
    log_bad = math.log(epsilon)
    base = np.sum(counts * log_bad + (10.0 - counts) * log_good, axis=1)
    delta = (2.0 * counts - 10.0) * (log_good - log_bad)
    log_binomial = np.sum(
        np.vectorize(math.lgamma)(np.full_like(counts, 11.0))
        - np.vectorize(math.lgamma)(counts + 1.0)
        - np.vectorize(math.lgamma)(11.0 - counts),
        axis=1,
    )
    return base[:, None] + delta @ correct.T + log_binomial[:, None]


class DerivationObjective:
    def __init__(self, arrays: dict, correct_counts: np.ndarray, device: str) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.energies = torch.as_tensor(
            arrays["energies"], dtype=torch.float64, device=device
        )
        self.mask_counts = torch.as_tensor(
            arrays["mask_counts"], dtype=torch.float64, device=device
        )
        self.order_correct = torch.as_tensor(
            arrays["order_correct"], dtype=torch.float64, device=device
        )
        self.correct_counts = torch.as_tensor(
            correct_counts, dtype=torch.float64, device=device
        )
        self.log_binomial = torch.sum(
            torch.lgamma(torch.full_like(self.correct_counts, 11.0))
            - torch.lgamma(self.correct_counts + 1.0)
            - torch.lgamma(11.0 - self.correct_counts),
            dim=1,
        )

    def __call__(self, values: np.ndarray) -> tuple[float, np.ndarray]:
        torch = self.torch
        parameters = torch.tensor(
            values, dtype=torch.float64, device=self.device, requires_grad=True
        )
        rho, tau, epsilon = parameters.unbind()
        log_mask = self.mask_counts * torch.log(rho)
        log_mask = log_mask + (8.0 - self.mask_counts) * torch.log1p(-rho)
        log_conditional = torch.log_softmax(-self.energies / tau, dim=1)
        log_order = torch.logsumexp(log_mask[:, None] + log_conditional, dim=0)
        log_good = torch.log1p(-epsilon)
        log_bad = torch.log(epsilon)
        base = torch.sum(
            self.correct_counts * log_bad + (10.0 - self.correct_counts) * log_good,
            dim=1,
        )
        delta = (2.0 * self.correct_counts - 10.0) * (log_good - log_bad)
        response = (
            base[:, None] + delta @ self.order_correct.T + self.log_binomial[:, None]
        )
        log_likelihood = torch.sum(
            torch.logsumexp(log_order[None, :] + response, dim=1)
        )
        loss = -log_likelihood
        loss.backward()
        gradient = parameters.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.detach().cpu()), gradient


def fit_parameters(
    specification: dict,
    arrays: dict,
    correct_counts: np.ndarray,
    *,
    device: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    contract = specification["derivation_contract"]
    optimizer_contract = contract["optimizer"]
    bounds_map = contract["bounds"]
    bounds = [
        tuple(float(value) for value in bounds_map[name])
        for name in ("rho", "tau", "epsilon")
    ]
    starts = list(
        product(
            optimizer_contract["starts"]["rho"],
            optimizer_contract["starts"]["tau"],
            optimizer_contract["starts"]["epsilon"],
        )
    )
    if len(starts) != 27 or optimizer_contract["starts"]["cartesian_total"] != 27:
        raise RuntimeError("registered optimizer start grid changed")
    objective = DerivationObjective(arrays, correct_counts, device)
    records = []
    for start in starts:
        result = optimize.minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": int(optimizer_contract["max_iterations"]),
                "maxls": int(optimizer_contract["max_line_search_steps"]),
                "ftol": float(optimizer_contract["ftol"]),
                "gtol": float(optimizer_contract["gtol"]),
            },
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        _order, field, identities = distribution_and_field(
            parameters, arrays, device=device
        )
        records.append(
            {
                "start": [float(value) for value in start],
                "parameters": parameters.tolist(),
                "log_likelihood": float(-result.fun),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "gradient_evaluations": int(result.njev),
                "gradient_max_abs": float(np.max(np.abs(result.jac))),
                "field": field.tolist(),
                "probability_identities": identities,
            }
        )
    converged = [record for record in records if record["success"]]
    if converged:
        best_log_likelihood = max(record["log_likelihood"] for record in converged)
        ties = [
            record
            for record in converged
            if best_log_likelihood - record["log_likelihood"] <= 1e-10
        ]
        selected = min(ties, key=lambda record: tuple(record["parameters"]))
        near = [
            record
            for record in converged
            if best_log_likelihood - record["log_likelihood"] <= 1e-6
        ]
        selected_field = np.asarray(selected["field"], dtype=np.float64)
        near_field_error = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(record["field"], dtype=np.float64) - selected_field
                    )
                )
            )
            for record in near
        )
        selected_parameters = np.asarray(selected["parameters"], dtype=np.float64)
        order, field, identities = distribution_and_field(
            selected_parameters, arrays, device=device
        )
    else:
        best_log_likelihood = float("-inf")
        near_field_error = float("inf")
        selected = None
        selected_parameters = np.full(3, np.nan)
        order = np.full(len(arrays["orders"]), np.nan)
        field = np.full(len(arrays["pairs"]), np.nan)
        identities = {}
    stability_pass = bool(
        len(converged) >= 3
        and np.isfinite(best_log_likelihood)
        and near_field_error <= 1e-5
        and np.all(np.isfinite(selected_parameters))
    )
    summary = {
        "starts": records,
        "converged_starts": len(converged),
        "selected": selected,
        "best_log_likelihood": best_log_likelihood,
        "near_optimum_field_max_abs_error": near_field_error,
        "stability_pass": stability_pass,
        "selected_probability_identities": identities,
        "selected_at_parameter_bound": {
            name: bool(
                np.isclose(selected_parameters[index], bounds[index][0], atol=1e-8)
                or np.isclose(selected_parameters[index], bounds[index][1], atol=1e-8)
            )
            for index, name in enumerate(("rho", "tau", "epsilon"))
        },
    }
    return summary, order, field


def interval_summary(point: float, bootstrap: np.ndarray) -> dict:
    values = np.asarray(bootstrap, dtype=np.float64)
    return {
        "point": float(point),
        "bootstrap": {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "lower90": float(np.quantile(values, 0.05)),
            "upper90": float(np.quantile(values, 0.95)),
            "lower95": float(np.quantile(values, 0.025)),
            "upper95": float(np.quantile(values, 0.975)),
        },
    }


def bootstrap_counts(samples: int, subjects: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.zeros((samples, subjects), dtype=np.int16)
    for row in output:
        row[:] = np.bincount(
            rng.choice(subjects, subjects, replace=True), minlength=subjects
        )
    return output


def residualize(values: np.ndarray, residualizer: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array @ residualizer.T


def row_correlations(
    first: np.ndarray, second: np.ndarray, minimum: float
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim == 1:
        left = np.broadcast_to(left, right.shape)
    left = left - np.mean(left, axis=1, keepdims=True)
    right = right - np.mean(right, axis=1, keepdims=True)
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if np.any(norms <= minimum):
        raise RuntimeError("pair correlation vector is degenerate")
    return np.sum(left * right, axis=1) / norms, norms


def vector_correlation(first: np.ndarray, second: np.ndarray, minimum: float) -> float:
    values, _norms = row_correlations(
        np.asarray(first)[None, :], np.asarray(second)[None, :], minimum
    )
    return float(values[0])


def confirmation_statistics(
    specification: dict,
    trials: np.ndarray,
    candidate_field_28: np.ndarray,
    metadata: dict,
) -> tuple[dict, dict, dict]:
    selected = metadata["selected"]
    full = np.mean(trials, axis=1)[:, selected]
    odd = np.mean(trials[:, 0::2], axis=1)[:, selected]
    even = np.mean(trials[:, 1::2], axis=1)[:, selected]
    candidate = np.asarray(candidate_field_28, dtype=np.float64)[selected]
    subjects = len(full)
    counts = bootstrap_counts(
        10000,
        subjects,
        36501,
    )
    full_boot = counts @ full / subjects
    odd_boot = counts @ odd / subjects
    even_boot = counts @ even / subjects
    human = np.mean(full, axis=0)
    human_odd = np.mean(odd, axis=0)
    human_even = np.mean(even, axis=0)
    weights = metadata["distance_weights"]
    human_slope = float(human @ weights)
    candidate_slope = float(candidate @ weights)
    human_slope_boot = full_boot @ weights
    slope_summary = interval_summary(human_slope, human_slope_boot)
    distance_adequate = bool(
        slope_summary["bootstrap"]["lower95"]
        <= candidate_slope
        <= slope_summary["bootstrap"]["upper95"]
    )
    residualizer = metadata["residualizer"]
    residual_human = residualize(human, residualizer)
    residual_odd = residualize(human_odd, residualizer)
    residual_even = residualize(human_even, residualizer)
    residual_candidate = residualize(candidate, residualizer)
    residual_full_boot = residualize(full_boot, residualizer)
    residual_odd_boot = residualize(odd_boot, residualizer)
    residual_even_boot = residualize(even_boot, residualizer)
    minimum = 1e-12
    r_ch = vector_correlation(residual_candidate, residual_human, minimum)
    r_hh = vector_correlation(residual_odd, residual_even, minimum)
    if r_hh <= 0.0:
        raise RuntimeError("point replication split-half reliability is nonpositive")
    rho_h = 2.0 * r_hh / (1.0 + r_hh)
    eta = r_ch / np.sqrt(rho_h)
    r_ch_boot, norm_ch = row_correlations(
        residual_candidate, residual_full_boot, minimum
    )
    r_hh_boot, norm_hh = row_correlations(
        residual_odd_boot, residual_even_boot, minimum
    )
    if np.any(r_hh_boot <= 0.0):
        raise RuntimeError("replication split-half bootstrap is nonpositive")
    rho_h_boot = 2.0 * r_hh_boot / (1.0 + r_hh_boot)
    eta_boot = r_ch_boot / np.sqrt(rho_h_boot)
    eta_summary = interval_summary(eta, eta_boot)
    pair_adequate = bool(eta_summary["bootstrap"]["lower90"] >= 0.80)
    statistics = {
        "distance": {
            "human_S_H": slope_summary,
            "candidate_S_C": candidate_slope,
            "adequate": distance_adequate,
            "status": "adequate"
            if distance_adequate
            else (
                "inadequate_below"
                if candidate_slope < slope_summary["bootstrap"]["lower95"]
                else "inadequate_above"
            ),
        },
        "pair": {
            "r_CH": interval_summary(r_ch, r_ch_boot),
            "r_HH": interval_summary(r_hh, r_hh_boot),
            "rho_H_spearman_brown": interval_summary(rho_h, rho_h_boot),
            "eta_pair": eta_summary,
            "threshold": 0.80,
            "adequate": pair_adequate,
        },
        "pair_vectors": {
            "labels": list(metadata["selected_labels"]),
            "human": human.tolist(),
            "candidate": candidate.tolist(),
            "human_residual": residual_human.tolist(),
            "candidate_residual": residual_candidate.tolist(),
        },
    }
    integrity = {
        "bootstrap_samples": len(counts),
        "bootstrap_subjects": subjects,
        "bootstrap_count_row_sum_max_abs_error": int(
            np.max(np.abs(np.sum(counts, axis=1) - subjects))
        ),
        "full_half_identity_max_abs_error": float(
            np.max(np.abs(full - 0.5 * (odd + even)))
        ),
        "human_slope_linear_identity_max_abs_error": float(
            abs(human_slope - float(np.mean(full @ weights)))
        ),
        "minimum_correlation_norm": float(min(np.min(norm_ch), np.min(norm_hh))),
        "minimum_bootstrap_r_HH": float(np.min(r_hh_boot)),
        "all_primary_arrays_finite": bool(
            all(
                np.all(np.isfinite(value))
                for value in (
                    full_boot,
                    odd_boot,
                    even_boot,
                    human_slope_boot,
                    r_ch_boot,
                    r_hh_boot,
                    rho_h_boot,
                    eta_boot,
                )
            )
        ),
    }
    raw = {
        "full_nonlearned": full.tolist(),
        "odd_nonlearned": odd.tolist(),
        "even_nonlearned": even.tolist(),
        "bootstrap_counts": counts.tolist(),
        "bootstrap_human_S_H": human_slope_boot.tolist(),
        "bootstrap_r_CH": r_ch_boot.tolist(),
        "bootstrap_r_HH": r_hh_boot.tolist(),
        "bootstrap_rho_H": rho_h_boot.tolist(),
        "bootstrap_eta_pair": eta_boot.tolist(),
    }
    return statistics, integrity, raw


def maximum_circular_triads(n_items: int) -> int:
    return (n_items**3 - 4 * n_items) // 24


def efficient_inter_subject_tau(rank_positions: np.ndarray) -> float:
    positions = np.asarray(rank_positions, dtype=np.int16)
    subjects = len(positions)
    if subjects < 2:
        raise RuntimeError("ranking-diversity cohort has fewer than two subjects")
    signs = np.stack(
        [
            np.sign(positions[:, first] - positions[:, second])
            for first, second in combinations(range(positions.shape[1]), 2)
        ],
        axis=1,
    ).astype(np.float64)
    sums = np.sum(signs, axis=0)
    return float(np.mean((sums**2 - subjects) / (subjects * (subjects - 1))))


def behavioral_components(
    correct_counts: np.ndarray,
    metadata: dict,
    *,
    tie_correct: np.ndarray | None = None,
) -> dict:
    counts = np.asarray(correct_counts, dtype=np.float64)
    pair_accuracy = counts / 10.0
    majority_correct = counts > 5.0
    ties = counts == 5.0
    if tie_correct is not None:
        tie_values = np.asarray(tie_correct, dtype=bool)
        if tie_values.shape != majority_correct.shape:
            raise ValueError("tie orientation array has the wrong shape")
        majority_correct[ties] = tie_values[ties]
    true_first = np.asarray(
        [
            higher == pair[0]
            for higher, pair in zip(
                metadata["true_higher"], metadata["pairs"], strict=True
            )
        ]
    )
    first_wins = majority_correct == true_first[None, :]
    pair_to_index = {pair: index for index, pair in enumerate(metadata["pairs"])}
    cycles = np.zeros(len(counts), dtype=np.int16)
    for first, second, third in combinations(range(8), 3):
        first_second = first_wins[:, pair_to_index[(first, second)]]
        second_third = first_wins[:, pair_to_index[(second, third)]]
        first_third = first_wins[:, pair_to_index[(first, third)]]
        cycles += (first_second & second_third & ~first_third) | (
            ~first_second & ~second_third & first_third
        )
    preference = np.where(
        true_first[None, :], 2.0 * pair_accuracy - 1.0, 1.0 - 2.0 * pair_accuracy
    )
    scores = np.zeros((len(counts), 8), dtype=np.float64)
    for edge, (first, second) in enumerate(metadata["pairs"]):
        scores[:, first] += preference[:, edge]
        scores[:, second] -= preference[:, edge]
    order = np.argsort(-scores, axis=1, kind="stable")
    rank_positions = np.empty_like(order)
    rank_positions[np.arange(len(order))[:, None], order] = np.arange(8)
    overall = np.mean(pair_accuracy, axis=1)
    eligible = overall >= 0.5
    correct_ranker = np.all(pair_accuracy > 0.5, axis=1)
    analysis = eligible & ~correct_ranker
    stable_80 = np.any((1.0 - pair_accuracy) >= 0.8 - 1e-9, axis=1)
    stable_100 = np.any((1.0 - pair_accuracy) >= 1.0 - 1e-9, axis=1)
    self_consistency = 1.0 - cycles / float(maximum_circular_triads(8))
    learned = ~metadata["selected"]
    analysis_subjects = int(np.sum(analysis))
    metrics = {
        "eligible_subjects": int(np.sum(eligible)),
        "analysis_subjects": analysis_subjects,
        "overall_accuracy": float(np.mean(overall[eligible])),
        "learned_accuracy": float(np.mean(pair_accuracy[eligible][:, learned])),
        "nonlearned_accuracy": float(np.mean(pair_accuracy[eligible][:, ~learned])),
        "mean_self_consistency_coefficient": float(np.mean(self_consistency[eligible])),
        "stable_error_80_analysis_proportion": (
            float(np.mean(stable_80[analysis])) if analysis_subjects else None
        ),
        "stable_error_100_analysis_proportion": (
            float(np.mean(stable_100[analysis])) if analysis_subjects else None
        ),
        "mean_inter_subject_kendall_tau": (
            efficient_inter_subject_tau(rank_positions[analysis])
            if analysis_subjects >= 2
            else None
        ),
        "ranking_class_counts": {
            "correct": int(np.sum(eligible & correct_ranker)),
            "self_consistent_incorrect": int(np.sum(analysis & (cycles == 0))),
            "self_inconsistent": int(np.sum(analysis & (cycles > 0))),
        },
    }
    return {
        "metrics": metrics,
        "eligible": eligible,
        "analysis": analysis,
        "self_consistency": self_consistency,
        "stable_80": stable_80,
        "stable_100": stable_100,
        "rank_positions": rank_positions,
    }


def human_qualification_bootstrap(components: dict, counts: np.ndarray) -> dict:
    eligible = components["eligible"].astype(np.float64)
    analysis = components["analysis"].astype(np.float64)
    eligible_n = counts @ eligible
    analysis_n = counts @ analysis
    if np.any(eligible_n == 0) or np.any(analysis_n < 2):
        raise RuntimeError("qualification bootstrap produced an empty cohort")
    self_consistency = counts @ (components["self_consistency"] * eligible) / eligible_n
    stable_80 = counts @ (components["stable_80"] * analysis) / analysis_n
    stable_100 = counts @ (components["stable_100"] * analysis) / analysis_n
    positions = components["rank_positions"]
    signs = np.stack(
        [
            np.sign(positions[:, first] - positions[:, second])
            for first, second in combinations(range(8), 2)
        ],
        axis=1,
    ).astype(np.float64)
    weighted = counts * analysis[None, :]
    sign_sums = weighted @ signs
    tau = np.mean(
        (sign_sums**2 - analysis_n[:, None])
        / (analysis_n[:, None] * (analysis_n[:, None] - 1.0)),
        axis=1,
    )
    return {
        "mean_self_consistency_coefficient": self_consistency,
        "stable_error_80_analysis_proportion": stable_80,
        "stable_error_100_analysis_proportion": stable_100,
        "mean_inter_subject_kendall_tau": tau,
    }


def predictive_qualification(
    specification: dict,
    order_probability: np.ndarray,
    parameters: np.ndarray,
    arrays: dict,
    human_trials: np.ndarray,
    primary_bootstrap_counts: np.ndarray,
) -> tuple[dict, dict]:
    contract = specification["individual_qualification"]
    simulation_subjects = 200000
    seed = 36601
    rng = np.random.default_rng(seed)
    order_indices = rng.choice(
        len(order_probability), simulation_subjects, p=order_probability
    )
    latent_correct = arrays["order_correct"][order_indices]
    epsilon = float(parameters[2])
    probability = np.where(latent_correct, 1.0 - epsilon, epsilon)
    simulated_counts = rng.binomial(10, probability).astype(np.int8)
    candidate_components = behavioral_components(
        simulated_counts, arrays, tie_correct=latent_correct
    )
    human_components = behavioral_components(np.sum(human_trials, axis=1), arrays)
    human_bootstrap = human_qualification_bootstrap(
        human_components, primary_bootstrap_counts
    )
    axes = {}
    all_pass = True
    for name in contract["mandatory_axes"]:
        human_point = human_components["metrics"][name]
        human_summary = interval_summary(human_point, human_bootstrap[name])
        candidate_point = candidate_components["metrics"][name]
        passed = bool(
            human_summary["bootstrap"]["lower95"]
            <= candidate_point
            <= human_summary["bootstrap"]["upper95"]
        )
        axes[name] = {
            "human": human_summary,
            "candidate_point": candidate_point,
            "passed": passed,
        }
        all_pass = all_pass and passed
    integrity = {
        "simulation_subjects": simulation_subjects,
        "simulation_seed": seed,
        "simulated_count_min": int(np.min(simulated_counts)),
        "simulated_count_max": int(np.max(simulated_counts)),
        "candidate_eligible_subjects": candidate_components["metrics"][
            "eligible_subjects"
        ],
        "candidate_analysis_subjects": candidate_components["metrics"][
            "analysis_subjects"
        ],
        "human_eligible_subjects": human_components["metrics"]["eligible_subjects"],
        "human_analysis_subjects": human_components["metrics"]["analysis_subjects"],
        "all_bootstrap_arrays_finite": bool(
            all(np.all(np.isfinite(value)) for value in human_bootstrap.values())
        ),
    }
    return {
        "passed": all_pass,
        "axes": axes,
        "human_descriptive": human_components["metrics"],
        "candidate_descriptive": candidate_components["metrics"],
        "simulation": {
            "subjects": simulation_subjects,
            "seed": seed,
            "order_index_counts": np.bincount(
                order_indices, minlength=len(order_probability)
            ).tolist(),
            "ranking_class_counts": candidate_components["metrics"][
                "ranking_class_counts"
            ],
        },
    }, integrity


def decide(
    distance_adequate: bool,
    pair_adequate: bool,
    qualification_passed: bool,
    all_gates_pass: bool,
) -> dict:
    if not all_gates_pass:
        outcome = "noninterpretable"
    elif distance_adequate and pair_adequate and qualification_passed:
        outcome = "metric_constructive_comparator_externally_adequate"
    elif distance_adequate and pair_adequate:
        outcome = "field_adequate_individual_qualification_failed"
    elif distance_adequate:
        outcome = "distance_adequate_pair_inadequate"
    elif pair_adequate:
        outcome = "pair_adequate_distance_inadequate"
    else:
        outcome = "metric_constructive_comparator_externally_inadequate"
    return {
        "outcome": outcome,
        "distance_adequate": distance_adequate if all_gates_pass else None,
        "pair_adequate": pair_adequate if all_gates_pass else None,
        "individual_qualification_passed": qualification_passed
        if all_gates_pass
        else None,
        "candidate_is_provisional_external_comparator": outcome
        == "metric_constructive_comparator_externally_adequate",
        "neural_intervention_authorized": False,
        "conditional_next_step": (
            "separately_register_read_only_neural_comparison"
            if outcome == "metric_constructive_comparator_externally_adequate"
            else "stop_existing_holdout_comparator_search_and_design_magnitude_placement_experiment"
        ),
    }


def derive(
    specification: dict,
    runtime: dict,
    source_validation: dict,
    git_freeze: dict,
) -> tuple[dict, dict]:
    protocol = load_ranking_protocol(
        resolve_registered_path(
            specification["registered_sources"]["liu_protocol"]["path"]
        )
    )
    arrays = model_arrays(protocol, specification)
    trials, labels = load_trial_cohort(
        resolve_registered_path(
            specification["registered_sources"]["derivation_trials"]["path"]
        ),
        "preregistered",
        arrays,
        expected_subjects=40,
    )
    correct_counts = np.sum(trials, axis=1)
    optimizer, order_probability, field = fit_parameters(
        specification,
        arrays,
        correct_counts,
        device=runtime["device"],
    )
    selected = optimizer["selected"]
    parameters = (
        np.asarray(selected["parameters"], dtype=np.float64)
        if selected is not None
        else np.full(3, np.nan)
    )
    identities = optimizer.get("selected_probability_identities", {})
    tolerance = 1e-10
    gates = {
        "source_validation": source_validation["passed"],
        "runtime": bool(
            runtime["active"]
            and runtime["cuda_available"]
            and runtime["torch_intraop_threads"] == 1
            and runtime["torch_interop_threads"] == 1
        ),
        "git_freeze": git_freeze["worktree_clean"],
        "cohort_isolation": source_validation["opened_trial_sources"]
        == ["derivation_trials"]
        and not source_validation["confirmation_trial_contents_opened"],
        "human_completeness": bool(
            trials.shape == (40, 10, 28) and np.all((trials == 0.0) | (trials == 1.0))
        ),
        "model_completeness": bool(
            arrays["masks"].shape == (256, 8)
            and arrays["orders"].shape == (40320, 8)
            and arrays["order_correct"].shape == (40320, 28)
            and len(optimizer["starts"]) == 27
        ),
        "optimizer_stability": optimizer["stability_pass"],
        "probability_identity": bool(
            identities
            and identities["order_probability_sum_abs_error"] <= tolerance
            and identities["order_probability_min"] >= 0.0
            and 0.0 <= identities["field_min"]
            and identities["field_max"] <= 1.0
        ),
    }
    passed = all(gates.values())
    artifact = {
        "schema_version": 1,
        "study_id": specification["study_id"],
        "registration_status": specification["registration_status"],
        "runtime": runtime,
        "source_validation": source_validation,
        "git_freeze_validation": git_freeze,
        "derivation_cohort": {
            "participant_labels": labels,
            "correct_counts": correct_counts.astype(int).tolist(),
            "subjects": 40,
            "blocks": 10,
            "pairs": 28,
        },
        "model_identity": {
            "pair_labels": list(arrays["pair_labels"]),
            "relation_labels": list(arrays["relation_labels"]),
            "support_magnitudes": arrays["support_magnitudes"].tolist(),
            "orders_sha256": array_sha256(arrays["orders"]),
            "masks_sha256": array_sha256(arrays["masks"]),
            "energies_sha256": array_sha256(arrays["energies"]),
            "order_correct_sha256": array_sha256(arrays["order_correct"]),
            "orders": arrays["orders"].astype(int).tolist(),
            "masks": arrays["masks"].astype(int).tolist(),
        },
        "optimization": optimizer,
        "selected_parameters": {
            name: float(parameters[index])
            for index, name in enumerate(("rho", "tau", "epsilon"))
        },
        "selected_order_probability": order_probability.tolist(),
        "selected_pair_field": field.tolist(),
        "derivation_gates": gates,
        "derivation_decision": {
            "passed": passed,
            "status": "derivation_frozen" if passed else "noninterpretable_derivation",
            "confirmation_authorized": passed,
        },
        "claim_boundary": specification["claim_boundary"],
    }
    parameter_lock = {
        "schema_version": 1,
        "study_id": specification["study_id"],
        "freeze_status": "derivation_frozen_before_confirmation",
        "specification_sha256": file_sha256(DEFAULT_SPECIFICATION_PATH),
        "implementation_lock_sha256": file_sha256(DEFAULT_IMPLEMENTATION_LOCK_PATH),
        "parameter_artifact": {
            "path": legacy_identifier(DEFAULT_PARAMETER_PATH),
            "sha256": "populated_after_exclusive_artifact_write",
        },
        "derivation_passed": passed,
        "confirmation_source_opened_during_derivation": False,
    }
    return artifact, parameter_lock


def confirm(
    specification: dict,
    parameters_artifact: dict,
    runtime: dict,
    source_validation: dict,
    parameter_validation: dict,
    git_freeze: dict,
) -> dict:
    protocol = load_ranking_protocol(
        resolve_registered_path(
            specification["registered_sources"]["liu_protocol"]["path"]
        )
    )
    arrays = model_arrays(protocol, specification)
    trials, labels = load_trial_cohort(
        resolve_registered_path(
            specification["registered_sources"]["confirmation_trials"]["path"]
        ),
        "replication",
        arrays,
        expected_subjects=37,
    )
    parameters = np.asarray(
        [
            parameters_artifact["selected_parameters"][name]
            for name in ("rho", "tau", "epsilon")
        ],
        dtype=np.float64,
    )
    order_probability, candidate_field, probability_identities = distribution_and_field(
        parameters, arrays, device=runtime["device"]
    )
    frozen_order = np.asarray(
        parameters_artifact["selected_order_probability"], dtype=np.float64
    )
    frozen_field = np.asarray(
        parameters_artifact["selected_pair_field"], dtype=np.float64
    )
    parameter_replay = {
        "order_probability_max_abs_error": float(
            np.max(np.abs(order_probability - frozen_order))
        ),
        "pair_field_max_abs_error": float(
            np.max(np.abs(candidate_field - frozen_field))
        ),
        "orders_sha256_matches": array_sha256(arrays["orders"])
        == parameters_artifact["model_identity"]["orders_sha256"],
        "masks_sha256_matches": array_sha256(arrays["masks"])
        == parameters_artifact["model_identity"]["masks_sha256"],
        "energies_sha256_matches": array_sha256(arrays["energies"])
        == parameters_artifact["model_identity"]["energies_sha256"],
        "order_correct_sha256_matches": array_sha256(arrays["order_correct"])
        == parameters_artifact["model_identity"]["order_correct_sha256"],
    }
    primary, primary_integrity, raw_primary = confirmation_statistics(
        specification, trials, candidate_field, arrays
    )
    primary_counts = np.asarray(raw_primary["bootstrap_counts"], dtype=np.int16)
    qualification, qualification_integrity = predictive_qualification(
        specification,
        order_probability,
        parameters,
        arrays,
        trials,
        primary_counts,
    )
    benchmark = load_json(
        resolve_registered_path(
            specification["registered_sources"]["human_benchmark"]["path"]
        )
    )
    benchmark_means = np.asarray(
        [row["mean_accuracy"] for row in benchmark["cohorts"]["replication"]["pairs"]],
        dtype=np.float64,
    )
    human_field_28 = np.mean(trials, axis=(0, 1))
    tolerance = 1e-10
    gates = {
        "source_validation": source_validation["passed"],
        "runtime": bool(
            runtime["active"]
            and runtime["cuda_available"]
            and runtime["torch_intraop_threads"] == 1
            and runtime["torch_interop_threads"] == 1
        ),
        "git_freeze": git_freeze["worktree_clean"],
        "parameter_lock": parameter_validation["passed"],
        "derivation_passed": parameters_artifact["derivation_decision"]["passed"],
        "cohort_isolation": source_validation["opened_trial_sources"]
        == ["derivation_trials", "confirmation_trials"],
        "human_completeness": bool(
            trials.shape == (37, 10, 28) and np.all((trials == 0.0) | (trials == 1.0))
        ),
        "human_benchmark_identity": bool(
            benchmark_means.shape == (28,)
            and np.max(np.abs(human_field_28 - benchmark_means)) <= tolerance
        ),
        "parameter_replay": bool(
            parameter_replay["order_probability_max_abs_error"] <= tolerance
            and parameter_replay["pair_field_max_abs_error"] <= tolerance
            and all(
                parameter_replay[name]
                for name in (
                    "orders_sha256_matches",
                    "masks_sha256_matches",
                    "energies_sha256_matches",
                    "order_correct_sha256_matches",
                )
            )
        ),
        "probability_identity": bool(
            probability_identities["order_probability_sum_abs_error"] <= tolerance
            and probability_identities["order_probability_min"] >= 0.0
            and 0.0 <= probability_identities["field_min"]
            and probability_identities["field_max"] <= 1.0
        ),
        "primary_integrity": bool(
            primary_integrity["bootstrap_samples"] == 10000
            and primary_integrity["bootstrap_subjects"] == 37
            and primary_integrity["bootstrap_count_row_sum_max_abs_error"] == 0
            and primary_integrity["full_half_identity_max_abs_error"] <= tolerance
            and primary_integrity["human_slope_linear_identity_max_abs_error"]
            <= tolerance
            and primary_integrity["minimum_correlation_norm"] > 1e-12
            and primary_integrity["minimum_bootstrap_r_HH"] > 0.0
            and primary_integrity["all_primary_arrays_finite"]
        ),
        "qualification_integrity": bool(
            qualification_integrity["simulation_subjects"] == 200000
            and qualification_integrity["simulation_seed"] == 36601
            and qualification_integrity["simulated_count_min"] >= 0
            and qualification_integrity["simulated_count_max"] <= 10
            and qualification_integrity["candidate_eligible_subjects"] > 0
            and qualification_integrity["candidate_analysis_subjects"] > 1
            and qualification_integrity["human_eligible_subjects"] == 37
            and qualification_integrity["human_analysis_subjects"] == 32
            and qualification_integrity["all_bootstrap_arrays_finite"]
        ),
    }
    all_gates_pass = all(gates.values())
    decision = decide(
        primary["distance"]["adequate"],
        primary["pair"]["adequate"],
        qualification["passed"],
        all_gates_pass,
    )
    return {
        "schema_version": 1,
        "study_id": specification["study_id"],
        "registration_status": specification["registration_status"],
        "runtime": runtime,
        "source_validation": source_validation,
        "parameter_validation": parameter_validation,
        "git_freeze_validation": git_freeze,
        "confirmation_cohort": {
            "participant_labels": labels,
            "subjects": 37,
            "blocks": 10,
            "pairs": 28,
        },
        "selected_parameters": parameters_artifact["selected_parameters"],
        "candidate_probability_identities": probability_identities,
        "parameter_replay": parameter_replay,
        "primary": primary,
        "individual_qualification": qualification,
        "integrity": {
            **primary_integrity,
            **qualification_integrity,
            "gates": gates,
            "passed": all_gates_pass,
        },
        "decision": decision,
        "raw_arrays": {
            "confirmation_trials": trials.astype(int).tolist(),
            "candidate_order_probability": order_probability.tolist(),
            "candidate_pair_field": candidate_field.tolist(),
            "primary": raw_primary,
            "qualification_simulation": qualification["simulation"],
        },
        "claim_boundaries": {
            "registered": specification["claim_boundary"],
            "outcome_contingent_route": specification["outcome_contingent_route"],
            "required": specification["reporting"]["required_claim_boundaries"],
        },
    }


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen human metric constructive comparator study."
    )
    parser.add_argument("phase", choices=("derive", "confirm"))
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument(
        "--implementation-lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETER_PATH)
    parser.add_argument(
        "--parameter-lock", type=Path, default=DEFAULT_PARAMETER_LOCK_PATH
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    _canonical_paths(parsed)
    runtime = require_formal_runtime()
    specification = load_json(parsed.specification)
    source_validation = validate_sources(
        parsed.specification, parsed.implementation_lock, phase=parsed.phase
    )
    if parsed.phase == "derive":
        git_freeze = require_pushed_freeze(
            (
                parsed.specification,
                INITIAL_IMPLEMENTATION_LOCK_PATH,
                DEFAULT_REPAIR_PATH,
                NONINTERPRETABLE_ATTEMPT_PATH,
                parsed.implementation_lock,
            )
        )
        artifact, parameter_lock = derive(
            specification, runtime, source_validation, git_freeze
        )
        write_json_exclusive(parsed.parameters, artifact)
        parameter_lock["parameter_artifact"]["sha256"] = file_sha256(parsed.parameters)
        write_json_exclusive(parsed.parameter_lock, parameter_lock)
    else:
        parameters, parameter_validation = validate_parameter_lock(
            parsed.specification,
            parsed.implementation_lock,
            parsed.parameters,
            parsed.parameter_lock,
        )
        git_freeze = require_pushed_freeze(
            (
                parsed.specification,
                INITIAL_IMPLEMENTATION_LOCK_PATH,
                DEFAULT_REPAIR_PATH,
                NONINTERPRETABLE_ATTEMPT_PATH,
                parsed.implementation_lock,
                parsed.parameters,
                parsed.parameter_lock,
            )
        )
        result = confirm(
            specification,
            parameters,
            runtime,
            source_validation,
            parameter_validation,
            git_freeze,
        )
        write_json_exclusive(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
