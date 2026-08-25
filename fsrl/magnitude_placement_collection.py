"""Locked synthetic collection core for magnitude-placement behavior v1.1."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path

import jsonschema
import numpy as np

from .study_registry import resolve_record

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = resolve_record("benchmarks/magnitude_placement_behavior_v1_1.json")
READINESS_PATH = (
    resolve_record("benchmarks/magnitude_placement_behavior_v1_1_collection_readiness.json")
)
RAW_SCHEMA_PATH = (
    resolve_record("benchmarks/magnitude_placement_behavior_v1_1_raw.schema.json")
)
REPAIR_PATH = (
    resolve_record("benchmarks/magnitude_placement_behavior_v1_1_collection_readiness_repair1.json")
)
MANIFEST_PATH = (
    resolve_record("benchmarks/magnitude_placement_behavior_v1_1_randomization.json.gz")
)
RESULT_PATH = (
    resolve_record("results/magnitude_placement_behavior_v1_1_collection_readiness.json")
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_seed(readiness: dict, identity: str, domain: str) -> int:
    randomization = readiness["randomization"]
    payload = "|".join(
        (
            readiness["readiness_id"],
            str(randomization["master_seed"]),
            identity,
            domain,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def matrix_sha256(matrix: list[list[int]]) -> str:
    return bytes_sha256(canonical_json_bytes(matrix))


def _dihedral_orbit(matrix: np.ndarray) -> set[tuple[int, ...]]:
    values = set()
    for turns in range(4):
        rotated = np.rot90(matrix, turns)
        values.add(tuple(int(value) for value in rotated.ravel()))
        values.add(tuple(int(value) for value in np.fliplr(rotated).ravel()))
    return values


def validate_codebooks(protocol: dict) -> dict:
    contract = protocol["binary_stimulus_contract"]
    books = {
        "C1": contract["codebook_1"],
        "C2": contract["codebook_2"],
    }
    arrays = {
        name: np.asarray(list(book.values()), dtype=np.int64)
        for name, book in books.items()
    }
    distances = {
        name: np.sum(array[:, None] != array[None, :], axis=(2, 3))
        for name, array in arrays.items()
    }
    expected_distance = int(contract["within_set_hamming_distance"])
    expected = np.full((8, 8), expected_distance, dtype=np.int64)
    np.fill_diagonal(expected, 0)
    flattened = [
        tuple(int(value) for value in matrix.ravel())
        for array in arrays.values()
        for matrix in array
    ]
    cross = np.sum(arrays["C1"][:, None] != arrays["C2"][None, :], axis=(2, 3))
    all_arrays = [matrix for array in arrays.values() for matrix in array]
    no_dihedral_matches = all(
        tuple(int(value) for value in all_arrays[second].ravel())
        not in _dihedral_orbit(all_arrays[first])
        for first in range(len(all_arrays))
        for second in range(first + 1, len(all_arrays))
    )
    gates = {
        "two_8_by_4_by_4_codebooks": all(
            array.shape == (8, 4, 4) for array in arrays.values()
        ),
        "binary_alphabet": all(
            {int(value) for value in np.unique(array)} == {0, 1}
            for array in arrays.values()
        ),
        "constant_weight_8": all(
            np.all(np.sum(array, axis=(1, 2)) == 8) for array in arrays.values()
        ),
        "within_codebook_distance_8": all(
            np.array_equal(distance, expected) for distance in distances.values()
        ),
        "distance_matrices_equal": np.array_equal(distances["C1"], distances["C2"]),
        "all_16_distinct": len(set(flattened)) == 16,
        "no_cross_complements": not any(
            tuple(1 - value for value in first) == second
            for first in flattened[:8]
            for second in flattened[8:]
        ),
        "cross_hamming_4_to_12": int(np.min(cross)) == 4
        and int(np.max(cross)) == 12,
        "row_column_occupancy_1_to_3": all(
            np.all((np.sum(array, axis=1) >= 1) & (np.sum(array, axis=1) <= 3))
            and np.all((np.sum(array, axis=2) >= 1) & (np.sum(array, axis=2) <= 3))
            for array in arrays.values()
        ),
        "no_dihedral_matches": no_dihedral_matches,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "distance_matrices": {
            name: distance.tolist() for name, distance in distances.items()
        },
        "cross_hamming_min": int(np.min(cross)),
        "cross_hamming_max": int(np.max(cross)),
        "matrix_sha256": {
            codebook: {
                code_id: matrix_sha256(matrix)
                for code_id, matrix in books[codebook].items()
            }
            for codebook in books
        },
    }


def _item_svg(matrix: list[list[int]], x: int, y: int, renderer: dict) -> str:
    item = renderer["code_item"]
    border = int(item["border_px"])
    cell = int(item["cell_px"])
    parts = [
        (
            f'<rect x="{x}" y="{y}" width="{item["outer_px"]}" '
            f'height="{item["outer_px"]}" fill="{item["border_color"]}"/>'
        )
    ]
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            fill = item["one_color"] if value else item["zero_color"]
            parts.append(
                f'<rect x="{x + border + column * cell}" '
                f'y="{y + border + row * cell}" width="{cell}" '
                f'height="{cell}" fill="{fill}"/>'
            )
    return "".join(parts)


def _bar_svg(x: int, level: int, renderer: dict) -> str:
    support = renderer["support_layout"]
    baseline = int(support["bar_baseline_y"])
    width = int(support["bar_width_px"])
    unit = int(support["pixels_per_unit"])
    maximum = int(support["bar_max_level"])
    height = level * unit
    top = baseline - height
    container_top = baseline - maximum * unit
    parts = [
        (
            f'<rect x="{x}" y="{container_top}" width="{width}" '
            f'height="{maximum * unit}" fill="none" '
            f'stroke="{support["bar_outline"]}" stroke-width="2"/>'
        ),
        (
            f'<rect x="{x}" y="{top}" width="{width}" height="{height}" '
            f'fill="{support["bar_fill"]}" stroke="{support["bar_outline"]}" '
            'stroke-width="2"/>'
        ),
    ]
    for tick in range(maximum + 1):
        tick_y = baseline - tick * unit
        parts.append(
            f'<line x1="{x - 8}" y1="{tick_y}" x2="{x}" y2="{tick_y}" '
            f'stroke="{support["bar_outline"]}" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{x + width}" y1="{tick_y}" x2="{x + width + 8}" '
            f'y2="{tick_y}" stroke="{support["bar_outline"]}" '
            'stroke-width="2"/>'
        )
    return "".join(parts)


def render_trial_svg(trial: dict, protocol: dict, readiness: dict) -> str:
    renderer = readiness["renderer"]
    canvas = renderer["canvas"]
    codebooks = {
        "C1": protocol["binary_stimulus_contract"]["codebook_1"],
        "C2": protocol["binary_stimulus_contract"]["codebook_2"],
    }
    codebook = trial["codebook"]
    if trial["code_left"] not in codebooks[codebook] or trial["code_right"] not in codebooks[codebook]:
        raise ValueError("trial code does not belong to its codebook")
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{canvas["width_px"]}" height="{canvas["height_px"]}" '
            f'viewBox="0 0 {canvas["width_px"]} {canvas["height_px"]}" '
            'shape-rendering="crispEdges">'
        ),
        (
            f'<rect x="0" y="0" width="{canvas["width_px"]}" '
            f'height="{canvas["height_px"]}" fill="{canvas["background"]}"/>'
        ),
    ]
    if trial["phase"] == "support":
        layout = renderer["support_layout"]
        left_xy = layout["left_item_xy"]
        right_xy = layout["right_item_xy"]
        parts.append(
            _item_svg(codebooks[codebook][trial["code_left"]], *left_xy, renderer)
        )
        parts.append(
            _item_svg(codebooks[codebook][trial["code_right"]], *right_xy, renderer)
        )
        parts.append(_bar_svg(int(layout["left_bar_x"]), trial["bar_left_level"], renderer))
        parts.append(_bar_svg(int(layout["right_bar_x"]), trial["bar_right_level"], renderer))
    elif trial["phase"] == "query":
        layout = renderer["query_layout"]
        parts.append(
            _item_svg(
                codebooks[codebook][trial["code_left"]],
                *layout["left_item_xy"],
                renderer,
            )
        )
        parts.append(
            _item_svg(
                codebooks[codebook][trial["code_right"]],
                *layout["right_item_xy"],
                renderer,
            )
        )
    else:
        raise ValueError("unknown phase")
    parts.append("</svg>")
    return "".join(parts)


def _cell_schedule(readiness: dict) -> list[str]:
    cells = list(readiness["randomization"]["cell_ids"])
    schedule = []
    for group in range(30):
        rng = np.random.default_rng(child_seed(readiness, f"group-{group + 1:02d}", "cell_order"))
        order = list(cells)
        rng.shuffle(order)
        schedule.extend(order)
    return schedule


def _role_map(protocol: dict, readiness: dict, slot_id: str, codebook: str) -> dict:
    roles = list(protocol["inherited_frozen_contract"]["assignment_A_low_to_high"])
    book_key = "codebook_1" if codebook == "C1" else "codebook_2"
    code_ids = list(protocol["binary_stimulus_contract"][book_key])
    rng = np.random.default_rng(child_seed(readiness, slot_id, f"{codebook}_role_mapping"))
    order = rng.permutation(len(code_ids))
    return {role: code_ids[int(index)] for role, index in zip(roles, order, strict=True)}


def _condition_maps(cell: str) -> tuple[list[str], dict[str, str]]:
    condition_order = ["A", "B"] if cell.startswith("A_first") else ["B", "A"]
    codebook_to_a = cell.endswith("C1_to_A")
    condition_codebook = {
        "A": "C1" if codebook_to_a else "C2",
        "B": "C2" if codebook_to_a else "C1",
    }
    return condition_order, condition_codebook


def _support_trials(
    slot_id: str,
    condition: str,
    codebook: str,
    role_map: dict,
    protocol: dict,
    readiness: dict,
) -> list[dict]:
    inherited = protocol["inherited_frozen_contract"]
    relations = list(inherited["support_relation_order"])
    gaps = dict(
        zip(relations, inherited[f"assignment_{condition}_gaps"], strict=True)
    )
    layout = readiness["renderer"]["support_layout"]
    rng = np.random.default_rng(child_seed(readiness, slot_id, f"{condition}_support"))
    trials = []
    for block in range(1, inherited["support_blocks"] + 1):
        for within, relation_index in enumerate(rng.permutation(len(relations)), 1):
            relation = relations[int(relation_index)]
            higher, lower = relation.split(">")
            gap = int(gaps[relation])
            low_level = int(
                rng.integers(
                    int(layout["bar_min_level"]),
                    int(layout["bar_max_level"]) - gap + 1,
                )
            )
            high_level = low_level + gap
            higher_left = bool(rng.integers(0, 2))
            role_left, role_right = (higher, lower) if higher_left else (lower, higher)
            bar_left, bar_right = (
                (high_level, low_level) if higher_left else (low_level, high_level)
            )
            trial = {
                "phase": "support",
                "condition": condition,
                "codebook": codebook,
                "block_index": block,
                "trial_index_within_block": within,
                "support_relation": relation,
                "registered_higher_role": higher,
                "registered_lower_role": lower,
                "registered_gap_units": gap,
                "role_left": role_left,
                "role_right": role_right,
                "code_left": role_map[role_left],
                "code_right": role_map[role_right],
                "bar_left_level": bar_left,
                "bar_right_level": bar_right,
            }
            trial["renderer_sha256"] = bytes_sha256(
                render_trial_svg(trial, protocol, readiness).encode("utf-8")
            )
            trials.append(trial)
    return trials


def _query_trials(
    slot_id: str,
    condition: str,
    codebook: str,
    role_map: dict,
    protocol: dict,
    readiness: dict,
) -> list[dict]:
    roles = list(protocol["inherited_frozen_contract"]["assignment_A_low_to_high"])
    pairs = list(combinations(roles, 2))
    blocks = int(protocol["inherited_frozen_contract"]["query_blocks"])
    rng = np.random.default_rng(child_seed(readiness, slot_id, f"{condition}_query"))
    trials = []
    for block in range(1, blocks + 1):
        for within, pair_index in enumerate(rng.permutation(len(pairs)), 1):
            first, second = pairs[int(pair_index)]
            first_left = bool(rng.integers(0, 2))
            role_left, role_right = (first, second) if first_left else (second, first)
            trial = {
                "phase": "query",
                "condition": condition,
                "codebook": codebook,
                "block_index": block,
                "trial_index_within_block": within,
                "pair": f"{first}-{second}",
                "role_left": role_left,
                "role_right": role_right,
                "code_left": role_map[role_left],
                "code_right": role_map[role_right],
                "bar_left_level": None,
                "bar_right_level": None,
            }
            trial["renderer_sha256"] = bytes_sha256(
                render_trial_svg(trial, protocol, readiness).encode("utf-8")
            )
            trials.append(trial)
    return trials


def build_manifest(protocol: dict, readiness: dict) -> dict:
    codebook_validation = validate_codebooks(protocol)
    if not codebook_validation["passed"]:
        raise RuntimeError("binary codebook contract failed")
    slots = []
    for index, cell in enumerate(_cell_schedule(readiness), 1):
        slot_id = f"MPB-{index:03d}"
        role_maps = {
            codebook: _role_map(protocol, readiness, slot_id, codebook)
            for codebook in ("C1", "C2")
        }
        condition_order, condition_codebook = _condition_maps(cell)
        sessions = []
        for session_index, condition in enumerate(condition_order, 1):
            codebook = condition_codebook[condition]
            sessions.append(
                {
                    "session_index": session_index,
                    "condition": condition,
                    "codebook": codebook,
                    "support_trials": _support_trials(
                        slot_id,
                        condition,
                        codebook,
                        role_maps[codebook],
                        protocol,
                        readiness,
                    ),
                    "query_trials": _query_trials(
                        slot_id,
                        condition,
                        codebook,
                        role_maps[codebook],
                        protocol,
                        readiness,
                    ),
                }
            )
        slots.append(
            {
                "slot_id": slot_id,
                "counterbalance_cell": cell,
                "condition_order": condition_order,
                "condition_codebook": condition_codebook,
                "role_maps": role_maps,
                "sessions": sessions,
            }
        )
    return {
        "schema_version": 1,
        "manifest_id": readiness["readiness_id"] + "-randomization",
        "master_seed": readiness["randomization"]["master_seed"],
        "runtime_entropy_used": False,
        "codebook_validation": codebook_validation,
        "slots": slots,
    }


def validate_manifest(manifest: dict, protocol: dict, readiness: dict) -> dict:
    roles = list(protocol["inherited_frozen_contract"]["assignment_A_low_to_high"])
    relations = set(protocol["inherited_frozen_contract"]["support_relation_order"])
    pairs = {f"{first}-{second}" for first, second in combinations(roles, 2)}
    slots = manifest["slots"]
    cell_counts = Counter(slot["counterbalance_cell"] for slot in slots)
    slot_ids = [slot["slot_id"] for slot in slots]
    group_balanced = all(
        {slot["counterbalance_cell"] for slot in slots[start : start + 4]}
        == set(readiness["randomization"]["cell_ids"])
        for start in range(0, len(slots), 4)
    )
    role_maps_valid = True
    session_maps_valid = True
    support_complete = True
    query_complete = True
    bars_valid = True
    frames_valid = True
    code_assignment_valid = True
    support_total = 0
    query_total = 0
    unit = int(readiness["renderer"]["support_layout"]["pixels_per_unit"])
    minimum = int(readiness["renderer"]["support_layout"]["bar_min_level"])
    maximum = int(readiness["renderer"]["support_layout"]["bar_max_level"])
    for slot in slots:
        expected_order, expected_mapping = _condition_maps(slot["counterbalance_cell"])
        session_maps_valid &= slot["condition_order"] == expected_order
        session_maps_valid &= slot["condition_codebook"] == expected_mapping
        for codebook, role_map in slot["role_maps"].items():
            expected_codes = set(
                protocol["binary_stimulus_contract"][
                    "codebook_1" if codebook == "C1" else "codebook_2"
                ]
            )
            role_maps_valid &= set(role_map) == set(roles)
            role_maps_valid &= set(role_map.values()) == expected_codes
        for session_index, session in enumerate(slot["sessions"], 1):
            session_maps_valid &= session["session_index"] == session_index
            session_maps_valid &= session["condition"] == expected_order[session_index - 1]
            session_maps_valid &= session["codebook"] == expected_mapping[session["condition"]]
            role_map = slot["role_maps"][session["codebook"]]
            support = session["support_trials"]
            query = session["query_trials"]
            support_total += len(support)
            query_total += len(query)
            support_complete &= len(support) == 32
            query_complete &= len(query) == 280
            for block in range(1, 5):
                support_complete &= {
                    trial["support_relation"]
                    for trial in support
                    if trial["block_index"] == block
                } == relations
            for block in range(1, 11):
                query_complete &= {
                    trial["pair"]
                    for trial in query
                    if trial["block_index"] == block
                } == pairs
            for trial in support:
                code_assignment_valid &= trial["code_left"] == role_map[trial["role_left"]]
                code_assignment_valid &= trial["code_right"] == role_map[trial["role_right"]]
                difference = abs(trial["bar_left_level"] - trial["bar_right_level"])
                bars_valid &= minimum <= trial["bar_left_level"] <= maximum
                bars_valid &= minimum <= trial["bar_right_level"] <= maximum
                bars_valid &= difference == trial["registered_gap_units"]
                bars_valid &= difference * unit == abs(
                    trial["bar_left_level"] * unit - trial["bar_right_level"] * unit
                )
                frames_valid &= trial["renderer_sha256"] == bytes_sha256(
                    render_trial_svg(trial, protocol, readiness).encode("utf-8")
                )
            for trial in query:
                code_assignment_valid &= trial["code_left"] == role_map[trial["role_left"]]
                code_assignment_valid &= trial["code_right"] == role_map[trial["role_right"]]
                frames_valid &= trial["renderer_sha256"] == bytes_sha256(
                    render_trial_svg(trial, protocol, readiness).encode("utf-8")
                )
    gates = {
        "codebooks": bool(manifest["codebook_validation"]["passed"]),
        "120_unique_chronological_slots": len(slots) == 120
        and len(set(slot_ids)) == 120
        and slot_ids == [f"MPB-{index:03d}" for index in range(1, 121)],
        "30_slots_per_cell": set(cell_counts.values()) == {30}
        and set(cell_counts) == set(readiness["randomization"]["cell_ids"]),
        "every_four_slots_balanced": group_balanced,
        "role_maps_are_permutations": role_maps_valid,
        "condition_session_codebook_maps": session_maps_valid,
        "support_blocks_complete": support_complete,
        "query_blocks_complete": query_complete,
        "bar_gaps_and_pixels_exact": bars_valid,
        "role_code_assignment_exact": code_assignment_valid,
        "renderer_hashes_exact": frames_valid,
        "7680_support_trials": support_total == 7680,
        "67200_query_trials": query_total == 67200,
        "no_runtime_entropy": manifest["runtime_entropy_used"] is False,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "cell_counts": dict(sorted(cell_counts.items())),
        "support_trials": support_total,
        "query_trials": query_total,
    }


def write_gzip_json_exclusive(path: Path, value: dict) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle,
    ):
        handle.write(payload)


def load_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def build_synthetic_session_bundle(
    slot: dict,
    session: dict,
    manifest_sha256: str,
    protocol: dict,
    readiness: dict,
) -> dict:
    if not slot["slot_id"].startswith("MPB-"):
        raise ValueError("invalid slot")
    timing = readiness["renderer"]["screen_structure_and_timing"]
    unit = int(readiness["renderer"]["support_layout"]["pixels_per_unit"])
    left_bounds = readiness["renderer"]["query_layout"]["left_click_bounds"]
    right_bounds = readiness["renderer"]["query_layout"]["right_click_bounds"]
    trials = []
    onset = 0.0
    for index, trial in enumerate(session["support_trials"], 1):
        trials.append(
            {
                "trial_id": f'{slot["slot_id"]}-S{session["session_index"]}-SUP-{index:03d}',
                "phase": "support",
                "block_index": trial["block_index"],
                "trial_index": index,
                "trial_index_within_block": trial["trial_index_within_block"],
                "role_left": trial["role_left"],
                "role_right": trial["role_right"],
                "code_left": trial["code_left"],
                "code_right": trial["code_right"],
                "scheduled_onset_ms": onset,
                "actual_onset_ms": onset,
                "focus_ok": True,
                "reload_count": 0,
                "network_interruption": False,
                "renderer_sha256": trial["renderer_sha256"],
                "support_relation": trial["support_relation"],
                "registered_higher_role": trial["registered_higher_role"],
                "registered_lower_role": trial["registered_lower_role"],
                "registered_gap_units": trial["registered_gap_units"],
                "bar_left_level": trial["bar_left_level"],
                "bar_right_level": trial["bar_right_level"],
                "bar_left_pixels": trial["bar_left_level"] * unit,
                "bar_right_pixels": trial["bar_right_level"] * unit,
                "choice_side": None,
                "choice_x_px": None,
                "choice_y_px": None,
                "response_time_ms": None,
            }
        )
        onset += float(timing["support_display_ms"] + timing["support_iti_ms"])
    for query_index, trial in enumerate(session["query_trials"], 1):
        choose_left = (query_index + session["session_index"]) % 2 == 0
        bounds = left_bounds if choose_left else right_bounds
        response_time = float(600 + 50 * (query_index % 5))
        trials.append(
            {
                "trial_id": f'{slot["slot_id"]}-S{session["session_index"]}-QUE-{query_index:03d}',
                "phase": "query",
                "block_index": trial["block_index"],
                "trial_index": 32 + query_index,
                "trial_index_within_block": trial["trial_index_within_block"],
                "role_left": trial["role_left"],
                "role_right": trial["role_right"],
                "code_left": trial["code_left"],
                "code_right": trial["code_right"],
                "scheduled_onset_ms": onset,
                "actual_onset_ms": onset,
                "focus_ok": True,
                "reload_count": 0,
                "network_interruption": False,
                "renderer_sha256": trial["renderer_sha256"],
                "support_relation": None,
                "registered_higher_role": None,
                "registered_lower_role": None,
                "registered_gap_units": None,
                "bar_left_level": None,
                "bar_right_level": None,
                "bar_left_pixels": None,
                "bar_right_pixels": None,
                "choice_side": "left" if choose_left else "right",
                "choice_x_px": 0.5 * (bounds[0] + bounds[2]),
                "choice_y_px": 0.5 * (bounds[1] + bounds[3]),
                "response_time_ms": response_time,
            }
        )
        onset += response_time + float(timing["query_iti_ms"])
    return {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "participant_id": f'SYNTH-{slot["slot_id"]}',
        "enrollment_slot": slot["slot_id"],
        "counterbalance_cell": slot["counterbalance_cell"],
        "session_index": session["session_index"],
        "condition": session["condition"],
        "codebook": session["codebook"],
        "manifest_sha256": manifest_sha256,
        "trials": trials,
        "acquisition_events": [
            {
                "event_index": 1,
                "event_kind": "session_start",
                "monotonic_ms": 0.0,
                "detail": "synthetic dry run",
            },
            {
                "event_index": 2,
                "event_kind": "session_end",
                "monotonic_ms": onset,
                "detail": "synthetic dry run complete",
            },
        ],
    }


def validate_session_bundle(
    bundle: dict,
    slot: dict,
    session: dict,
    protocol: dict,
    readiness: dict,
    raw_schema: dict,
) -> dict:
    jsonschema.Draft202012Validator(raw_schema).validate(bundle)
    trial_schedule = session["support_trials"] + session["query_trials"]
    identity_exact = (
        bundle["enrollment_slot"] == slot["slot_id"]
        and bundle["counterbalance_cell"] == slot["counterbalance_cell"]
        and bundle["session_index"] == session["session_index"]
        and bundle["condition"] == session["condition"]
        and bundle["codebook"] == session["codebook"]
    )
    schedule_exact = True
    raw_choice_valid = True
    onset_monotonic = True
    prior_onset = -1.0
    layout = readiness["renderer"]["query_layout"]
    for raw, scheduled in zip(bundle["trials"], trial_schedule, strict=True):
        schedule_exact &= raw["phase"] == scheduled["phase"]
        schedule_exact &= raw["block_index"] == scheduled["block_index"]
        schedule_exact &= raw["trial_index_within_block"] == scheduled["trial_index_within_block"]
        schedule_exact &= raw["role_left"] == scheduled["role_left"]
        schedule_exact &= raw["role_right"] == scheduled["role_right"]
        schedule_exact &= raw["code_left"] == scheduled["code_left"]
        schedule_exact &= raw["code_right"] == scheduled["code_right"]
        schedule_exact &= raw["renderer_sha256"] == scheduled["renderer_sha256"]
        onset_monotonic &= raw["actual_onset_ms"] >= prior_onset
        prior_onset = raw["actual_onset_ms"]
        if raw["phase"] == "support":
            schedule_exact &= raw["support_relation"] == scheduled["support_relation"]
            schedule_exact &= raw["registered_gap_units"] == scheduled["registered_gap_units"]
        else:
            bounds = (
                layout["left_click_bounds"]
                if raw["choice_side"] == "left"
                else layout["right_click_bounds"]
            )
            raw_choice_valid &= bounds[0] <= raw["choice_x_px"] <= bounds[2]
            raw_choice_valid &= bounds[1] <= raw["choice_y_px"] <= bounds[3]
    gates = {
        "synthetic_participant_only": bundle["participant_id"].startswith("SYNTH-"),
        "header_matches_manifest": identity_exact,
        "312_trials": len(bundle["trials"]) == 312,
        "schedule_matches_manifest": schedule_exact,
        "raw_clicks_inside_registered_item": raw_choice_valid,
        "onsets_monotonic": onset_monotonic,
        "session_start_and_end_events": [
            event["event_kind"] for event in bundle["acquisition_events"]
        ]
        == ["session_start", "session_end"],
    }
    return {"passed": all(gates.values()), "gates": gates}


def write_session_bundle_exclusive(path: Path, bundle: dict) -> dict:
    payload = json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest = bytes_sha256(payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    with sidecar.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "sha256": digest, "sidecar": str(sidecar)}


def read_locked_session_bundle(path: Path) -> dict:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected, filename = sidecar.read_text(encoding="ascii").strip().split("  ", 1)
    if filename != path.name or file_sha256(path) != expected:
        raise RuntimeError("raw session bundle lock mismatch")
    return load_json(path)


def run_readiness(
    manifest: dict,
    manifest_sha256: str,
    protocol: dict,
    readiness: dict,
    repair: dict,
    raw_schema: dict,
) -> dict:
    from .magnitude_placement_analysis import (
        profile_from_session_bundles,
        validate_all_synthetic_branches,
    )

    protocol_hash = file_sha256(PROTOCOL_PATH)
    readiness_hash = file_sha256(READINESS_PATH)
    manifest_validation = validate_manifest(manifest, protocol, readiness)
    jsonschema.Draft202012Validator.check_schema(raw_schema)
    selected_slots = []
    seen_cells = set()
    for slot in manifest["slots"]:
        cell = slot["counterbalance_cell"]
        if cell not in seen_cells:
            selected_slots.append(slot)
            seen_cells.add(cell)
    session_gates = {}
    import_gates = {}
    synthetic_ids = []
    with tempfile.TemporaryDirectory(prefix="magnitude-placement-readiness-") as directory:
        dry_run_root = Path(directory)
        for slot in selected_slots:
            bundles = []
            for session in slot["sessions"]:
                bundle = build_synthetic_session_bundle(
                    slot,
                    session,
                    manifest_sha256,
                    protocol,
                    readiness,
                )
                synthetic_ids.append(bundle["participant_id"])
                validation = validate_session_bundle(
                    bundle,
                    slot,
                    session,
                    protocol,
                    readiness,
                    raw_schema,
                )
                key = f'{slot["slot_id"]}/session-{session["session_index"]}'
                path = dry_run_root / slot["slot_id"] / f'session-{session["session_index"]}.json'
                write_session_bundle_exclusive(path, bundle)
                reopened = read_locked_session_bundle(path)
                session_gates[key] = validation["passed"] and reopened == bundle
                bundles.append(reopened)
            profile = profile_from_session_bundles(bundles)
            import_gates[slot["slot_id"]] = (
                profile["complete"]
                and set(profile["pair_probability"]) == {"A", "B"}
                and all(
                    len(profile["pair_probability"][condition]) == 28
                    for condition in ("A", "B")
                )
            )
    branch_validation = validate_all_synthetic_branches(protocol)
    source_files = (
        ROOT / "fsrl" / "magnitude_placement_collection.py",
        ROOT / "fsrl" / "magnitude_placement_analysis.py",
        ROOT / "tests" / "test_magnitude_placement_collection.py",
        ROOT / "tests" / "test_magnitude_placement_analysis.py",
        ROOT / "requirements.txt",
    )
    gates = {
        "scientific_protocol_hash": protocol_hash
        == readiness["scientific_protocol"]["sha256"],
        "readiness_repair_parent_hash": readiness_hash
        == repair["original_readiness_contract"]["sha256"],
        "paper_source_hash": file_sha256(
            resolve_record(readiness["source_audit"]["paper"]["path"])
        )
        == readiness["source_audit"]["paper"]["sha256"],
        "raw_schema_is_valid_draft_2020_12": True,
        "raw_schema_has_repaired_pseudonym_interface": raw_schema["properties"][
            "participant_id"
        ]["pattern"]
        == "^(SYNTH|MPBP)-[A-Z0-9-]+$",
        "full_manifest": manifest_validation["passed"],
        "four_counterbalance_cells_dry_run": len(selected_slots) == 4
        and len(seen_cells) == 4,
        "eight_session_bundles_validate_and_reopen": len(session_gates) == 8
        and all(session_gates.values()),
        "four_participant_profiles_import": len(import_gates) == 4
        and all(import_gates.values()),
        "all_five_analysis_branches": branch_validation["passed"],
        "dry_run_contains_only_synthetic_ids": bool(synthetic_ids)
        and all(identifier.startswith("SYNTH-") for identifier in synthetic_ids),
        "collection_remains_no_go": readiness["readiness_decision"][
            "collection_status"
        ]
        == "NO_GO"
        and repair["collection_status"] == "NO_GO",
    }
    return {
        "schema_version": 1,
        "readiness_id": readiness["readiness_id"],
        "implementation_status": "pass" if all(gates.values()) else "fail",
        "collection_status": "NO_GO",
        "gates": gates,
        "manifest_validation": manifest_validation,
        "dry_run": {
            "human_data_used": False,
            "temporary_files_retained": False,
            "selected_slots": [slot["slot_id"] for slot in selected_slots],
            "session_gates": session_gates,
            "profile_import_gates": import_gates,
        },
        "synthetic_analysis_branches": branch_validation["gates"],
        "external_go_requirements": {
            requirement: "pending"
            for requirement in readiness["readiness_decision"][
                "external_go_requirements"
            ]
        },
        "provenance": {
            "scientific_protocol_sha256": protocol_hash,
            "readiness_contract_sha256": readiness_hash,
            "readiness_repair_sha256": file_sha256(REPAIR_PATH),
            "raw_schema_sha256": file_sha256(RAW_SCHEMA_PATH),
            "randomization_manifest_sha256": manifest_sha256,
            "source_sha256": {
                str(path.relative_to(ROOT)): file_sha256(path)
                for path in source_files
            },
        },
        "claim_boundary": (
            "Synthetic implementation readiness only. No human responses were "
            "collected, and no ethics, consent, recruitment, privacy, platform, "
            "or explicit collection-GO requirement is satisfied by this artifact."
        ),
    }


def write_json_exclusive(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    validate_parser = subparsers.add_parser("validate-manifest")
    validate_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    readiness_parser = subparsers.add_parser("readiness")
    readiness_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    readiness_parser.add_argument("--output", type=Path, default=RESULT_PATH)
    arguments = parser.parse_args(argv)
    protocol = load_json(PROTOCOL_PATH)
    readiness = load_json(READINESS_PATH)
    if arguments.command == "manifest":
        manifest = build_manifest(protocol, readiness)
        validation = validate_manifest(manifest, protocol, readiness)
        if not validation["passed"]:
            raise RuntimeError("generated randomization manifest failed validation")
        write_gzip_json_exclusive(arguments.output, manifest)
        print(json.dumps({"path": str(arguments.output), "sha256": file_sha256(arguments.output)}))
        return 0
    manifest = load_gzip_json(arguments.manifest)
    if arguments.command == "readiness":
        result = run_readiness(
            manifest,
            file_sha256(arguments.manifest),
            protocol,
            readiness,
            load_json(REPAIR_PATH),
            load_json(RAW_SCHEMA_PATH),
        )
        write_json_exclusive(arguments.output, result)
        print(
            json.dumps(
                {
                    "path": str(arguments.output),
                    "sha256": file_sha256(arguments.output),
                    "implementation_status": result["implementation_status"],
                    "collection_status": result["collection_status"],
                },
                sort_keys=True,
            )
        )
        return 0 if result["implementation_status"] == "pass" else 2
    result = validate_manifest(manifest, protocol, readiness)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
