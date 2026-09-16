"""Read-only completion of Phase 1 from the locked 72-unit attempt."""

from __future__ import annotations

import copy

from fsrl.experiments.clean_single_p.protocol import inherited_recipe
from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.write_cost.evaluation import summarize_generic
from fsrl.infra.provenance import load_json, write_json_exclusive

from .baseline import (
    _aggregate_outcome,
    _compare_parent,
    _inputs,
    _pair,
    _parent_raw,
    _raw_cpu,
)
from .budgets import audit_unit, merge_caps
from .locks import (
    reference,
    validate_attempt2_artifact_lock,
    validate_source_lock,
    verify_reference,
)
from .protocol import (
    ATTEMPT2_ARTIFACT_LOCK,
    BASELINE_RESULT,
    BASELINE_RUNS,
    PROTOCOL_SHA256,
    SOURCE_LOCK,
    specification,
)
from .storage import load_npz

CELLS = ("A0", "Ae", "C0", "Ce")


def _group(flat: dict, prefix: str) -> dict:
    marker = prefix + "__"
    return {
        key.removeprefix(marker): value
        for key, value in flat.items()
        if key.startswith(marker)
    }


def _liu_tree(flat: dict) -> dict:
    bundles = {}
    for name in ("intact", "local_off", "P_off", "evidence_shuffle"):
        bundles[name] = {"logits": flat[f"bundles__{name}__logits"]}
    return {
        "bundles": bundles,
        "removed": flat["removed"],
        "removed_global": flat["removed_global"],
        "evidence_route": flat["evidence_route"],
        "cost": flat["cost"],
        "total_write": flat["total_write"],
        "storage": _group(flat, "storage"),
    }


def _reconstruct_rows(
    external: dict, liu_cpu, recipe: dict, seed: int, panel: int
) -> dict:
    generic = {name: value.copy() for name, value in external["generic"].items()}
    sample_seed = seed + panel * 1000000
    generic_result = summarize_generic(generic, recipe, sample_seed)
    global_arrays = {**generic, "margins": generic["global_margins"]}
    generic_result["global"] = summarize_generic(global_arrays, recipe, sample_seed)
    liu_raw = _liu_tree(external["liu"])
    liu_result, _, _ = primary_analysis(
        liu_raw,
        liu_cpu,
        size_protocol(recipe, 8),
        recipe,
        sample_seed,
        generic_result,
    )
    return {"generic": generic_result, "liu": liu_result}


def _load_external(seed: int, panel: int, condition: str, cell: str) -> dict:
    flat = load_npz(
        BASELINE_RUNS / str(seed) / str(panel) / condition / cell / "external.npz"
    )
    return {"generic": _group(flat, "generic"), "liu": _group(flat, "liu")}


def run(runtime: dict) -> dict:
    spec = specification()
    lock = validate_source_lock(runtime)
    attempt_lock = validate_attempt2_artifact_lock()
    if BASELINE_RESULT.exists():
        raise RuntimeError("canonical propagated baseline result is write-once")
    panels_by_seed, raws_by_seed, parent_raws_by_seed = {}, {}, {}
    replay_errors, caps = {}, []
    try:
        for seed in spec["design"]["network_seeds"]:
            seed_key = str(seed)
            panels_by_seed[seed_key], raws_by_seed[seed_key] = {}, {}
            parent_raws_by_seed[seed_key] = {}
            for panel in spec["design"]["evaluation_panels"]:
                panel_key = str(panel)
                recipe = inherited_recipe(panel)
                inputs = _inputs(lock, panel)
                liu_cpu = _raw_cpu(inputs["liu-8"])
                panels_by_seed[seed_key][panel_key] = {}
                raws_by_seed[seed_key][panel_key] = {}
                parent_raws_by_seed[seed_key][panel_key] = {}
                for condition in spec["design"]["architecture_conditions"]:
                    panels_by_seed[seed_key][panel_key][condition] = {}
                    raws_by_seed[seed_key][panel_key][condition] = {}
                    parent_raws_by_seed[seed_key][panel_key][condition] = {}
                    for cell in CELLS:
                        identity = f"{seed}/{panel}/{condition}/{cell}"
                        direct = _load_external(seed, panel, condition, cell)
                        parent = _parent_raw(lock, identity)
                        replay_errors[identity], budget = audit_unit(
                            direct, parent, liu_cpu, recipe
                        )
                        caps.append(budget)
                        panels_by_seed[seed_key][panel_key][condition][cell] = (
                            _reconstruct_rows(direct, liu_cpu, recipe, seed, panel)
                        )
                        raws_by_seed[seed_key][panel_key][condition][cell] = direct
                        parent_raws_by_seed[seed_key][panel_key][condition][cell] = (
                            parent
                        )
        pairs, direct_draws = {}, {}
        for seed in spec["design"]["network_seeds"]:
            pairs[str(seed)], trace = _pair(
                seed, panels_by_seed[str(seed)], raws_by_seed[str(seed)]
            )
            direct_draws.update(
                {f"{seed}/{key}": value for key, value in trace.items()}
            )
        outcome = _aggregate_outcome(pairs)
        parent_result = load_json(verify_reference(lock["parents"]["parent_result"]))
        parent_panels = {
            str(seed): {
                str(panel): {
                    condition: copy.deepcopy(
                        parent_result["pairs"][str(seed)]["panels"][str(panel)][
                            condition
                        ]["cells"]
                    )
                    for condition in spec["design"]["architecture_conditions"]
                }
                for panel in spec["design"]["evaluation_panels"]
            }
            for seed in spec["design"]["network_seeds"]
        }
        reconstructed, parent_draws = {}, {}
        for seed in spec["design"]["network_seeds"]:
            reconstructed[str(seed)], trace = _pair(
                seed, parent_panels[str(seed)], parent_raws_by_seed[str(seed)]
            )
            parent_draws.update(
                {f"{seed}/{key}": value for key, value in trace.items()}
            )
        compatibility = _compare_parent(
            pairs,
            reconstructed,
            parent_result,
            outcome,
            merge_caps(caps),
            direct_draws,
            parent_draws,
        )
        manifest = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_lock": reference(SOURCE_LOCK),
            "attempt2_artifact_lock": reference(ATTEMPT2_ARTIFACT_LOCK),
            "members": attempt_lock["artifacts"],
            "model_replay": False,
        }
        manifest_path = BASELINE_RUNS / "manifest.json"
        write_json_exclusive(manifest_path, manifest)
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_lock": reference(SOURCE_LOCK),
            "attempt2_artifact_lock": reference(ATTEMPT2_ARTIFACT_LOCK),
            "runtime": runtime,
            "outcome": "baseline_compatible",
            "self_reproducibility": "bitwise_identical_in_locked_attempt2",
            "external_compatibility": compatibility,
            "maximum_external_errors": {
                name: max(row[name] for row in replay_errors.values())
                for name in next(iter(replay_errors.values()))
            },
            "pairs": json_ready(pairs),
            "evaluation_units": len(replay_errors),
            "artifact_members": attempt_lock["artifact_count"],
            "manifest": reference(manifest_path),
            "parent_outcome_unchanged": "time_removal_failure",
            "model_replay_during_finalization": False,
            "mechanism_estimates_exposed": False,
        }
        write_json_exclusive(BASELINE_RESULT, result)
        return {
            "outcome": result["outcome"],
            "evaluation_units": result["evaluation_units"],
            "artifact_members": result["artifact_members"],
            **compatibility,
        }
    except Exception as error:
        failure = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_lock": reference(SOURCE_LOCK),
            "attempt2_artifact_lock": reference(ATTEMPT2_ARTIFACT_LOCK),
            "runtime": runtime,
            "outcome": "baseline_incompatible",
            "failure": str(error),
            "completed_units": len(replay_errors),
            "model_replay_during_finalization": False,
            "mechanism_estimates_exposed": False,
            "parent_outcome_unchanged": "time_removal_failure",
        }
        if not BASELINE_RESULT.exists():
            write_json_exclusive(BASELINE_RESULT, failure)
        raise


__all__ = ["run"]
