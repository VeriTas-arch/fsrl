"""Rebuild original cohort endpoints and behavior from portable raw arrays."""

import numpy as np

from fsrl.experiments.training_strategy.behavior import evaluate_behavior
from fsrl.experiments.training_strategy.estimands import query_endpoints
from fsrl.experiments.training_strategy.locks import verify_reference
from fsrl.experiments.training_strategy.summaries import summarize_endpoints
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .analysis import query_groups, query_signs
from .decisions import fit_decision
from .inputs import load_group


def raw_arrays(ref: dict) -> dict:
    with np.load(verify_reference(ref), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def assess_groups(files: dict, cohorts: dict, seed: int, spec: dict) -> tuple:
    protocol = load_registered_protocol(spec["evaluation"]["liu"]["protocol_id"])
    domains, internal = {}, {}
    behavior = None
    for domain, groups in cohorts.items():
        size = spec["evaluation"][domain][
            "episodes" if domain == "generic" else "subjects"
        ]
        endpoints = {}
        strict, ties = np.zeros(size, dtype=bool), np.zeros(size, dtype=bool)
        for name, input_record in groups.items():
            batch, auxiliary = load_group(input_record)
            raw = raw_arrays(files[f"{domain}-{name}"])
            indices = auxiliary["subject_indices"]
            signs = query_signs(batch, protocol if domain == "liu" else None)
            masks = query_groups(batch)
            np.testing.assert_array_equal(raw["correct_signs"], signs)
            for condition in ("intact", "shuffled", "z_off", "fixed_parameter"):
                margins = raw[f"outputs__{condition}__margins"]
                values = query_endpoints(
                    margins[..., None],
                    signs[..., None],
                    masks,
                    temperature=spec["evaluation"][domain]["temperature"],
                )
                destination = endpoints.setdefault(condition, {})
                for metric, rows in values.items():
                    for group, value in rows.items():
                        np.testing.assert_allclose(
                            raw[f"endpoints__{condition}__{metric}__{group}"],
                            value,
                            atol=0,
                            rtol=0,
                            equal_nan=True,
                        )
                        destination.setdefault(metric, {}).setdefault(
                            group, np.full(size, np.nan)
                        )[indices] = value
            strict[indices] = raw["strict_correct_order"]
            ties[indices] = raw["score_ties"]
            if domain == "liu":
                behavior = evaluate_behavior(
                    {"logits": raw["outputs__intact__margins"]}, protocol, seed, spec
                )
        domains[domain] = endpoints
        internal[domain] = {
            "strict_correct_order_count": int(strict.sum()),
            "score_tie_count": int(ties.sum()),
            "subjects": size,
            "strict_correct_order_fraction": float(strict.mean()),
        }
    assert behavior is not None
    record = {
        "raw_endpoints": domains,
        "endpoints": {
            domain: {
                name: summarize_endpoints(
                    value,
                    (86000 if domain == "generic" else 85000) + seed,
                    spec["statistics"],
                )
                for name, value in rows.items()
            }
            for domain, rows in domains.items()
        },
        "behavior": behavior["record"],
        "internal": internal,
    }
    record["decision"] = fit_decision(record, spec, seed)
    return record, behavior["sampled_behavior"]
