"""Narrow decomposition of already frozen full and global query margins."""

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry, gradient_energy_fraction
from fsrl.experiments.evidence_routing.measurement import internal_preferences
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import STUDIES_ROOT

from .protocol import PROTOCOL_SHA256, RECORDS, specification


def prepare():
    spec = specification()
    protocol = size_protocol(spec, 8)
    geometry = build_complete_graph_geometry(protocol)
    parent = STUDIES_ROOT / "weak_evidence_routing" / "records"
    parent_result = load_json(parent / "results/result.json")
    rows = {}
    for seed in (2132, 2133, 2134):
        for arm in ("shared", "isolated"):
            name = f"seed-{seed}.{arm}.evaluation.raw.npz"
            path = parent / "results" / name
            verify_reference(parent_result["artifacts"][name])
            with np.load(path, allow_pickle=False) as data:
                intact = data["N8__bundles__intact__logits"]
                global_margin = data["N8__bundles__local_off__logits"]
            fields = {
                "intact": intact,
                "global": global_margin,
                "local": intact - global_margin,
            }
            row = {"source": reference(path), "components": {}}
            for label, margin in fields.items():
                summary, raw = internal_preferences(margin, protocol, spec, seed)
                summary["coherence"] = estimate(
                    gradient_energy_fraction(raw["field"], geometry),
                    seed=spec["statistics"]["seed_offset"] + seed,
                    statistics=spec["statistics"],
                )
                row["components"][label] = summary
            rows[f"{seed}/{arm}"] = row
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "rows": rows,
        "boundary": "Fixed 77 subjects; global is local_off, not P alone; pure local is a margin difference, not P_off. No new forward evaluation.",
    }
    write_json_exclusive(RECORDS / "results/preparation.json", result)
    return result
