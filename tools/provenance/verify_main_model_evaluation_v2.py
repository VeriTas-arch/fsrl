"""Read-only audit preserving main-model v2 pre-JSON report order."""

import json

import numpy as np

from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import load_json
from tools.provenance.main_model_evaluation_v2 import (
    ARRAYS,
    CONTRACT_HASH,
    FIT_SEEDS,
    LOCK,
    REPORT,
    RESULT,
    _load_arrays,
    reconstruct,
    render_report,
    summarize,
    validate_lock,
)


def check_summary_and_report(result: dict, rebuilt: dict, saved_report: str) -> None:
    for name, value in rebuilt.items():
        if json_ready(value) != result[name]:
            raise RuntimeError(f"saved main-model audit summary differs: {name}")
    # publish() rendered before sorted-key JSON serialization. Restore the
    # freshly rebuilt insertion order without changing any value or report byte.
    if saved_report != render_report({**result, **rebuilt}):
        raise RuntimeError("saved main-model evaluation report differs")


def audit() -> dict:
    lock = validate_lock()
    result = load_json(RESULT)
    if (
        result["execution_lock"] != reference(LOCK)
        or result["contract_sha256"] != CONTRACT_HASH
        or result["source_commit"] != lock["source_commit"]
        or result["arrays"] != reference(ARRAYS)
        or result["training_performed"]
        or result["new_simulation_performed"]
    ):
        raise RuntimeError("saved main-model evaluation provenance differs")
    rebuilt_arrays = reconstruct()
    saved_arrays = _load_arrays(result["arrays"])
    if set(saved_arrays) != set(rebuilt_arrays):
        raise RuntimeError("saved main-model audit array inventory differs")
    for name, value in rebuilt_arrays.items():
        np.testing.assert_array_equal(saved_arrays[name], value)
    rebuilt = summarize(rebuilt_arrays)
    check_summary_and_report(result, rebuilt, REPORT.read_text(encoding="utf-8"))
    return {
        "passed": True,
        "outcome": result["outcome"],
        "sampling_localization": result["sampling_localization"],
        "fits": len(FIT_SEEDS),
        "cohorts_per_fit": lock["cohorts_per_fit"],
        "training_performed": False,
        "new_simulation_performed": False,
        "arrays_exactly_reconstructed": True,
        "summaries_exactly_reconstructed": True,
        "report_exact_bytes_reconstructed": True,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
