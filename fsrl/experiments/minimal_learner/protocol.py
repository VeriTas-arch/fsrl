"""Prospectively fixed candidate and data authority."""

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT
from fsrl.tasks.holdouts import registered_holdout_signatures
from fsrl.tasks.sparse_ranking import GenericRankingTaskGenerator

RECORD_ROOT = STUDIES_ROOT / "minimal_relational_learner" / "records"
RUN_ROOT = RUNS_ROOT / "minimal_relational_learner_v1"
PROTOCOL_PATH = RECORD_ROOT / "benchmarks" / "minimal_relational_learner_v1.json"
PROTOCOL_SHA256 = "6586f2e51eef80446c3b109cce386914178dd01efbf8df6a8fda69cfb4d6f7c2"
PROTOCOL_COMMIT = "0969d3fd6f6436b41a24f5b9d4cbea33147d452a"


def specification() -> dict:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("the registered minimal learner contract changed")
    return load_json(PROTOCOL_PATH)


def task_generator() -> GenericRankingTaskGenerator:
    task = specification()["task"]
    return GenericRankingTaskGenerator(
        **{
            key: task[key]
            for key in (
                "n_items",
                "cue_size",
                "min_edges",
                "max_edges",
                "support_blocks",
                "subject_encoding_mode",
            )
        },
        excluded_signatures=registered_holdout_signatures(),
    )


def run_directory(seed: int, condition: str):
    contract = specification()["seeds"]
    if seed not in contract["mandatory"] or condition not in contract["conditions"]:
        raise ValueError("unregistered training stream or condition")
    return RUN_ROOT / f"seed-{seed}" / condition
