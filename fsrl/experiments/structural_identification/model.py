"""Structural input deletions and an independent NumPy score recurrence."""

import numpy as np

from fsrl.experiments.adaptive_plasticity.data import relation_slots
from fsrl.experiments.adaptive_plasticity.model import make_model as adaptive_model
from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.quantized_learner.encoding import encode_batch

from .protocol import CODEBOOK, SCHEDULES, STRUCTURES, model_specification


def encode(batch: ModelBatch, structure: str, uniforms: np.ndarray) -> ModelBatch:
    if structure not in STRUCTURES:
        raise ValueError("unregistered structure")
    arrays = dict(batch.arrays)
    if structure[1] == "0":
        arrays["retention"] = np.ones_like(arrays["retention"])
    arrays["local_evidence"] = np.zeros_like(arrays["signed"])
    condition = "resampled" if structure[2] == "1" else "exact"
    return encode_batch(ModelBatch(arrays), condition, uniforms, CODEBOOK)[0]


def make_model(schedule: str, device: str = "cpu"):
    if schedule not in SCHEDULES:
        raise ValueError("unregistered learning schedule")
    condition = (
        "adaptive_eta_resampled" if schedule == "decay" else "fixed_eta_resampled"
    )
    return adaptive_model(condition, model_specification(), device, scheduler="global")


def reference(batch: ModelBatch, eta: float, gain: float, schedule: str) -> tuple:
    """Independent float64 recurrence, with no Torch implementation calls."""
    if schedule not in SCHEDULES:
        raise ValueError("unregistered learning schedule")
    a = batch.arrays
    width = a["support_cues"].shape[-1] // 2
    features = np.asarray(a["support_cues"], dtype=np.float64)
    features = features[..., :width] - features[..., width:]
    subjects = features.shape[1]
    state = np.zeros((subjects, width), dtype=np.float64)
    count = relation_slots(a["support_cues"]).max(axis=0) + 1
    for t, x in enumerate(features):
        rate = eta / (1 + (t // count) * eta) if schedule == "decay" else eta
        factor = rate * a["retention"][t] / (1e-8 + np.einsum("bi,bi->b", x, x))
        state -= (factor * np.einsum("bi,bi->b", state, x))[:, None] * x
        state += (factor * a["signed"][t])[:, None] * x
    query = np.asarray(a["query_cues"], dtype=np.float64)
    query = query[..., :width] - query[..., width:]
    return gain * np.einsum("bi,bqi->bq", state, query), state
