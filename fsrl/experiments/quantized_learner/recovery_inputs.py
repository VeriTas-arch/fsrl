"""Observed-only recovery designs and independent hidden-state prior draws."""

from dataclasses import dataclass

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.tasks.subject_encoding import sample_subject_encoding_states

from .encoding import canonical_addresses


@dataclass(frozen=True)
class ObservedDesign:
    """No generating admission, code, item rank, target or terminal state fields."""

    support_cues: np.ndarray
    signed: np.ndarray
    query_cues: np.ndarray

    @classmethod
    def from_batch(cls, batch: ModelBatch, subject: int):
        values = (
            batch.arrays["support_cues"][:, subject],
            batch.arrays["signed"][:, subject],
            batch.arrays["query_cues"][subject],
        )
        copies = [np.array(value, dtype=np.float64, copy=True) for value in values]
        for value in copies:
            value.setflags(write=False)
        return cls(*copies)

    def relations(self) -> dict:
        """Exchangeable item IDs from cues; distances from displayed values."""
        keys, orientation = canonical_addresses(self.support_cues)
        canonical_addresses(self.query_cues)
        if self.signed.shape != keys.shape or not np.all(
            np.isfinite(self.signed) & (np.abs(self.signed) <= 1)
        ):
            raise ValueError("recovery requires finite observed support values")
        width = self.support_cues.shape[-1] // 2
        items = np.unique(
            np.concatenate((self.support_cues, self.query_cues)).reshape(-1, width),
            axis=0,
        )
        item_ids = {tuple(cue): index for index, cue in enumerate(items)}
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        signed = orientation * self.signed
        if not np.array_equal(signed, signed[first][inverse]):
            raise ValueError("recovery assumes unchanged repeated observations")
        distances = np.abs(self.signed[first]) * (len(items) - 1)
        if not np.allclose(distances, np.rint(distances), atol=1e-7, rtol=0):
            raise ValueError("observed gaps do not follow the registered task scale")
        return {
            "item_count": len(items),
            "pairs": np.asarray(
                [
                    [item_ids[tuple(cue[:width])], item_ids[tuple(cue[width:])]]
                    for cue in self.support_cues[first]
                ]
            ),
            "distances": np.rint(distances).astype(np.int64),
            "first": first,
            "inverse": inverse,
        }


def nuisance_pool(design: ObservedDesign, rng, draws: int) -> tuple[ModelBatch, dict]:
    """Draw the unchanged admission law, never condition on generating latents.

    All budgets use prefixes of ONE maximum-budget pool. Generating data must
    use a separate RNG. Item saliences are exchangeable under the original prior,
    so lexicographic cue IDs need no access to the experiment's hidden ranking.
    """
    relations = design.relations()
    states = sample_subject_encoding_states(rng, draws, relations["item_count"])
    probabilities = np.asarray(
        [
            [
                state.relation_reliability(int(i), int(j), int(distance))
                for (i, j), distance in zip(
                    relations["pairs"], relations["distances"], strict=True
                )
            ]
            for state in states
        ]
    )
    admission_uniforms = rng.random(probabilities.shape)
    retained = admission_uniforms < probabilities
    signed = np.broadcast_to(design.signed[:, None], (len(design.signed), draws))
    arrays = {
        "support_cues": np.broadcast_to(
            design.support_cues[:, None], (*signed.shape, design.support_cues.shape[-1])
        ),
        "signed": signed,
        "retention": retained[:, relations["inverse"]].T.astype(np.float64),
        "local_evidence": np.zeros_like(signed),
        "query_cues": np.broadcast_to(
            design.query_cues[None], (draws, *design.query_cues.shape)
        ),
    }
    witness = {
        "encoding_uniforms": rng.random(signed.shape),
        "admission_uniforms": admission_uniforms,
        "probabilities": probabilities,
        "prior_latents": np.asarray(
            [
                [state.baseline_logit, *state.item_salience, state.distance_slope]
                for state in states
            ]
        ),
    }
    return ModelBatch(arrays), witness
