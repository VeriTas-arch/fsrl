import numpy as np
import torch

from fsrl.experiments.q_only_aligned_score.algebra import recurrence_and_kernel
from fsrl.experiments.q_only_aligned_score.decisions import (
    generic_category,
    generic_panel_passed,
    liu_evidence_binding,
)
from fsrl.experiments.q_only_aligned_score.model import QOnlyScore


def test_score_is_additive_and_kernel_exact():
    rng = np.random.default_rng(7)
    support = rng.normal(size=(5, 3, 30)).astype(np.float32)
    q = rng.normal(size=(5, 3)).astype(np.float32)
    query = rng.normal(size=(3, 8, 30)).astype(np.float32)
    model = QOnlyScore()
    margins, state = model(
        torch.from_numpy(support), torch.from_numpy(q), torch.from_numpy(query)
    )
    expected, _, reconstructed = recurrence_and_kernel(
        support, q, query, eta=0.5, gamma=1.0, epsilon=1e-8
    )
    np.testing.assert_allclose(margins.detach(), expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(expected, reconstructed, atol=1e-10, rtol=1e-10)
    assert state.shape == (3, 15)


def test_registered_generic_categories():
    assert generic_category(0) == "noncompetent"
    assert generic_category(1) == "panel_variable"
    assert generic_category(2) == "panel_variable"
    assert generic_category(3) == "stable_competent"


def test_generic_decision_reads_summarize_subjects_schema():
    def estimate(lower):
        return {"bootstrap": {"lower": lower}}

    result = {
        "competence": {"learned": estimate(0.6), "nonlearned": estimate(0.6)},
        "state_dependence": {
            "learned": estimate(0.1),
            "nonlearned": estimate(0.1),
        },
        "evidence_binding": {
            "learned": estimate(0.1),
            "nonlearned": estimate(0.1),
        },
    }
    assert generic_panel_passed(result)


def test_liu_binding_reads_primary_analysis_schema():
    def result(lower):
        return {
            "liu": {
                "effects": {
                    "intact_minus_evidence_shuffle_learned": {
                        "bootstrap": {"lower": lower}
                    }
                }
            }
        }

    assert liu_evidence_binding(result(0.1))
    assert not liu_evidence_binding(result(0.0))
