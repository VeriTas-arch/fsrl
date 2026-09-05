import unittest
from unittest.mock import patch

import numpy as np
import torch

from fsrl.core.local_trace import antisymmetric_conjunctive_key
from fsrl.experiments.learner_diagnostics.algebra import (
    component_counts,
    global_references,
    keys,
    least_squares_state,
    local_decomposition,
    online_state,
    sigmoid_attribution,
)
from fsrl.experiments.learner_diagnostics.analysis import (
    global_analysis,
    local_analysis,
)
from fsrl.experiments.learner_diagnostics.estimands import (
    direction,
    endpoints,
    global_contrasts,
    readout_accounting,
    scoring,
    summarize,
)
from fsrl.experiments.learner_diagnostics.verification import (
    verify_numeric,
    verify_summaries,
)
from fsrl.experiments.minimal_learner.model import MetricScoreLearner
from fsrl.experiments.training_strategy.evaluation import flatten_arrays
from fsrl.experiments.training_strategy.summaries import liu_endpoints
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs


def fixture():
    rng = np.random.default_rng(919001)
    protocol = RankingProtocol(
        "synthetic-diagnosis",
        tuple("abcdefgh"),
        tuple(range(8)),
        tuple((i, i + 1) for i in range(7)) + ((0, 2),),
        4,
        10,
        {},
    )
    codes = rng.normal(size=(4, 8, 15))
    support_pairs = np.tile(protocol.support_pairs_higher_lower, (4, 4, 1))
    query_pairs = np.broadcast_to(ordered_pairs(8), (4, 56, 2)).copy()
    subjects = np.arange(4)[:, None]

    def cues(pairs):
        return np.concatenate(
            (codes[subjects, pairs[..., 0]], codes[subjects, pairs[..., 1]]), -1
        )

    d = ((support_pairs[..., 1] - support_pairs[..., 0]) / 7).T
    retained = rng.integers(0, 2, size=(4, 8))
    retained[0] = 0
    retained[1] = 1
    z = np.tile(retained, (1, 4)).T
    inputs = {
        "support_cues": cues(support_pairs).transpose(1, 0, 2),
        "signed": d,
        "retention": z,
        "local_evidence": d * (z + (1 - z) * 0.4),
        "query_cues": cues(query_pairs),
        "support_pairs": support_pairs,
        "query_pairs": query_pairs,
    }
    parameters = {"eta": 0.6, "gamma_G": 2.0, "gamma_L": 0.2}
    spec = {
        "integrity": {
            "svd_rcond": 1e-12,
            "algebra_atol": 1e-10,
            "float32_bridge_atol": 1e-5,
            "rank_tie_tolerance": 1e-10,
        },
        "statistics": {"samples": 40, "interval": 0.95},
    }
    return protocol, inputs, retained, parameters, spec


class LearnerDiagnosticTests(unittest.TestCase):
    def test_omission_is_not_zero_observation(self):
        x = np.ones((1, 2, 1))
        d = np.asarray([[1.0, 90.0]])
        np.testing.assert_allclose(
            online_state(x, d, np.asarray([[1, 0]]), 0.5, 1e-8), [[0.5]]
        )
        np.testing.assert_array_equal(
            online_state(x, d, np.zeros((1, 2)), 0.5, 1e-8), 0
        )

    def test_lstsq_empty_rank_deficient_and_inconsistent(self):
        x = np.asarray([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
        d = np.asarray([[2.0, 4.0], [2.0, 4.0]])
        state, rank = least_squares_state(x, d, np.asarray([[0, 0], [1, 1]]), 1e-12)
        np.testing.assert_allclose(state, [[0.0, 0.0], [3.0, 0.0]], atol=1e-14)
        np.testing.assert_array_equal(rank, [0, 1])

    def test_numpy_replay_and_keys_match_frozen_model(self):
        _, inputs, _, _, spec = fixture()
        model = MetricScoreLearner(
            15,
            with_local=True,
            initial_eta=0.6,
            initial_global_gain=2,
            initial_local_gain=0.2,
            epsilon=1e-8,
        ).double()
        args = [
            torch.as_tensor(inputs[k], dtype=torch.float64)
            for k in (
                "support_cues",
                "signed",
                "retention",
                "local_evidence",
                "query_cues",
            )
        ]
        with torch.no_grad():
            _, g, l, w, _ = model(*args)
        params = {
            "eta": model.eta.item(),
            "gamma_G": model.global_gain.item(),
            "gamma_L": model.local.gain.item(),
        }
        references = global_references(inputs, params, spec["integrity"])
        pieces = local_decomposition(inputs, params["gamma_L"])
        np.testing.assert_allclose(references["RF"]["state"], w, atol=1e-12)
        np.testing.assert_allclose(references["RF"]["margin"], g, atol=1e-12)
        np.testing.assert_allclose(
            pieces["self_margin"] + pieces["cross_margin"], l, atol=1e-12
        )
        q = inputs["query_cues"].reshape(-1, 30)
        np.testing.assert_allclose(
            keys(q), antisymmetric_conjunctive_key(torch.tensor(q), 15), atol=1e-12
        )

    def test_no_admission_all_finite_and_limit_zero(self):
        _, inputs, _, parameters, spec = fixture()
        cells = global_references(inputs, parameters, spec["integrity"])
        for name in ("RF", "RL"):
            np.testing.assert_array_equal(cells[name]["state"][0], 0)
        self.assertGreater(np.linalg.norm(cells["AL"]["state"][0]), 0)

    def test_local_nonlearned_self_zero_and_orientation(self):
        protocol, inputs, retained, _, _ = fixture()
        pieces = local_decomposition(inputs, 0.2)
        context = scoring(protocol, retained)
        nonlearned = np.repeat(context["groups"]["nonlearned"], 2)
        np.testing.assert_array_equal(pieces["self_margin"][:, nonlearned], 0)
        self.assertGreater(np.max(np.abs(pieces["cross_margin"][:, nonlearned])), 0)
        np.testing.assert_allclose(
            pieces["trial_contribution"][..., ::2],
            -pieces["trial_contribution"][..., 1::2],
            atol=1e-15,
        )

    def test_sigmoid_decomposition_and_between_recipe_identity(self):
        rng = np.random.default_rng(22)
        g, s, c, baseline = rng.normal(size=(4, 2, 6))
        effects, cells = sigmoid_attribution(g, s, c, np.ones(6), 0.25)
        np.testing.assert_allclose(
            effects["self"] + effects["cross"], effects["total"], atol=1e-15
        )
        from fsrl.analysis.statistics import stable_sigmoid

        p = {name: stable_sigmoid(m / 0.25) for name, m in cells.items()}
        b = stable_sigmoid(baseline / 0.25)
        np.testing.assert_allclose(
            p["GSC"] - b, effects["total"] + p["G"] - b, atol=1e-15
        )

    def test_fixed_scoring_and_scalar_rank_invariance(self):
        protocol, inputs, retained, params, spec = fixture()
        context = scoring(protocol, retained)
        margin = global_references(inputs, params, spec["integrity"])["RF"]["margin"]
        first, _ = endpoints(margin, 2, context, 0.25, 1e-10)
        second, _ = endpoints(margin * 3, 6, context, 0.25, 1e-10)
        for key in first:
            if key.startswith("latent/"):
                np.testing.assert_array_equal(first[key], second[key])
        self.assertAlmostEqual(context["slope_weight"].sum(), 0)
        self.assertAlmostEqual(context["serial_weight"].sum(), 0)
        self.assertAlmostEqual(context["distance"] @ context["slope_weight"], 1)
        readout = readout_accounting(margin, context, 0.25)
        for group in context["groups"]:
            np.testing.assert_allclose(
                readout["total"][group],
                readout["correct_shortfall"][group]
                + readout["wrong_rescue"][group]
                + readout["ties"][group],
                atol=1e-15,
            )
        contrasts = global_contrasts({name: first for name in ("RF", "AF", "RL", "AL")})
        self.assertEqual(contrasts["total"]["latent/strict_correct_order"].sum(), 0)

    def test_cohorts_and_bootstrap_exclusions(self):
        protocol, inputs, _, _, spec = fixture()
        count = component_counts(
            inputs["support_pairs"], inputs["retention"].T, protocol.n_items
        )
        self.assertEqual(count[0], 8)
        self.assertEqual(count[1], 1)
        row = summarize({"x": np.asarray([np.nan, 1.0, 2.0])}, 1, spec["statistics"])[
            "x"
        ]
        self.assertEqual(row["subjects"], 2)
        self.assertEqual(row["excluded_subject_indices"], [0])
        self.assertEqual(direction([row] * 3), "consistently_positive")

    def test_complete_synthetic_analysis_with_parent_bridge(self):
        protocol, inputs, retained, params, spec = fixture()
        g = global_references(inputs, params, spec["integrity"])["RF"]
        local = local_decomposition(inputs, params["gamma_L"])
        l = local["self_margin"] + local["cross_margin"]
        arrays = {
            "liu__w": g["state"],
            "liu__retention": retained,
            "liu__bundles__local_off__logits": g["margin"],
            "liu__bundles__global_off__logits": l,
            "liu__bundles__intact__logits": g["margin"] + l,
        }
        parent = liu_endpoints(
            {"intact": {"logits": g["margin"]}}, retained, protocol, 0.25
        )
        row = {"parameters": params, "raw_endpoints": {"liu": parent}}
        raw, result, context = global_analysis(arrays, inputs, row, protocol, spec, 42)
        _, summary = local_analysis(
            arrays, inputs, row, raw["cells"]["RF"]["margin"], context, spec, 42
        )
        self.assertLess(result["checks"]["global_margin"], 1e-12)
        self.assertEqual(summary["checks"]["nonlearned_self"], 0)

    def test_independent_numeric_and_interval_verifier(self):
        protocol, inputs, retained, params, spec = fixture()
        g = global_references(inputs, params, spec["integrity"])["RF"]
        pieces = local_decomposition(inputs, params["gamma_L"])
        l = pieces["self_margin"] + pieces["cross_margin"]
        source = {
            "liu__w": g["state"],
            "liu__retention": retained,
            "liu__bundles__local_off__logits": g["margin"],
            "liu__bundles__global_off__logits": l,
            "liu__bundles__intact__logits": g["margin"] + l,
        }
        parent = liu_endpoints(
            {"intact": {"logits": g["margin"]}}, retained, protocol, 0.25
        )
        row = {"parameters": params, "raw_endpoints": {"liu": parent}}
        gr, gs, context = global_analysis(source, inputs, row, protocol, spec, 42)
        lr, _ = local_analysis(source, inputs, row, g["margin"], context, spec, 42)
        arrays = flatten_arrays({"1": {"global": gr, "local": lr, "scoring": context}})
        original = {"conditions": {"1/score_only": row, "1/score_trace": row}}
        score_source = {**source, "liu__bundles__intact__logits": g["margin"]}
        with patch(
            "fsrl.experiments.learner_diagnostics.verification.load_arrays",
            side_effect=[(score_source, inputs), (source, inputs)],
        ):
            check = verify_numeric(1, arrays, original, spec)
        self.assertTrue(check["passed"])
        count, error = verify_summaries(
            arrays, "1__global__endpoints", gs["cells"], 42, 40
        )
        self.assertGreater(count, 40)
        self.assertLess(error, 1e-12)


if __name__ == "__main__":
    unittest.main()
