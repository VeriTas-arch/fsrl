"""Matched architecture noninferiority; independent panels, no network pooling."""

import numpy as np

from fsrl.analysis.policy import exact_probability
from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.training_strategy.estimands import subject_means


def probabilities(raw):
    generic = raw["generic"]
    probability = exact_probability(generic["margins"] * generic["signs"], 1.0)
    values = {
        "generic_" + label: subject_means(probability, mask)
        for label, mask in (
            ("learned", generic["learned"]),
            ("nonlearned", ~generic["learned"]),
        )
    }
    values.update(
        {
            "liu_" + label: raw["liu"]["endpoints__intact__probability__" + label]
            for label in ("learned", "nonlearned", "omitted")
        }
    )
    return values


def paired_probability(single, dual, boot, samples=2000):
    points, draws, subjects = {}, {}, {}
    for name in single:
        a, b = np.asarray(single[name]), np.asarray(dual[name])
        if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
            raise RuntimeError("architecture endpoint eligibility differs")
        mask = np.isfinite(a) & np.isfinite(b)
        if not mask.any():
            raise RuntimeError("no paired endpoint subjects")
        difference = a[mask] - b[mask]
        counts = bootstrap_counts(np.random.default_rng(boot), samples, len(difference))
        points[name] = difference.mean()
        draws[name] = counts @ difference / len(difference)
        subjects[name] = {
            "total": len(a),
            "included": np.flatnonzero(mask).tolist(),
            "excluded": np.flatnonzero(~mask).tolist(),
        }
    return (points, draws), subjects


def noninferiority(summary):
    return {
        name: {
            "noninferior": row["interval"]["lower"] >= -0.02,
            "materially_inferior": row["interval"]["upper"] < -0.02,
        }
        for name, row in summary.items()
    }


def qualification_checks():
    a = {"omitted": np.array([0.8, np.nan, 0.6, 0.4])}
    b = {"omitted": np.array([0.7, np.nan, 0.62, 0.5])}
    (point, draws), sizes = paired_probability(a, b, 12, 60)
    counts = np.random.default_rng(12).multinomial(3, [1 / 3] * 3, size=60)
    expected = [np.mean(np.repeat([0.1, -0.02, -0.1], row)) for row in counts]
    np.testing.assert_allclose(draws["omitted"], expected, atol=1e-14)
    np.testing.assert_allclose(
        point["omitted"], np.mean([0.1, -0.02, -0.1]), atol=1e-14
    )
    assert sizes["omitted"]["excluded"] == [1]
    return {"complete_case_paired_bootstrap": True}
