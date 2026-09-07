"""Independent within-panel resampling followed by equal averaging of effects."""

import numpy as np

from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.evidence_routing.measurement import tau_matrix, weighted_tau


def panel_draws(raw, endpoints, shifted, seed, spec):
    """Share subject draws across cells, but keep generic and Liu units distinct."""
    counts = {}
    for phase, n in (
        ("generic", len(endpoints["A0"]["generic_global"])),
        ("liu", len(shifted["A"])),
    ):
        counts[phase] = bootstrap_counts(
            np.random.default_rng(seed), spec["statistics"]["samples"], n
        )
    tau = {}
    for cell in raw:
        matrix = tau_matrix(raw[cell]["liu"]["routes__global__internal__orders"])
        n = matrix.shape[0]
        tau[cell] = (
            weighted_tau(matrix, np.ones((1, n)))[0],
            weighted_tau(matrix, counts["liu"]),
        )
    points, draws = {}, {}
    for name, coefficient in spec["estimands"]["contrasts"].items():
        key = name + "/global_all77_tau"
        points[key] = sum(c * tau[cell][0] for cell, c in coefficient.items())
        draws[key] = sum(c * tau[cell][1] for cell, c in coefficient.items())
        for metric in endpoints["A0"]:
            values = np.stack(
                [c * endpoints[cell][metric] for cell, c in coefficient.items()]
            ).sum(0)
            weights = counts[metric.split("_")[0]]
            key = name + "/" + metric
            points[key], draws[key] = values.mean(), weights @ values / len(values)
    for name, values in shifted.items():
        key = "order_shift/" + name
        points[key], draws[key] = values.mean(), counts["liu"] @ values / len(values)
    return points, draws


def panel_mean(panels):
    """Panels are fixed; neither networks nor panel IDs are bootstrap units."""
    points = {key: np.mean([p[key] for p, _ in panels]) for key in panels[0][0]}
    draws = {key: np.mean([d[key] for _, d in panels], axis=0) for key in points}
    if not all(np.isfinite(value).all() for value in draws.values()):
        raise ValueError("undefined complete-panel effect")
    summary = {
        key: {
            "point": float(point),
            "interval": dict(
                zip(
                    ("lower", "upper"),
                    np.quantile(draws[key], [0.025, 0.975]).tolist(),
                    strict=True,
                )
            ),
        }
        for key, point in points.items()
    }
    return summary, draws
