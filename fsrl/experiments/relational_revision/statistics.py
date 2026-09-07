"""Episode-paired summaries; no pooling of independently trained networks."""

import numpy as np
from scipy.special import expit

from .inputs import QUERIES


def mean_mask(values, mask):
    if not np.all(mask.sum(-1)):
        raise ValueError("empty query group")
    return (values * mask).sum(-1) / mask.sum(-1)


def antisymmetric(margins, pair):
    forward = np.flatnonzero(np.all(QUERIES == pair, axis=1))[0]
    reverse = np.flatnonzero(np.all(QUERIES == pair[::-1], axis=1))[0]
    return (margins[..., forward] - margins[..., reverse]) / 2


def endpoints(stable, revision, outlier, panel):
    remote, same, seen = (
        panel[k] for k in ("changed_remote", "unchanged", "post_seen")
    )
    old_sign, final_sign = panel["old_sign"], panel["final_sign"]
    ce = np.logaddexp(0, -revision * final_sign[:, None])
    old_ce = np.logaddexp(0, -revision * old_sign[:, None])
    difference = abs(expit(outlier) - expit(stable))
    raw_delta = revision[:, 1] - revision[:, 0]
    delta = raw_delta - (stable[:, 1] - stable[:, 0])
    return {
        "competence_seen": mean_mask(expit(stable[:, -1] * old_sign), seen),
        "competence_remote": mean_mask(expit(stable[:, -1] * old_sign), remote),
        "revision_gain": mean_mask(ce[:, 0] - ce[:, -1], remote),
        "revision_specificity": mean_mask(
            np.logaddexp(0, -stable[:, -1] * final_sign) - ce[:, -1], remote
        ),
        "preservation_cost": mean_mask(
            old_ce[:, -1] - np.logaddexp(0, -stable[:, -1] * old_sign), same
        ),
        "preservation_from_before": mean_mask(old_ce[:, -1] - old_ce[:, 0], same),
        "outlier_recovery": mean_mask(difference[:, 1] - difference[:, -1], remote),
        "outlier_initial": mean_mask(difference[:, 1], remote),
        "outlier_final": mean_mask(difference[:, -1], remote),
        "AD_change": antisymmetric(delta, (0, 3)),
        "BC_change": antisymmetric(delta, (1, 2)),
        "AD_raw_change": antisymmetric(raw_delta, (0, 3)),
        "BC_raw_change": antisymmetric(raw_delta, (1, 2)),
        "AD_before": antisymmetric(revision[:, 0], (0, 3)),
        "BC_before": antisymmetric(revision[:, 0], (1, 2)),
    }


def interval(values, draws):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite endpoint")
    samples = np.mean(
        [row[index].mean(1) for row, index in zip(values, draws, strict=True)], axis=0
    )
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def decision(summary, history):
    low = lambda key: summary[key]["ci95"][0]
    high = lambda key: summary[key]["ci95"][1]
    return {
        "competence": low("competence_seen") > 0.5 and low("competence_remote") > 0.5,
        "revision": low("revision_gain") > 0 and low("revision_specificity") > 0,
        "preservation": high("preservation_cost") <= 0.05,
        "crossover": (low("AD_change") > 0 if history == 0 else high("AD_change") < 0)
        and low("BC_change") > 0,
        "outlier_recovery": low("outlier_recovery") > 0,
    }
