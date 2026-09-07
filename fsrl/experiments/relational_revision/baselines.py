"""Information-matched mathematical controls; no hidden group/change labels."""

import numpy as np


def trajectory(panel, rule):
    count = len(panel["old_scores"])
    score = np.zeros((count, 8))
    covariance = np.broadcast_to(np.eye(8) * 0.25, (count, 8, 8)).copy()
    margins = []
    query = panel["query_pairs"]
    for t, pairs in enumerate(panel["pairs"]):
        feature = np.eye(8)[pairs[:, 0]] - np.eye(8)[pairs[:, 1]]
        residual = panel["q"][t] - np.einsum("bi,bi->b", feature, score)
        if rule == "adaptive_delta":
            rate = 0.2 + 0.6 * np.minimum(abs(residual) / 0.4, 1)
            score += 0.5 * (rate * residual)[:, None] * feature
        elif rule == "covariance_filter":
            covariance += np.eye(8)[None] * 0.0005
            projected = np.einsum("bij,bj->bi", covariance, feature)
            variance = np.einsum("bi,bi->b", feature, projected) + 0.05**2
            gain = projected / variance[:, None]
            score += gain * residual[:, None]
            # Joseph form is stable and retains the complete covariance geometry.
            transform = np.eye(8)[None] - gain[:, :, None] * feature[:, None, :]
            covariance = transform @ covariance @ transform.transpose(0, 2, 1)
            covariance += 0.05**2 * gain[:, :, None] * gain[:, None, :]
        else:
            raise ValueError("unknown rule")
        if t >= 23:
            margins.append((score[:, query[:, 0]] - score[:, query[:, 1]]) / 0.1)
    return {"margins": np.stack(margins, axis=1)}
