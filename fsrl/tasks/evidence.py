"""Evidence-admission rules for the maintained global/local model."""

from __future__ import annotations

import numpy as np


def broader_local_admission(
    global_admission: np.ndarray, reliability: np.ndarray
) -> np.ndarray:
    """Return ``z + (1-z)p`` from the confirmed differential-access equation."""

    global_values, reliability_values = np.broadcast_arrays(
        np.asarray(global_admission, dtype=np.float64),
        np.asarray(reliability, dtype=np.float64),
    )
    if np.any((global_values < 0.0) | (global_values > 1.0)):
        raise ValueError("global admission must lie in [0, 1]")
    if np.any((reliability_values < 0.0) | (reliability_values > 1.0)):
        raise ValueError("reliability must lie in [0, 1]")
    return global_values + (1.0 - global_values) * reliability_values
