"""Exact algebraic identities for global-policy field factorial estimands."""

from __future__ import annotations

import numpy as np


def factorial_identity_errors(
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Evaluate every registered dependent contrast identity row by row."""

    return {
        "D_equals_A_plus_R": np.abs(values["D"] - values["A"] - values["R"]),
        "D_equals_Delta_A_plus_C_A": np.abs(
            values["D"] - values["Delta_A"] - values["C_A"]
        ),
        "D_equals_Delta_R_plus_C_R": np.abs(
            values["D"] - values["Delta_R"] - values["C_R"]
        ),
        "D_equals_Q_shape_plus_C_shape": np.abs(
            values["D"] - values["Q_shape"] - values["C_shape"]
        ),
        "I_equals_Delta_A_minus_C_R": np.abs(
            values["I"] - values["Delta_A"] + values["C_R"]
        ),
        "I_equals_Delta_R_minus_C_A": np.abs(
            values["I"] - values["Delta_R"] + values["C_A"]
        ),
        "Delta_A_equals_A_plus_half_I": np.abs(
            values["Delta_A"] - values["A"] - 0.5 * values["I"]
        ),
        "C_A_equals_R_minus_half_I": np.abs(
            values["C_A"] - values["R"] + 0.5 * values["I"]
        ),
        "Delta_R_equals_R_plus_half_I": np.abs(
            values["Delta_R"] - values["R"] - 0.5 * values["I"]
        ),
        "C_R_equals_A_minus_half_I": np.abs(
            values["C_R"] - values["A"] + 0.5 * values["I"]
        ),
        "Delta_A_equals_Q_shape_plus_Q_amp": np.abs(
            values["Delta_A"] - values["Q_shape"] - values["Q_amp"]
        ),
        "C_shape_equals_C_A_plus_Q_amp": np.abs(
            values["C_shape"] - values["C_A"] - values["Q_amp"]
        ),
    }
