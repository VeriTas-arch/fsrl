"""Independent CPU affine-exponential reference and original discrete rule."""

import numpy as np
from scipy.linalg import expm


def differences(cues: np.ndarray) -> np.ndarray:
    width = cues.shape[-1] // 2
    return cues[..., :width].astype(float) - cues[..., width:]


def discrete(inputs: dict, eta: float, gain: float) -> dict:
    x = differences(inputs["support_cues"])
    w = np.zeros(x.shape[1:])
    history = []
    for t, activity in enumerate(x):
        error = inputs["signed"][t] - np.einsum("sf,sf->s", w, activity)
        amount = eta * inputs["retention"][t] * error / (1e-8 + (activity**2).sum(-1))
        w = w + amount[:, None] * activity
        history.append(w.copy())
    return {
        "trajectory": np.stack(history, axis=1),
        "margin": gain * np.einsum("sf,sqf->sq", w, differences(inputs["query_cues"])),
    }


def affine_matrix(
    x, teaching, admission, eta, scale, prediction, *, availability=1.0, unclamped=1.0
) -> np.ndarray:
    """State order D+/D-/S+/S-/E+/E-/integral(E+)/integral(E-)/constant."""
    alpha = -np.log1p(-eta) / 0.5
    td, ts, te = 0.002 * scale, 0.002 * scale, 0.06 * scale
    norm = np.dot(x, x)
    kappa = 0.5 * availability
    matrix = np.zeros((9, 9))
    for branch, sign in enumerate((1.0, -1.0)):
        matrix[branch, branch] = -1 / td
        matrix[branch, 6 + branch] = alpha * availability * admission * norm / td
        matrix[branch, 8] = prediction[branch] / td
        matrix[2 + branch, branch] = (1 - kappa) / ts
        matrix[2 + branch, 2 + branch] = -1 / ts
        matrix[2 + branch, 8] = sign * kappa * teaching / (2 * ts)
        matrix[4 + branch, :4] = sign * unclamped * np.array([-1, 1, 1, -1]) / (2 * te)
        matrix[4 + branch, 4 + branch] = -(1e-8 + norm) / te
        matrix[6 + branch, 4 + branch] = 1
    return matrix


def affine_support(
    inputs: dict,
    eta: float,
    scale: float,
    *,
    duration: float = 1,
    control: str = "intact",
) -> np.ndarray:
    x = differences(inputs["support_cues"])
    width = x.shape[-1]
    y = np.concatenate(
        (np.ones((x.shape[1], 2 * width)), np.zeros((x.shape[1], 6))), axis=1
    )
    alpha = -np.log1p(-eta) / 0.5
    teaching = np.asarray(inputs["signed"])
    if control == "teaching_shuffle":
        teaching = np.roll(teaching, 1, axis=0)
    availability = float(control != "teacher_off")
    history = []
    for t, activity in enumerate(x):
        for s, cue in enumerate(activity):
            prediction = [
                np.dot(y[s, :width], cue),
                np.dot(y[s, width : 2 * width], cue),
            ]
            matrix = affine_matrix(
                cue,
                teaching[t, s],
                inputs["retention"][t, s],
                eta,
                scale,
                prediction,
                availability=availability,
                unclamped=float(control != "mismatch_clamp"),
            )
            propagated = expm(duration * matrix) @ np.r_[y[s, 2 * width :], 0, 0, 1]
            change = alpha * availability * inputs["retention"][t, s] * cue
            y[s, :width] += change * propagated[6]
            y[s, width : 2 * width] += change * propagated[7]
            y[s, 2 * width :] = propagated[:6]
        history.append(y.copy())
    return np.stack(history, axis=1)
