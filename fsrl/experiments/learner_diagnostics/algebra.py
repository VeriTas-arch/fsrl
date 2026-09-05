"""Input-only state references and exact local expression decomposition."""

from __future__ import annotations

import numpy as np

from fsrl.analysis.statistics import stable_sigmoid


def differences(cues: np.ndarray) -> np.ndarray:
    left, right = np.split(np.asarray(cues, dtype=np.float64), 2, axis=-1)
    return left - right


def online_state(x, evidence, admitted, eta: float, epsilon: float) -> np.ndarray:
    """Subject-first inputs; omission skips the whole error update."""
    state = np.zeros((x.shape[0], x.shape[2]))
    for t in range(x.shape[1]):
        error = evidence[:, t] - np.einsum("sd,sd->s", state, x[:, t])
        rate = eta * admitted[:, t] / (epsilon + np.square(x[:, t]).sum(-1))
        state += (rate * error)[:, None] * x[:, t]
    return state


def least_squares_state(x, evidence, admitted, rcond: float) -> tuple:
    state = np.zeros((x.shape[0], x.shape[2]))
    ranks = np.zeros(x.shape[0], dtype=np.int64)
    for s in range(x.shape[0]):
        selected = admitted[s].astype(bool)
        if np.any(selected):
            state[s], _, ranks[s], _ = np.linalg.lstsq(
                x[s, selected], evidence[s, selected], rcond=rcond
            )
    return state, ranks


def global_references(inputs: dict, parameters: dict, integrity: dict) -> dict:
    x = differences(inputs["support_cues"]).transpose(1, 0, 2)
    q = differences(inputs["query_cues"])
    d, z = (inputs[name].T for name in ("signed", "retention"))
    cells = {}
    for letter, admitted in (("R", z), ("A", np.ones_like(z))):
        finite = online_state(x, d, admitted, parameters["eta"], 1e-8)
        limit, rank = least_squares_state(x, d, admitted, integrity["svd_rcond"])
        for mode, state in (("F", finite), ("L", limit)):
            residual = np.einsum("sd,std->st", state, x) - d
            if mode == "L":
                # Least-squares stationarity also covers rounded metric inputs.
                np.testing.assert_allclose(
                    np.einsum("st,std->sd", residual * admitted, x),
                    0,
                    atol=integrity["algebra_atol"],
                    rtol=0,
                )
            cells[letter + mode] = {
                "state": state,
                "margin": parameters["gamma_G"] * np.einsum("sd,sqd->sq", state, q),
                "support_residual": residual,
                "admitted": admitted,
                "design_rank": rank,
            }
    return cells


def keys(cues: np.ndarray) -> np.ndarray:
    left, right = np.split(np.asarray(cues, dtype=np.float64), 2, axis=-1)
    raw = left[..., :, None] * right[..., None, :]
    raw = raw - np.swapaxes(raw, -1, -2)
    flat = raw.reshape(*raw.shape[:-2], -1)
    return flat / np.maximum(np.linalg.norm(flat, axis=-1, keepdims=True), 1e-8)


def local_decomposition(inputs: dict, gain: float) -> dict:
    support = keys(inputs["support_cues"]).transpose(1, 0, 2)
    query = keys(inputs["query_cues"])
    contributions = gain * np.einsum("std,sqd->stq", support, query)
    contributions *= inputs["local_evidence"].T[..., None]
    support_pairs = np.sort(inputs["support_pairs"], axis=-1)
    query_pairs = np.sort(inputs["query_pairs"], axis=-1)
    same = np.all(support_pairs[:, :, None] == query_pairs[:, None], axis=-1)
    self_margin = np.where(same, contributions, 0).sum(axis=1)
    cross_margin = np.where(~same, contributions, 0).sum(axis=1)
    return {
        "trial_contribution": contributions,
        "same_relation": same,
        "self_margin": self_margin,
        "cross_margin": cross_margin,
    }


def sigmoid_attribution(global_margin, self_margin, cross_margin, signs, temperature):
    """Symmetric two-order allocation of an exact, nonlinear probability change."""
    cells = {
        "G": global_margin,
        "GS": global_margin + self_margin,
        "GC": global_margin + cross_margin,
        "GSC": global_margin + self_margin + cross_margin,
    }
    p = {
        name: stable_sigmoid(signs * margin / temperature)
        for name, margin in cells.items()
    }
    return {
        "self": 0.5 * (p["GS"] - p["G"] + p["GSC"] - p["GC"]),
        "cross": 0.5 * (p["GC"] - p["G"] + p["GSC"] - p["GS"]),
        "total": p["GSC"] - p["G"],
        "full_minus_self_only": p["GSC"] - p["GS"],
    }, cells


def component_counts(support_pairs, admitted, n_items: int) -> np.ndarray:
    """Connectivity is computed only from admitted observed support edges."""
    counts = []
    for pairs, selected in zip(support_pairs, admitted, strict=True):
        labels = np.arange(n_items)
        for (i, j), keep in zip(pairs, selected, strict=True):
            if keep:
                labels[labels == labels[j]] = labels[i]
        counts.append(len(np.unique(labels)))
    return np.asarray(counts, dtype=np.int64)
