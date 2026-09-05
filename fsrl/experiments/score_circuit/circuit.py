"""Direct bounded-efficacy integration using actual compartment mismatch."""

from collections.abc import Callable

import numpy as np
import torch

CHUNK_STEPS = 16


def initial_state(subjects: int, width: int, device: str) -> torch.Tensor:
    efficacy = torch.ones((subjects, 2 * width), dtype=torch.float64, device=device)
    activity = efficacy.new_zeros((subjects, 6))
    return torch.cat((efficacy, activity), dim=-1)


def coefficients(eta: float, scale: float, steps: int, device: str) -> torch.Tensor:
    return torch.tensor(
        [-np.log1p(-eta) / 0.5, 0.002 * scale, 0.002 * scale, 0.06 * scale, 1 / steps],
        dtype=torch.float64,
        device=device,
    )


def derivative(y, rates, teaching, admission, availability, unclamped, coefficients):
    width = rates.shape[-1]
    plus, minus = y[:, :width], y[:, width : 2 * width]
    dp, dm, sp, sm, ep, em = y[:, 2 * width :].unbind(-1)
    alpha, tau_d, tau_s, tau_e, _ = coefficients.unbind()
    activity = rates - 2.0
    pooled = 1e-8 + 2 * activity.abs().sum(-1)
    # The learning signal has no direct access to teaching or a computed target error.
    mismatch = ((sp - dp) - (sm - dm)) * unclamped
    write = alpha * availability * admission
    kappa = 0.5 * availability
    currents = torch.stack(
        (
            ((plus * activity).sum(-1) - dp) / tau_d,
            ((minus * activity).sum(-1) - dm) / tau_d,
            ((1 - kappa) * dp + kappa * teaching / 2 - sp) / tau_s,
            ((1 - kappa) * dm - kappa * teaching / 2 - sm) / tau_s,
            (mismatch / 2 - pooled * ep) / tau_e,
            (-mismatch / 2 - pooled * em) / tau_e,
        ),
        dim=-1,
    )
    return torch.cat(
        (
            write[:, None] * activity * ep[:, None],
            write[:, None] * activity * em[:, None],
            currents,
        ),
        dim=-1,
    )


def rk4_step(y, rates, teaching, admission, availability, unclamped, coefficients):
    arguments = (rates, teaching, admission, availability, unclamped, coefficients)
    h = coefficients[-1]
    k1 = derivative(y, *arguments)
    k2 = derivative(y + h * k1 / 2, *arguments)
    k3 = derivative(y + h * k2 / 2, *arguments)
    k4 = derivative(y + h * k3, *arguments)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integration_chunk(
    y, diagnostics, rates, teaching, admission, availability, unclamped, coefficients
):
    width = rates.shape[-1]
    for _ in range(CHUNK_STEPS):
        raw = rk4_step(
            y, rates, teaching, admission, availability, unclamped, coefficients
        )
        g = raw[:, : 2 * width].clamp(0, 2)
        y = torch.cat((g, raw[:, 2 * width :]), dim=-1)
        observations = torch.stack(
            (
                g.amin(-1),
                -g.amax(-1),
                (64 + y[:, 2 * width :]).amin(-1),
                -(g[:, :width] + g[:, width:] - 2).abs().amax(-1),
            ),
            dim=-1,
        )
        extrema = torch.minimum(diagnostics[:, :4], observations)
        hits = diagnostics[:, 4] + (raw[:, : 2 * width] != g).sum(-1)
        diagnostics = torch.cat((extrema, hits[:, None]), dim=-1)
    return y, diagnostics


def integrate_support(
    inputs: dict,
    eta: float,
    scale: float,
    steps: int,
    runner: Callable,
    *,
    device: str = "cuda",
    control: str = "intact",
    duration: float = 1.0,
) -> dict:
    cues = torch.as_tensor(inputs["support_cues"], dtype=torch.float64, device=device)
    width = cues.shape[-1] // 2
    rates = 2 + cues[..., :width] - cues[..., width:]
    teaching = torch.as_tensor(inputs["signed"], dtype=torch.float64, device=device)
    if control == "teaching_shuffle":
        teaching = teaching.roll(1, dims=0)
    admission = torch.as_tensor(inputs["retention"], dtype=torch.float64, device=device)
    availability = teaching.new_full((cues.shape[1],), float(control != "teacher_off"))
    unclamped = teaching.new_tensor(float(control != "mismatch_clamp"))
    constants = coefficients(eta, scale, steps, device)
    constants[-1] *= duration
    y = initial_state(cues.shape[1], width, device)
    diagnostics = teaching.new_full((cues.shape[1], 5), float("inf"))
    diagnostics[:, 4] = 0
    history = []
    with torch.no_grad():
        for trial in range(cues.shape[0]):
            for _ in range(steps // CHUNK_STEPS):
                y, diagnostics = runner(
                    y,
                    diagnostics,
                    rates[trial],
                    teaching[trial],
                    admission[trial],
                    availability,
                    unclamped,
                    constants,
                )
            history.append(y.clone())
    return {
        "trajectory": torch.stack(history, dim=1).cpu().numpy(),
        "diagnostics": diagnostics.cpu().numpy(),
        "minimum_input_rate": float(rates.min().item()),
    }


def query_read(
    state: np.ndarray,
    cues: np.ndarray,
    gain: float,
    tau_d: float,
    duration: float = 0.1,
    *,
    reverse: bool = False,
) -> np.ndarray:
    """Exact no-write query ODE, with neural history retained between queries."""
    width = cues.shape[-1] // 2
    activity = cues[..., :width] - cues[..., width:]
    targets = np.stack(
        (
            np.einsum("sf,sqf->sq", state[:, :width], activity),
            np.einsum("sf,sqf->sq", state[:, width : 2 * width], activity),
        ),
        axis=-1,
    )
    prediction = state[:, 2 * width : 2 * width + 2].copy()
    decay = np.exp(-duration / tau_d)
    result = np.empty(cues.shape[:2])
    indices = range(cues.shape[1] - 1, -1, -1) if reverse else range(cues.shape[1])
    for index in indices:
        prediction = targets[:, index] + (prediction - targets[:, index]) * decay
        result[:, index] = gain * (prediction[:, 0] - prediction[:, 1])
    return result
