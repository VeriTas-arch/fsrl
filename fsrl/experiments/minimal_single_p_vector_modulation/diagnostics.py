"""Pure support-write geometry diagnostics for vector modulation."""

from __future__ import annotations

import numpy as np
import torch

TOLERANCE = 1e-12


def geometry_records(model, support_inputs: torch.Tensor) -> dict[str, np.ndarray]:
    """Return one record per subject and support microstep; never include queries."""
    subjects = support_inputs.shape[2]
    fast_weights = model.initial_fast_weights(subjects)
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "trial",
            "step",
            "active",
            "R",
            "c_star",
            "U_norm",
            "E_norm",
            "modulation_mean",
            "modulation_variance",
            "positive_fraction",
            "negative_fraction",
            "actual_delta_norm",
            "clipping_residual_ratio",
            "clamp_fraction",
        )
    }
    for trial, inputs in enumerate(support_inputs.unbind(0)):
        hidden = model.initial_hidden(subjects)
        eligibility = model.initial_eligibility(subjects)
        for step, current in enumerate(inputs.unbind(0)):
            old_fast = fast_weights
            _, modulation, hidden, next_eligibility, proposal = model.step(
                current, hidden, eligibility, fast_weights
            )
            write = modulation[:, :, None] * eligibility
            unbounded = old_fast + write
            e2 = eligibility.square().sum((1, 2))
            u2 = write.square().sum((1, 2))
            active = (e2.sqrt() > TOLERANCE) & (u2.sqrt() > TOLERANCE)
            c_star = (write * eligibility).sum((1, 2)) / e2.clamp_min(TOLERANCE)
            residual = write - c_star[:, None, None] * eligibility
            ratio = residual.square().sum((1, 2)) / u2.clamp_min(TOLERANCE)
            row_energy = eligibility.square().sum(2)
            weighted_mean = (modulation * row_energy).sum(1) / e2.clamp_min(TOLERANCE)
            weighted_variance = (
                (modulation - weighted_mean[:, None]).square() * row_energy
            ).sum(1) / e2.clamp_min(TOLERANCE)
            clip_residual = proposal - unbounded
            values = {
                "trial": torch.full_like(e2, trial),
                "step": torch.full_like(e2, step),
                "active": active,
                "R": torch.where(active, ratio, torch.nan),
                "c_star": torch.where(active, c_star, torch.nan),
                "U_norm": u2.sqrt(),
                "E_norm": e2.sqrt(),
                "modulation_mean": modulation.mean(1),
                "modulation_variance": weighted_variance,
                "positive_fraction": (modulation > 0).float().mean(1),
                "negative_fraction": (modulation < 0).float().mean(1),
                "actual_delta_norm": (proposal - old_fast).square().sum((1, 2)).sqrt(),
                "clipping_residual_ratio": clip_residual.square().sum((1, 2)).sqrt()
                / u2.sqrt().clamp_min(TOLERANCE),
                "clamp_fraction": (proposal.abs() == 50.0).float().mean((1, 2)),
            }
            for name, value in values.items():
                rows[name].append(value.detach().cpu().numpy())
            eligibility = next_eligibility
            fast_weights = proposal
    return {name: np.concatenate(values) for name, values in rows.items()}


def summarize_geometry(records: dict[str, np.ndarray]) -> dict:
    active = records["active"].astype(bool)
    result = {
        "records": int(active.size),
        "active_records": int(active.sum()),
        "active_fraction": float(active.mean()),
        "vector_used": bool(active.any() and np.nanmax(records["R"]) > 1e-10),
    }
    for name in (
        "R",
        "c_star",
        "U_norm",
        "E_norm",
        "modulation_mean",
        "modulation_variance",
        "positive_fraction",
        "negative_fraction",
        "actual_delta_norm",
        "clipping_residual_ratio",
        "clamp_fraction",
    ):
        values = records[name]
        finite = values[np.isfinite(values)]
        result[name] = {
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "maximum": float(finite.max()) if finite.size else None,
        }
    return result


__all__ = ["TOLERANCE", "geometry_records", "summarize_geometry"]
