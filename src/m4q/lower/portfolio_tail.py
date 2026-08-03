from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def var_es_integral(losses: np.ndarray, confidence: float) -> tuple[float, float, float]:
    ordered = np.sort(np.asarray(losses, dtype=float))
    if ordered.ndim != 1 or ordered.size == 0 or not np.all(np.isfinite(ordered)):
        raise ValueError("VaR/ES requires finite one-dimensional losses")
    if not 0.0 < confidence < 1.0:
        raise ValueError("VaR/ES confidence must lie strictly between zero and one")
    index = int(np.ceil(confidence * ordered.size)) - 1
    value_at_risk = float(ordered[index])
    tail_mass = (1.0 - confidence) * ordered.size
    full_count = int(np.floor(tail_mass))
    fractional = tail_mass - full_count
    tail_sum = float(ordered[-full_count:].sum()) if full_count else 0.0
    if fractional > 1e-15:
        tail_sum += fractional * float(ordered[-full_count - 1])
    expected_shortfall = tail_sum / tail_mass
    lower = float(ordered[max(index - 1, 0)])
    upper = float(ordered[min(index + 1, ordered.size - 1)])
    return value_at_risk, expected_shortfall, 0.5 * (upper - lower)


@dataclass(frozen=True)
class TailRiskResult:
    value_at_risk: float
    expected_shortfall: float
    effective_tail_observations: float
    quantile_resolution: float
    es_interval: tuple[float, float]
    status: str


def empirical_tail_risk(
    losses: np.ndarray,
    confidence: float,
    *,
    minimum_tail_observations: int,
    warning_tail_observations: int | None = None,
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> TailRiskResult:
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0 or not np.all(np.isfinite(losses)):
        raise ValueError("tail sample requires finite one-dimensional losses")
    if not 0.0 < confidence < 1.0 or minimum_tail_observations <= 0:
        raise ValueError("tail confidence and minimum observations are invalid")
    warning = warning_tail_observations or minimum_tail_observations
    if warning < minimum_tail_observations:
        raise ValueError("warning tail observations cannot be below the rejection threshold")
    effective = losses.size * (1.0 - confidence)
    if effective + 1e-12 < minimum_tail_observations:
        raise ValueError(
            "insufficient effective tail observations: "
            f"observed={effective:.6g}, required={minimum_tail_observations}"
        )
    value_at_risk, expected_shortfall, resolution = var_es_integral(losses, confidence)
    status = "warn" if effective + 1e-12 < warning else "pass"
    if bootstrap_samples:
        if bootstrap_samples < 50:
            raise ValueError("tail bootstrap requires at least 50 resamples")
        generator = np.random.default_rng(seed)
        estimates = np.empty(bootstrap_samples)
        for index in range(bootstrap_samples):
            draw = generator.choice(losses, size=losses.size, replace=True)
            estimates[index] = var_es_integral(draw, confidence)[1]
        interval = tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))
    else:
        interval = (expected_shortfall, expected_shortfall)
    return TailRiskResult(
        value_at_risk, expected_shortfall, effective, resolution, interval, status
    )


def nonlinear_portfolio_loss(
    shocks: np.ndarray, linear_exposure: np.ndarray, gamma_exposure: np.ndarray
) -> float:
    shocks = np.asarray(shocks, dtype=float)
    linear = np.asarray(linear_exposure, dtype=float)
    gamma = np.asarray(gamma_exposure, dtype=float)
    if shocks.ndim != 1 or linear.shape != shocks.shape or gamma.shape != shocks.shape:
        raise ValueError("stress shocks and exposures must be aligned vectors")
    return float(-(linear @ shocks + 0.5 * gamma @ (shocks * shocks)))


def reverse_stress_scale(
    shock_direction: np.ndarray,
    linear_exposure: np.ndarray,
    gamma_exposure: np.ndarray,
    *,
    loss_threshold: float,
    maximum_scale: float,
) -> float:
    if loss_threshold <= 0.0 or maximum_scale <= 0.0:
        raise ValueError("reverse stress threshold and maximum scale must be positive")

    direction = np.asarray(shock_direction, dtype=float)
    linear = np.asarray(linear_exposure, dtype=float)
    gamma = np.asarray(gamma_exposure, dtype=float)
    nonlinear_portfolio_loss(direction, linear, gamma)  # validate aligned vectors
    first_order = float(linear @ direction)
    second_order = 0.5 * float(gamma @ (direction * direction))
    # L(s)-C = -second_order*s^2 - first_order*s - C.  Solving the
    # directional quadratic exactly avoids missing a narrow excursion between
    # arbitrary grid points.
    if abs(second_order) <= 1e-15:
        if first_order >= 0.0:
            raise ValueError("loss threshold is not reached within the maximum stress scale")
        roots = [-loss_threshold / first_order]
    else:
        discriminant = first_order * first_order - 4.0 * second_order * loss_threshold
        if discriminant < 0.0:
            raise ValueError("loss threshold is not reached within the maximum stress scale")
        square_root = float(np.sqrt(max(discriminant, 0.0)))
        roots = [
            (-first_order - square_root) / (2.0 * second_order),
            (-first_order + square_root) / (2.0 * second_order),
        ]
    feasible = sorted(root for root in roots if -1e-12 <= root <= maximum_scale + 1e-12)
    if not feasible:
        raise ValueError("loss threshold is not reached within the maximum stress scale")
    return float(max(0.0, feasible[0]))


__all__ = [
    "TailRiskResult",
    "empirical_tail_risk",
    "nonlinear_portfolio_loss",
    "reverse_stress_scale",
    "var_es_integral",
]
