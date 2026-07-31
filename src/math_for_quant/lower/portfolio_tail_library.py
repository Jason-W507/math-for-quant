from __future__ import annotations

import numpy as np


def library_historical_var_es(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    sample = np.sort(np.asarray(losses, dtype=float))
    value_at_risk = float(np.quantile(sample, confidence, method="lower"))
    tail_count = int(round(sample.size * (1.0 - confidence)))
    if tail_count <= 0 or not np.isclose(tail_count, sample.size * (1.0 - confidence)):
        raise ValueError("library comparison requires an integer tail count")
    return value_at_risk, float(sample[-tail_count:].mean())


def library_nonlinear_loss(
    shocks: np.ndarray, linear_exposure: np.ndarray, gamma_exposure: np.ndarray
) -> float:
    terms = [
        -(float(delta) * float(shock) + 0.5 * float(gamma) * float(shock) ** 2)
        for shock, delta, gamma in zip(shocks, linear_exposure, gamma_exposure, strict=True)
    ]
    return float(sum(terms))


def library_reverse_stress_roots(
    direction: np.ndarray,
    linear_exposure: np.ndarray,
    gamma_exposure: np.ndarray,
    *,
    threshold: float,
    maximum_scale: float,
) -> float:
    direction = np.asarray(direction, dtype=float)
    linear = float(np.asarray(linear_exposure, dtype=float) @ direction)
    quadratic = 0.5 * float(np.asarray(gamma_exposure, dtype=float) @ (direction * direction))
    roots = np.roots([-quadratic, -linear, -threshold]) if abs(quadratic) > 1e-15 else np.array([-threshold / linear])
    feasible = sorted(
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1e-10 and -1e-12 <= float(root.real) <= maximum_scale + 1e-12
    )
    if not feasible:
        raise ValueError("independent reverse-stress polynomial has no feasible root")
    return max(0.0, feasible[0])


__all__ = [
    "library_historical_var_es",
    "library_nonlinear_loss",
    "library_reverse_stress_roots",
]
