from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def library_poisson_mle(interarrivals: list[float]) -> float:
    observations = np.asarray(interarrivals, dtype=float)
    if observations.size == 0 or np.any(observations <= 0.0):
        raise ValueError("positive interarrivals are required")

    score = lambda intensity: observations.size / intensity - observations.sum()
    return float(brentq(score, 1e-8, 100.0, xtol=1e-14))


def library_queue_fill_probability(required_depletion: int, mean_depletion: float) -> float:
    if required_depletion <= 0 or mean_depletion < 0.0:
        raise ValueError("queue parameters are invalid")
    cumulative = sum(
        math.exp(-mean_depletion) * mean_depletion**count / math.factorial(count)
        for count in range(required_depletion)
    )
    return 1.0 - cumulative


def library_joint_beta(signs: np.ndarray, changes: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(signs)), np.asarray(signs, dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(changes, dtype=float), rcond=None)
    return float(coefficients[1])


__all__ = ["library_joint_beta", "library_poisson_mle", "library_queue_fill_probability"]
