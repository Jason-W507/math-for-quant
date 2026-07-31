from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad


def library_poisson_mle(interarrivals: list[float]) -> float:
    observations = np.asarray(interarrivals, dtype=float)
    if observations.size == 0 or np.any(observations <= 0.0):
        raise ValueError("positive interarrivals are required")

    total = float(observations.sum())
    log_score = lambda log_intensity: observations.size - math.exp(log_intensity) * total
    log_lower = math.log(np.nextafter(0.0, 1.0))
    log_upper = math.log(max(1.0, 2.0 / float(observations.mean())))
    return math.exp(brentq(log_score, log_lower, log_upper, xtol=1e-13))


def library_seasonal_poisson_mle(
    interarrivals: list[float], multipliers: list[float]
) -> tuple[float, np.ndarray]:
    waiting = np.asarray(interarrivals, dtype=float)
    seasonal = np.asarray(multipliers, dtype=float)
    exposure = waiting * seasonal
    total_exposure = float(exposure.sum())
    log_score = lambda log_intensity: waiting.size - math.exp(log_intensity) * total_exposure
    log_lower = math.log(np.nextafter(0.0, 1.0))
    log_upper = math.log(max(1.0, 2.0 * waiting.size / total_exposure))
    estimate = math.exp(brentq(log_score, log_lower, log_upper, xtol=1e-13))
    return estimate, estimate * exposure


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


def library_hawkes_log_likelihood(
    times: np.ndarray,
    baseline: float,
    alpha: float,
    beta: float,
    horizon: float,
    initial_excitation: float = 0.0,
) -> float:
    observations = np.asarray(times, dtype=float)

    def intensity(time: float) -> float:
        past = observations[observations < time]
        excitation = initial_excitation * np.exp(-beta * time)
        if past.size:
            excitation += np.exp(-beta * (time - past)).sum()
        return float(baseline + alpha * excitation)

    log_terms = sum(math.log(intensity(float(time))) for time in observations)
    compensator, _ = quad(intensity, 0.0, horizon, points=observations.tolist())
    return float(log_terms - compensator)


__all__ = [
    "library_hawkes_log_likelihood",
    "library_joint_beta",
    "library_poisson_mle",
    "library_queue_fill_probability",
    "library_seasonal_poisson_mle",
]
