from __future__ import annotations

import numpy as np

from m4q.lower.portfolio_tail import var_es_integral


def library_black_litterman_posterior(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    views: np.ndarray,
    view_returns: np.ndarray,
    view_covariance: np.ndarray,
    *,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(prior_mean, dtype=float)
    covariance = np.asarray(prior_covariance, dtype=float)
    views = np.asarray(views, dtype=float)
    omega = np.asarray(view_covariance, dtype=float)
    scaled = tau * covariance
    innovation_covariance = views @ scaled @ views.T + omega
    gain = scaled @ views.T @ np.linalg.inv(innovation_covariance)
    posterior_mean = mean + gain @ (np.asarray(view_returns, dtype=float) - views @ mean)
    mean_uncertainty = scaled - gain @ views @ scaled
    return posterior_mean, covariance + mean_uncertainty


def enumerate_two_asset_cvar(
    scenario_returns: np.ndarray, *, confidence: float, grid_step: float,
    maximum_weight: float,
) -> tuple[np.ndarray, float]:
    scenarios = np.asarray(scenario_returns, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != 2:
        raise ValueError("enumerated CVaR oracle requires two assets")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not 0.0 < maximum_weight <= 1.0:
        raise ValueError("maximum weight must lie in (0, 1]")
    if 2.0 * maximum_weight < 1.0 - 1e-12:
        raise ValueError("maximum weight makes the enumerated problem infeasible")
    if grid_step <= 0.0 or not np.isclose(round(1.0 / grid_step) * grid_step, 1.0):
        raise ValueError("grid step must divide one exactly")
    best_weights: np.ndarray | None = None
    best_cvar = np.inf
    for first in np.arange(0.0, 1.0 + grid_step / 2.0, grid_step):
        weights = np.array([first, 1.0 - first])
        if np.any(weights > maximum_weight + 1e-12):
            continue
        cvar = var_es_integral(-(scenarios @ weights), confidence)[1]
        if cvar < best_cvar - 1e-15:
            best_weights, best_cvar = weights, cvar
    if best_weights is None:
        raise ValueError("enumerated CVaR problem is infeasible on the requested grid")
    return best_weights, float(best_cvar)


def enumerate_robust_rebalance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    uncertainty: np.ndarray,
    *,
    risk_aversion: float,
    uncertainty_penalty: float,
    cost_rate: float,
    maximum_weight: float,
    tradable: np.ndarray,
    grid_step: float,
) -> tuple[np.ndarray, float]:
    steps = round(1.0 / grid_step)
    best: tuple[np.ndarray, float] | None = None
    for integer_weight in range(steps + 1):
        candidate = np.array([integer_weight / steps, 1.0 - integer_weight / steps])
        if np.any(candidate > maximum_weight + 1e-12):
            continue
        if np.any((np.asarray(tradable) == 0) & (np.abs(candidate - current_weights) > 1e-12)):
            continue
        turnover = float(np.linalg.norm(candidate - current_weights, ord=1))
        score = float(
            candidate @ expected_returns
            - uncertainty_penalty * np.linalg.norm(uncertainty * candidate)
            - 0.5 * risk_aversion * candidate @ covariance @ candidate
            - cost_rate * turnover
        )
        if best is None or score > best[1] + 1e-15:
            best = candidate, score
    if best is None:
        raise ValueError("independent rebalance enumeration found no feasible portfolio")
    return best


__all__ = [
    "enumerate_robust_rebalance",
    "enumerate_two_asset_cvar",
    "library_black_litterman_posterior",
]
