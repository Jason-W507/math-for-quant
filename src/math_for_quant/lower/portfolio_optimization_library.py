from __future__ import annotations

import numpy as np

from math_for_quant.lower.portfolio_tail import var_es_integral


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
    assert best_weights is not None
    return best_weights, float(best_cvar)


__all__ = ["enumerate_two_asset_cvar", "library_black_litterman_posterior"]
