from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from m4q.lower.portfolio_estimation import validate_covariance


def black_litterman_posterior(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    views: np.ndarray,
    view_returns: np.ndarray,
    view_covariance: np.ndarray,
    *,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    covariance = validate_covariance(prior_covariance)
    prior_mean = np.asarray(prior_mean, dtype=float)
    views = np.asarray(views, dtype=float)
    view_returns = np.asarray(view_returns, dtype=float)
    omega = validate_covariance(view_covariance)
    assets = covariance.shape[0]
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if prior_mean.shape != (assets,) or views.ndim != 2 or views.shape[1] != assets:
        raise ValueError("Black--Litterman asset dimensions disagree")
    if view_returns.shape != (views.shape[0],) or omega.shape != (views.shape[0], views.shape[0]):
        raise ValueError("Black--Litterman view dimensions disagree")
    prior_precision = np.linalg.inv(tau * covariance)
    view_precision = np.linalg.inv(omega)
    posterior_covariance = np.linalg.inv(prior_precision + views.T @ view_precision @ views)
    posterior_mean = posterior_covariance @ (
        prior_precision @ prior_mean + views.T @ view_precision @ view_returns
    )
    return posterior_mean, validate_covariance(covariance + posterior_covariance)


@dataclass(frozen=True)
class CvarResult:
    weights: np.ndarray
    threshold: float
    objective: float
    recomputed_cvar: float


def cvar_optimize(
    scenario_returns: np.ndarray, *, confidence: float, maximum_weight: float
) -> CvarResult:
    scenarios = np.asarray(scenario_returns, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[0] < 2 or not np.all(np.isfinite(scenarios)):
        raise ValueError("CVaR requires a scenario-by-asset matrix")
    if not 0.0 < confidence < 1.0 or not 0.0 < maximum_weight <= 1.0:
        raise ValueError("CVaR confidence and maximum weight are invalid")
    rows, assets = scenarios.shape
    if assets * maximum_weight < 1.0 - 1e-12:
        raise ValueError("maximum weight makes the full-investment constraint infeasible")
    objective = np.concatenate(
        [np.zeros(assets), [1.0], np.full(rows, 1.0 / ((1.0 - confidence) * rows))]
    )
    inequalities = np.zeros((rows, assets + 1 + rows))
    inequalities[:, :assets] = -scenarios
    inequalities[:, assets] = -1.0
    inequalities[:, assets + 1 :] = -np.eye(rows)
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(rows),
        A_eq=np.concatenate([np.ones(assets), np.zeros(1 + rows)])[None, :],
        b_eq=np.array([1.0]),
        bounds=[(0.0, maximum_weight)] * assets + [(None, None)] + [(0.0, None)] * rows,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"CVaR optimization failed: {result.message}")
    weights = np.asarray(result.x[:assets], dtype=float)
    threshold = float(result.x[assets])
    losses = -(scenarios @ weights)
    recomputed = threshold + float(np.maximum(losses - threshold, 0.0).mean()) / (1.0 - confidence)
    return CvarResult(weights, threshold, float(result.fun), recomputed)


@dataclass(frozen=True)
class RebalanceResult:
    weights: np.ndarray
    turnover: float
    cash_cost: float
    robust_score: float


def robust_cost_aware_rebalance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    return_uncertainty: np.ndarray,
    *,
    risk_aversion: float,
    uncertainty_penalty: float,
    cost_rate: float,
    capital: float,
    maximum_weight: float,
    tradable: np.ndarray,
    grid_step: float,
) -> RebalanceResult:
    covariance = validate_covariance(covariance)
    expected_returns = np.asarray(expected_returns, dtype=float)
    current_weights = np.asarray(current_weights, dtype=float)
    uncertainty = np.asarray(return_uncertainty, dtype=float)
    tradable = np.asarray(tradable, dtype=int)
    assets = covariance.shape[0]
    if any(value.shape != (assets,) for value in (expected_returns, current_weights, uncertainty, tradable)):
        raise ValueError("rebalance inputs must share the asset dimension")
    if not all(np.all(np.isfinite(value)) for value in (expected_returns, current_weights, uncertainty)):
        raise ValueError("rebalance numeric inputs must be finite")
    if np.any((tradable != 0) & (tradable != 1)):
        raise ValueError("tradable indicators must be zero or one")
    if not np.isclose(current_weights.sum(), 1.0) or np.any(current_weights < 0.0):
        raise ValueError("current weights must be nonnegative and sum to one")
    if risk_aversion <= 0.0:
        raise ValueError("risk aversion must be positive")
    if uncertainty_penalty < 0.0 or cost_rate < 0.0 or capital <= 0.0:
        raise ValueError("uncertainty, cost and capital contract is invalid")
    if np.any(uncertainty < 0.0) or not 0.0 < maximum_weight <= 1.0:
        raise ValueError("uncertainty and maximum weight contract is invalid")
    if grid_step <= 0.0 or not np.isclose(round(1.0 / grid_step) * grid_step, 1.0):
        raise ValueError("grid step must divide one exactly")
    if assets != 2:
        raise ValueError("transparent grid oracle is intentionally limited to two assets")
    best_weights: np.ndarray | None = None
    best_score = -np.inf
    for first in np.arange(0.0, 1.0 + grid_step / 2.0, grid_step):
        candidate = np.array([first, 1.0 - first])
        if np.any(candidate > maximum_weight + 1e-12):
            continue
        if np.any((tradable == 0) & (np.abs(candidate - current_weights) > 1e-12)):
            continue
        turnover = float(np.abs(candidate - current_weights).sum())
        score = float(
            candidate @ expected_returns
            - uncertainty_penalty * np.linalg.norm(uncertainty * candidate)
            - 0.5 * risk_aversion * candidate @ covariance @ candidate
            - cost_rate * turnover
        )
        if score > best_score + 1e-15:
            best_weights, best_score = candidate, score
    if best_weights is None:
        raise ValueError("no feasible portfolio under tradability and weight constraints")
    turnover = float(np.abs(best_weights - current_weights).sum())
    return RebalanceResult(best_weights, turnover, capital * cost_rate * turnover, best_score)


__all__ = [
    "CvarResult",
    "RebalanceResult",
    "black_litterman_posterior",
    "cvar_optimize",
    "robust_cost_aware_rebalance",
]
