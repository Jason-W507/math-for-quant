from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import bootstrap
from sklearn.covariance import ShrunkCovariance

from math_for_quant.lower.portfolio_estimation import risk_contributions, validate_covariance


def library_risk_parity_weights(covariance: np.ndarray) -> np.ndarray:
    matrix = validate_covariance(covariance)
    assets = matrix.shape[0]

    def objective(weights: np.ndarray) -> float:
        contributions = risk_contributions(weights, matrix)
        target = float(contributions.sum()) / assets
        return float(np.sum((contributions - target) ** 2))

    result = minimize(
        objective,
        np.full(assets, 1.0 / assets),
        method="SLSQP",
        bounds=[(1e-10, 1.0)] * assets,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"ftol": 1e-18, "maxiter": 2000},
    )
    if not result.success:
        raise ValueError(f"library risk parity optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def library_shrink_covariance(returns: np.ndarray, intensity: float) -> np.ndarray:
    observations = np.asarray(returns, dtype=float)
    if observations.ndim != 2:
        raise ValueError("returns must be a matrix")
    return np.asarray(ShrunkCovariance(shrinkage=intensity).fit(observations).covariance_)


def library_factor_covariance(
    loadings: np.ndarray, factor_covariance: np.ndarray, specific_variance: np.ndarray
) -> np.ndarray:
    loadings = np.asarray(loadings, dtype=float)
    factors = np.asarray(factor_covariance, dtype=float)
    specific = np.asarray(specific_variance, dtype=float)
    systematic = np.einsum("ik,kl,jl->ij", loadings, factors, loadings)
    return systematic + np.diag(specific)


def library_portfolio_volatility(returns: np.ndarray, weights: np.ndarray) -> float:
    portfolio = np.asarray(returns, dtype=float) @ np.asarray(weights, dtype=float)
    return float(np.sqrt(np.cov(portfolio, ddof=1)))


def library_bootstrap_volatility_interval(
    returns: np.ndarray,
    weights: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    portfolio = np.asarray(returns, dtype=float) @ np.asarray(weights, dtype=float)
    result = bootstrap(
        (portfolio,),
        lambda sample: float(np.std(sample, ddof=1)),
        n_resamples=samples,
        confidence_level=confidence,
        method="percentile",
        vectorized=False,
        rng=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


__all__ = [
    "library_factor_covariance",
    "library_bootstrap_volatility_interval",
    "library_portfolio_volatility",
    "library_risk_parity_weights",
    "library_shrink_covariance",
]
