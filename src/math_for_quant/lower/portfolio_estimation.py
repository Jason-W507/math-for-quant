from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def validate_covariance(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("covariance must be finite and symmetric")
    if float(np.linalg.eigvalsh(matrix).min()) < -1e-12:
        raise ValueError("covariance must be positive semidefinite")
    return matrix


def shrink_covariance(sample: np.ndarray, target: np.ndarray, intensity: float) -> np.ndarray:
    sample_matrix = validate_covariance(sample)
    target_matrix = validate_covariance(target)
    if sample_matrix.shape != target_matrix.shape:
        raise ValueError("sample and target covariance dimensions disagree")
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("shrinkage intensity must lie in [0, 1]")
    result = (1.0 - intensity) * sample_matrix + intensity * target_matrix
    return validate_covariance(result)


def factor_covariance(
    loadings: np.ndarray, factor_covariance_matrix: np.ndarray, idiosyncratic_variance: np.ndarray
) -> np.ndarray:
    loadings = np.asarray(loadings, dtype=float)
    factors = validate_covariance(factor_covariance_matrix)
    specific = np.asarray(idiosyncratic_variance, dtype=float)
    if loadings.ndim != 2 or loadings.shape[1] != factors.shape[0]:
        raise ValueError("factor covariance dimensions disagree")
    if specific.shape != (loadings.shape[0],) or np.any(specific < 0.0):
        raise ValueError("idiosyncratic variances must be nonnegative and asset-aligned")
    return validate_covariance(loadings @ factors @ loadings.T + np.diag(specific))


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    matrix = validate_covariance(covariance)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (matrix.shape[0],):
        raise ValueError("weights and covariance dimensions disagree")
    return weights * (matrix @ weights)


def risk_parity_weights(covariance: np.ndarray) -> np.ndarray:
    matrix = validate_covariance(covariance)
    if np.any(np.diag(matrix) <= 0.0):
        raise ValueError("risk parity requires positive marginal variances")
    assets = matrix.shape[0]

    weights = 1.0 / np.sqrt(np.diag(matrix))
    weights /= weights.sum()
    for _ in range(10_000):
        contributions = risk_contributions(weights, matrix)
        if np.any(contributions <= 0.0):
            raise ValueError("positive risk contributions are required by this iteration")
        target = float(contributions.sum()) / assets
        updated = weights * np.sqrt(target / contributions)
        updated /= updated.sum()
        if float(np.max(np.abs(updated - weights))) < 1e-13:
            return updated
        weights = updated
    raise ValueError("risk parity iteration did not converge")


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    samples: int


def bootstrap_portfolio_volatility(
    returns: np.ndarray,
    weights: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapInterval:
    observations = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if observations.ndim != 2 or observations.shape[0] < 3:
        raise ValueError("bootstrap requires at least three rows of asset returns")
    if weights.shape != (observations.shape[1],) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("bootstrap weights must be asset-aligned and sum to one")
    if samples < 50 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap samples and confidence are outside the teaching contract")
    portfolio = observations @ weights
    point = float(np.std(portfolio, ddof=1))
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        draw = generator.integers(0, portfolio.size, size=portfolio.size)
        estimates[index] = np.std(portfolio[draw], ddof=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return BootstrapInterval(point, float(lower), float(upper), samples)


__all__ = [
    "BootstrapInterval",
    "bootstrap_portfolio_volatility",
    "factor_covariance",
    "risk_contributions",
    "risk_parity_weights",
    "shrink_covariance",
    "validate_covariance",
]
