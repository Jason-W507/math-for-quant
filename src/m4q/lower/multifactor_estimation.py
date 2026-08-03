from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PredictiveFamaMacBethResult:
    coefficients: np.ndarray
    mean_coefficient: float
    standard_error: float


@dataclass(frozen=True)
class ClassicTwoPassResult:
    betas: np.ndarray
    alphas: np.ndarray
    risk_prices: np.ndarray
    cross_sectional_intercept: float
    shanken_multiplier: float


def predictive_fama_macbeth(
    signals: np.ndarray, future_returns: np.ndarray
) -> PredictiveFamaMacBethResult:
    """Estimate one predictive cross-sectional slope per date, then average it."""
    x = np.asarray(signals, dtype=float)
    y = np.asarray(future_returns, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("signals and future_returns must be equal two-dimensional panels")
    coefficients: list[float] = []
    for signal, realized in zip(x, y, strict=True):
        design = np.column_stack((np.ones(signal.size), signal))
        coefficients.append(float(np.linalg.lstsq(design, realized, rcond=None)[0][1]))
    values = np.asarray(coefficients)
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    )
    return PredictiveFamaMacBethResult(values, float(values.mean()), standard_error)


def classic_two_pass_fama_macbeth(
    asset_returns: np.ndarray, factor_returns: np.ndarray
) -> ClassicTwoPassResult:
    """Estimate time-series betas, then cross-sectional prices of factor risk.

    The reported Shanken multiplier is the familiar scalar diagnostic
    ``sqrt(1 + lambda' Sigma_f^-1 lambda)``.  It is not presented as a
    universally valid generated-regressor standard-error correction.
    """
    returns = np.asarray(asset_returns, dtype=float)
    factors = np.asarray(factor_returns, dtype=float)
    if returns.ndim != 2 or factors.ndim != 2 or returns.shape[0] != factors.shape[0]:
        raise ValueError("asset and factor returns must share a two-dimensional time axis")
    time_design = np.column_stack((np.ones(factors.shape[0]), factors))
    coefficients = np.linalg.lstsq(time_design, returns, rcond=None)[0]
    alphas = coefficients[0]
    betas = coefficients[1:].T
    cross_design = np.column_stack((np.ones(returns.shape[1]), betas))
    cross_coefficients = np.linalg.lstsq(
        cross_design, returns.mean(axis=0), rcond=None
    )[0]
    risk_prices = cross_coefficients[1:]
    covariance = np.atleast_2d(np.cov(factors, rowvar=False, ddof=1))
    quadratic = float(risk_prices @ np.linalg.pinv(covariance) @ risk_prices)
    return ClassicTwoPassResult(
        betas=betas,
        alphas=alphas,
        risk_prices=risk_prices,
        cross_sectional_intercept=float(cross_coefficients[0]),
        shanken_multiplier=math.sqrt(1.0 + max(0.0, quadratic)),
    )


def ridge_closed_form(design: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(target, dtype=float)
    if penalty < 0.0:
        raise ValueError("ridge penalty must be nonnegative")
    return np.linalg.solve(
        matrix.T @ matrix + penalty * np.eye(matrix.shape[1]), matrix.T @ values
    )


def _soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def lasso_coordinate_descent(
    design: np.ndarray,
    target: np.ndarray,
    penalty: float,
    iterations: int = 1_000,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Minimize 0.5 ||y-Xb||^2 + penalty ||b||_1 by coordinate descent."""
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(target, dtype=float)
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],):
        raise ValueError("lasso design and target shapes are inconsistent")
    if penalty < 0.0 or iterations <= 0:
        raise ValueError("lasso penalty must be nonnegative and iterations positive")
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    squared_norms = np.sum(matrix * matrix, axis=0)
    if np.any(squared_norms == 0.0):
        raise ValueError("lasso design contains a zero column")
    for _ in range(iterations):
        previous = coefficients.copy()
        for column in range(matrix.shape[1]):
            residual = values - matrix @ coefficients + matrix[:, column] * coefficients[column]
            score = float(matrix[:, column] @ residual)
            coefficients[column] = _soft_threshold(score, penalty) / squared_norms[column]
        if float(np.max(np.abs(coefficients - previous))) <= tolerance:
            break
    return coefficients


def within_fixed_effects(
    design: np.ndarray, target: np.ndarray, entities: np.ndarray
) -> np.ndarray:
    """Estimate slopes after demeaning every variable within each entity."""
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(target, dtype=float)
    labels = np.asarray(entities)
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],) or labels.shape != values.shape:
        raise ValueError("fixed-effect design, target and entity labels are inconsistent")
    demeaned_x = matrix.copy()
    demeaned_y = values.copy()
    for entity in np.unique(labels):
        rows = labels == entity
        demeaned_x[rows] -= matrix[rows].mean(axis=0)
        demeaned_y[rows] -= values[rows].mean()
    return np.linalg.lstsq(demeaned_x, demeaned_y, rcond=None)[0]
