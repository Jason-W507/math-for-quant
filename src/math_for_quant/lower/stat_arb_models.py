from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CointegratingRelation:
    intercept: float
    slope: float
    residuals: np.ndarray
    residual_adf_statistic: float


@dataclass(frozen=True)
class ErrorCorrectionFit:
    intercept: float
    adjustment_speed: float
    short_run_slope: float


@dataclass(frozen=True)
class OUDiagnostics:
    discrete_phi: float
    mean_reversion_rate: float
    half_life: float
    expected_first_passage: float


def _as_series(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 4 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional series")
    return result


def adf_statistic(values: np.ndarray) -> float:
    """One-lag Dickey--Fuller t statistic with an intercept and no lag augmentation."""
    series = _as_series(values, name="ADF input")
    delta = np.diff(series)
    design = np.column_stack([np.ones(delta.size), series[:-1]])
    coefficients, _, rank, _ = np.linalg.lstsq(design, delta, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("ADF regression is unidentified")
    residuals = delta - design @ coefficients
    degrees = delta.size - design.shape[1]
    if degrees <= 0:
        raise ValueError("ADF regression has no residual degrees of freedom")
    variance = float(residuals @ residuals / degrees)
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
    if standard_error == 0.0:
        return -math.inf if coefficients[1] < 0.0 else math.inf
    return float(coefficients[1] / standard_error)


def engle_granger(y: np.ndarray, x: np.ndarray) -> CointegratingRelation:
    dependent = _as_series(y, name="y")
    regressor = _as_series(x, name="x")
    if dependent.size != regressor.size:
        raise ValueError("Engle--Granger series lengths differ")
    design = np.column_stack([np.ones(regressor.size), regressor])
    coefficients, _, rank, _ = np.linalg.lstsq(design, dependent, rcond=None)
    if rank != 2:
        raise ValueError("cointegrating regression is unidentified")
    residuals = dependent - design @ coefficients
    return CointegratingRelation(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        residuals=residuals,
        residual_adf_statistic=adf_statistic(residuals),
    )


def johansen_rank(levels: np.ndarray, *, tolerance: float = 0.1) -> int:
    """Transparent rank diagnostic from the singular values of differenced levels.

    For a two-series teaching example, one small singular direction relative to the
    largest indicates one common stochastic trend and hence cointegration rank one.
    This is a diagnostic, not a replacement for Johansen critical-value tables.
    """
    matrix = np.asarray(levels, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 5 or matrix.shape[1] < 2:
        raise ValueError("Johansen diagnostic requires a finite T-by-k level matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Johansen input contains non-finite values")
    singular = np.linalg.svd(np.diff(matrix, axis=0), compute_uv=False)
    if singular[0] == 0.0:
        raise ValueError("Johansen diagnostic is unidentified")
    return int(np.sum(singular / singular[0] < tolerance))


def fit_ecm(
    y: np.ndarray, x: np.ndarray, relation: CointegratingRelation
) -> ErrorCorrectionFit:
    dependent = _as_series(y, name="y")
    regressor = _as_series(x, name="x")
    if dependent.size != regressor.size or relation.residuals.size != dependent.size:
        raise ValueError("ECM inputs do not share a common timeline")
    design = np.column_stack(
        [np.ones(dependent.size - 1), relation.residuals[:-1], np.diff(regressor)]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, np.diff(dependent), rcond=None)
    if rank != design.shape[1]:
        raise ValueError("ECM regression is unidentified")
    return ErrorCorrectionFit(*(float(value) for value in coefficients))


def ou_diagnostics(values: np.ndarray, *, step: float) -> OUDiagnostics:
    series = _as_series(values, name="OU input")
    if step <= 0.0:
        raise ValueError("OU step must be positive")
    centered = series - series.mean()
    denominator = float(centered[:-1] @ centered[:-1])
    if denominator == 0.0:
        raise ValueError("OU fit requires variation")
    phi = float(centered[:-1] @ centered[1:] / denominator)
    if not 0.0 < phi < 1.0:
        raise ValueError("continuous mean-reverting OU mapping requires 0 < phi < 1")
    rate = -math.log(phi) / step
    return OUDiagnostics(
        discrete_phi=phi,
        mean_reversion_rate=rate,
        half_life=math.log(2.0) / rate,
        expected_first_passage=1.0 / rate,
    )
