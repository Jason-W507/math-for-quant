from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from math_for_quant.lower.stat_arb_models import engle_granger


@dataclass(frozen=True)
class LibraryCrossCheck:
    transparent_slope: float
    library_slope: float
    transparent_adf: float
    library_adf: float
    engle_granger_p_value: float
    johansen_rank: int


def cross_check_long_run(y: np.ndarray, x: np.ndarray) -> LibraryCrossCheck:
    relation = engle_granger(y, x)
    design = np.column_stack([np.ones(x.size), x])
    library_coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    library_adf = float(adfuller(relation.residuals, maxlag=0, regression="c", autolag=None)[0])
    _, coint_p_value, _ = coint(y, x, trend="c", maxlag=0, autolag=None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)
        johansen = coint_johansen(
            np.column_stack([y, x]), det_order=0, k_ar_diff=1
        )
    rank = int(np.sum(johansen.lr1 > johansen.cvt[:, 1]))
    return LibraryCrossCheck(
        transparent_slope=relation.slope,
        library_slope=float(library_coefficients[1]),
        transparent_adf=relation.residual_adf_statistic,
        library_adf=library_adf,
        engle_granger_p_value=float(coint_p_value),
        johansen_rank=rank,
    )


def fit_markov_switching(values: np.ndarray) -> np.ndarray:
    """Fit a two-regime mean model and return filtered state probabilities."""
    series = np.asarray(values, dtype=float)
    result = MarkovRegression(series, k_regimes=2, trend="c", switching_variance=True).fit(
        disp=False, maxiter=500
    )
    probabilities = np.asarray(result.filtered_marginal_probabilities)
    if probabilities.shape != (series.size, 2):
        probabilities = probabilities.T
    return probabilities
