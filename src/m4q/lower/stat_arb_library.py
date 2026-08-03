from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from m4q.lower.stat_arb_models import engle_granger
from m4q.lower.stat_arb_estimation import (
    ScalarStateSpaceSpec,
    StateEstimates,
)


@dataclass(frozen=True)
class LibraryCrossCheck:
    transparent_slope: float
    library_slope: float
    transparent_adf: float
    library_adf: float
    engle_granger_p_value: float
    johansen_rank: int


@dataclass(frozen=True)
class MarkovLibraryFit:
    filtered_probabilities: np.ndarray
    log_likelihood: float


def cross_check_long_run(y: np.ndarray, x: np.ndarray) -> LibraryCrossCheck:
    relation = engle_granger(y, x)
    design = sm.add_constant(np.asarray(x, dtype=float))
    library_coefficients = sm.OLS(np.asarray(y, dtype=float), design).fit().params
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


def library_kalman_filter_and_smooth(
    observations: np.ndarray,
    *,
    spec: ScalarStateSpaceSpec,
) -> StateEstimates:
    """Run statsmodels under exactly the transparent scalar-model contract."""
    spec.validate()
    values = np.asarray(observations, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("observations must be a finite series")
    model = KalmanSmoother(k_endog=1, k_states=1, k_posdef=1)
    model.bind(values[:, None])
    model.design = np.asarray([[spec.observation_loading]])
    model.transition = np.asarray([[spec.transition]])
    model.selection = np.asarray([[1.0]])
    model.state_cov = np.asarray([[spec.process_variance]])
    model.obs_cov = np.asarray([[spec.observation_variance]])
    model.initialize_known(
        np.asarray([spec.transition * spec.initial_mean]),
        np.asarray(
            [[spec.transition**2 * spec.initial_variance + spec.process_variance]]
        ),
    )
    result = model.smooth()
    return StateEstimates(
        filtered=np.asarray(result.filtered_state[0]),
        smoothed=np.asarray(result.smoothed_state[0]),
        filtered_variances=np.asarray(result.filtered_state_cov[0, 0]),
    )


def fit_markov_switching(values: np.ndarray) -> MarkovLibraryFit:
    """Fit a two-regime MLE model, returning probabilities and likelihood."""
    series = np.asarray(values, dtype=float)
    result = MarkovRegression(series, k_regimes=2, trend="c", switching_variance=True).fit(
        disp=False, maxiter=500
    )
    probabilities = np.asarray(result.filtered_marginal_probabilities)
    if probabilities.shape != (series.size, 2):
        probabilities = probabilities.T
    return MarkovLibraryFit(
        filtered_probabilities=probabilities,
        log_likelihood=float(result.llf),
    )
