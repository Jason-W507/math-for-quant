from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, Ridge
from statsmodels.stats.multitest import multipletests

from m4q.lower.multifactor import (
    bh_rejections,
    correlation,
    neutralize,
    ranks,
)
from m4q.lower.multifactor_estimation import within_fixed_effects


@dataclass(frozen=True)
class LibraryCrossCheck:
    predictive_mean: float
    classic_betas: np.ndarray
    classic_risk_prices: np.ndarray
    ridge_coefficients: np.ndarray
    lasso_coefficients: np.ndarray


@dataclass(frozen=True)
class RouteStatisticsCrossCheck:
    panel_slope_gap: float
    neutralization_gap: float
    ic_gap: float
    rank_ic_gap: float
    decay_gap: float
    bh_count_gap: int


def cross_check_estimators(
    *,
    signals: np.ndarray,
    future_returns: np.ndarray,
    asset_returns: np.ndarray,
    factor_returns: np.ndarray,
    regularization_design: np.ndarray,
    regularization_target: np.ndarray,
    ridge_penalty: float,
    lasso_penalty: float,
) -> LibraryCrossCheck:
    """Re-estimate the transparent route examples with mature libraries."""
    slopes = [
        float(sm.OLS(realized, sm.add_constant(signal)).fit().params[1])
        for signal, realized in zip(signals, future_returns, strict=True)
    ]
    factor_design = sm.add_constant(factor_returns)
    beta_rows = [
        sm.OLS(asset_returns[:, asset], factor_design).fit().params[1:]
        for asset in range(asset_returns.shape[1])
    ]
    betas = np.asarray(beta_rows)
    risk_prices = np.asarray(
        sm.OLS(asset_returns.mean(axis=0), sm.add_constant(betas)).fit().params[1:]
    )
    ridge = Ridge(alpha=ridge_penalty, fit_intercept=False, solver="cholesky").fit(
        regularization_design, regularization_target
    )
    lasso = Lasso(
        alpha=lasso_penalty / regularization_design.shape[0],
        fit_intercept=False,
        max_iter=100_000,
        tol=1e-12,
        selection="cyclic",
    ).fit(regularization_design, regularization_target)
    return LibraryCrossCheck(
        predictive_mean=float(np.mean(slopes)),
        classic_betas=betas,
        classic_risk_prices=risk_prices,
        ridge_coefficients=np.asarray(ridge.coef_),
        lasso_coefficients=np.asarray(lasso.coef_),
    )


def cross_check_route_statistics(
    *,
    panel_design: np.ndarray,
    panel_target: np.ndarray,
    entities: np.ndarray,
    signal: np.ndarray,
    size: np.ndarray,
    industry: np.ndarray,
    future_returns: np.ndarray,
    horizon_returns: np.ndarray,
    p_values: list[float],
    alpha: float,
    condition_limit: float = 1e10,
) -> RouteStatisticsCrossCheck:
    """Compare transparent route statistics with SciPy/statsmodels paths."""
    transparent_panel = within_fixed_effects(panel_design, panel_target, entities)
    unique_entities = np.unique(entities)
    dummies = np.column_stack(
        [(np.asarray(entities) == entity).astype(float) for entity in unique_entities[1:]]
    )
    library_panel = sm.OLS(
        panel_target, sm.add_constant(np.column_stack((panel_design, dummies)))
    ).fit().params[1 : 1 + panel_design.shape[1]]

    transparent_residual, _ = neutralize(
        signal, size, industry, condition_limit
    )
    neutral_design = np.column_stack((np.ones(signal.size), size, industry))
    library_residual = sm.OLS(signal, neutral_design).fit().resid
    transparent_ic = correlation(signal, future_returns)
    library_ic = float(stats.pearsonr(signal, future_returns).statistic)
    transparent_rank_ic = correlation(ranks(signal), ranks(future_returns))
    library_rank_ic = float(stats.spearmanr(signal, future_returns).statistic)
    transparent_decay = np.asarray(
        [correlation(signal, horizon) for horizon in horizon_returns]
    )
    library_decay = np.asarray(
        [stats.pearsonr(signal, horizon).statistic for horizon in horizon_returns],
        dtype=float,
    )
    transparent_bh = bh_rejections(p_values, alpha)
    library_bh = int(np.sum(multipletests(p_values, alpha=alpha, method="fdr_bh")[0]))
    return RouteStatisticsCrossCheck(
        panel_slope_gap=float(np.max(np.abs(transparent_panel - library_panel))),
        neutralization_gap=float(
            np.max(np.abs(transparent_residual - library_residual))
        ),
        ic_gap=abs(transparent_ic - library_ic),
        rank_ic_gap=abs(transparent_rank_ic - library_rank_ic),
        decay_gap=float(np.max(np.abs(transparent_decay - library_decay))),
        bh_count_gap=abs(transparent_bh - library_bh),
    )
