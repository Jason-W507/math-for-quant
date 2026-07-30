from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from math_for_quant.lower.ml_alpha_research import AlphaLedger, PortfolioPolicy


def library_alpha_ledger(
    *,
    scores: np.ndarray,
    realized_returns: np.ndarray,
    fill_fractions: np.ndarray,
    policy: PortfolioPolicy,
) -> AlphaLedger:
    values = np.asarray(scores, dtype=float)
    returns = np.asarray(realized_returns, dtype=float)
    fills = np.asarray(fill_fractions, dtype=float)
    if values.ndim != 2 or returns.shape != values.shape or fills.shape != values.shape:
        raise ValueError("alpha ledger matrices are misaligned")
    targets = np.zeros_like(values)
    long_weight = policy.gross_limit / (2.0 * policy.long_count)
    short_weight = policy.gross_limit / (2.0 * policy.short_count)
    for index, row in enumerate(values):
        order = np.argsort(row)
        targets[index, order[: policy.short_count]] = -short_weight
        targets[index, order[-policy.long_count :]] = long_weight
    positions = np.empty_like(targets)
    for asset in range(values.shape[1]):
        system = np.eye(values.shape[0])
        if values.shape[0] > 1:
            system[np.arange(1, values.shape[0]), np.arange(values.shape[0] - 1)] = -(
                1.0 - fills[1:, asset]
            )
        positions[:, asset] = solve_triangular(
            system, fills[:, asset] * targets[:, asset], lower=True
        )
    previous = np.vstack([np.zeros(values.shape[1]), positions[:-1]])
    turnover = float(np.abs(positions - previous).sum())
    period_gross = np.sum(positions * returns, axis=1)
    gross = float(period_gross.sum())
    cost = turnover * policy.cost_per_unit_turnover
    return AlphaLedger(targets, positions, period_gross, turnover, cost, gross, gross - cost)
