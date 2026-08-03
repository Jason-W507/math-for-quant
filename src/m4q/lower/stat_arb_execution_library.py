from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from m4q.lower.stat_arb_research import (
    ExecutionPolicy,
    ForecastLedger,
)


def library_forecast_ledger(
    *,
    forecasts: np.ndarray,
    realized_returns: np.ndarray,
    fill_fractions: np.ndarray,
    policy: ExecutionPolicy,
) -> ForecastLedger:
    """Cross-check the execution recursion as a triangular linear system."""
    forecasts = np.asarray(forecasts, dtype=float)
    realized = np.asarray(realized_returns, dtype=float)
    fills = np.asarray(fill_fractions, dtype=float)
    if forecasts.ndim != 1 or forecasts.shape != realized.shape or forecasts.shape != fills.shape:
        raise ValueError("forecast, return and fill timelines differ")
    if not (np.isfinite(forecasts).all() and np.isfinite(realized).all() and np.isfinite(fills).all()):
        raise ValueError("ledger inputs must be finite")
    if np.any((fills < 0.0) | (fills > 1.0)):
        raise ValueError("fill fractions must lie in [0, 1]")
    if policy.entry_threshold < 0.0 or policy.position_limit <= 0.0:
        raise ValueError("entry threshold and position limit are invalid")
    if policy.holding_period < 1 or policy.rebalance_every < 1 or policy.cost_per_unit_turnover < 0.0:
        raise ValueError("execution policy is invalid")

    targets = np.zeros_like(forecasts)
    active, age = 0.0, policy.holding_period
    for index, forecast in enumerate(forecasts):
        if index % policy.rebalance_every == 0 and age >= policy.holding_period:
            direction = int(forecast >= policy.entry_threshold) - int(
                forecast <= -policy.entry_threshold
            )
            active = policy.position_limit * float(direction)
            age = 0
        targets[index] = active
        age += 1

    system = np.eye(forecasts.size)
    if forecasts.size > 1:
        system[np.arange(1, forecasts.size), np.arange(forecasts.size - 1)] = -(
            1.0 - fills[1:]
        )
    positions = solve_triangular(
        system,
        fills * targets,
        lower=True,
        check_finite=True,
    )
    previous = np.concatenate([[0.0], positions[:-1]])
    period_gross = positions * realized
    turnover = float(np.abs(positions - previous).sum())
    cost = turnover * policy.cost_per_unit_turnover
    gross = float(period_gross.sum())
    return ForecastLedger(
        forecasts=forecasts,
        target_positions=targets,
        fill_fractions=fills,
        filled_positions=positions,
        period_gross_returns=period_gross,
        turnover=turnover,
        cost=cost,
        gross_return=gross,
        net_return=gross - cost,
    )
