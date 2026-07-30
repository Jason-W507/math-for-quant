from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionPolicy:
    entry_threshold: float
    position_limit: float
    holding_period: int
    rebalance_every: int
    cost_per_unit_turnover: float


@dataclass(frozen=True)
class ForecastLedger:
    forecasts: np.ndarray
    target_positions: np.ndarray
    fill_fractions: np.ndarray
    filled_positions: np.ndarray
    period_gross_returns: np.ndarray
    turnover: float
    cost: float
    gross_return: float
    net_return: float


def validate_purged_walk_forward(
    *,
    train_indices: range,
    validation_indices: range,
    trade_indices: range,
    label_horizon: int,
    embargo: int,
) -> None:
    train = list(train_indices)
    validation = list(validation_indices)
    trade = list(trade_indices)
    if not train or not validation or not trade or label_horizon < 0 or embargo < 0:
        raise ValueError("walk-forward inputs are invalid")
    if max(train) + label_horizon >= min(validation):
        raise ValueError("purge gap does not cover the training label horizon")
    if max(validation) + embargo >= min(trade):
        raise ValueError("embargo gap does not separate validation and trade")


def build_forecast_ledger(
    *,
    forecasts: np.ndarray,
    realized_returns: np.ndarray,
    fill_fractions: np.ndarray,
    policy: ExecutionPolicy,
) -> ForecastLedger:
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
    positions = targets * fills
    previous = np.concatenate([[0.0], positions[:-1]])
    turnover = float(np.abs(positions - previous).sum())
    period_gross = positions * realized
    gross = float(period_gross.sum())
    cost = turnover * policy.cost_per_unit_turnover
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
