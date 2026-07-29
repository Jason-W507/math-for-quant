from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TurnoverConvention(Enum):
    CROSS_SECTIONAL_OPEN = "cross-sectional-open"
    SEQUENTIAL_REBALANCE = "sequential-rebalance"


@dataclass(frozen=True)
class TradingLedger:
    positions: np.ndarray
    gross_return: float
    turnover: float
    cost: float
    net_return: float


def evaluate_trading_ledger(
    *,
    positions: np.ndarray,
    realized_returns: np.ndarray,
    cost_per_unit_turnover: float | np.ndarray,
    turnover_convention: TurnoverConvention,
) -> TradingLedger:
    positions = np.asarray(positions, dtype=float)
    realized_returns = np.asarray(realized_returns, dtype=float)
    if positions.ndim != 1 or realized_returns.shape != positions.shape:
        raise ValueError("positions and realized returns must be aligned vectors")
    costs = np.broadcast_to(np.asarray(cost_per_unit_turnover, dtype=float), positions.shape)
    if np.any(costs < 0.0):
        raise ValueError("turnover cost must be nonnegative")
    if turnover_convention is TurnoverConvention.CROSS_SECTIONAL_OPEN:
        turnover_by_observation = np.abs(positions)
    elif turnover_convention is TurnoverConvention.SEQUENTIAL_REBALANCE:
        turnover_by_observation = np.abs(np.diff(np.concatenate(([0.0], positions))))
    else:  # pragma: no cover
        raise ValueError(f"unsupported turnover convention: {turnover_convention!r}")
    gross = float(positions @ realized_returns)
    turnover = float(turnover_by_observation.sum())
    cost = float(turnover_by_observation @ costs)
    return TradingLedger(positions, gross, turnover, cost, gross - cost)
