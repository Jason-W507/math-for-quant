from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

import numpy as np

from math_for_quant.lower.multifactor import correlation


@dataclass(frozen=True)
class GroupPortfolioLedger:
    weights: np.ndarray
    period_gross_returns: np.ndarray
    gross_return: float
    turnover: float
    cost: float
    capacity_impact: float
    net_return: float


@dataclass(frozen=True)
class RealCrossSection:
    countries: tuple[str, ...]
    signal_year: int
    outcome_year: int
    signals: np.ndarray
    outcomes: np.ndarray
    correlation: float


class Weighting(str, Enum):
    EQUAL = "equal"
    CAPITALIZATION = "capitalization"


@dataclass(frozen=True)
class PortfolioInputs:
    signals: np.ndarray
    realized_returns: np.ndarray
    market_caps: np.ndarray


@dataclass(frozen=True)
class PortfolioPolicy:
    quantiles: int
    weighting: Weighting
    holding_periods: int
    cost_per_unit_turnover: float
    capacity_impact: float


def _vintage_weights(
    signal: np.ndarray,
    market_cap: np.ndarray,
    quantiles: int,
    weighting: Weighting,
) -> np.ndarray:
    if quantiles < 2 or signal.size % quantiles != 0:
        raise ValueError("quantiles must evenly divide the cross section")
    order = np.argsort(signal, kind="stable")
    group_size = signal.size // quantiles
    short_indices = order[:group_size]
    long_indices = order[-group_size:]
    weights = np.zeros(signal.size, dtype=float)
    if weighting is Weighting.EQUAL:
        weights[long_indices] = 1.0 / group_size
        weights[short_indices] = -1.0 / group_size
    elif weighting is Weighting.CAPITALIZATION:
        if np.any(market_cap <= 0.0):
            raise ValueError("capitalization weights require positive market caps")
        weights[long_indices] = market_cap[long_indices] / market_cap[long_indices].sum()
        weights[short_indices] = -market_cap[short_indices] / market_cap[short_indices].sum()
    return weights


def build_group_portfolio_ledger(
    *,
    inputs: PortfolioInputs,
    policy: PortfolioPolicy,
) -> GroupPortfolioLedger:
    signal_panel = np.asarray(inputs.signals, dtype=float)
    return_panel = np.asarray(inputs.realized_returns, dtype=float)
    cap_panel = np.asarray(inputs.market_caps, dtype=float)
    if signal_panel.shape != return_panel.shape or signal_panel.shape != cap_panel.shape:
        raise ValueError("signals, returns and market caps must have identical shapes")
    if signal_panel.ndim != 2 or policy.holding_periods <= 0:
        raise ValueError("panels must be two-dimensional and holding_periods positive")
    if policy.cost_per_unit_turnover < 0.0 or policy.capacity_impact < 0.0:
        raise ValueError("cost and capacity impact must be nonnegative")
    vintages = [
        _vintage_weights(signal, cap, policy.quantiles, policy.weighting)
        for signal, cap in zip(signal_panel, cap_panel, strict=True)
    ]
    effective: list[np.ndarray] = []
    for index in range(len(vintages)):
        active = vintages[max(0, index - policy.holding_periods + 1) : index + 1]
        effective.append(np.mean(active, axis=0))
    weights = np.asarray(effective)
    period_gross = np.sum(weights * return_panel, axis=1)
    previous = np.zeros(weights.shape[1], dtype=float)
    turnovers: list[float] = []
    for current in weights:
        turnovers.append(float(np.sum(np.abs(current - previous))))
        previous = current
    average_turnover = float(np.mean(turnovers))
    gross = float(np.mean(period_gross))
    cost = policy.cost_per_unit_turnover * average_turnover
    return GroupPortfolioLedger(
        weights=weights,
        period_gross_returns=period_gross,
        gross_return=gross,
        turnover=average_turnover,
        cost=cost,
        capacity_impact=policy.capacity_impact,
        net_return=gross - cost - policy.capacity_impact,
    )


def load_real_cross_section(path: Path) -> RealCrossSection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    signal_year = int(payload["signal_year"])
    outcome_year = int(payload["outcome_year"])
    if signal_year >= outcome_year:
        raise ValueError("real-data signal year must precede outcome year")
    rows = payload["rows"]
    countries = tuple(str(row["country"]) for row in rows)
    signals = np.asarray([row["signal"] for row in rows], dtype=float)
    outcomes = np.asarray([row["outcome"] for row in rows], dtype=float)
    return RealCrossSection(
        countries=countries,
        signal_year=signal_year,
        outcome_year=outcome_year,
        signals=signals,
        outcomes=outcomes,
        correlation=correlation(signals, outcomes),
    )
