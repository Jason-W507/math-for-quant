from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioPolicy:
    long_count: int
    short_count: int
    gross_limit: float
    cost_per_unit_turnover: float


@dataclass(frozen=True)
class AlphaLedgerInputs:
    scores: np.ndarray
    realized_returns: np.ndarray
    fill_fractions: np.ndarray
    policy: PortfolioPolicy

    def validated_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(self.scores, dtype=float)
        returns = np.asarray(self.realized_returns, dtype=float)
        fills = np.asarray(self.fill_fractions, dtype=float)
        if values.ndim != 2 or returns.shape != values.shape or fills.shape != values.shape:
            raise ValueError("alpha ledger matrices are misaligned")
        if not all(np.isfinite(item).all() for item in (values, returns, fills)):
            raise ValueError("alpha ledger inputs must be finite")
        if np.any((fills < 0.0) | (fills > 1.0)):
            raise ValueError("fill fractions must lie in [0, 1]")
        assets = values.shape[1]
        if self.policy.long_count < 1 or self.policy.short_count < 1 or self.policy.long_count + self.policy.short_count > assets:
            raise ValueError("portfolio counts are invalid")
        if self.policy.gross_limit <= 0.0 or self.policy.cost_per_unit_turnover < 0.0:
            raise ValueError("portfolio limits or costs are invalid")
        return values, returns, fills


@dataclass(frozen=True)
class AlphaLedger:
    target_positions: np.ndarray
    filled_positions: np.ndarray
    period_gross_returns: np.ndarray
    turnover: float
    cost: float
    gross_return: float
    net_return: float


def pairwise_ranking_loss(scores: np.ndarray, realized_returns: np.ndarray) -> float:
    values = np.asarray(scores, dtype=float)
    returns = np.asarray(realized_returns, dtype=float)
    if values.shape != returns.shape or values.ndim != 1:
        raise ValueError("ranking inputs must be aligned vectors")
    losses = []
    for left in range(values.size):
        for right in range(left + 1, values.size):
            direction = np.sign(returns[left] - returns[right])
            if direction != 0.0:
                losses.append(np.logaddexp(0.0, -direction * (values[left] - values[right])))
    if not losses:
        raise ValueError("ranking loss needs at least one ordered return pair")
    return float(np.mean(losses))


def return_weighted_loss(scores: np.ndarray, realized_returns: np.ndarray) -> float:
    values = np.asarray(scores, dtype=float)
    returns = np.asarray(realized_returns, dtype=float)
    if values.shape != returns.shape:
        raise ValueError("return-weighted inputs are misaligned")
    return -float(np.mean(values * returns))


def cross_sectional_mse(scores: np.ndarray, realized_returns: np.ndarray) -> float:
    values = np.asarray(scores, dtype=float)
    returns = np.asarray(realized_returns, dtype=float)
    if values.shape != returns.shape:
        raise ValueError("cross-sectional inputs are misaligned")
    return float(np.mean((values - returns) ** 2))


def build_alpha_ledger(
    *,
    inputs: AlphaLedgerInputs,
) -> AlphaLedger:
    values, returns, fills = inputs.validated_arrays()
    policy = inputs.policy
    assets = values.shape[1]
    targets = np.zeros_like(values)
    long_weight = policy.gross_limit / (2.0 * policy.long_count)
    short_weight = policy.gross_limit / (2.0 * policy.short_count)
    for index, row in enumerate(values):
        order = np.argsort(row)
        targets[index, order[: policy.short_count]] = -short_weight
        targets[index, order[-policy.long_count :]] = long_weight
    positions = np.empty_like(targets)
    current = np.zeros(assets)
    turnover = 0.0
    for index in range(values.shape[0]):
        order_delta = targets[index] - current
        filled_delta = fills[index] * order_delta
        current = current + filled_delta
        positions[index] = current
        turnover += float(np.abs(filled_delta).sum())
    period_gross = np.sum(positions * returns, axis=1)
    gross = float(period_gross.sum())
    cost = turnover * policy.cost_per_unit_turnover
    return AlphaLedger(targets, positions, period_gross, turnover, cost, gross, gross - cost)
