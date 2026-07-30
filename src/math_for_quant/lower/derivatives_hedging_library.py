from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from math_for_quant.lower.derivatives_numerics_library import library_black_scholes_call


def _library_delta(
    spot: np.ndarray | float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
) -> np.ndarray:
    values = np.asarray(spot, dtype=float)
    d1 = (
        np.log(values / strike) + (rate + 0.5 * sigma * sigma) * maturity
    ) / (sigma * math.sqrt(maturity))
    return np.asarray(ndtr(d1), dtype=float)


def vectorized_delta_hedges(
    *,
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    normals: np.ndarray,
    cost_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths, steps = normals.shape
    dt = maturity / steps
    option = library_black_scholes_call(spot, strike, rate, sigma, maturity)
    delta = np.full(paths, float(_library_delta(spot, strike, rate, sigma, maturity)))
    current = np.full(paths, spot, dtype=float)
    cash_no_cost = np.full(paths, option, dtype=float) - delta * current
    initial_cost = cost_rate * np.abs(delta) * current
    cash_after_cost = cash_no_cost - initial_cost
    raw_cost = initial_cost.copy()
    for index in range(1, steps + 1):
        growth = math.exp(rate * dt)
        cash_no_cost *= growth
        cash_after_cost *= growth
        current *= np.exp(
            (rate - 0.5 * sigma * sigma) * dt
            + sigma * math.sqrt(dt) * normals[:, index - 1]
        )
        if index < steps:
            remaining = maturity - index * dt
            next_delta = _library_delta(current, strike, rate, sigma, remaining)
            trade = next_delta - delta
            cost = cost_rate * np.abs(trade) * current
            cash_no_cost -= trade * current
            cash_after_cost -= trade * current + cost
            raw_cost += cost
            delta = next_delta
    payoff = np.maximum(current - strike, 0.0)
    return (
        delta * current + cash_no_cost - payoff,
        delta * current + cash_after_cost - payoff,
        raw_cost,
    )


__all__ = ["vectorized_delta_hedges"]
