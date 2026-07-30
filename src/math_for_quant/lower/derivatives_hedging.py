from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from math_for_quant.lower.derivatives import delta_hedge
from math_for_quant.lower.derivatives_hedging_library import vectorized_delta_hedges


@dataclass(frozen=True)
class HedgingDistribution:
    paths: int
    steps: int
    no_cost_bias: float
    no_cost_rmse: float
    after_cost_bias: float
    after_cost_rmse: float
    error_q05: float
    error_q50: float
    error_q95: float
    mean_cost: float
    cost_q95: float
    summary_gap: float


def simulate_hedging_distribution(
    *,
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    paths: int,
    steps: int,
    cost_rate: float,
    seed: int,
) -> HedgingDistribution:
    if paths < 2 or steps < 1:
        raise ValueError("hedging distribution requires multiple paths and positive steps")
    if cost_rate < 0.0:
        raise ValueError("hedging transaction-cost rate cannot be negative")
    normals = np.random.default_rng(seed).standard_normal((paths, steps))
    transparent = np.asarray(
        [
            delta_hedge(
                spot, strike, rate, sigma, maturity, path, cost_rate
            )[:2]
            for path in normals
        ]
    )
    no_cost, after_cost, raw_cost = vectorized_delta_hedges(
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        normals=normals,
        cost_rate=cost_rate,
    )
    summary_gap = float(
        max(
            np.max(np.abs(transparent[:, 0] - no_cost)),
            np.max(np.abs(transparent[:, 1] - after_cost)),
        )
    )
    q05, q50, q95 = np.quantile(after_cost, [0.05, 0.5, 0.95])
    return HedgingDistribution(
        paths=paths,
        steps=steps,
        no_cost_bias=float(np.mean(no_cost)),
        no_cost_rmse=float(np.sqrt(np.mean(no_cost * no_cost))),
        after_cost_bias=float(np.mean(after_cost)),
        after_cost_rmse=float(np.sqrt(np.mean(after_cost * after_cost))),
        error_q05=float(q05),
        error_q50=float(q50),
        error_q95=float(q95),
        mean_cost=float(np.mean(raw_cost)),
        cost_q95=float(np.quantile(raw_cost, 0.95)),
        summary_gap=summary_gap,
    )


__all__ = ["HedgingDistribution", "simulate_hedging_distribution"]
