from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from m4q.lower.derivatives import (
    black_scholes_call,
    call_delta,
    call_gamma,
    call_vega,
    delta_hedge,
)
from m4q.lower.derivatives_hedging_library import vectorized_delta_hedges


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


@dataclass(frozen=True)
class GreekConvergence:
    steps: tuple[float, ...]
    delta_errors: tuple[float, ...]
    gamma_errors: tuple[float, ...]
    vega_errors: tuple[float, ...]
    delta_gap: float
    gamma_gap: float
    vega_gap: float


def greek_convergence(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    *,
    steps: tuple[float, ...],
) -> GreekConvergence:
    if not steps or any(step <= 0.0 or step >= spot for step in steps):
        raise ValueError("Greek finite-difference steps must be positive and below spot")
    ordered_steps = tuple(float(step) for step in steps)
    center = black_scholes_call(spot, strike, rate, sigma, maturity)
    analytic_delta = call_delta(spot, strike, rate, sigma, maturity)
    analytic_gamma = call_gamma(spot, strike, rate, sigma, maturity)
    analytic_vega = call_vega(spot, strike, rate, sigma, maturity)
    delta_errors: list[float] = []
    gamma_errors: list[float] = []
    vega_errors: list[float] = []
    for spot_step in ordered_steps:
        up = black_scholes_call(spot + spot_step, strike, rate, sigma, maturity)
        down = black_scholes_call(spot - spot_step, strike, rate, sigma, maturity)
        delta_fd = (up - down) / (2.0 * spot_step)
        gamma_fd = (up - 2.0 * center + down) / (spot_step * spot_step)
        volatility_step = spot_step / 100.0
        vega_fd = (
            black_scholes_call(spot, strike, rate, sigma + volatility_step, maturity)
            - black_scholes_call(spot, strike, rate, sigma - volatility_step, maturity)
        ) / (2.0 * volatility_step)
        delta_errors.append(abs(delta_fd - analytic_delta))
        gamma_errors.append(abs(gamma_fd - analytic_gamma))
        vega_errors.append(abs(vega_fd - analytic_vega))
    return GreekConvergence(
        steps=ordered_steps,
        delta_errors=tuple(delta_errors),
        gamma_errors=tuple(gamma_errors),
        vega_errors=tuple(vega_errors),
        delta_gap=delta_errors[-1],
        gamma_gap=gamma_errors[-1],
        vega_gap=vega_errors[-1],
    )


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


__all__ = [
    "GreekConvergence",
    "HedgingDistribution",
    "greek_convergence",
    "simulate_hedging_distribution",
]
