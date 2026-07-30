from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from math_for_quant.lower.derivatives import (
    black_scholes_call,
    implied_volatility,
    validate_surface_constraints,
)
from math_for_quant.lower.derivatives_numerics_library import (
    library_total_variance_coefficients,
)


@dataclass(frozen=True)
class SurfaceNode:
    strike: float
    maturity: float
    price: float
    weight: float = 1.0


@dataclass(frozen=True)
class ParametricSurfaceFit:
    coefficients: np.ndarray
    point_implied_volatilities: np.ndarray
    fitted_volatilities: np.ndarray
    maximum_price_error: float
    weighted_price_loss: float
    library_coefficient_gap: float


def _solve_tridiagonal(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    diagonal_work = np.asarray(diagonal, dtype=float).copy()
    rhs_work = np.asarray(rhs, dtype=float).copy()
    upper_work = np.asarray(upper, dtype=float)
    lower_work = np.asarray(lower, dtype=float)
    for index in range(1, diagonal_work.size):
        multiplier = lower_work[index - 1] / diagonal_work[index - 1]
        diagonal_work[index] -= multiplier * upper_work[index - 1]
        rhs_work[index] -= multiplier * rhs_work[index - 1]
    solution = np.empty_like(rhs_work)
    solution[-1] = rhs_work[-1] / diagonal_work[-1]
    for index in range(solution.size - 2, -1, -1):
        solution[index] = (
            rhs_work[index] - upper_work[index] * solution[index + 1]
        ) / diagonal_work[index]
    return solution


def implicit_fd_call(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    *,
    space_steps: int,
    time_steps: int,
    spot_max: float,
) -> float:
    """Price a European call with an implicit finite-difference PDE solver."""
    if min(spot, strike, sigma, maturity, spot_max) <= 0.0:
        raise ValueError("PDE inputs must be positive")
    if space_steps < 3 or time_steps < 1 or spot >= spot_max:
        raise ValueError("PDE grid must contain the spot in a nontrivial domain")
    ds = spot_max / space_steps
    dt = maturity / time_steps
    grid = np.linspace(0.0, spot_max, space_steps + 1)
    values = np.maximum(grid - strike, 0.0)
    index = np.arange(1, space_steps, dtype=float)
    left_generator = 0.5 * sigma * sigma * index * index - 0.5 * rate * index
    center_generator = -sigma * sigma * index * index - rate
    right_generator = 0.5 * sigma * sigma * index * index + 0.5 * rate * index
    lower_full = -dt * left_generator
    diagonal = 1.0 - dt * center_generator
    upper_full = -dt * right_generator
    for step in range(1, time_steps + 1):
        tau = step * dt
        rhs = values[1:-1].copy()
        upper_boundary = spot_max - strike * math.exp(-rate * tau)
        rhs[-1] -= upper_full[-1] * upper_boundary
        interior = _solve_tridiagonal(
            lower_full[1:], diagonal, upper_full[:-1], rhs
        )
        values[0] = 0.0
        values[1:-1] = interior
        values[-1] = upper_boundary
    return float(np.interp(spot, grid, values))


def point_implied_volatilities(
    spot: float, rate: float, nodes: list[SurfaceNode]
) -> np.ndarray:
    if not nodes:
        raise ValueError("point inversion requires at least one quote")
    return np.asarray(
        [
            implied_volatility(node.price, spot, node.strike, rate, node.maturity)
            for node in nodes
        ],
        dtype=float,
    )


def _surface_design(
    spot: float, rate: float, nodes: list[SurfaceNode]
) -> np.ndarray:
    rows = []
    for node in nodes:
        forward = spot * math.exp(rate * node.maturity)
        log_moneyness = math.log(node.strike / forward)
        rows.append(
            [node.maturity, log_moneyness * log_moneyness, node.maturity**2]
        )
    return np.asarray(rows, dtype=float)


def fit_parametric_total_variance(
    spot: float, rate: float, nodes: list[SurfaceNode]
) -> ParametricSurfaceFit:
    """Fit w(k,T)=aT+b k^2+cT^2 after point-wise IV inversion.

    The point inversions are observations for the second-stage surface model;
    they are not themselves called a calibrated parametric surface.
    """
    point_vols = point_implied_volatilities(spot, rate, nodes)
    design = _surface_design(spot, rate, nodes)
    target = point_vols * point_vols * np.asarray(
        [node.maturity for node in nodes], dtype=float
    )
    weights = np.sqrt(np.asarray([node.weight for node in nodes], dtype=float))
    if np.any(weights <= 0.0):
        raise ValueError("surface weights must be positive")
    coefficients, *_ = np.linalg.lstsq(
        design * weights[:, None], target * weights, rcond=None
    )
    library_coefficients = library_total_variance_coefficients(
        design, target, weights * weights
    )
    fitted_total_variance = design @ coefficients
    maturities = np.asarray([node.maturity for node in nodes], dtype=float)
    if np.any(fitted_total_variance <= 0.0):
        raise ValueError("parametric surface produced nonpositive total variance")
    fitted_vols = np.sqrt(fitted_total_variance / maturities)
    fitted_prices = np.asarray(
        [
            black_scholes_call(
                spot, node.strike, rate, float(sigma), node.maturity
            )
            for node, sigma in zip(nodes, fitted_vols)
        ]
    )
    observed_prices = np.asarray([node.price for node in nodes], dtype=float)
    price_errors = fitted_prices - observed_prices
    validate_surface_constraints(
        [
            {
                "strike": node.strike,
                "maturity": node.maturity,
                "price": float(price),
            }
            for node, price in zip(nodes, fitted_prices)
        ],
        rate=rate,
        dividend_yield=0.0,
    )
    return ParametricSurfaceFit(
        coefficients=np.asarray(coefficients),
        point_implied_volatilities=point_vols,
        fitted_volatilities=fitted_vols,
        maximum_price_error=float(np.max(np.abs(price_errors))),
        weighted_price_loss=float(np.sum((weights * price_errors) ** 2)),
        library_coefficient_gap=float(
            np.max(np.abs(coefficients - library_coefficients))
        ),
    )


__all__ = [
    "ParametricSurfaceFit",
    "SurfaceNode",
    "black_scholes_call",
    "fit_parametric_total_variance",
    "implicit_fd_call",
    "point_implied_volatilities",
]
