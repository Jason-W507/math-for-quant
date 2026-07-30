from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_banded
from scipy.special import ndtr
from scipy.stats import binom, norm
from sklearn.linear_model import LinearRegression


def library_black_scholes_call(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    *,
    dividend_yield: float = 0.0,
) -> float:
    root_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * sigma * sigma) * maturity
    ) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    return float(
        spot * math.exp(-dividend_yield * maturity) * ndtr(d1)
        - strike * math.exp(-rate * maturity) * ndtr(d2)
    )


def library_binomial_call(
    spot: float, strike: float, rate: float, sigma: float, maturity: float, steps: int
) -> float:
    dt = maturity / steps
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    probability = (math.exp(rate * dt) - down) / (up - down)
    down_moves = np.arange(steps + 1)
    terminal = spot * up ** (steps - down_moves) * down**down_moves
    expectation = np.sum(
        binom.pmf(down_moves, steps, 1.0 - probability)
        * np.maximum(terminal - strike, 0.0)
    )
    return float(math.exp(-rate * maturity) * expectation)


def library_implicit_fd_call(
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
    ds = spot_max / space_steps
    dt = maturity / time_steps
    grid = np.linspace(0.0, spot_max, space_steps + 1)
    values = np.maximum(grid - strike, 0.0)
    index = np.arange(1, space_steps, dtype=float)
    lower = -dt * (0.5 * sigma * sigma * index * index - 0.5 * rate * index)
    diagonal = 1.0 + dt * (sigma * sigma * index * index + rate)
    upper = -dt * (0.5 * sigma * sigma * index * index + 0.5 * rate * index)
    banded = np.zeros((3, diagonal.size))
    banded[0, 1:] = upper[:-1]
    banded[1] = diagonal
    banded[2, :-1] = lower[1:]
    for step in range(1, time_steps + 1):
        rhs = values[1:-1].copy()
        boundary = spot_max - strike * math.exp(-rate * step * dt)
        rhs[-1] -= upper[-1] * boundary
        values[1:-1] = solve_banded((1, 1), banded, rhs)
        values[0] = 0.0
        values[-1] = boundary
    return float(np.interp(spot, grid, values))


def library_monte_carlo_call(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    del samples, seed
    root_t = math.sqrt(maturity)

    def discounted_payoff(z_value: float) -> float:
        terminal = spot * math.exp(
            (rate - 0.5 * sigma * sigma) * maturity + sigma * root_t * z_value
        )
        return (
            math.exp(-rate * maturity)
            * max(terminal - strike, 0.0)
            * float(norm.pdf(z_value))
        )

    price, quadrature_error = quad(
        discounted_payoff, -10.0, 10.0, epsabs=1e-11, epsrel=1e-11, limit=200
    )
    return float(price), float(quadrature_error)


def library_total_variance_coefficients(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    model = LinearRegression(fit_intercept=False).fit(
        np.asarray(design, dtype=float),
        np.asarray(target, dtype=float),
        sample_weight=np.asarray(weights, dtype=float),
    )
    return np.asarray(model.coef_, dtype=float)


__all__ = [
    "library_black_scholes_call",
    "library_binomial_call",
    "library_implicit_fd_call",
    "library_monte_carlo_call",
    "library_total_variance_coefficients",
]
