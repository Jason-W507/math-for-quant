from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from math_for_quant.evidence import load_oracle_bundle


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def black_scholes_call(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    if min(spot, strike, sigma, maturity) <= 0.0:
        raise ValueError("positive spot, strike, volatility, and maturity required")
    root_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * maturity) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    return spot * normal_cdf(d1) - strike * math.exp(-rate * maturity) * normal_cdf(d2)


def call_delta(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    if maturity <= 0.0:
        return float(spot > strike)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * maturity) / (sigma * math.sqrt(maturity))
    return normal_cdf(d1)


def call_vega(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * maturity) / (sigma * math.sqrt(maturity))
    return spot * normal_pdf(d1) * math.sqrt(maturity)


def binomial_call(spot: float, strike: float, rate: float, sigma: float, maturity: float, steps: int) -> float:
    if steps <= 0:
        raise ValueError("binomial grid requires positive steps")
    dt = maturity / steps
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    probability = (math.exp(rate * dt) - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("binomial risk-neutral probability is invalid")
    terminal = np.array([max(spot * up ** (steps - index) * down**index - strike, 0.0) for index in range(steps + 1)])
    discount = math.exp(-rate * dt)
    for _ in range(steps):
        terminal = discount * (probability * terminal[:-1] + (1.0 - probability) * terminal[1:])
    return float(terminal[0])


def monte_carlo_call(
    spot: float, strike: float, rate: float, sigma: float, maturity: float, samples: int, seed: int
) -> tuple[float, float]:
    if samples < 2:
        raise ValueError("Monte Carlo confidence interval requires at least two paths")
    normals = np.random.default_rng(seed).standard_normal(samples)
    terminal = spot * np.exp((rate - 0.5 * sigma * sigma) * maturity + sigma * math.sqrt(maturity) * normals)
    discounted = math.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
    return float(discounted.mean()), float(1.96 * discounted.std(ddof=1) / math.sqrt(samples))


def implied_volatility(price: float, spot: float, strike: float, rate: float, maturity: float) -> float:
    lower_bound = max(spot - strike * math.exp(-rate * maturity), 0.0)
    if not lower_bound <= price <= spot:
        raise ValueError("call quote violates static arbitrage bounds")
    lower, upper = 1e-6, 3.0
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if black_scholes_call(spot, strike, rate, middle, maturity) < price:
            lower = middle
        else:
            upper = middle
    result = 0.5 * (lower + upper)
    if call_vega(spot, strike, rate, result, maturity) < 1e-6:
        raise ValueError("implied-volatility calibration is ill-conditioned")
    return result


def quadratic_variation_and_ito(normals: np.ndarray, maturity: float) -> tuple[float, float, float]:
    dt = maturity / normals.size
    increments = math.sqrt(dt) * normals
    brownian = np.concatenate(([0.0], np.cumsum(increments)))
    qv = float(increments @ increments)
    left = float(brownian[-1] ** 2)
    right = float(2.0 * (brownian[:-1] @ increments) + qv)
    return qv, left, right


def gbm_terminals(spot: float, drift: float, sigma: float, maturity: float, normals: np.ndarray) -> tuple[float, float]:
    dt = maturity / normals.size
    increments = math.sqrt(dt) * normals
    exact = spot * math.exp((drift - 0.5 * sigma * sigma) * maturity + sigma * float(increments.sum()))
    euler = spot
    for increment in increments:
        euler += drift * euler * dt + sigma * euler * float(increment)
    return exact, euler


def validate_measure_change(theta_energy: float, maximum_energy: float) -> None:
    if not math.isfinite(theta_energy) or theta_energy > maximum_energy:
        raise ValueError("Novikov-style integrability budget rejected")


def validate_risk_neutral_drift(physical_drift: float, rate: float, sigma: float, theta: float) -> None:
    transformed = physical_drift - sigma * theta
    if abs(transformed - rate) > 1e-12:
        raise ValueError("wrong risk-neutral drift transform")


def finite_difference(function: Callable[[float], float], center: float, step: float) -> float:
    if step <= 0.0:
        raise ValueError("finite-difference step must be positive")
    return (function(center + step) - function(center - step)) / (2.0 * step)


def delta_hedge(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    normals: np.ndarray,
    cost_rate: float,
) -> tuple[float, float]:
    steps = normals.size
    dt = maturity / steps
    option = black_scholes_call(spot, strike, rate, sigma, maturity)
    delta = call_delta(spot, strike, rate, sigma, maturity)
    cash = option - delta * spot
    total_cost = 0.0
    current = spot
    for index, normal in enumerate(normals, start=1):
        cash *= math.exp(rate * dt)
        current *= math.exp((rate - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * float(normal))
        remaining = maturity - index * dt
        next_delta = call_delta(current, strike, rate, sigma, remaining)
        trade = next_delta - delta
        cost = cost_rate * abs(trade) * current
        cash -= trade * current + cost
        total_cost += cost
        delta = next_delta
    error = delta * current + cash - max(current - strike, 0.0)
    return float(error), float(total_cost)


def expect_rejection(action: Callable[[], object], diagnostic: str) -> int:
    try:
        action()
    except ValueError as error:
        if diagnostic not in str(error):
            raise AssertionError(f"wrong diagnostic: {error!s}") from error
        return 1
    raise AssertionError(f"expected rejection containing {diagnostic!r}")


def render_report(observed: dict[str, float | int]) -> str:
    return f"""# 衍生品定价与对冲可复现研究包

- 二次变差：离散值 {observed['quadratic_variation']:.6f}；Itô 离散恒等式两侧为 {observed['ito_left']:.6f} 与 {observed['ito_right']:.6f}。
- SDE：GBM 解析终值 {observed['gbm_exact']:.6f}，Euler 终值 {observed['gbm_euler']:.6f}；差异属于离散化误差。
- Black--Scholes 闭式价格 {observed['closed_price']:.6f}；二叉树 {observed['tree_price']:.6f}；Monte Carlo {observed['mc_price']:.6f}，95% 半宽 {observed['mc_half_width']:.6f}。
- 隐含波动率：由合成报价反解为 {observed['implied_volatility']:.6f}；套利不一致报价与病态校准必须拒绝。
- Greeks：解析 Delta {observed['analytic_delta']:.6f}，差分 Delta {observed['fd_delta']:.6f}；解析 Vega {observed['analytic_vega']:.6f}。
- 离散对冲：误差 {observed['hedge_error']:.6f}，交易成本 {observed['hedge_cost']:.6f}。
- 模型风险：GBM、常波动率、连续交易和无冲击均是限制；本实验不可声称实盘盈利、可部署性或覆盖跳跃与波动率曲面风险。
- 复现命令：`uv run python notebooks/lower/ch04_derivatives.py evidence/lower-ch04/oracle.json`。
"""


def main(oracle_path: Path = Path("evidence/lower-ch04/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    parameters = oracle["parameters"]
    spot, strike, rate, sigma, maturity = (float(parameters[name]) for name in ("spot", "strike", "rate", "sigma", "maturity"))
    normals = np.asarray(oracle["path_normals"], dtype=float)
    qv, ito_left, ito_right = quadratic_variation_and_ito(normals, maturity)
    gbm_exact, gbm_euler = gbm_terminals(spot, float(parameters["physical_drift"]), sigma, maturity, normals)
    closed = black_scholes_call(spot, strike, rate, sigma, maturity)
    tree = binomial_call(spot, strike, rate, sigma, maturity, int(oracle["tree_steps"]))
    mc_price, mc_half_width = monte_carlo_call(spot, strike, rate, sigma, maturity, int(oracle["mc_samples"]), int(oracle["seed"]))
    implied = implied_volatility(closed, spot, strike, rate, maturity)
    analytic_delta = call_delta(spot, strike, rate, sigma, maturity)
    fd_delta = finite_difference(lambda value: black_scholes_call(value, strike, rate, sigma, maturity), spot, float(oracle["fd_step"]))
    analytic_vega = call_vega(spot, strike, rate, sigma, maturity)
    hedge_error, hedge_cost = delta_hedge(spot, strike, rate, sigma, maturity, normals, float(oracle["cost_rate"]))
    theta = (float(parameters["physical_drift"]) - rate) / sigma
    validate_measure_change(0.5 * theta * theta * maturity, float(oracle["maximum_theta_energy"]))
    validate_risk_neutral_drift(float(parameters["physical_drift"]), rate, sigma, theta)
    tree_agrees = int(abs(tree - closed) <= float(oracle["tree_error_budget"]))
    mc_agrees = int(abs(mc_price - closed) <= mc_half_width)
    failures = (
        expect_rejection(lambda: validate_measure_change(float("inf"), 1.0), "integrability"),
        expect_rejection(lambda: validate_risk_neutral_drift(float(parameters["physical_drift"]), rate, sigma, -theta), "wrong risk-neutral drift"),
        expect_rejection(lambda: implied_volatility(spot + 1.0, spot, strike, rate, maturity), "arbitrage bounds"),
        expect_rejection(lambda: finite_difference(lambda x: x * x, 1.0, 0.0), "step must be positive"),
        expect_rejection(lambda: monte_carlo_call(spot, strike, rate, sigma, maturity, 1, 1), "at least two paths"),
    )
    observed: dict[str, float | int] = {
        "quadratic_variation": qv, "ito_left": ito_left, "ito_right": ito_right,
        "gbm_exact": gbm_exact, "gbm_euler": gbm_euler,
        "closed_price": closed, "tree_price": tree, "mc_price": mc_price, "mc_half_width": mc_half_width,
        "tree_agrees": tree_agrees, "mc_agrees": mc_agrees,
        "implied_volatility": implied, "analytic_delta": analytic_delta, "fd_delta": fd_delta, "analytic_vega": analytic_vega,
        "hedge_error": hedge_error, "hedge_cost": hedge_cost,
        "integrability_rejected": failures[0], "wrong_drift_rejected": failures[1], "arbitrage_rejected": failures[2],
        "grid_rejected": failures[3], "mc_budget_rejected": failures[4],
    }
    tolerance = float(oracle["absolute_tolerance"])
    for name, expected in oracle["expected"].items():
        value = observed[name]
        if isinstance(expected, int):
            if value != expected:
                raise SystemExit(f"{name} failed: {value} != {expected}")
        elif abs(float(value) - float(expected)) > tolerance:
            raise SystemExit(f"{name} failed: {value} != {expected}")
    report_path = Path(oracle["report"])
    if report_path.read_text(encoding="utf-8") != render_report(observed):
        raise SystemExit(f"reproducible report drifted: {report_path}")
    print(
        "oracle=passed "
        f"ito=({qv:.6f},{ito_left:.6f},{ito_right:.6f}) "
        f"gbm=({gbm_exact:.6f},{gbm_euler:.6f}) "
        f"prices=({closed:.6f},{tree:.6f},{mc_price:.6f},{mc_half_width:.6f}) "
        f"calibration=({implied:.6f}) greeks=({analytic_delta:.6f},{fd_delta:.6f},{analytic_vega:.6f}) "
        f"hedge=({hedge_error:.6f},{hedge_cost:.6f}) failures=({','.join(str(value) for value in failures)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch04/oracle.json")))
