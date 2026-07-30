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


def call_gamma(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    if min(spot, strike, sigma, maturity) <= 0.0:
        raise ValueError("positive spot, strike, volatility, and maturity required")
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * maturity) / (sigma * math.sqrt(maturity))
    return normal_pdf(d1) / (spot * sigma * math.sqrt(maturity))


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
) -> tuple[float, float, float]:
    if samples < 2:
        raise ValueError("Monte Carlo confidence interval requires at least two paths")
    normals = np.random.default_rng(seed).standard_normal(samples)
    terminal = spot * np.exp((rate - 0.5 * sigma * sigma) * maturity + sigma * math.sqrt(maturity) * normals)
    discounted = math.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
    variance = float(discounted.var(ddof=1))
    standard_error = math.sqrt(variance / samples)
    return float(discounted.mean()), standard_error, variance


def implied_volatility(price: float, spot: float, strike: float, rate: float, maturity: float) -> float:
    lower_bound = max(spot - strike * math.exp(-rate * maturity), 0.0)
    if not lower_bound <= price <= spot:
        raise ValueError("call quote violates static arbitrage bounds")
    lower, upper = 1e-6, 0.5
    while black_scholes_call(spot, strike, rate, upper, maturity) < price and upper < 8.0:
        upper *= 2.0
    lower_price = black_scholes_call(spot, strike, rate, lower, maturity)
    upper_price = black_scholes_call(spot, strike, rate, upper, maturity)
    if not lower_price <= price <= upper_price:
        raise ValueError("implied-volatility calibration root is not bracketed")
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if black_scholes_call(spot, strike, rate, middle, maturity) < price:
            lower = middle
        else:
            upper = middle
    result = 0.5 * (lower + upper)
    if abs(black_scholes_call(spot, strike, rate, result, maturity) - price) > 1e-10:
        raise ValueError("implied-volatility calibration residual is too large")
    if call_vega(spot, strike, rate, result, maturity) < 1e-6:
        raise ValueError("implied-volatility calibration is ill-conditioned")
    return result


def calibrate_surface(
    spot: float,
    rate: float,
    nodes: list[dict[str, float]],
) -> tuple[list[float], float]:
    implied = [implied_volatility(node["price"], spot, node["strike"], rate, node["maturity"]) for node in nodes]
    weighted_squared_error = 0.0
    for node, sigma in zip(nodes, implied):
        fitted = black_scholes_call(spot, node["strike"], rate, sigma, node["maturity"])
        weighted_squared_error += node["weight"] * (fitted - node["price"]) ** 2
    validate_surface_constraints(nodes, rate=rate, dividend_yield=0.0)
    return implied, weighted_squared_error


def validate_surface_constraints(
    nodes: list[dict[str, float]],
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    calendar_mode: str = "nonnegative-rate-no-dividend",
) -> None:
    if not nodes:
        raise ValueError("surface validation requires at least one node")
    maturities = sorted({float(node["maturity"]) for node in nodes})
    by_maturity = {
        maturity: sorted(
            (node for node in nodes if float(node["maturity"]) == maturity),
            key=lambda node: float(node["strike"]),
        )
        for maturity in maturities
    }
    for maturity in maturities:
        maturity_nodes = by_maturity[maturity]
        strikes = [float(node["strike"]) for node in maturity_nodes]
        prices = [float(node["price"]) for node in maturity_nodes]
        if len(set(strikes)) != len(strikes):
            raise ValueError("surface contains duplicate strike/maturity nodes")
        if any(left < right for left, right in zip(prices, prices[1:])):
            raise ValueError("surface violates strike monotonicity")
        if len(strikes) >= 3:
            slopes = [
                (right_price - left_price) / (right_strike - left_strike)
                for left_strike, right_strike, left_price, right_price in zip(
                    strikes, strikes[1:], prices, prices[1:]
                )
            ]
            if any(right < left - 1e-12 for left, right in zip(slopes, slopes[1:])):
                raise ValueError("surface violates butterfly convexity")
    if calendar_mode == "skip":
        return
    if calendar_mode == "forward-normalized":
        normalized: dict[float, list[tuple[float, float]]] = {}
        for maturity in maturities:
            for node in by_maturity[maturity]:
                forward = float(node.get("forward", 0.0))
                discount = float(node.get("discount_factor", 0.0))
                if forward <= 0.0 or discount <= 0.0:
                    raise ValueError(
                        "forward-normalized calendar gate requires positive forward and discount_factor"
                    )
                moneyness = round(float(node["strike"]) / forward, 12)
                normalized_price = float(node["price"]) / (discount * forward)
                normalized.setdefault(moneyness, []).append((maturity, normalized_price))
        if any(len(values) != len(maturities) for values in normalized.values()):
            raise ValueError(
                "forward-normalized calendar gate requires a complete moneyness grid"
            )
        for values in normalized.values():
            prices = [price for _, price in sorted(values)]
            if any(later < earlier - 1e-12 for earlier, later in zip(prices, prices[1:])):
                raise ValueError("surface violates forward-normalized calendar monotonicity")
        return
    if calendar_mode != "nonnegative-rate-no-dividend":
        raise ValueError("unknown calendar monotonicity mode")
    if rate < 0.0 or abs(dividend_yield) > 1e-15:
        raise ValueError(
            "calendar monotonicity requires nonnegative rates and no dividends; "
            "use forward-normalized quotes or skip this gate"
        )
    strike_sets = [
        {float(node["strike"]) for node in by_maturity[maturity]}
        for maturity in maturities
    ]
    if any(strikes != strike_sets[0] for strikes in strike_sets[1:]):
        raise ValueError("raw calendar gate requires a complete fixed-strike grid")
    for strike in sorted(strike_sets[0]):
        prices = [
            next(float(node["price"]) for node in by_maturity[maturity] if float(node["strike"]) == strike)
            for maturity in maturities
        ]
        if any(later < earlier for earlier, later in zip(prices, prices[1:])):
            raise ValueError("surface violates calendar monotonicity")


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


def validate_market_price_risk_energy(theta_energy: float, maximum_energy: float) -> None:
    if not math.isfinite(theta_energy) or theta_energy > maximum_energy:
        raise ValueError("market-price-of-risk energy budget rejected")


def validate_novikov_exponential_moment(
    theta_energies: list[float], probabilities: list[float]
) -> float:
    if len(theta_energies) != len(probabilities) or not theta_energies:
        raise ValueError("Novikov condition requires matched nonempty scenarios")
    if any(probability < 0.0 for probability in probabilities) or not math.isclose(
        sum(probabilities), 1.0, abs_tol=1e-12
    ):
        raise ValueError("Novikov condition requires probabilities summing to one")
    if any(not math.isfinite(energy) for energy in theta_energies):
        raise ValueError("Novikov condition failed: exponential moment is not finite")
    try:
        moment = sum(
            probability * math.exp(energy)
            for energy, probability in zip(theta_energies, probabilities)
        )
    except OverflowError as error:
        raise ValueError(
            "Novikov condition failed: exponential moment is not finite"
        ) from error
    if not math.isfinite(moment):
        raise ValueError("Novikov condition failed: exponential moment is not finite")
    return moment


def terminal_singular_theta_energy(horizon: float, cutoff: float) -> float:
    """Return half the energy of theta_t=(horizon-t)^(-1/2) up to T-cutoff."""
    if horizon <= 0.0:
        raise ValueError("Novikov witness requires a positive horizon")
    if cutoff <= 0.0:
        raise ValueError(
            "Novikov condition failed: integral of theta_t^2 diverges at the horizon"
        )
    if cutoff >= horizon:
        raise ValueError("Novikov witness cutoff must lie inside the horizon")
    return 0.5 * math.log(horizon / cutoff)


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
) -> tuple[float, float, float, float]:
    steps = normals.size
    dt = maturity / steps
    option = black_scholes_call(spot, strike, rate, sigma, maturity)
    delta = call_delta(spot, strike, rate, sigma, maturity)
    cash_without_cost = option - delta * spot
    initial_cost = cost_rate * abs(delta) * spot
    cash_with_cost = cash_without_cost - initial_cost
    raw_cost = initial_cost
    current = spot
    for index, normal in enumerate(normals, start=1):
        cash_without_cost *= math.exp(rate * dt)
        cash_with_cost *= math.exp(rate * dt)
        current *= math.exp((rate - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * float(normal))
        if index < steps:
            remaining = maturity - index * dt
            next_delta = call_delta(current, strike, rate, sigma, remaining)
            trade = next_delta - delta
            cost = cost_rate * abs(trade) * current
            cash_without_cost -= trade * current
            cash_with_cost -= trade * current + cost
            raw_cost += cost
            delta = next_delta
    payoff = max(current - strike, 0.0)
    no_cost_error = delta * current + cash_without_cost - payoff
    cost_after_error = delta * current + cash_with_cost - payoff
    financed_cost_drag = no_cost_error - cost_after_error
    return float(no_cost_error), float(cost_after_error), float(financed_cost_drag), float(raw_cost)


def expect_rejection(action: Callable[[], object], diagnostic: str) -> int:
    try:
        action()
    except ValueError as error:
        if diagnostic not in str(error):
            raise AssertionError(f"wrong diagnostic: {error!s}") from error
        return 1
    raise AssertionError(f"expected rejection containing {diagnostic!r}")


def render_experiment_budget(observed: dict[str, float | int]) -> str:
    return (
        f"二叉树 {int(observed['tree_steps'])} 步，$\\Delta t={observed['tree_dt']:.6f}$；"
        f"Monte Carlo {int(observed['mc_samples'])} 路径，seed={int(observed['seed'])}；"
        f"波动率曲面 {int(observed['surface_nodes'])} 个执行价/期限节点"
    )


def render_report(observed: dict[str, float | int]) -> str:
    experiment_budget = render_experiment_budget(observed)
    return f"""# 衍生品定价与对冲可复现研究包

- 二次变差：离散值 {observed['quadratic_variation']:.6f}；Itô 离散恒等式两侧为 {observed['ito_left']:.6f} 与 {observed['ito_right']:.6f}。
- SDE：GBM 解析终值 {observed['gbm_exact']:.6f}，Euler 终值 {observed['gbm_euler']:.6f}；差异属于离散化误差。
- Black--Scholes 闭式价格 {observed['closed_price']:.6f}；二叉树 {observed['tree_price']:.6f}；Monte Carlo {observed['mc_price']:.6f}，样本方差 {observed['mc_variance']:.6f}，标准误 {observed['mc_standard_error']:.6f}，95% 半宽 {observed['mc_half_width']:.6f}。实验预算：{experiment_budget}。
- 波动率曲面：加权价格平方误差 {observed['surface_objective']:.12f}，最大隐波恢复误差 {observed['surface_max_sigma_error']:.12f}；执行价单调、蝶式凸性与日历单调约束均通过。
- 隐含波动率：由合成报价反解为 {observed['implied_volatility']:.6f}；套利不一致报价与病态校准必须拒绝。
- Greeks：解析 Delta {observed['analytic_delta']:.6f}，差分 Delta {observed['fd_delta']:.6f}；解析 Vega {observed['analytic_vega']:.6f}。
- 离散对冲：无成本复制误差 {observed['hedge_no_cost_error']:.6f}，成本后误差 {observed['hedge_after_cost_error']:.6f}，融资后成本拖累 {observed['hedge_cost_drag']:.6f}，名义成本现金流 {observed['hedge_raw_cost']:.6f}。
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
    mc_price, mc_standard_error, mc_variance = monte_carlo_call(spot, strike, rate, sigma, maturity, int(oracle["mc_samples"]), int(oracle["seed"]))
    mc_half_width = 1.96 * mc_standard_error
    implied = implied_volatility(closed, spot, strike, rate, maturity)
    analytic_delta = call_delta(spot, strike, rate, sigma, maturity)
    fd_delta = finite_difference(lambda value: black_scholes_call(value, strike, rate, sigma, maturity), spot, float(oracle["fd_step"]))
    analytic_vega = call_vega(spot, strike, rate, sigma, maturity)
    surface_nodes = [
        {"strike": float(strike_node), "maturity": float(maturity_node), "weight": 1.0,
         "price": black_scholes_call(spot, float(strike_node), rate, float(oracle["surface_sigma"]), float(maturity_node))}
        for maturity_node in oracle["surface_maturities"] for strike_node in oracle["surface_strikes"]
    ]
    surface_sigmas, surface_objective = calibrate_surface(spot, rate, surface_nodes)
    surface_max_sigma_error = max(abs(value - float(oracle["surface_sigma"])) for value in surface_sigmas)
    hedge_no_cost_error, hedge_after_cost_error, hedge_cost_drag, hedge_raw_cost = delta_hedge(
        spot, strike, rate, sigma, maturity, normals, float(oracle["cost_rate"])
    )
    theta = (float(parameters["physical_drift"]) - rate) / sigma
    theta_energy = 0.5 * theta * theta * maturity
    validate_market_price_risk_energy(theta_energy, float(oracle["maximum_theta_energy"]))
    validate_novikov_exponential_moment([theta_energy], [1.0])
    validate_risk_neutral_drift(float(parameters["physical_drift"]), rate, sigma, theta)
    tree_agrees = int(abs(tree - closed) <= float(oracle["tree_error_budget"]))
    mc_agrees = int(abs(mc_price - closed) <= mc_half_width)
    bad_surface = [dict(node) for node in surface_nodes]
    bad_surface[1]["price"] += 5.0
    failures = (
        expect_rejection(
            lambda: terminal_singular_theta_energy(maturity, 0.0),
            "Novikov condition",
        ),
        expect_rejection(lambda: validate_risk_neutral_drift(float(parameters["physical_drift"]), rate, sigma, -theta), "wrong risk-neutral drift"),
        expect_rejection(lambda: implied_volatility(spot, spot, strike, rate, maturity), "not bracketed"),
        expect_rejection(lambda: validate_surface_constraints(bad_surface), "butterfly convexity"),
        expect_rejection(lambda: finite_difference(lambda x: x * x, 1.0, 0.0), "step must be positive"),
        expect_rejection(lambda: monte_carlo_call(spot, strike, rate, sigma, maturity, 1, 1), "at least two paths"),
    )
    observed: dict[str, float | int] = {
        "quadratic_variation": qv, "ito_left": ito_left, "ito_right": ito_right,
        "gbm_exact": gbm_exact, "gbm_euler": gbm_euler,
        "closed_price": closed, "tree_price": tree, "mc_price": mc_price, "mc_standard_error": mc_standard_error,
        "mc_variance": mc_variance, "mc_half_width": mc_half_width,
        "tree_agrees": tree_agrees, "mc_agrees": mc_agrees,
        "surface_objective": surface_objective, "surface_max_sigma_error": surface_max_sigma_error,
        "tree_steps": int(oracle["tree_steps"]), "tree_dt": maturity / int(oracle["tree_steps"]),
        "mc_samples": int(oracle["mc_samples"]), "seed": int(oracle["seed"]),
        "surface_nodes": len(surface_nodes),
        "implied_volatility": implied, "analytic_delta": analytic_delta, "fd_delta": fd_delta, "analytic_vega": analytic_vega,
        "hedge_no_cost_error": hedge_no_cost_error, "hedge_after_cost_error": hedge_after_cost_error,
        "hedge_cost_drag": hedge_cost_drag, "hedge_raw_cost": hedge_raw_cost,
        "integrability_rejected": failures[0], "wrong_drift_rejected": failures[1], "calibration_bracket_rejected": failures[2],
        "surface_arbitrage_rejected": failures[3], "grid_rejected": failures[4], "mc_budget_rejected": failures[5],
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
        f"surface=({surface_objective:.12f},{surface_max_sigma_error:.12f}) "
        f"hedge=({hedge_no_cost_error:.6f},{hedge_after_cost_error:.6f},{hedge_cost_drag:.6f},{hedge_raw_cost:.6f}) "
        f"failures=({','.join(str(value) for value in failures)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch04/oracle.json")))
