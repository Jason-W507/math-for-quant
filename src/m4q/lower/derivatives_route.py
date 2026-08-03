from __future__ import annotations

import math
from typing import Any

import numpy as np

from m4q.reporting import stable_gap

from m4q.lower.derivatives import (
    binomial_call,
    black_scholes_call,
    delta_hedge,
    monte_carlo_call,
    quadratic_variation_and_ito,
    validate_surface_constraints,
)
from m4q.lower.derivatives_hedging import (
    greek_convergence,
    simulate_hedging_distribution,
)
from m4q.lower.derivatives_numerics import (
    SurfaceNode,
    fit_parametric_total_variance,
    implicit_fd_call,
    point_implied_volatilities,
)
from m4q.lower.derivatives_numerics_library import (
    library_black_scholes_call,
    library_binomial_call,
    library_implicit_fd_call,
    library_quadrature_call,
)
from m4q.lower.derivatives_stochastic import (
    nested_quadratic_variation,
    terminal_singular_theta_energy,
    validate_novikov_exponential_moment,
    validate_risk_neutral_drift,
)
from m4q.lower.derivatives_stochastic_library import (
    measure_change_density_gap,
)
from m4q.lower.notebook_evidence import expect_value_error


def run_stochastic(fixture: dict[str, Any]) -> dict[str, float | int]:
    size = int(fixture["fine_grid_size"])
    maturity = float(fixture["maturity"])
    normals = np.random.default_rng(int(fixture["seed"])).standard_normal(size)
    increments = math.sqrt(maturity / size) * normals
    block_sizes = tuple(int(value) for value in fixture["block_sizes"])
    levels = nested_quadratic_variation(increments, block_sizes=block_sizes)
    _, ito_left, ito_right = quadratic_variation_and_ito(normals, maturity)
    physical_drift = float(fixture["physical_drift"])
    rate = float(fixture["rate"])
    sigma = float(fixture["sigma"])
    theta = (physical_drift - rate) / sigma
    validate_risk_neutral_drift(physical_drift, rate, sigma, theta)
    large_energy_moment = validate_novikov_exponential_moment(
        [float(fixture["large_finite_energy"])], [1.0]
    )
    coarse_singular = terminal_singular_theta_energy(
        maturity, float(fixture["singular_coarse_cutoff"])
    )
    fine_singular = terminal_singular_theta_energy(
        maturity, float(fixture["singular_fine_cutoff"])
    )
    singular_rejected = expect_value_error(
        lambda: terminal_singular_theta_energy(maturity, 0.0),
        "Novikov condition",
    )
    return {
        "qv_coarse": levels[block_sizes[0]],
        "qv_medium": levels[block_sizes[1]],
        "qv_fine": levels[block_sizes[-1]],
        "qv_fine_error": abs(levels[block_sizes[-1]] - maturity),
        "ito_identity_gap": abs(ito_left - ito_right),
        "risk_neutral_drift": physical_drift - sigma * theta,
        "large_energy_finite": int(math.isfinite(large_energy_moment)),
        "singular_coarse_energy": coarse_singular,
        "singular_fine_energy": fine_singular,
        "singular_rejected": singular_rejected,
        "measure_change_library_gap": measure_change_density_gap(
            theta=theta, observation=float(fixture["density_observation"])
        ),
    }


def _surface_nodes(fixture: dict[str, Any]) -> list[SurfaceNode]:
    spot = float(fixture["spot"])
    rate = float(fixture["rate"])
    coefficients = np.asarray(fixture["surface_coefficients"], dtype=float)
    nodes: list[SurfaceNode] = []
    for maturity in fixture["surface_maturities"]:
        maturity_value = float(maturity)
        forward = spot * math.exp(rate * maturity_value)
        for strike in fixture["surface_strikes"]:
            strike_value = float(strike)
            log_moneyness = math.log(strike_value / forward)
            total_variance = (
                coefficients[0] * maturity_value
                + coefficients[1] * log_moneyness**2
                + coefficients[2] * maturity_value**2
            )
            sigma = math.sqrt(total_variance / maturity_value)
            nodes.append(
                SurfaceNode(
                    strike=strike_value,
                    maturity=maturity_value,
                    price=black_scholes_call(
                        spot, strike_value, rate, sigma, maturity_value
                    ),
                    weight=1.0,
                )
            )
    return nodes


def run_numerics(fixture: dict[str, Any]) -> dict[str, float | int]:
    spot = float(fixture["spot"])
    strike = float(fixture["strike"])
    rate = float(fixture["rate"])
    sigma = float(fixture["sigma"])
    maturity = float(fixture["maturity"])
    closed = black_scholes_call(spot, strike, rate, sigma, maturity)
    closed_library = library_black_scholes_call(
        spot, strike, rate, sigma, maturity
    )
    tree = binomial_call(
        spot, strike, rate, sigma, maturity, int(fixture["tree_steps"])
    )
    tree_library = library_binomial_call(
        spot, strike, rate, sigma, maturity, int(fixture["tree_steps"])
    )
    pde = implicit_fd_call(
        spot,
        strike,
        rate,
        sigma,
        maturity,
        space_steps=int(fixture["pde_space_steps"]),
        time_steps=int(fixture["pde_time_steps"]),
        spot_max=float(fixture["pde_spot_max"]),
    )
    space_steps = int(fixture["pde_space_steps"])
    time_steps = int(fixture["pde_time_steps"])
    spot_max = float(fixture["pde_spot_max"])
    pde_library = library_implicit_fd_call(
        spot, strike, rate, sigma, maturity,
        space_steps=space_steps, time_steps=time_steps, spot_max=spot_max,
    )
    pde_coarse_space = implicit_fd_call(
        spot, strike, rate, sigma, maturity,
        space_steps=space_steps // 2, time_steps=time_steps, spot_max=spot_max,
    )
    pde_coarse_time = implicit_fd_call(
        spot, strike, rate, sigma, maturity,
        space_steps=space_steps, time_steps=time_steps // 2, spot_max=spot_max,
    )
    boundary_spot_max = 0.75 * spot_max
    boundary_space_steps = round(space_steps * boundary_spot_max / spot_max)
    pde_short_boundary = implicit_fd_call(
        spot, strike, rate, sigma, maturity,
        space_steps=boundary_space_steps,
        time_steps=time_steps,
        spot_max=boundary_spot_max,
    )
    mc_price, mc_se, _ = monte_carlo_call(
        spot,
        strike,
        rate,
        sigma,
        maturity,
        int(fixture["mc_paths"]),
        int(fixture["mc_seed"]),
    )
    mc_library, mc_quadrature_error = library_quadrature_call(
        spot, strike, rate, sigma, maturity
    )
    nodes = _surface_nodes(fixture)
    point_vols = point_implied_volatilities(spot, rate, nodes)
    fit = fit_parametric_total_variance(spot, rate, nodes)
    dividend_yield = float(fixture["calendar_dividend_yield"])
    normalized_nodes: list[dict[str, float]] = []
    for calendar_maturity in (0.5, 1.0):
        forward = spot * math.exp((rate - dividend_yield) * calendar_maturity)
        discount = math.exp(-rate * calendar_maturity)
        for moneyness in (0.9, 1.0, 1.1):
            calendar_strike = moneyness * forward
            normalized_nodes.append(
                {
                    "maturity": calendar_maturity,
                    "strike": calendar_strike,
                    "price": library_black_scholes_call(
                        spot, calendar_strike, rate, sigma, calendar_maturity,
                        dividend_yield=dividend_yield,
                    ),
                    "forward": forward,
                    "discount_factor": discount,
                }
            )
    validate_surface_constraints(
        normalized_nodes,
        rate=rate,
        dividend_yield=dividend_yield,
        calendar_mode="forward-normalized",
    )
    return {
        "closed_price": closed,
        "closed_library_gap": abs(closed_library - closed),
        "tree_price": tree,
        "tree_error": abs(tree - closed),
        "tree_library_gap": abs(tree_library - tree),
        "pde_price": pde,
        "pde_error": abs(pde - closed),
        "pde_library_gap": abs(pde_library - pde),
        "pde_space_gap": abs(pde - pde_coarse_space),
        "pde_time_gap": abs(pde - pde_coarse_time),
        "pde_boundary_gap": abs(pde - pde_short_boundary),
        "mc_price": mc_price,
        "mc_standard_error": mc_se,
        "mc_error": abs(mc_price - closed),
        "mc_library_gap": abs(mc_library - mc_price),
        "mc_quadrature_error": mc_quadrature_error,
        "point_iv_count": int(point_vols.size),
        "surface_a": float(fit.coefficients[0]),
        "surface_b": float(fit.coefficients[1]),
        "surface_c": float(fit.coefficients[2]),
        "surface_max_price_error": fit.maximum_price_error,
        "surface_weighted_loss": fit.weighted_price_loss,
        "surface_library_gap": fit.library_coefficient_gap,
        "forward_calendar_passed": 1,
        "raw_calendar_rejected": expect_value_error(
            lambda: validate_surface_constraints(
                normalized_nodes,
                rate=rate,
                dividend_yield=dividend_yield,
                calendar_mode="nonnegative-rate-no-dividend",
            ),
            "requires nonnegative rates and no dividends",
        ),
        "nonuniform_convexity_rejected": expect_value_error(
            lambda: validate_surface_constraints(
                [
                    {"maturity": 1.0, "strike": 90.0, "price": 15.0},
                    {"maturity": 1.0, "strike": 100.0, "price": 13.0},
                    {"maturity": 1.0, "strike": 130.0, "price": 3.0},
                ],
                calendar_mode="skip",
            ),
            "butterfly convexity",
        ),
    }


def run_hedging(fixture: dict[str, Any]) -> dict[str, float | int]:
    result = simulate_hedging_distribution(
        spot=float(fixture["spot"]),
        strike=float(fixture["strike"]),
        rate=float(fixture["rate"]),
        sigma=float(fixture["sigma"]),
        maturity=float(fixture["maturity"]),
        paths=int(fixture["paths"]),
        steps=int(fixture["steps"]),
        cost_rate=float(fixture["cost_rate"]),
        seed=int(fixture["seed"]),
    )
    greeks = greek_convergence(
        float(fixture["spot"]),
        float(fixture["strike"]),
        float(fixture["rate"]),
        float(fixture["sigma"]),
        float(fixture["maturity"]),
        steps=tuple(float(value) for value in fixture["greek_steps"]),
    )
    coarse = simulate_hedging_distribution(
        spot=float(fixture["spot"]), strike=float(fixture["strike"]),
        rate=float(fixture["rate"]), sigma=float(fixture["sigma"]),
        maturity=float(fixture["maturity"]), paths=int(fixture["paths"]),
        steps=int(fixture["coarse_steps"]), cost_rate=float(fixture["cost_rate"]),
        seed=int(fixture["seed"]),
    )
    fine = simulate_hedging_distribution(
        spot=float(fixture["spot"]), strike=float(fixture["strike"]),
        rate=float(fixture["rate"]), sigma=float(fixture["sigma"]),
        maturity=float(fixture["maturity"]), paths=int(fixture["paths"]),
        steps=int(fixture["fine_steps"]), cost_rate=float(fixture["cost_rate"]),
        seed=int(fixture["seed"]),
    )
    negative_cost_rejected = expect_value_error(
        lambda: simulate_hedging_distribution(
            spot=100.0, strike=100.0, rate=0.02, sigma=0.2, maturity=1.0,
            paths=8, steps=2, cost_rate=-0.001, seed=1,
        ),
        "cannot be negative",
    )
    one_step_no_cost, one_step_after_cost, one_step_drag, one_step_raw_cost = (
        delta_hedge(
            float(fixture["spot"]), float(fixture["strike"]),
            float(fixture["rate"]), float(fixture["sigma"]),
            float(fixture["maturity"]), np.asarray([0.0]),
            float(fixture["cost_rate"]),
        )
    )
    return {
        "paths": result.paths,
        "steps": result.steps,
        "no_cost_bias": result.no_cost_bias,
        "no_cost_rmse": result.no_cost_rmse,
        "after_cost_bias": result.after_cost_bias,
        "after_cost_rmse": result.after_cost_rmse,
        "error_q05": result.error_q05,
        "error_q50": result.error_q50,
        "error_q95": result.error_q95,
        "mean_cost": result.mean_cost,
        "cost_q95": result.cost_q95,
        "summary_gap": result.summary_gap,
        "delta_gap": greeks.delta_gap,
        "delta_coarse_gap": greeks.delta_errors[0],
        "gamma_gap": greeks.gamma_gap,
        "gamma_coarse_gap": greeks.gamma_errors[0],
        "vega_gap": greeks.vega_gap,
        "vega_coarse_gap": greeks.vega_errors[0],
        "coarse_after_cost_rmse": coarse.after_cost_rmse,
        "fine_after_cost_rmse": fine.after_cost_rmse,
        "coarse_mean_cost": coarse.mean_cost,
        "fine_mean_cost": fine.mean_cost,
        "negative_cost_rejected": negative_cost_rejected,
        "one_step_raw_cost": one_step_raw_cost,
        "one_step_financed_drag": one_step_drag,
        "one_step_cost_identity_gap": abs(
            (one_step_no_cost - one_step_after_cost) - one_step_drag
        ),
    }


def render_route_report(
    stochastic: dict[str, float | int],
    numerics: dict[str, float | int],
    hedging: dict[str, float | int],
) -> str:
    return f"""# 衍生品定价与对冲 v0.3 路线报告

- 随机分析：嵌套分割二次变差由 {stochastic['qv_coarse']:.6f} 细化到 {stochastic['qv_fine']:.6f}；离散 Itô 恒等式误差 {stable_gap(stochastic['ito_identity_gap']):.3e}；终端奇点能量从 {stochastic['singular_coarse_energy']:.6f} 增至 {stochastic['singular_fine_energy']:.6f}，完整端点以稳定 Novikov 诊断拒绝；指数密度与 SciPy 正态密度比差 {stable_gap(stochastic['measure_change_library_gap']):.3e}。
- 无套利：物理漂移经市场价格风险变换为 {stochastic['risk_neutral_drift']:.6f}；有限但很大的确定性能量仍通过 Novikov 指数矩检查，教材能量预算另行报告。
- 定价：Black--Scholes 闭式 {numerics['closed_price']:.6f}；树 {numerics['tree_price']:.6f}（误差 {numerics['tree_error']:.6f}）；隐式 PDE {numerics['pde_price']:.6f}（总偏差 {numerics['pde_error']:.6f}，空间/时间/边界扰动 {numerics['pde_space_gap']:.6f}/{numerics['pde_time_gap']:.6f}/{stable_gap(numerics['pde_boundary_gap']):.3e}）；Monte Carlo {numerics['mc_price']:.6f}（标准误 {numerics['mc_standard_error']:.6f}，相对独立 SciPy 积分差 {numerics['mc_library_gap']:.6f}，即 {numerics['mc_library_gap'] / numerics['mc_standard_error']:.2f} 个标准误）；闭式/树/PDE 成熟库差分别为 {stable_gap(numerics['closed_library_gap']):.3e}/{stable_gap(numerics['tree_library_gap']):.3e}/{stable_gap(numerics['pde_library_gap']):.3e}，积分误差估计 {stable_gap(numerics['mc_quadrature_error']):.3e}。
- 校准：先逐点反演 {int(numerics['point_iv_count'])} 个隐含波动率，再拟合参数化总方差系数 ({numerics['surface_a']:.6f}, {numerics['surface_b']:.6f}, {numerics['surface_c']:.6f})；最大价格误差 {stable_gap(numerics['surface_max_price_error']):.3e}，透明/成熟库系数差 {stable_gap(numerics['surface_library_gap']):.3e}；含分红的 forward/discount 归一化日历门禁通过。
- 对冲：{int(hedging['paths'])} 路径、{int(hedging['steps'])} 次离散步；无成本误差 bias/RMSE=({hedging['no_cost_bias']:.6f}, {hedging['no_cost_rmse']:.6f})，成本后=({hedging['after_cost_bias']:.6f}, {hedging['after_cost_rmse']:.6f})；成本后误差 5%/50%/95%=({hedging['error_q05']:.6f}, {hedging['error_q50']:.6f}, {hedging['error_q95']:.6f})；平均成本 {hedging['mean_cost']:.6f}，95% 成本 {hedging['cost_q95']:.6f}；Delta/Gamma/Vega 差分差从 {hedging['delta_coarse_gap']:.3e}/{hedging['gamma_coarse_gap']:.3e}/{hedging['vega_coarse_gap']:.3e} 收敛到 {hedging['delta_gap']:.3e}/{hedging['gamma_gap']:.3e}/{hedging['vega_gap']:.3e}；12/52 次调仓成本后 RMSE {hedging['coarse_after_cost_rmse']:.6f}/{hedging['fine_after_cost_rmse']:.6f}。
- 限制：合成 GBM、常波动率、无跳跃和简化交易成本只支持方法验证；财政部收益率快照只演示贴现输入的来源、日期、许可与哈希，不是期权曲面或盈利证据。
"""


__all__ = ["render_route_report", "run_hedging", "run_numerics", "run_stochastic"]
