from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from m4q.reporting import stable_gap

from m4q.lower.portfolio_estimation import (
    bootstrap_portfolio_volatility,
    factor_covariance,
    risk_contributions,
    risk_parity_weights,
    shrink_covariance,
)
from m4q.lower.portfolio_estimation_library import (
    library_factor_covariance,
    library_bootstrap_volatility_interval,
    library_portfolio_volatility,
    library_risk_parity_weights,
    library_shrink_covariance,
)
from m4q.lower.portfolio_optimization import (
    black_litterman_posterior,
    cvar_optimize,
    robust_cost_aware_rebalance,
)
from m4q.lower.portfolio_optimization_library import (
    enumerate_robust_rebalance,
    enumerate_two_asset_cvar,
    library_black_litterman_posterior,
)
from m4q.lower.portfolio_tail import (
    empirical_tail_risk,
    nonlinear_portfolio_loss,
    reverse_stress_scale,
)
from m4q.lower.portfolio_tail_library import (
    library_historical_var_es,
    library_nonlinear_loss,
    library_reverse_stress_roots,
)


def run_estimation(fixture: dict[str, Any]) -> dict[str, float | int]:
    covariance = np.asarray(fixture["covariance"], dtype=float)
    weights = risk_parity_weights(covariance)
    contributions = risk_contributions(weights, covariance)
    library_weights = library_risk_parity_weights(covariance)
    returns = np.asarray(fixture["returns"], dtype=float)
    sample = np.cov(returns, rowvar=False, ddof=0)
    target = np.eye(sample.shape[0]) * float(np.trace(sample) / sample.shape[0])
    intensity = float(fixture["shrinkage_intensity"])
    shrunk = shrink_covariance(sample, target, intensity)
    library_shrunk = library_shrink_covariance(returns, intensity)
    modeled = factor_covariance(
        np.asarray(fixture["factor_loadings"], dtype=float),
        np.asarray(fixture["factor_covariance"], dtype=float),
        np.asarray(fixture["idiosyncratic_variance"], dtype=float),
    )
    library_modeled = library_factor_covariance(
        np.asarray(fixture["factor_loadings"], dtype=float),
        np.asarray(fixture["factor_covariance"], dtype=float),
        np.asarray(fixture["idiosyncratic_variance"], dtype=float),
    )
    interval = bootstrap_portfolio_volatility(
        returns,
        np.asarray(fixture["bootstrap_weights"], dtype=float),
        samples=int(fixture["bootstrap_samples"]),
        seed=int(fixture["bootstrap_seed"]),
        confidence=float(fixture["bootstrap_confidence"]),
    )
    library_interval = library_bootstrap_volatility_interval(
        returns,
        np.asarray(fixture["bootstrap_weights"], dtype=float),
        samples=int(fixture["bootstrap_samples"]),
        seed=int(fixture["bootstrap_seed"]),
        confidence=float(fixture["bootstrap_confidence"]),
    )
    normalized = contributions / contributions.sum()
    return {
        "risk_parity_weight_1": weights[0],
        "risk_parity_weight_2": weights[1],
        "risk_parity_weight_3": weights[2],
        "maximum_risk_budget_gap": float(np.max(np.abs(normalized - 1.0 / 3.0))),
        "risk_parity_library_gap": float(np.max(np.abs(weights - library_weights))),
        "shrinkage_library_gap": float(np.max(np.abs(shrunk - library_shrunk))),
        "shrunk_minimum_eigenvalue": float(np.linalg.eigvalsh(shrunk).min()),
        "factor_trace": float(np.trace(modeled)),
        "factor_library_gap": float(np.max(np.abs(modeled - library_modeled))),
        "bootstrap_volatility": interval.point,
        "bootstrap_confidence": float(fixture["bootstrap_confidence"]),
        "bootstrap_library_point_gap": abs(
            interval.point
            - library_portfolio_volatility(
                returns, np.asarray(fixture["bootstrap_weights"], dtype=float)
            )
        ),
        "bootstrap_lower": interval.lower,
        "bootstrap_upper": interval.upper,
        "bootstrap_library_lower": library_interval[0],
        "bootstrap_library_upper": library_interval[1],
        "bootstrap_library_interval_gap": max(
            abs(interval.lower - library_interval[0]),
            abs(interval.upper - library_interval[1]),
        ),
        "bootstrap_samples": interval.samples,
    }


def run_optimization(fixture: dict[str, Any]) -> dict[str, float | int]:
    prior_mean = np.asarray(fixture["prior_mean"], dtype=float)
    prior_covariance = np.asarray(fixture["prior_covariance"], dtype=float)
    views = np.asarray(fixture["views"], dtype=float)
    view_returns = np.asarray(fixture["view_returns"], dtype=float)
    view_covariance = np.asarray(fixture["view_covariance"], dtype=float)
    tau = float(fixture["tau"])
    posterior_mean, posterior_covariance = black_litterman_posterior(
        prior_mean, prior_covariance, views, view_returns, view_covariance, tau=tau
    )
    library_mean, library_covariance = library_black_litterman_posterior(
        prior_mean, prior_covariance, views, view_returns, view_covariance, tau=tau
    )
    scenarios = np.asarray(fixture["scenario_returns"], dtype=float)
    confidence = float(fixture["cvar_confidence"])
    cvar = cvar_optimize(
        scenarios, confidence=confidence, maximum_weight=float(fixture["maximum_weight"])
    )
    enumerated_weights, enumerated_cvar = enumerate_two_asset_cvar(
        scenarios,
        confidence=confidence,
        grid_step=float(fixture["grid_step"]),
        maximum_weight=float(fixture["maximum_weight"]),
    )
    rebalance = robust_cost_aware_rebalance(
        posterior_mean,
        posterior_covariance,
        np.asarray(fixture["current_weights"], dtype=float),
        np.asarray(fixture["return_uncertainty"], dtype=float),
        risk_aversion=float(fixture["risk_aversion"]),
        uncertainty_penalty=float(fixture["uncertainty_penalty"]),
        cost_rate=float(fixture["cost_rate"]),
        capital=float(fixture["capital"]),
        maximum_weight=float(fixture["maximum_weight"]),
        tradable=np.asarray(fixture["tradable"], dtype=int),
        grid_step=float(fixture["grid_step"]),
    )
    library_rebalance_weights, library_rebalance_score = enumerate_robust_rebalance(
        posterior_mean,
        posterior_covariance,
        np.asarray(fixture["current_weights"], dtype=float),
        np.asarray(fixture["return_uncertainty"], dtype=float),
        risk_aversion=float(fixture["risk_aversion"]),
        uncertainty_penalty=float(fixture["uncertainty_penalty"]),
        cost_rate=float(fixture["cost_rate"]),
        maximum_weight=float(fixture["maximum_weight"]),
        tradable=np.asarray(fixture["tradable"], dtype=int),
        grid_step=float(fixture["grid_step"]),
    )
    return {
        "posterior_mean_1": posterior_mean[0],
        "posterior_mean_2": posterior_mean[1],
        "posterior_minimum_eigenvalue": float(np.linalg.eigvalsh(posterior_covariance).min()),
        "black_litterman_library_gap": float(
            max(np.max(np.abs(posterior_mean - library_mean)), np.max(np.abs(posterior_covariance - library_covariance)))
        ),
        "cvar_weight_1": cvar.weights[0],
        "cvar_weight_2": cvar.weights[1],
        "cvar_objective": cvar.objective,
        "cvar_identity_gap": abs(cvar.objective - cvar.recomputed_cvar),
        "cvar_enumeration_gap": abs(cvar.objective - enumerated_cvar),
        "cvar_weight_gap": float(np.max(np.abs(cvar.weights - enumerated_weights))),
        "rebalance_weight_1": rebalance.weights[0],
        "rebalance_weight_2": rebalance.weights[1],
        "turnover": rebalance.turnover,
        "cash_cost": rebalance.cash_cost,
        "robust_score": rebalance.robust_score,
        "rebalance_library_gap": float(
            max(
                np.max(np.abs(rebalance.weights - library_rebalance_weights)),
                abs(rebalance.robust_score - library_rebalance_score),
            )
        ),
    }


def run_tail(fixture: dict[str, Any]) -> dict[str, float | int | str]:
    losses = np.linspace(
        float(fixture["loss_start"]), float(fixture["loss_end"]), int(fixture["loss_count"])
    )
    confidence = float(fixture["confidence"])
    tail = empirical_tail_risk(
        losses,
        confidence,
        minimum_tail_observations=int(fixture["minimum_tail_observations"]),
        warning_tail_observations=int(fixture["warning_tail_observations"]),
        bootstrap_samples=int(fixture["bootstrap_samples"]),
        seed=int(fixture["bootstrap_seed"]),
    )
    library_var, library_es = library_historical_var_es(losses, confidence)
    direction = np.asarray(fixture["shock_direction"], dtype=float)
    linear = np.asarray(fixture["linear_exposure"], dtype=float)
    gamma = np.asarray(fixture["gamma_exposure"], dtype=float)
    unit_loss = nonlinear_portfolio_loss(direction, linear, gamma)
    library_unit_loss = library_nonlinear_loss(direction, linear, gamma)
    linear_loss = float(-(linear @ direction))
    scale = reverse_stress_scale(
        direction,
        linear,
        gamma,
        loss_threshold=float(fixture["loss_threshold"]),
        maximum_scale=float(fixture["maximum_scale"]),
    )
    reached = nonlinear_portfolio_loss(scale * direction, linear, gamma)
    library_scale = library_reverse_stress_roots(
        direction,
        linear,
        gamma,
        threshold=float(fixture["loss_threshold"]),
        maximum_scale=float(fixture["maximum_scale"]),
    )
    return {
        "confidence": confidence,
        "value_at_risk": tail.value_at_risk,
        "expected_shortfall": tail.expected_shortfall,
        "effective_tail_observations": tail.effective_tail_observations,
        "quantile_resolution": tail.quantile_resolution,
        "es_interval_lower": tail.es_interval[0],
        "es_interval_upper": tail.es_interval[1],
        "es_interval_confidence": 0.95,
        "tail_status": tail.status,
        "tail_library_gap": max(abs(tail.value_at_risk - library_var), abs(tail.expected_shortfall - library_es)),
        "linear_unit_loss": linear_loss,
        "nonlinear_unit_loss": unit_loss,
        "nonlinear_library_gap": abs(unit_loss - library_unit_loss),
        "reverse_stress_scale": scale,
        "reverse_stress_library_gap": abs(scale - library_scale),
        "loss_threshold": float(fixture["loss_threshold"]),
        "reverse_stress_loss": reached,
        "reverse_stress_identity_gap": abs(reached - float(fixture["loss_threshold"])),
    }


def render_route_report(
    estimation: dict[str, float | int],
    optimization: dict[str, float | int],
    tail: dict[str, float | int | str],
    real_data: dict[str, float | int | str],
) -> str:
    if real_data["tail_status"] == "reject":
        real_tail = (
            f"等权损失的 {100.0 * float(real_data['tail_confidence']):.0f}% 尾部门禁仅有 "
            f"{float(real_data['tail_effective_observations']):.2f} 个有效观察，状态 reject，"
            "因此不发布 VaR/ES"
        )
    else:
        real_tail = (
            f"等权损失的 {100.0 * float(real_data['tail_confidence']):.0f}% 尾部门禁状态 "
            f"{real_data['tail_status']}"
        )
    return f"""# 组合与风险 v0.3 路线报告

- 风险估计：风险平价权重 ({estimation['risk_parity_weight_1']:.6f}, {estimation['risk_parity_weight_2']:.6f}, {estimation['risk_parity_weight_3']:.6f})，最大风险预算差 {stable_gap(estimation['maximum_risk_budget_gap']):.3e}，独立 SLSQP 权重差 {stable_gap(estimation['risk_parity_library_gap']):.3e}；收缩矩阵最小特征值 {estimation['shrunk_minimum_eigenvalue']:.3e}，sklearn 差 {stable_gap(estimation['shrinkage_library_gap']):.3e}；因子矩阵迹 {estimation['factor_trace']:.6f}、独立差 {stable_gap(estimation['factor_library_gap']):.3e}；组合波动率 bootstrap {100.0 * estimation['bootstrap_confidence']:.0f}% 区间 [{estimation['bootstrap_lower']:.6f}, {estimation['bootstrap_upper']:.6f}]，SciPy 区间端点最大差 {stable_gap(estimation['bootstrap_library_interval_gap']):.3e}。
- 优化实施：Black--Litterman 后验均值 ({optimization['posterior_mean_1']:.6f}, {optimization['posterior_mean_2']:.6f})、后验预测协方差最小特征值 {optimization['posterior_minimum_eigenvalue']:.6f}，Woodbury 对照差 {stable_gap(optimization['black_litterman_library_gap']):.3e}；CVaR LP 权重 ({optimization['cvar_weight_1']:.6f}, {optimization['cvar_weight_2']:.6f})、目标 {optimization['cvar_objective']:.6f}、枚举差 {stable_gap(optimization['cvar_enumeration_gap']):.3e}；稳健再平衡权重 ({optimization['rebalance_weight_1']:.6f}, {optimization['rebalance_weight_2']:.6f})，换手 {optimization['turnover']:.6f}，现金成本 {optimization['cash_cost']:.2f}，独立枚举差 {stable_gap(optimization['rebalance_library_gap']):.3e}。
- 尾部风险：{100.0 * float(tail['confidence']):.0f}% VaR/ES={tail['value_at_risk']:.6f}/{tail['expected_shortfall']:.6f}，有效尾部观察 {tail['effective_tail_observations']:.1f}，状态 {tail['tail_status']}，分位离散误差 {tail['quantile_resolution']:.6f}，ES bootstrap {100.0 * float(tail['es_interval_confidence']):.0f}% 区间 [{tail['es_interval_lower']:.6f}, {tail['es_interval_upper']:.6f}]，独立历史实现差 {tail['tail_library_gap']:.3e}。
- 压力测试：单位方向线性/非线性损失 {tail['linear_unit_loss']:.6f}/{tail['nonlinear_unit_loss']:.6f}，逐项独立差 {stable_gap(tail['nonlinear_library_gap']):.3e}；使损失达到 {float(tail['loss_threshold']):g} 的最小反向压力尺度 {tail['reverse_stress_scale']:.6f}，独立根差 {stable_gap(tail['reverse_stress_library_gap']):.3e}，重估恒等差 {stable_gap(tail['reverse_stress_identity_gap']):.3e}。
- 真实数据轨：冻结宏观快照含 {real_data['rows']} 个水平观察和 {real_data['growth_rows']} 个增长率观察，增长率协方差迹为 {real_data['covariance_trace']:.6e}，两列风险平价权重 ({real_data['risk_parity_weight_1']:.6f}, {real_data['risk_parity_weight_2']:.6f})；{100.0 * float(real_data['cvar_confidence']):.0f}% CVaR 权重 ({real_data['cvar_weight_1']:.6f}, {real_data['cvar_weight_2']:.6f})、目标 {real_data['cvar_objective']:.6f}；{real_tail}。
- 限制：合成收益、两资产网格和二阶重估用于验证定义与协议；宏观真实快照只演示有来源的时间序列如何进入协方差、CVaR 与历史尾部计算，不代表可交易资产、尾部稳定性或盈利能力。
"""


__all__ = ["render_route_report", "run_estimation", "run_optimization", "run_tail"]
