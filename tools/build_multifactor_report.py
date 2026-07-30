from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from math_for_quant.lower.multifactor import bh_rejections, null_search_p_values
from math_for_quant.lower.multifactor_estimation import (
    classic_two_pass_fama_macbeth,
    lasso_coordinate_descent,
    predictive_fama_macbeth,
    ridge_closed_form,
)
from math_for_quant.lower.multifactor_library import (
    cross_check_estimators,
    cross_check_route_statistics,
)
from math_for_quant.lower.multifactor_research import (
    PortfolioInputs,
    PortfolioPolicy,
    Weighting,
    build_group_portfolio_ledger,
    load_real_cross_section,
)


ROOT = Path(__file__).resolve().parents[1]


def build_report() -> str:
    model = json.loads((ROOT / "data/fixtures/multifactor-model.json").read_text(encoding="utf-8"))
    signals = np.asarray(model["signals"], dtype=float)
    future = np.asarray(model["future_returns"], dtype=float)
    factors = np.asarray(model["factor_returns"], dtype=float)
    betas = np.asarray(model["betas"], dtype=float)
    assets = float(model["asset_alpha"]) + factors @ betas[None, :]
    predictive = predictive_fama_macbeth(signals, future)
    classic = classic_two_pass_fama_macbeth(assets, factors)
    estimation = json.loads((ROOT / "data/fixtures/multifactor-estimation.json").read_text(encoding="utf-8"))
    regularization_design = np.asarray(estimation["design"], dtype=float)
    regularization_target = np.asarray(estimation["target"], dtype=float)
    ridge = ridge_closed_form(
        regularization_design,
        regularization_target,
        float(estimation["ridge_penalty"]),
    )
    lasso = lasso_coordinate_descent(
        regularization_design,
        regularization_target,
        float(estimation["lasso_penalty"]),
        100_000,
    )
    library_estimators = cross_check_estimators(
        signals=signals,
        future_returns=future,
        asset_returns=assets,
        factor_returns=factors,
        regularization_design=regularization_design,
        regularization_target=regularization_target,
        ridge_penalty=float(estimation["ridge_penalty"]),
        lasso_penalty=float(estimation["lasso_penalty"]),
    )
    estimator_gap = max(
        abs(library_estimators.predictive_mean - predictive.mean_coefficient),
        float(np.max(np.abs(library_estimators.classic_betas - classic.betas))),
        float(
            np.max(
                np.abs(
                    library_estimators.classic_risk_prices
                    - classic.risk_prices
                )
            )
        ),
        float(np.max(np.abs(library_estimators.ridge_coefficients - ridge))),
        float(np.max(np.abs(library_estimators.lasso_coefficients - lasso))),
    )
    route_check = cross_check_route_statistics(
        panel_design=np.asarray([[0.0], [1.0], [0.0], [1.0], [1.0], [2.0]]),
        panel_target=np.asarray([1.0, 1.4, -2.0, -1.6, 0.9, 1.3]),
        entities=np.asarray([0, 0, 1, 1, 2, 2]),
        signal=np.asarray([-2.0, -1.0, 1.0, 2.0]),
        size=np.asarray([-1.0, -1.0, 1.0, 1.0]),
        industry=np.asarray([0.0, 1.0, 0.0, 1.0]),
        future_returns=np.asarray([-0.03, -0.01, 0.02, 0.04]),
        horizon_returns=np.asarray([
            [-0.03, -0.01, 0.02, 0.04],
            [-0.015, -0.004, 0.009, 0.02],
        ]),
        p_values=[0.001, 0.02, 0.3, 0.8],
        alpha=0.05,
    )
    route_gap = max(
        estimator_gap,
        route_check.panel_slope_gap,
        route_check.neutralization_gap,
        route_check.ic_gap,
        route_check.rank_ic_gap,
        route_check.decay_gap,
        float(route_check.bh_count_gap),
    )

    research = json.loads((ROOT / "data/fixtures/multifactor-research.json").read_text(encoding="utf-8"))
    ledger = build_group_portfolio_ledger(
        inputs=PortfolioInputs(
            signals=np.asarray(research["signals"], dtype=float),
            realized_returns=np.asarray(research["realized_returns"], dtype=float),
            market_caps=np.asarray(research["market_caps"], dtype=float),
        ),
        policy=PortfolioPolicy(
            quantiles=int(research["quantiles"]),
            weighting=Weighting.EQUAL,
            holding_periods=int(research["holding_periods"]),
            cost_per_unit_turnover=float(research["cost_per_unit_turnover"]),
            capacity_impact=float(research["capacity_impact"]),
        ),
    )
    real = load_real_cross_section(ROOT / "data/real/multifactor-wdi-2013-2014.json")
    p_values = null_search_p_values(seed=11, observations=60, attempts=20)
    naive = sum(value < 0.05 for value in p_values)
    bh = bh_rejections(p_values, 0.05)
    return (
        "# 多因子与计量 v0.3 冻结研究报告\n\n"
        "## 模型边界\n\n"
        f"- 预测型横截面斜率均值：{predictive.mean_coefficient:.6f}\n"
        f"- 经典两遍法因子风险价格：{classic.risk_prices[0]:.6f}\n"
        "- 前者是下一期收益预测关系；后者依赖因子资产定价模型，不能互换。\n\n"
        "## 估计与数值实现\n\n"
        f"- Ridge 透明系数：{ridge[0]:.6f}\n"
        f"- Lasso 透明系数：{lasso[0]:.6f}\n"
        f"- 全路线透明/成熟库最大差：{route_gap:.3e}\n"
        "- statsmodels/SciPy/scikit-learn 对照覆盖横截面、面板、中性化、IC、Rank IC、衰减、BH 与正则化。\n\n"
        "## 选择与多重检验\n\n"
        f"- 固定 20 次零信号搜索：朴素显著 {naive}，BH 拒绝 {bh}。\n"
        "- BH 的 FDR 保证依赖独立或适当正依赖条件；任意相关搜索需更保守协议。\n\n"
        "## 从信号到净收益\n\n"
        f"- 两期毛收益：{ledger.period_gross_returns[0]:.6f}, {ledger.period_gross_returns[1]:.6f}\n"
        f"- 期均毛收益：{ledger.gross_return:.6f}\n"
        f"- 双边换手：{ledger.turnover:.6f}\n"
        f"- 换手成本：{ledger.cost:.6f}\n"
        f"- 容量冲击：{ledger.capacity_impact:.6f}\n"
        f"- 期均净收益：{ledger.net_return:.6f}\n\n"
        "## 双轨数据与限制\n\n"
        "- 合成小面板是唯一正确性 oracle。\n"
        f"- WDI 外部截面覆盖 {len(real.countries)} 个国家，信号年 {real.signal_year}，结果年 {real.outcome_year}，相关系数 {real.correlation:.6f}。\n"
        "- WDI 只验证数据来源、许可、哈希与时间协议，不构成股票 alpha 或因果证据。\n"
    )


if __name__ == "__main__":
    print(build_report(), end="")
