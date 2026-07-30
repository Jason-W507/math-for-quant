from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from math_for_quant.lower.stat_arb import validate_scaler, validate_walk_forward
from math_for_quant.lower.stat_arb_estimation import ScalarStateSpaceSpec, kalman_filter_and_smooth, regime_filter
from math_for_quant.lower.stat_arb_library import cross_check_long_run, fit_markov_switching, library_kalman_filter_and_smooth
from math_for_quant.lower.stat_arb_models import engle_granger, fit_ecm, ou_diagnostics
from math_for_quant.lower.stat_arb_research import ExecutionPolicy, build_forecast_ledger, validate_failure_state, validate_purged_walk_forward
from math_for_quant.lower.stat_arb_execution_library import library_forecast_ledger


ROOT = Path(__file__).resolve().parents[1]


def rejects(callable_) -> int:
    try:
        callable_()
    except ValueError:
        return 1
    return 0


def build_report() -> str:
    model = json.loads((ROOT / "data/fixtures/stat-arb-model.json").read_text(encoding="utf-8"))
    count = int(model["observation_count"])
    x = np.cumsum(np.resize(np.asarray(model["x_increments"], dtype=float), count))
    residual = float(model["residual_phi"]) ** np.arange(count)
    y = float(model["cointegration_intercept"]) + float(model["cointegration_slope"]) * x + residual
    relation = engle_granger(y, x)
    ecm = fit_ecm(y, x, relation)
    ou = ou_diagnostics(relation.residuals, step=float(model["step"]))
    real = json.loads((ROOT / "data/real/stat-arb-us-macro-1999q4-2009q3.json").read_text(encoding="utf-8"))
    real_y = np.log(np.asarray([row["realgdp"] for row in real["rows"]], dtype=float))
    real_x = np.log(np.asarray([row["realcons"] for row in real["rows"]], dtype=float))
    library = cross_check_long_run(real_y, real_x)

    estimation = json.loads((ROOT / "data/fixtures/stat-arb-estimation.json").read_text(encoding="utf-8"))
    observations = np.asarray(estimation["observations"], dtype=float)
    spec = ScalarStateSpaceSpec(
        float(estimation["transition"]), float(estimation["observation_loading"]),
        float(estimation["process_variance"]), float(estimation["observation_variance"]),
        float(estimation["initial_mean"]), float(estimation["initial_variance"]),
    )
    states = kalman_filter_and_smooth(observations, spec=spec)
    library_states = library_kalman_filter_and_smooth(observations, spec=spec)
    high_noise = kalman_filter_and_smooth(
        observations,
        spec=ScalarStateSpaceSpec(spec.transition, spec.observation_loading, 0.8, spec.observation_variance, spec.initial_mean, spec.initial_variance),
    )
    regimes = regime_filter(
        observations,
        transition=np.asarray(estimation["regime_transition"]),
        means=np.asarray(estimation["regime_means"]),
        variances=np.asarray(estimation["regime_variances"]),
        initial=np.asarray(estimation["regime_initial"]),
    )
    fitted_regimes = fit_markov_switching(np.tile(observations, 6))

    research = json.loads((ROOT / "data/fixtures/stat-arb-research.json").read_text(encoding="utf-8"))
    ledger = build_forecast_ledger(
        forecasts=np.asarray(research["forecasts"]),
        realized_returns=np.asarray(research["realized_returns"]),
        fill_fractions=np.asarray(research["fill_fractions"]),
        policy=ExecutionPolicy(
            entry_threshold=float(research["entry_threshold"]),
            position_limit=float(research["position_limit"]),
            holding_period=int(research["holding_period"]),
            rebalance_every=int(research["rebalance_every"]),
            cost_per_unit_turnover=float(research["cost_per_unit_turnover"]),
        ),
    )
    high_cost = build_forecast_ledger(
        forecasts=np.asarray(research["forecasts"]),
        realized_returns=np.asarray(research["realized_returns"]),
        fill_fractions=np.asarray(research["fill_fractions"]),
        policy=ExecutionPolicy(
            float(research["entry_threshold"]), float(research["position_limit"]),
            int(research["holding_period"]), int(research["rebalance_every"]),
            2.0 * float(research["cost_per_unit_turnover"]),
        ),
    )
    library_ledger = library_forecast_ledger(
        forecasts=np.asarray(research["forecasts"]),
        realized_returns=np.asarray(research["realized_returns"]),
        fill_fractions=np.asarray(research["fill_fractions"]),
        policy=ExecutionPolicy(
            float(research["entry_threshold"]), float(research["position_limit"]),
            int(research["holding_period"]), int(research["rebalance_every"]),
            float(research["cost_per_unit_turnover"]),
        ),
    )
    negative_checks = {
        "purge": rejects(lambda: validate_purged_walk_forward(train_indices=range(0, 7), validation_indices=range(8, 10), trade_indices=range(12, 14), label_horizon=2, embargo=2)),
        "embargo": rejects(lambda: validate_purged_walk_forward(train_indices=range(0, 6), validation_indices=range(8, 10), trade_indices=range(11, 14), label_horizon=2, embargo=2)),
        "walk-forward": rejects(lambda: validate_walk_forward([
            {"name":"train", "start":"2020-01", "end":"2020-06"},
            {"name":"validation", "start":"2020-06", "end":"2020-08"},
            {"name":"trade", "start":"2020-09", "end":"2020-12"},
        ])),
        "scaler": rejects(lambda: validate_scaler("2020-08-01", "2020-06-01")),
        "failure-state": rejects(lambda: validate_failure_state(half_life=9.0, maximum_half_life=4.0, fill_rate=0.9, minimum_fill_rate=0.5)),
    }
    return (
        "# 时间序列与统计套利 v0.3 冻结研究报告\n\n"
        "## 长期关系与动态\n\n"
        f"- Engle--Granger 长期斜率：{relation.slope:.6f}\n"
        f"- 残差 DF 统计量：{relation.residual_adf_statistic:.6f}\n"
        f"- ECM 调整速度：{ecm.adjustment_speed:.6f}\n"
        f"- OU 半衰期：{ou.half_life:.6f}\n"
        f"- OU 从冻结起点命中均值的期望时间：{ou.expected_first_passage:.6f}\n"
        f"- 公共快照透明/成熟库斜率差：{abs(library.transparent_slope-library.library_slope):.3e}\n"
        f"- statsmodels Engle--Granger p 值：{library.engle_granger_p_value:.6f}；Johansen rank：{library.johansen_rank}\n\n"
        "## 状态推断\n\n"
        f"- 在线滤波末值：{states.filtered[-1]:.6f}\n"
        f"- 第二期平滑值：{states.smoothed[1]:.6f}\n"
        f"- 给定参数状态 1 末概率：{regimes[-1,1]:.6f}\n"
        f"- 透明/成熟库 Kalman 最大差：{max(np.max(np.abs(states.filtered-library_states.filtered)),np.max(np.abs(states.smoothed-library_states.smoothed))):.3e}\n"
        f"- 过程噪声敏感性（末期状态差）：{abs(high_noise.filtered[-1]-states.filtered[-1]):.6f}\n"
        f"- 成熟库 Markov MLE 对数似然：{fitted_regimes.log_likelihood:.6f}；概率和最大偏差：{np.max(np.abs(fitted_regimes.filtered_probabilities.sum(axis=1)-1.0)):.3e}\n"
        "- 平滑使用未来信息，只能用于事后解释；交易决策使用在线滤波。\n\n"
        "## 从预测到净收益\n\n"
        f"- 目标仓位：{ledger.target_positions.tolist()}\n"
        f"- 实际仓位：{ledger.filled_positions.tolist()}\n"
        f"- 毛收益：{ledger.gross_return:.6f}\n"
        f"- 换手：{ledger.turnover:.6f}\n"
        f"- 成本：{ledger.cost:.6f}\n"
        f"- 净收益：{ledger.net_return:.6f}\n"
        f"- 单位成本加倍后的净收益下降：{ledger.net_return-high_cost.net_return:.6f}\n"
        f"- 透明递推/SciPy 三角求解最大差：{max(np.max(np.abs(ledger.filled_positions-library_ledger.filled_positions)),abs(ledger.net_return-library_ledger.net_return)):.3e}\n"
        f"- 可执行负例拒绝结果：{negative_checks}\n\n"
        "## 双轨数据与限制\n\n"
        "- 合成数据是唯一正确性 oracle。\n"
        f"- 公共领域宏观快照含 {len(real['rows'])} 个季度，观察截止 {real['observed_through']}。\n"
        "- 宏观快照只验证来源、哈希、时间顺序与外部估计管线，不是可交易统计套利证据。\n"
    )


if __name__ == "__main__":
    print(build_report(), end="")
