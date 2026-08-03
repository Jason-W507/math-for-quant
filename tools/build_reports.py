"""Build the frozen lower-volume route reports.

The report builders share one command-line entry point because they all do the
same thing: run a route's transparent and library implementations against the
frozen fixtures and print the resulting Markdown report.  The route code stays
in ``src/m4q``; this file only selects the requested route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from m4q.lower.derivatives_route import (
    render_route_report as render_derivatives_report,
    run_hedging,
    run_numerics,
    run_stochastic,
)
from m4q.lower.microstructure_route import build_route_report as build_microstructure_route_report
from m4q.lower.ml_alpha_execution_library import library_alpha_ledger
from m4q.lower.ml_alpha_library import cross_check_classical_models
from m4q.lower.ml_alpha_models import TorchTrainingConfig, sequence_order_sensitivity, train_tiny_mlp
from m4q.lower.ml_alpha_research import (
    AlphaLedgerInputs,
    PortfolioPolicy as AlphaPortfolioPolicy,
    build_alpha_ledger,
    cross_sectional_mse,
    pairwise_ranking_loss,
    return_weighted_loss,
)
from m4q.lower.ml_alpha_text import compare_text_adaptation
from m4q.lower.ml_alpha_validation import (
    cross_fitted_ridge_predictions,
    importance_jaccard,
    platt_calibrate,
    validate_model_selection,
    validate_preprocessing_cutoff,
    validate_target_alignment,
)
from m4q.lower.ml_alpha_validation_library import (
    library_cross_fitted_ridge_predictions,
    maximum_prediction_gap,
)
from m4q.lower.multifactor import bh_rejections, null_search_p_values
from m4q.lower.multifactor_estimation import (
    classic_two_pass_fama_macbeth,
    lasso_coordinate_descent,
    predictive_fama_macbeth,
    ridge_closed_form,
)
from m4q.lower.multifactor_library import (
    cross_check_estimators,
    cross_check_route_statistics,
)
from m4q.lower.multifactor_research import (
    PortfolioInputs,
    PortfolioPolicy,
    Weighting,
    build_group_portfolio_ledger,
    load_real_cross_section,
)
from m4q.lower.notebook_evidence import expect_value_error
from m4q.lower.portfolio_real_data import run_portfolio_real_data
from m4q.lower.portfolio_route import (
    render_route_report as render_portfolio_report,
    run_estimation,
    run_optimization,
    run_tail,
)
from m4q.lower.stat_arb import validate_scaler, validate_walk_forward
from m4q.lower.stat_arb_estimation import (
    ScalarStateSpaceSpec,
    kalman_filter_and_smooth,
    regime_filter,
)
from m4q.lower.stat_arb_execution_library import library_forecast_ledger
from m4q.lower.stat_arb_library import (
    cross_check_long_run,
    fit_markov_switching,
    library_kalman_filter_and_smooth,
)
from m4q.lower.stat_arb_models import engle_granger, fit_ecm, ou_diagnostics
from m4q.lower.stat_arb_research import (
    ExecutionPolicy,
    build_forecast_ledger,
    validate_failure_state,
    validate_purged_walk_forward,
)
from m4q.reporting import stable_gap


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, object]:
    return json.loads((ROOT / "data" / "fixtures" / name).read_text(encoding="utf-8"))


def build_derivatives() -> str:
    return render_derivatives_report(
        run_stochastic(_load("derivatives-stochastic.json")),
        run_numerics(_load("derivatives-numerics.json")),
        run_hedging(_load("derivatives-hedging.json")),
    )


def build_microstructure() -> str:
    return build_microstructure_route_report()


def build_multifactor() -> str:
    model = _load("multifactor-model.json")
    signals = np.asarray(model["signals"], dtype=float)
    future = np.asarray(model["future_returns"], dtype=float)
    factors = np.asarray(model["factor_returns"], dtype=float)
    betas = np.asarray(model["betas"], dtype=float)
    assets = float(model["asset_alpha"]) + factors @ betas[None, :]
    predictive = predictive_fama_macbeth(signals, future)
    classic = classic_two_pass_fama_macbeth(assets, factors)
    estimation = _load("multifactor-estimation.json")
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
        float(np.max(np.abs(library_estimators.classic_risk_prices - classic.risk_prices))),
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
    route_gap = stable_gap(max(
        estimator_gap,
        route_check.panel_slope_gap,
        route_check.neutralization_gap,
        route_check.ic_gap,
        route_check.rank_ic_gap,
        route_check.decay_gap,
        float(route_check.bh_count_gap),
    ))

    research = _load("multifactor-research.json")
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


def _rejects(callable_) -> int:
    try:
        callable_()
    except ValueError:
        return 1
    return 0


def build_stat_arb() -> str:
    model = _load("stat-arb-model.json")
    count = int(model["observation_count"])
    x = np.cumsum(np.resize(np.asarray(model["x_increments"], dtype=float), count))
    residual = float(model["residual_phi"]) ** np.arange(count)
    y = float(model["cointegration_intercept"]) + float(model["cointegration_slope"]) * x + residual
    relation = engle_granger(y, x)
    ecm = fit_ecm(y, x, relation)
    ou = ou_diagnostics(relation.residuals, step=float(model["step"]))
    real = json.loads(
        (ROOT / "data" / "real" / "stat-arb-us-macro-1999q4-2009q3.json").read_text(
            encoding="utf-8"
        )
    )
    real_y = np.log(np.asarray([row["realgdp"] for row in real["rows"]], dtype=float))
    real_x = np.log(np.asarray([row["realcons"] for row in real["rows"]], dtype=float))
    library = cross_check_long_run(real_y, real_x)

    estimation = _load("stat-arb-estimation.json")
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
    slope_gap = stable_gap(abs(library.transparent_slope - library.library_slope))
    kalman_gap = stable_gap(max(
        np.max(np.abs(states.filtered - library_states.filtered)),
        np.max(np.abs(states.smoothed - library_states.smoothed)),
    ))
    probability_gap = stable_gap(float(np.max(np.abs(fitted_regimes.filtered_probabilities.sum(axis=1) - 1.0))))

    research = _load("stat-arb-research.json")
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
    execution_gap = stable_gap(max(
        np.max(np.abs(ledger.filled_positions - library_ledger.filled_positions)),
        abs(ledger.net_return - library_ledger.net_return),
    ))
    negative_checks = {
        "purge": _rejects(lambda: validate_purged_walk_forward(train_indices=range(0, 7), validation_indices=range(8, 10), trade_indices=range(12, 14), label_horizon=2, embargo=2)),
        "embargo": _rejects(lambda: validate_purged_walk_forward(train_indices=range(0, 6), validation_indices=range(8, 10), trade_indices=range(11, 14), label_horizon=2, embargo=2)),
        "walk-forward": _rejects(lambda: validate_walk_forward([
            {"name": "train", "start": "2020-01", "end": "2020-06"},
            {"name": "validation", "start": "2020-06", "end": "2020-08"},
            {"name": "trade", "start": "2020-09", "end": "2020-12"},
        ])),
        "scaler": _rejects(lambda: validate_scaler("2020-08-01", "2020-06-01")),
        "failure-state": _rejects(lambda: validate_failure_state(half_life=9.0, maximum_half_life=4.0, fill_rate=0.9, minimum_fill_rate=0.5)),
    }
    return (
        "# 时间序列与统计套利 v0.3 冻结研究报告\n\n"
        "## 长期关系与动态\n\n"
        f"- Engle--Granger 长期斜率：{relation.slope:.6f}\n"
        f"- 残差 DF 统计量：{relation.residual_adf_statistic:.6f}\n"
        f"- ECM 调整速度：{ecm.adjustment_speed:.6f}\n"
        f"- OU 半衰期：{ou.half_life:.6f}\n"
        f"- OU 从冻结起点命中均值的期望时间：{ou.expected_first_passage:.6f}\n"
        f"- 公共快照透明/成熟库斜率差：{slope_gap:.3e}\n"
        f"- statsmodels Engle--Granger p 值：{library.engle_granger_p_value:.6f}；Johansen rank：{library.johansen_rank}\n\n"
        "## 状态推断\n\n"
        f"- 在线滤波末值：{states.filtered[-1]:.6f}\n"
        f"- 第二期平滑值：{states.smoothed[1]:.6f}\n"
        f"- 给定参数状态 1 末概率：{regimes[-1,1]:.6f}\n"
        f"- 透明/成熟库 Kalman 最大差：{kalman_gap:.3e}\n"
        f"- 过程噪声敏感性（末期状态差）：{abs(high_noise.filtered[-1]-states.filtered[-1]):.6f}\n"
        f"- 成熟库 Markov MLE 对数似然：{fitted_regimes.log_likelihood:.6f}；概率和最大偏差：{probability_gap:.3e}\n"
        "- 平滑使用未来信息，只能用于事后解释；交易决策使用在线滤波。\n\n"
        "## 从预测到净收益\n\n"
        f"- 目标仓位：{ledger.target_positions.tolist()}\n"
        f"- 实际仓位：{ledger.filled_positions.tolist()}\n"
        f"- 毛收益：{ledger.gross_return:.6f}\n"
        f"- 换手：{ledger.turnover:.6f}\n"
        f"- 成本：{ledger.cost:.6f}\n"
        f"- 净收益：{ledger.net_return:.6f}\n"
        f"- 单位成本加倍后的净收益下降：{ledger.net_return-high_cost.net_return:.6f}\n"
        f"- 透明递推/SciPy 三角求解最大差：{execution_gap:.3e}\n"
        f"- 可执行负例拒绝结果：{negative_checks}\n\n"
        "## 双轨数据与限制\n\n"
        "- 合成数据是唯一正确性 oracle。\n"
        f"- 公共领域宏观快照含 {len(real['rows'])} 个季度，观察截止 {real['observed_through']}。\n"
        "- 宏观快照只验证来源、哈希、时间顺序与外部估计管线，不是可交易统计套利证据。\n"
    )


def build_ml_alpha() -> str:
    model = _load("ml-alpha-model.json")
    artifact = train_tiny_mlp(
        np.asarray(model["features"], dtype=np.float32),
        np.asarray(model["target"], dtype=np.float32),
        config=TorchTrainingConfig(
            int(model["seed"]), int(model["epochs"]), float(model["learning_rate"])
        ),
    )
    if artifact.loss > 1e-3 or len(artifact.checkpoint_sha256) != 64:
        raise RuntimeError("MLP evidence did not satisfy the frozen loss/checkpoint policy")
    order = sequence_order_sensitivity(np.asarray(model["sequence"], dtype=np.float32), seed=int(model["sequence_seed"]))
    if abs(order["mean_pool"]) > 1e-12:
        raise RuntimeError("mean pooling unexpectedly retained sequence order")
    if any(not np.isfinite(order[name]) or order[name] <= 0.1 for name in ("causal_conv", "rnn", "attention", "transformer")):
        raise RuntimeError("an order-aware sequence path did not detect permutation")
    classical = cross_check_classical_models(
        np.asarray(model["classical_features"]),
        np.asarray(model["classical_target"]),
        np.asarray(model["classical_evaluation"]),
        boosting_rounds=int(model["boosting_rounds"]),
    )
    linear_gap = stable_gap(classical.linear_max_gap)
    stump_gap = stable_gap(classical.stump_max_gap)
    boosting_gap = stable_gap(classical.boosting_max_gap)
    text = compare_text_adaptation(
        train_texts=model["train_texts"],
        train_labels=np.asarray(model["train_labels"]),
        inference_texts=model["inference_texts"],
        seed=int(model["text_seed"]),
        encoder_id=str(model["encoder_id"]),
        encoder_version=str(model["encoder_version"]),
        encoder_license=str(model["encoder_license"]),
    )
    text_parameter_ratio = text.lora_trainable_parameters / text.full_trainable_parameters
    if not 0.0 < text_parameter_ratio < 1.0:
        raise RuntimeError("LoRA parameter ratio must be strictly between zero and one")
    if not np.all(np.isfinite(text.full_finetune_scores)) or not np.all(np.isfinite(text.lora_scores)):
        raise RuntimeError("text adaptation produced non-finite scores")

    validation = _load("ml-alpha-validation.json")
    folds = [
        (range(min(item["train"]), max(item["train"]) + 1), range(min(item["validation"]), max(item["validation"]) + 1))
        for item in validation["crossfit_folds"]
    ]
    x = np.asarray(validation["crossfit_features"])
    y = np.asarray(validation["crossfit_target"])
    transparent_crossfit = cross_fitted_ridge_predictions(x, y, folds=folds, alpha=float(validation["ridge_alpha"]))
    library_crossfit = library_cross_fitted_ridge_predictions(x, y, folds=folds, alpha=float(validation["ridge_alpha"]))
    crossfit_gap = stable_gap(maximum_prediction_gap(transparent_crossfit, library_crossfit))
    fit_scores = np.asarray(validation["calibration_fit_scores"])
    fit_labels = np.asarray(validation["calibration_fit_labels"])
    evaluation_scores = np.asarray(validation["calibration_evaluation_scores"])
    evaluation_labels = np.asarray(validation["calibration_evaluation_labels"])
    raw = 1 / (1 + np.exp(-evaluation_scores))
    calibrated = platt_calibrate(
        fit_scores, fit_labels, evaluation_scores,
        fit_ends=int(validation["calibration_fit_ends"]),
        evaluation_starts=int(validation["calibration_evaluation_starts"]),
    )
    raw_brier = float(np.mean((raw - evaluation_labels) ** 2))
    calibrated_brier = float(np.mean((calibrated - evaluation_labels) ** 2))
    if not calibrated_brier < raw_brier:
        raise RuntimeError("calibration did not improve the frozen evaluation Brier score")
    stability = importance_jaccard(np.asarray(validation["importance_train"]), np.asarray(validation["importance_test"]), top_k=int(validation["top_k"]))
    drift = abs(float(np.mean(validation["monitor_values"])) - float(np.mean(validation["reference_values"])))
    protocol_rejections = [
        expect_value_error(lambda: validate_preprocessing_cutoff(fitted_through=int(validation["evaluation_starts"]), evaluation_starts=int(validation["evaluation_starts"])), "future preprocessing"),
        expect_value_error(lambda: validate_target_alignment(feature_time=int(validation["feature_time"]), target_time=int(validation["target_time"]) + 1, horizon=int(validation["prediction_horizon"])), "target misalignment"),
        expect_value_error(lambda: validate_model_selection(attempts=int(validation["selection_budget"]) + 1, budget=int(validation["selection_budget"]), test_reused=False), "selection budget"),
        expect_value_error(lambda: validate_model_selection(attempts=int(validation["selection_attempts"]), budget=int(validation["selection_budget"]), test_reused=True), "test reselection"),
    ]

    research = _load("ml-alpha-research.json")
    research_scores = np.asarray(research["scores"])
    returns = np.asarray(research["realized_returns"])
    fills = np.asarray(research["fill_fractions"])
    policy = AlphaPortfolioPolicy(
        int(research["long_count"]), int(research["short_count"]),
        float(research["gross_limit"]), float(research["cost_per_unit_turnover"]),
    )
    inputs = AlphaLedgerInputs(research_scores, returns, fills, policy)
    ledger = build_alpha_ledger(inputs=inputs)
    library_ledger = library_alpha_ledger(inputs=inputs)
    execution_gap = stable_gap(max(
        float(np.max(np.abs(ledger.filled_positions - library_ledger.filled_positions))),
        abs(ledger.net_return - library_ledger.net_return),
    ))
    return (
        "# 机器学习、深度学习与 NLP/LLM v0.3 冻结研究报告\n\n"
        "## 模型与表示\n\n"
        "- CPU PyTorch MLP 最终 MSE 不超过 0.001000；checkpoint 完整性已验证。\n"
        f"- 经典透明/成熟库最大差：linear={linear_gap:.3e}，stump={stump_gap:.3e}，boosting={boosting_gap:.3e}。\n"
        "- 时间置换门禁：均值池化对顺序不敏感；卷积、RNN、attention 与 transformer 四条顺序感知路径均检测到重排。\n"
        f"- 文本编码器：{text.encoder_id}@{text.encoder_version}（{text.encoder_license}）；全量微调与 LoRA 均产生有限分数，且 LoRA 只训练严格更少的参数。\n"
        "- 算力边界：单批次 CPU 固定数据；不从零预训练基础模型，真实编码器需另绑模型卡、许可与 tokenizer。\n\n"
        "## 验证与监控\n\n"
        f"- cross-fitted 行数：{len(transparent_crossfit)}；NumPy/sklearn 最大差：{crossfit_gap:.3e}。\n"
        "- 后期评价集门禁：冻结校准器严格改善 Brier 分数。\n"
        f"- Top-2 重要性重叠低于 0.5：{stability < 0.5}；均值漂移超过 0.5：{drift > 0.5}，触发停止自动上线。\n"
        f"- 未来预处理、目标错位、选择超预算、测试集再选择拒绝结果：{protocol_rejections}；purge 与 embargo 另由嵌套窗口契约拒绝。\n\n"
        "## 从分数到净收益\n\n"
        f"- 三种目标：MSE={cross_sectional_mse(research_scores[0], returns[0]):.6f}，ranking={pairwise_ranking_loss(research_scores[0], returns[0]):.6f}，return-weighted={return_weighted_loss(research_scores[0], returns[0]):.6f}。\n"
        f"- 目标仓位：{ledger.target_positions.tolist()}。\n"
        f"- 实际仓位：{ledger.filled_positions.tolist()}。\n"
        f"- 毛收益 {ledger.gross_return:.6f}；换手 {ledger.turnover:.6f}；成本 {ledger.cost:.6f}；净收益 {ledger.net_return:.6f}。\n"
        f"- 透明递推/SciPy 三角求解最大差：{execution_gap:.3e}。\n"
        "- 文本决策保存首次发布、抓取与修订时间；未来修订由稳定诊断拒绝。\n\n"
        "## 双轨数据与限制\n\n"
        "- 合成数据是正确性 oracle；WDI 公共快照只验证许可、哈希、时间顺序和留出流程。\n"
        "- 冻结研究包不声称模型或文本信号具有市场 alpha；容量、冲击、借券和供应商修订历史需在开放 Capstone 补足。\n"
    )


def build_portfolio_risk() -> str:
    return render_portfolio_report(
        run_estimation(_load("portfolio-risk-estimation.json")),
        run_optimization(_load("portfolio-risk-optimization.json")),
        run_tail(_load("portfolio-risk-tail.json")),
        run_portfolio_real_data(ROOT / "data/real/stat-arb-us-macro-1999q4-2009q3.json"),
    )


ROUTES = {
    "multifactor": build_multifactor,
    "stat-arb": build_stat_arb,
    "ml-alpha": build_ml_alpha,
    "derivatives": build_derivatives,
    "portfolio-risk": build_portfolio_risk,
    "microstructure": build_microstructure,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", choices=sorted(ROUTES), help="route report to print")
    args = parser.parse_args()
    print(ROUTES[args.route](), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
