from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from math_for_quant.lower.ml_alpha_execution_library import library_alpha_ledger
from math_for_quant.lower.ml_alpha_library import cross_check_classical_models
from math_for_quant.lower.ml_alpha_models import TorchTrainingConfig, sequence_order_sensitivity, train_tiny_mlp
from math_for_quant.lower.ml_alpha_research import AlphaLedgerInputs, PortfolioPolicy, build_alpha_ledger, cross_sectional_mse, pairwise_ranking_loss, return_weighted_loss
from math_for_quant.lower.ml_alpha_text import compare_text_adaptation
from math_for_quant.lower.ml_alpha_validation import cross_fitted_ridge_predictions, importance_jaccard, platt_calibrate, validate_model_selection, validate_preprocessing_cutoff, validate_target_alignment
from math_for_quant.lower.ml_alpha_validation_library import library_cross_fitted_ridge_predictions, maximum_prediction_gap
from math_for_quant.lower.notebook_evidence import expect_value_error


ROOT = Path(__file__).resolve().parents[1]


def stable_gap(value: float) -> float:
    return 0.0 if abs(value) < 1e-10 else value


def load(name: str) -> dict[str, object]:
    return json.loads((ROOT / "data" / "fixtures" / name).read_text(encoding="utf-8"))


def build_report() -> str:
    model = load("ml-alpha-model.json")
    artifact = train_tiny_mlp(np.asarray(model["features"],dtype=np.float32),np.asarray(model["target"],dtype=np.float32),config=TorchTrainingConfig(int(model["seed"]),int(model["epochs"]),float(model["learning_rate"])))
    order = sequence_order_sensitivity(np.asarray(model["sequence"],dtype=np.float32),seed=int(model["sequence_seed"]))
    classical = cross_check_classical_models(np.asarray(model["classical_features"]),np.asarray(model["classical_target"]),np.asarray(model["classical_evaluation"]),boosting_rounds=int(model["boosting_rounds"]))
    linear_gap = stable_gap(classical.linear_max_gap)
    stump_gap = stable_gap(classical.stump_max_gap)
    boosting_gap = stable_gap(classical.boosting_max_gap)
    text = compare_text_adaptation(train_texts=model["train_texts"],train_labels=np.asarray(model["train_labels"]),inference_texts=model["inference_texts"],seed=int(model["text_seed"]),encoder_id=str(model["encoder_id"]),encoder_version=str(model["encoder_version"]),encoder_license=str(model["encoder_license"]))

    validation = load("ml-alpha-validation.json")
    folds=[(range(min(item["train"]),max(item["train"])+1),range(min(item["validation"]),max(item["validation"])+1)) for item in validation["crossfit_folds"]]
    x=np.asarray(validation["crossfit_features"]); y=np.asarray(validation["crossfit_target"])
    transparent_crossfit=cross_fitted_ridge_predictions(x,y,folds=folds,alpha=float(validation["ridge_alpha"]))
    library_crossfit=library_cross_fitted_ridge_predictions(x,y,folds=folds,alpha=float(validation["ridge_alpha"]))
    crossfit_gap = stable_gap(
        maximum_prediction_gap(transparent_crossfit, library_crossfit)
    )
    fit_scores=np.asarray(validation["calibration_fit_scores"]); fit_labels=np.asarray(validation["calibration_fit_labels"])
    evaluation_scores=np.asarray(validation["calibration_evaluation_scores"]); evaluation_labels=np.asarray(validation["calibration_evaluation_labels"])
    raw=1/(1+np.exp(-evaluation_scores)); calibrated=platt_calibrate(fit_scores,fit_labels,evaluation_scores,fit_ends=int(validation["calibration_fit_ends"]),evaluation_starts=int(validation["calibration_evaluation_starts"]))
    stability=importance_jaccard(np.asarray(validation["importance_train"]),np.asarray(validation["importance_test"]),top_k=int(validation["top_k"]))
    drift=abs(float(np.mean(validation["monitor_values"]))-float(np.mean(validation["reference_values"])))
    protocol_rejections = [
        expect_value_error(lambda: validate_preprocessing_cutoff(fitted_through=int(validation["evaluation_starts"]), evaluation_starts=int(validation["evaluation_starts"])), "future preprocessing"),
        expect_value_error(lambda: validate_target_alignment(feature_time=int(validation["feature_time"]), target_time=int(validation["target_time"])+1, horizon=int(validation["prediction_horizon"])), "target misalignment"),
        expect_value_error(lambda: validate_model_selection(attempts=int(validation["selection_budget"])+1, budget=int(validation["selection_budget"]), test_reused=False), "selection budget"),
        expect_value_error(lambda: validate_model_selection(attempts=int(validation["selection_attempts"]), budget=int(validation["selection_budget"]), test_reused=True), "test reselection"),
    ]

    research=load("ml-alpha-research.json")
    research_scores=np.asarray(research["scores"]); returns=np.asarray(research["realized_returns"]); fills=np.asarray(research["fill_fractions"])
    policy=PortfolioPolicy(int(research["long_count"]),int(research["short_count"]),float(research["gross_limit"]),float(research["cost_per_unit_turnover"]))
    inputs=AlphaLedgerInputs(research_scores,returns,fills,policy)
    ledger=build_alpha_ledger(inputs=inputs)
    library_ledger=library_alpha_ledger(inputs=inputs)
    execution_gap=stable_gap(max(float(np.max(np.abs(ledger.filled_positions-library_ledger.filled_positions))),abs(ledger.net_return-library_ledger.net_return)))
    return (
        "# 机器学习、深度学习与 NLP/LLM v0.3 冻结研究报告\n\n"
        "## 模型与表示\n\n"
        f"- CPU PyTorch MLP 最终 MSE：{artifact.loss:.6f}；checkpoint：`{artifact.checkpoint_sha256[:12]}`。\n"
        f"- 经典透明/成熟库最大差：linear={linear_gap:.3e}，stump={stump_gap:.3e}，boosting={boosting_gap:.3e}。\n"
        f"- 时间置换差：mean={order['mean_pool']:.6f}，conv={order['causal_conv']:.6f}，RNN={order['rnn']:.6f}，attention={order['attention']:.6f}，transformer={order['transformer']:.6f}。\n"
        f"- 文本编码器：{text.encoder_id}@{text.encoder_version}（{text.encoder_license}）；全量微调/LoRA 均值分数：{np.mean(text.full_finetune_scores):.6f}/{np.mean(text.lora_scores):.6f}；LoRA 可训练参数占比：{text.lora_trainable_parameters/text.full_trainable_parameters:.6f}。\n"
        "- 算力边界：单批次 CPU 固定数据；不从零预训练基础模型，真实编码器需另绑模型卡、许可与 tokenizer。\n\n"
        "## 验证与监控\n\n"
        f"- cross-fitted 行数：{len(transparent_crossfit)}；NumPy/sklearn 最大差：{crossfit_gap:.3e}。\n"
        f"- 后期评价集 Brier：校准前 {np.mean((raw-evaluation_labels)**2):.6f}，校准后 {np.mean((calibrated-evaluation_labels)**2):.6f}。\n"
        f"- Top-2 重要性 Jaccard：{stability:.6f}；均值漂移：{drift:.6f}，触发停止自动上线。\n"
        f"- 未来预处理、目标错位、选择超预算、测试集再选择拒绝结果：{protocol_rejections}；purge 与 embargo 另由嵌套窗口契约拒绝。\n\n"
        "## 从分数到净收益\n\n"
        f"- 三种目标：MSE={cross_sectional_mse(research_scores[0],returns[0]):.6f}，ranking={pairwise_ranking_loss(research_scores[0],returns[0]):.6f}，return-weighted={return_weighted_loss(research_scores[0],returns[0]):.6f}。\n"
        f"- 目标仓位：{ledger.target_positions.tolist()}。\n"
        f"- 实际仓位：{ledger.filled_positions.tolist()}。\n"
        f"- 毛收益 {ledger.gross_return:.6f}；换手 {ledger.turnover:.6f}；成本 {ledger.cost:.6f}；净收益 {ledger.net_return:.6f}。\n"
        f"- 透明递推/SciPy 三角求解最大差：{execution_gap:.3e}。\n"
        "- 文本决策保存首次发布、抓取与修订时间；未来修订由稳定诊断拒绝。\n\n"
        "## 双轨数据与限制\n\n"
        "- 合成数据是正确性 oracle；WDI 公共快照只验证许可、哈希、时间顺序和留出流程。\n"
        "- 冻结研究包不声称模型或文本信号具有市场 alpha；容量、冲击、借券和供应商修订历史需在开放 Capstone 补足。\n"
    )


if __name__ == "__main__":
    print(build_report(),end="")
