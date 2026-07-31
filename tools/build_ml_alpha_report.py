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
from math_for_quant.reporting import stable_gap


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict[str, object]:
    return json.loads((ROOT / "data" / "fixtures" / name).read_text(encoding="utf-8"))


def build_report() -> str:
    model = load("ml-alpha-model.json")
    artifact = train_tiny_mlp(np.asarray(model["features"],dtype=np.float32),np.asarray(model["target"],dtype=np.float32),config=TorchTrainingConfig(int(model["seed"]),int(model["epochs"]),float(model["learning_rate"])))
    if artifact.loss > 1e-3 or len(artifact.checkpoint_sha256) != 64:
        raise RuntimeError("MLP evidence did not satisfy the frozen loss/checkpoint policy")
    order = sequence_order_sensitivity(np.asarray(model["sequence"],dtype=np.float32),seed=int(model["sequence_seed"]))
    if abs(order["mean_pool"]) > 1e-12:
        raise RuntimeError("mean pooling unexpectedly retained sequence order")
    order_aware = ("causal_conv", "rnn", "attention", "transformer")
    if any(not np.isfinite(order[name]) or order[name] <= 0.1 for name in order_aware):
        raise RuntimeError("an order-aware sequence path did not detect permutation")
    classical = cross_check_classical_models(np.asarray(model["classical_features"]),np.asarray(model["classical_target"]),np.asarray(model["classical_evaluation"]),boosting_rounds=int(model["boosting_rounds"]))
    linear_gap = stable_gap(classical.linear_max_gap)
    stump_gap = stable_gap(classical.stump_max_gap)
    boosting_gap = stable_gap(classical.boosting_max_gap)
    text = compare_text_adaptation(train_texts=model["train_texts"],train_labels=np.asarray(model["train_labels"]),inference_texts=model["inference_texts"],seed=int(model["text_seed"]),encoder_id=str(model["encoder_id"]),encoder_version=str(model["encoder_version"]),encoder_license=str(model["encoder_license"]))
    text_parameter_ratio = text.lora_trainable_parameters / text.full_trainable_parameters
    if not (0.0 < text_parameter_ratio < 1.0):
        raise RuntimeError("LoRA parameter ratio must be strictly between zero and one")
    if not np.all(np.isfinite(text.full_finetune_scores)) or not np.all(np.isfinite(text.lora_scores)):
        raise RuntimeError("text adaptation produced non-finite scores")

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
    raw_brier = float(np.mean((raw-evaluation_labels)**2))
    calibrated_brier = float(np.mean((calibrated-evaluation_labels)**2))
    if not calibrated_brier < raw_brier:
        raise RuntimeError("calibration did not improve the frozen evaluation Brier score")
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
