# %% [markdown]
# # 验证与监控：purge、embargo、嵌套时序、校准与漂移
#
# **研究目标。** 让模型选择、cross-fitting、概率校准、解释稳定性与漂移响应都服从时间边界。
# **假设。** 标签跨度和 embargo 事前冻结；校准集同时包含两类；重要性只描述模型依赖。
# **手算 oracle。** 训练末端 `7+2=9` 严格早于验证起点 `10`；验证末端 `11+2=13`
# 严格早于测试起点 `14`。
# **失败注入。** 将训练扩到索引 8 或把测试提前到 13，分别触发 purge 和 embargo。

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.ml_alpha_validation import PurgedNestedSplit, cross_fitted_ridge_predictions, importance_jaccard, platt_calibrate, validate_model_selection, validate_nested_time_split, validate_preprocessing_cutoff, validate_target_alignment
from math_for_quant.lower.ml_alpha_validation_library import library_cross_fitted_ridge_predictions, maximum_prediction_gap
from math_for_quant.lower.notebook_evidence import assert_expected, expect_value_error, load_oracle_and_fixture


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    split = PurgedNestedSplit(range(min(fixture["train"]), max(fixture["train"])+1), range(min(fixture["validation"]), max(fixture["validation"])+1), range(min(fixture["test"]), max(fixture["test"])+1), int(fixture["label_horizon"]), int(fixture["embargo"]))
    validate_nested_time_split(split)
    folds = [(range(min(item["train"]), max(item["train"])+1), range(min(item["validation"]), max(item["validation"])+1)) for item in fixture["crossfit_folds"]]
    crossfit = cross_fitted_ridge_predictions(np.asarray(fixture["crossfit_features"]), np.asarray(fixture["crossfit_target"]), folds=folds, alpha=float(fixture["ridge_alpha"]))
    library_crossfit = library_cross_fitted_ridge_predictions(np.asarray(fixture["crossfit_features"]), np.asarray(fixture["crossfit_target"]), folds=folds, alpha=float(fixture["ridge_alpha"]))
    fit_scores, fit_labels = np.asarray(fixture["calibration_fit_scores"]), np.asarray(fixture["calibration_fit_labels"])
    evaluation_scores, evaluation_labels = np.asarray(fixture["calibration_evaluation_scores"]), np.asarray(fixture["calibration_evaluation_labels"])
    calibrated = platt_calibrate(fit_scores, fit_labels, evaluation_scores, fit_ends=int(fixture["calibration_fit_ends"]), evaluation_starts=int(fixture["calibration_evaluation_starts"]))
    raw = 1.0 / (1.0 + np.exp(-evaluation_scores))
    raw_brier = float(np.mean((raw-evaluation_labels)**2)); calibrated_brier = float(np.mean((calibrated-evaluation_labels)**2))
    stability = importance_jaccard(np.asarray(fixture["importance_train"]), np.asarray(fixture["importance_test"]), top_k=int(fixture["top_k"]))
    drift = abs(float(np.mean(fixture["monitor_values"])) - float(np.mean(fixture["reference_values"])))
    purge_rejected = expect_value_error(lambda: validate_nested_time_split(PurgedNestedSplit(range(0,9), range(10,12), range(14,16), 2, 2)), "purge gap")
    embargo_rejected = expect_value_error(lambda: validate_nested_time_split(PurgedNestedSplit(range(0,8), range(10,12), range(13,16), 2, 2)), "embargo gap")
    future_preprocessing_rejected = expect_value_error(lambda: validate_preprocessing_cutoff(fitted_through=int(fixture["evaluation_starts"]), evaluation_starts=int(fixture["evaluation_starts"])), "future preprocessing")
    target_misalignment_rejected = expect_value_error(lambda: validate_target_alignment(feature_time=int(fixture["feature_time"]), target_time=int(fixture["target_time"])+1, horizon=int(fixture["prediction_horizon"])), "target misalignment")
    selection_budget_rejected = expect_value_error(lambda: validate_model_selection(attempts=int(fixture["selection_budget"])+1, budget=int(fixture["selection_budget"]), test_reused=False), "selection budget")
    test_reselection_rejected = expect_value_error(lambda: validate_model_selection(attempts=int(fixture["selection_attempts"]), budget=int(fixture["selection_budget"]), test_reused=True), "test reselection")
    plt.figure(figsize=(5,2.5)); plt.plot(sorted(crossfit), [crossfit[key] for key in sorted(crossfit)]); plt.close()
    observed = {"crossfit_count":len(crossfit), "crossfit_library_gap":maximum_prediction_gap(crossfit,library_crossfit), "raw_brier":raw_brier, "calibrated_brier":calibrated_brier, "importance_jaccard":stability, "drift":drift, "drift_triggered":int(drift >= float(fixture["drift_threshold"])), "purge_rejected":purge_rejected, "embargo_rejected":embargo_rejected, "future_preprocessing_rejected":future_preprocessing_rejected, "target_misalignment_rejected":target_misalignment_rejected, "selection_budget_rejected":selection_budget_rejected, "test_reselection_rejected":test_reselection_rejected}
    assert_expected(observed, oracle)
    print("ml-alpha-validation=passed " + " ".join(f"{key}={value:.6f}" for key,value in observed.items()))
    return 0


# %% [markdown]
# **敏感性。** 同时报告校准前后 Brier、Top-k 稳定性与均值漂移；三者不互相替代。
# **限制。** 小样本 Platt scaling 只验证管线；正式研究需嵌套选择校准器并保留可靠性图。

# %%
if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv)>1 else Path("evidence/ml-alpha-validation/oracle.json")))
