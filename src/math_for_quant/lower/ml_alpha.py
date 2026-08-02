from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from math_for_quant.evidence import load_oracle_bundle
from math_for_quant.lower.time_boundaries import validate_chronological_split, validate_fit_cutoff
from math_for_quant.lower.trading_ledger import (
    TradingLedger,
    TurnoverConvention,
    evaluate_trading_ledger,
)


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((target - prediction) ** 2))


def validate_time_split(train: list[int], validation: list[int], inference: list[int]) -> None:
    validate_chronological_split(train, validation, inference)


def validate_preprocessor(fit_end: str, train_end: str) -> None:
    validate_fit_cutoff(fit_end, train_end)


def known_dgp(features: np.ndarray) -> np.ndarray:
    return 0.2 * features + 0.8 * (features > 0.0)


def fit_stump(features: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    if features.shape != target.shape:
        raise ValueError("stump features and target must have the same shape")
    unique = np.unique(features)
    thresholds = (unique[:-1] + unique[1:]) / 2.0
    candidates: list[tuple[float, float, float, float]] = []
    for threshold in thresholds:
        left = target[features <= threshold]
        right = target[features > threshold]
        if left.size and right.size:
            left_value, right_value = float(left.mean()), float(right.mean())
            prediction = np.where(features <= threshold, left_value, right_value)
            candidates.append((mse(target, prediction), float(threshold), left_value, right_value))
    if not candidates:
        raise ValueError("stump fitting requires at least two distinct ordered features")
    _, threshold, left_value, right_value = min(candidates)
    return threshold, left_value, right_value


def prediction_to_return_ledger(
    *,
    scores: np.ndarray,
    realized_returns: np.ndarray,
    threshold: float,
    position_limit: float,
    cost_per_unit_turnover: float,
) -> TradingLedger:
    if scores.ndim != 1 or realized_returns.shape != scores.shape:
        raise ValueError("scores and realized returns must be aligned one-dimensional arrays")
    if position_limit <= 0.0:
        raise ValueError("position limit must be positive")
    positions = np.where(scores >= threshold, position_limit, -position_limit)
    return evaluate_trading_ledger(
        positions=positions,
        realized_returns=realized_returns,
        cost_per_unit_turnover=cost_per_unit_turnover,
        turnover_convention=TurnoverConvention.CROSS_SECTIONAL_OPEN,
    )


def predict_stump(features: np.ndarray, stump: tuple[float, float, float]) -> np.ndarray:
    threshold, left_value, right_value = stump
    return np.where(features <= threshold, left_value, right_value)


def fit_boosting(features: np.ndarray, target: np.ndarray, rounds: int) -> tuple[float, list[tuple[float, float, float]]]:
    base = float(target.mean())
    prediction = np.full_like(target, base, dtype=float)
    stumps: list[tuple[float, float, float]] = []
    for _ in range(rounds):
        stump = fit_stump(features, target - prediction)
        prediction += predict_stump(features, stump)
        stumps.append(stump)
    return base, stumps


def predict_boosting(features: np.ndarray, model: tuple[float, list[tuple[float, float, float]]]) -> np.ndarray:
    base, stumps = model
    prediction = np.full(features.shape, base, dtype=float)
    for stump in stumps:
        prediction += predict_stump(features, stump)
    return prediction


def masked_mean(sequences: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = mask[..., None]
    return (sequences * weights).sum(axis=1) / weights.sum(axis=1)


def fit_linear(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    return np.linalg.lstsq(design, target, rcond=None)[0]


def predict_linear(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(features.shape[0]), features)) @ weights


def fit_random_feature_model(features: np.ndarray, target: np.ndarray, seed: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    projection = generator.normal(size=(features.shape[1], width))
    hidden = np.maximum(features @ projection, 0.0)
    return projection, fit_linear(hidden, target)


def predict_random_feature_model(features: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    projection, weights = model
    return predict_linear(np.maximum(features @ projection, 0.0), weights)


def model_fingerprint(stump: tuple[float, float, float], boosting: tuple[float, list[tuple[float, float, float]]]) -> str:
    payload = json.dumps({"stump": stump, "boosting": boosting}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def sequence_dgp(representation: np.ndarray) -> np.ndarray:
    return representation[:, 0] ** 2 + 0.5 * representation[:, 1]


def last_valid_step(sequences: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = mask.sum(axis=1).astype(int) - 1
    if np.any(indices < 0):
        raise ValueError("every sequence requires at least one observed step")
    return sequences[np.arange(sequences.shape[0]), indices]


def expect_rejection(validator: object, diagnostic: str) -> int:
    try:
        validator()  # type: ignore[operator]
    except ValueError as error:
        if diagnostic not in str(error):
            raise AssertionError(f"wrong diagnostic: {error!s}") from error
        return 1
    raise AssertionError(f"expected rejection containing {diagnostic!r}")


def render_report(observed: dict[str, float | int], fingerprint: str) -> str:
    return f"""# 机器学习 Alpha 可复现研究包

- 数据生成过程：`y = 0.2 * x + 0.8 * 1[x > 0]`，目标由代码生成而非预填预测。
- 训练窗口：索引 0--4；验证窗口：索引 5--6；固定推理窗口：索引 7--8。
- 切分审计：随机切分：拒绝；全样本预处理：拒绝；未来目标对齐：拒绝；张量维度错误：拒绝。
- 模型选择：验证集 MSE 为 baseline={observed['validation_baseline_loss']:.4f}、tree={observed['validation_tree_loss']:.4f}、boosting={observed['validation_boosting_loss']:.4f}；冻结选择 boosting。
- 固定推理：MSE 为 baseline={observed['baseline_loss']:.4f}、tree={observed['tree_loss']:.4f}、boosting={observed['boosting_loss']:.4f}；模型指纹 `{fingerprint}`。
- 序列任务：非深度基线 MSE={observed['sequence_baseline_loss']:.6f}；冻结随机特征序列模型 MSE={observed['sequence_model_loss']:.6f}。
- 模型报告：只报告训练分数：拒绝；超过调参预算：拒绝；未冻结随机种子：拒绝。
- 漂移响应：均值漂移 {observed['drift']:.1f} 超过阈值 0.5，触发停止自动上线并重新校准。
- 校准：Brier 分数由 {observed['uncalibrated_brier']:.3f} 降至 {observed['calibrated_brier']:.3f}。
- 解释：特征重要性不等于因果；Top-2 Jaccard={observed['explanation_jaccard']:.6f}，因果主张由独立门禁拒绝。
- 收益：毛收益 {observed['gross']:.3f}；成本 0.006；成本后收益：{observed['net']:.3f}。
- 复现命令：`uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.allow_error_names=SystemExit notebooks/lower/ch03_ml_alpha.ipynb evidence/lower-ch03/oracle.json`。
"""


def validate_sequence(shape: list[int], mask: list[list[int]], target_offset: int) -> tuple[int, int, int, int]:
    if len(shape) != 3 or shape[0] != len(mask) or any(len(row) != shape[1] for row in mask):
        raise ValueError("sequence tensor and mask dimensions disagree")
    if target_offset <= 0:
        raise ValueError("sequence target must follow the input window")
    return shape[0], shape[1], shape[2], target_offset


def build_sequence_task(
    paths: np.ndarray,
    full_mask: np.ndarray,
    input_steps: int,
    target_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if paths.ndim != 3 or full_mask.shape != paths.shape[:2]:
        raise ValueError("sequence paths and full mask dimensions disagree")
    if target_offset <= 0:
        raise ValueError("sequence target must follow the input window")
    label_index = input_steps - 1 + target_offset
    if input_steps <= 0 or label_index >= paths.shape[1]:
        raise ValueError("future target lies outside the observed timeline")
    if np.any(full_mask[:, label_index] != 1):
        raise ValueError("future target is missing")
    return paths[:, :input_steps], full_mask[:, :input_steps], sequence_dgp(paths[:, label_index, :])


def validate_reproducibility(seed: int | None) -> None:
    if seed is None:
        raise ValueError("stochastic training seed is not frozen")


def validate_model_report(
    train_score: float | None,
    validation_score: float | None,
    attempts: int,
    attempt_limit: int,
) -> None:
    if train_score is None or validation_score is None:
        raise ValueError("train-only model report rejected")
    if attempts > attempt_limit:
        raise ValueError("model-selection budget exceeded")


def validate_explanation_claim(claim: str) -> None:
    if claim == "causal":
        raise ValueError("feature importance does not establish causality")


def brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def top_feature_jaccard(left: np.ndarray, right: np.ndarray, count: int) -> float:
    left_top = set(np.argsort(left)[-count:].tolist())
    right_top = set(np.argsort(right)[-count:].tolist())
    return len(left_top & right_top) / len(left_top | right_top)


def main(oracle_path: Path = Path("evidence/lower-ch03/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    tolerance = float(oracle["absolute_tolerance"])
    all_features = np.asarray(oracle["all_features"], dtype=float)
    split = oracle["time_split"]
    validate_time_split(**split)
    train_features = all_features[np.asarray(split["train"], dtype=int)]
    validation_features = all_features[np.asarray(split["validation"], dtype=int)]
    inference_features = all_features[np.asarray(split["inference"], dtype=int)]
    train_target = known_dgp(train_features)
    validation_target = known_dgp(validation_features)
    target = known_dgp(inference_features)
    baseline_value = float(train_target.mean())
    baseline_prediction = np.full(target.shape, baseline_value)
    train_risk = mse(train_target, np.full(train_target.shape, baseline_value))
    generalization_risk = mse(target, baseline_prediction)
    regularized_objective = train_risk + float(oracle["regularization_lambda"]) * float(oracle["weight_norm_squared"])
    stump = fit_stump(train_features, train_target)
    boosting = fit_boosting(train_features, train_target, int(oracle["boosting_rounds"]))
    model_losses = [
        generalization_risk,
        mse(target, predict_stump(inference_features, stump)),
        mse(target, predict_boosting(inference_features, boosting)),
    ]
    validation_losses = [
        mse(validation_target, np.full(validation_target.shape, baseline_value)),
        mse(validation_target, predict_stump(validation_features, stump)),
        mse(validation_target, predict_boosting(validation_features, boosting)),
    ]
    selected_model = int(np.argmin(validation_losses))
    fingerprint = model_fingerprint(stump, boosting)

    validate_preprocessor(**oracle["preprocessor"])
    validate_reproducibility(oracle["seed"])
    validate_model_report(**oracle["model_report"])
    sequence = validate_sequence(oracle["sequence_shape"], oracle["sequence_mask"], int(oracle["target_offset"]))
    sequence_train, sequence_train_mask, sequence_train_target = build_sequence_task(
        np.asarray(oracle["sequence_train_paths"], dtype=float),
        np.asarray(oracle["sequence_train_full_mask"], dtype=float),
        int(oracle["input_steps"]),
        int(oracle["target_offset"]),
    )
    sequence_inference, sequence_inference_mask, sequence_target = build_sequence_task(
        np.asarray(oracle["sequence_inference_paths"], dtype=float),
        np.asarray(oracle["sequence_inference_full_mask"], dtype=float),
        int(oracle["input_steps"]),
        int(oracle["target_offset"]),
    )
    sequence_baseline = fit_linear(last_valid_step(sequence_train, sequence_train_mask), sequence_train_target)
    sequence_model = fit_random_feature_model(
        masked_mean(sequence_train, sequence_train_mask), sequence_train_target, int(oracle["seed"]), int(oracle["representation_width"])
    )
    sequence_baseline_loss = mse(sequence_target, predict_linear(last_valid_step(sequence_inference, sequence_inference_mask), sequence_baseline))
    sequence_model_loss = mse(sequence_target, predict_random_feature_model(masked_mean(sequence_inference, sequence_inference_mask), sequence_model))
    drift = abs(float(np.mean(oracle["monitor_values"])) - float(np.mean(oracle["reference_values"])))
    drift_response = int(drift >= float(oracle["drift_threshold"]))
    calibrated = brier(np.asarray(oracle["calibrated_probabilities"], dtype=float), np.asarray(oracle["labels"], dtype=float))
    uncalibrated = brier(np.asarray(oracle["uncalibrated_probabilities"], dtype=float), np.asarray(oracle["labels"], dtype=float))
    explanation = top_feature_jaccard(np.asarray(oracle["importance_train"], dtype=float), np.asarray(oracle["importance_test"], dtype=float), 2)
    explanation_stable = int(explanation >= float(oracle["explanation_stability_threshold"]))

    failure_cases = (
        (lambda: validate_time_split(**oracle["bad_split"]), "random or overlapping time split"),
        (lambda: validate_preprocessor(**oracle["bad_preprocessor"]), "after the training window"),
        (lambda: build_sequence_task(np.asarray(oracle["sequence_inference_paths"], dtype=float), np.asarray(oracle["sequence_inference_full_mask"], dtype=float), int(oracle["input_steps"]), 0), "target must follow"),
        (lambda: validate_sequence(oracle["bad_sequence_shape"], oracle["sequence_mask"], int(oracle["target_offset"])), "dimensions disagree"),
        (lambda: validate_reproducibility(oracle["missing_seed"]), "seed is not frozen"),
        (lambda: validate_model_report(**oracle["bad_train_only_report"]), "train-only"),
        (lambda: validate_model_report(**oracle["bad_overtuned_report"]), "budget exceeded"),
        (lambda: validate_explanation_claim(oracle["bad_explanation_claim"]), "does not establish causality"),
    )
    failures = [expect_rejection(validator, diagnostic) for validator, diagnostic in failure_cases]

    return_ledger = prediction_to_return_ledger(
        scores=np.asarray(oracle["calibrated_probabilities"], dtype=float),
        realized_returns=np.asarray(oracle["realized_returns"], dtype=float),
        threshold=float(oracle["position_threshold"]),
        position_limit=float(oracle["position_limit"]),
        cost_per_unit_turnover=float(oracle["cost_per_unit_turnover"]),
    )
    gross, net = return_ledger.gross_return, return_ledger.net_return
    observed = {
        "train_risk": train_risk, "generalization_risk": generalization_risk, "regularized_objective": regularized_objective,
        "baseline_loss": model_losses[0], "tree_loss": model_losses[1], "boosting_loss": model_losses[2],
        "validation_baseline_loss": validation_losses[0], "validation_tree_loss": validation_losses[1],
        "validation_boosting_loss": validation_losses[2], "selected_model": selected_model,
        "batch": sequence[0], "steps": sequence[1], "features": sequence[2], "target_offset": sequence[3],
        "sequence_baseline_loss": sequence_baseline_loss, "sequence_model_loss": sequence_model_loss,
        "drift": drift, "drift_response": drift_response, "calibrated_brier": calibrated, "uncalibrated_brier": uncalibrated,
        "explanation_jaccard": explanation, "explanation_stable": explanation_stable,
        "gross": gross, "net": net,
        "random_split_rejected": failures[0], "preprocessing_rejected": failures[1], "future_target_rejected": failures[2],
        "bad_shape_rejected": failures[3], "missing_seed_rejected": failures[4], "train_only_rejected": failures[5],
        "overtuned_rejected": failures[6], "causal_claim_rejected": failures[7],
    }
    for name, expected in oracle["expected"].items():
        value = observed[name]
        if isinstance(expected, int):
            if value != expected: raise SystemExit(f"{name} failed: {value} != {expected}")
        elif abs(float(value) - float(expected)) > tolerance:
            raise SystemExit(f"{name} failed: {value} != {expected}")
    if fingerprint != oracle["model_fingerprint"]:
        raise SystemExit(f"model_fingerprint failed: {fingerprint} != {oracle['model_fingerprint']}")
    report_path = Path(oracle["report"])
    if report_path.read_text(encoding="utf-8") != render_report(observed, fingerprint):
        raise SystemExit(f"reproducible report drifted: {report_path}")
    print(
        "oracle=passed "
        f"risks=({train_risk:.6f},{generalization_risk:.6f},{regularized_objective:.6f}) "
        f"models=({model_losses[0]:.6f},{model_losses[1]:.6f},{model_losses[2]:.6f},{selected_model}) "
        f"sequence=({sequence[0]},{sequence[1]},{sequence[2]},{sequence[3]},{sequence_baseline_loss:.6f},{sequence_model_loss:.6f}) "
        f"drift=({drift:.6f},{drift_response}) calibration=({calibrated:.6f},{uncalibrated:.6f}) "
        f"explanation=({explanation:.6f},{explanation_stable}) returns=({gross:.6f},{net:.6f}) "
        f"fingerprint={fingerprint} failures=({','.join(str(value) for value in failures)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch03/oracle.json")))
