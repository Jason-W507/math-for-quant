from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np

from math_for_quant.evidence import load_oracle_bundle


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((target - prediction) ** 2))


def validate_time_split(train: list[int], validation: list[int], inference: list[int]) -> None:
    if not train or not validation or not inference or not (max(train) < min(validation) and max(validation) < min(inference)):
        raise ValueError("random or overlapping split rejected")


def validate_preprocessor(fit_end: str, train_end: str) -> None:
    if date.fromisoformat(fit_end + "-01") > date.fromisoformat(train_end + "-01"):
        raise ValueError("preprocessor used future observations")


def validate_sequence(shape: list[int], mask: list[list[int]], target_offset: int) -> tuple[int, int, int, int]:
    if len(shape) != 3 or shape[0] != len(mask) or any(len(row) != shape[1] for row in mask):
        raise ValueError("sequence tensor and mask dimensions disagree")
    if target_offset <= 0:
        raise ValueError("sequence target must follow the input window")
    return shape[0], shape[1], shape[2], target_offset


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
    target = np.asarray(oracle["target"], dtype=float)
    train_risk = mse(np.asarray(oracle["train_target"], dtype=float), np.asarray(oracle["train_prediction"], dtype=float))
    generalization_risk = mse(target, np.asarray(oracle["baseline_prediction"], dtype=float))
    regularized_objective = train_risk + float(oracle["regularization_lambda"]) * float(oracle["weight_norm_squared"])
    model_losses = [mse(target, np.asarray(values, dtype=float)) for values in oracle["model_predictions"]]

    validate_time_split(**oracle["time_split"])
    validate_preprocessor(**oracle["preprocessor"])
    validate_reproducibility(oracle["seed"])
    validate_model_report(**oracle["model_report"])
    sequence = validate_sequence(oracle["sequence_shape"], oracle["sequence_mask"], int(oracle["target_offset"]))
    sequence_target = np.asarray(oracle["sequence_target"], dtype=float)
    sequence_baseline_loss = mse(sequence_target, np.asarray(oracle["sequence_baseline_prediction"], dtype=float))
    sequence_model_loss = mse(sequence_target, np.asarray(oracle["sequence_model_prediction"], dtype=float))
    drift = abs(float(np.mean(oracle["monitor_values"])) - float(np.mean(oracle["reference_values"])))
    drift_response = int(drift >= float(oracle["drift_threshold"]))
    calibrated = brier(np.asarray(oracle["calibrated_probabilities"], dtype=float), np.asarray(oracle["labels"], dtype=float))
    uncalibrated = brier(np.asarray(oracle["uncalibrated_probabilities"], dtype=float), np.asarray(oracle["labels"], dtype=float))
    explanation = top_feature_jaccard(np.asarray(oracle["importance_train"], dtype=float), np.asarray(oracle["importance_test"], dtype=float), 2)
    explanation_stable = int(explanation >= float(oracle["explanation_stability_threshold"]))

    failures = []
    validators = (
        lambda: validate_time_split(**oracle["bad_split"]),
        lambda: validate_preprocessor(**oracle["bad_preprocessor"]),
        lambda: validate_sequence(oracle["bad_sequence_shape"], oracle["sequence_mask"], 0),
        lambda: validate_reproducibility(oracle["missing_seed"]),
        lambda: validate_model_report(**oracle["bad_model_report"]),
        lambda: validate_explanation_claim(oracle["bad_explanation_claim"]),
    )
    for validator in validators:
        try:
            validator()
        except ValueError:
            failures.append(1)

    gross = sum(float(value) for value in oracle["trade_returns"])
    net = gross - float(oracle["cost"])
    observed = {
        "train_risk": train_risk, "generalization_risk": generalization_risk, "regularized_objective": regularized_objective,
        "baseline_loss": model_losses[0], "tree_loss": model_losses[1], "boosting_loss": model_losses[2],
        "batch": sequence[0], "steps": sequence[1], "features": sequence[2], "target_offset": sequence[3],
        "sequence_baseline_loss": sequence_baseline_loss, "sequence_model_loss": sequence_model_loss,
        "drift": drift, "drift_response": drift_response, "calibrated_brier": calibrated, "uncalibrated_brier": uncalibrated,
        "explanation_jaccard": explanation, "explanation_stable": explanation_stable,
        "gross": gross, "net": net,
        "random_split_rejected": failures[0], "preprocessing_rejected": failures[1], "future_target_rejected": failures[2], "missing_seed_rejected": failures[3],
        "train_only_or_overtuned_rejected": failures[4], "causal_claim_rejected": failures[5],
    }
    for name, expected in oracle["expected"].items():
        value = observed[name]
        if isinstance(expected, int):
            if value != expected: raise SystemExit(f"{name} failed: {value} != {expected}")
        elif abs(float(value) - float(expected)) > tolerance:
            raise SystemExit(f"{name} failed: {value} != {expected}")
    print(
        "oracle=passed "
        f"risks=({train_risk:.6f},{generalization_risk:.6f},{regularized_objective:.6f}) "
        f"models=({model_losses[0]:.6f},{model_losses[1]:.6f},{model_losses[2]:.6f}) "
        f"sequence=({sequence[0]},{sequence[1]},{sequence[2]},{sequence[3]},{sequence_baseline_loss:.6f},{sequence_model_loss:.6f}) "
        f"drift=({drift:.6f},{drift_response}) calibration=({calibrated:.6f},{uncalibrated:.6f}) "
        f"explanation=({explanation:.6f},{explanation_stable}) returns=({gross:.6f},{net:.6f}) "
        f"failures=({failures[0]},{failures[1]},{failures[2]},{failures[3]},{failures[4]},{failures[5]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch03/oracle.json")))
