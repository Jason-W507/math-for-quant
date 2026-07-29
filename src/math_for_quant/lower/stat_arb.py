from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np

from math_for_quant.evidence import load_oracle_bundle


def ar1_coefficient(values: np.ndarray) -> float:
    lagged, current = values[:-1], values[1:]
    denominator = float(lagged @ lagged)
    if denominator == 0.0:
        raise ValueError("AR identification requires lag variation")
    return float(lagged @ current / denominator)


def centered_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left - left.mean(), right - right.mean()
    denominator = float(np.sqrt((a @ a) * (b @ b)))
    if denominator == 0.0:
        raise ValueError("correlation requires nonconstant inputs")
    return float(a @ b / denominator)


def dynamic_oracles(config: dict[str, object]) -> tuple[float, np.ndarray, float]:
    ma_forecast = float(config["ma_theta"]) * float(config["previous_innovation"])
    matrix = np.asarray(config["var_matrix"], dtype=float)
    state = np.asarray(config["var_state"], dtype=float)
    var_forecast = matrix @ state
    garch_variance = (
        float(config["garch_omega"])
        + float(config["garch_alpha"]) * float(config["previous_squared_innovation"])
        + float(config["garch_beta"]) * float(config["previous_variance"])
    )
    return ma_forecast, var_forecast, garch_variance


def kalman_update(prior_mean: float, prior_variance: float, observation: float, observation_variance: float) -> tuple[float, float, float]:
    gain = prior_variance / (prior_variance + observation_variance)
    return gain, prior_mean + gain * (observation - prior_mean), (1.0 - gain) * prior_variance


def detect_alarms(values: np.ndarray, minimum_segment: int, threshold: float) -> list[int]:
    baseline = float(values[:minimum_segment].mean())
    candidates = list(range(minimum_segment, values.size))
    scores = [abs(float(values[index]) - baseline) for index in candidates]
    return [index for index, score in zip(candidates, scores) if score >= threshold]


def evaluate_alarms(alarms: list[int], true_change: int) -> tuple[int, int, int, int, int]:
    first_alarm = alarms[0] if alarms else -1
    false_alarms = sum(i < true_change for i in alarms)
    valid = [i for i in alarms if i >= true_change]
    if not valid:
        return first_alarm, -1, -1, false_alarms, 1
    detected = valid[0]
    return first_alarm, detected, detected - true_change, false_alarms, 0


def validate_walk_forward(windows: list[dict[str, str]]) -> None:
    if [window["name"] for window in windows] != ["train", "validation", "trade"]:
        raise ValueError("walk-forward roles must be train, validation, trade")
    parsed = []
    for window in windows:
        start = date.fromisoformat(window["start"] + "-01")
        end = date.fromisoformat(window["end"] + "-01")
        if start > end:
            raise ValueError("walk-forward interval runs backward")
        parsed.append((start, end))
    if not (parsed[0][1] < parsed[1][0] and parsed[1][1] < parsed[2][0]):
        raise ValueError("walk-forward windows overlap or run backward")


def validate_split(train: list[int], validation: list[int], trade: list[int]) -> None:
    if not train or not validation or not trade or not (max(train) < min(validation) and max(validation) < min(trade)):
        raise ValueError("random or overlapping time split rejected")


def validate_scaler(fit_end: str, train_end: str) -> None:
    if date.fromisoformat(fit_end + "-01") > date.fromisoformat(train_end + "-01"):
        raise ValueError("scaler used observations after the training window")


def validate_online_state(observed_through: str, decision_date: str) -> None:
    if date.fromisoformat(observed_through) > date.fromisoformat(decision_date):
        raise ValueError("state estimate uses future observations")


def rolling_ledger(windows: list[dict[str, object]]) -> tuple[list[float], list[float], float, float]:
    coefficients, errors, gross, cost = [], [], 0.0, 0.0
    for window in windows:
        train = np.asarray(window["train_values"], dtype=float)
        coefficient = ar1_coefficient(train)
        forecast = coefficient * float(train[-1])
        coefficients.append(coefficient)
        errors.append(float(window["validation_value"]) - forecast)
        gross += float(window["trade_return"])
        cost += float(window["cost"])
    return coefficients, errors, gross, gross - cost


def main(oracle_path: Path = Path("evidence/lower-ch02/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    tolerance = float(oracle["absolute_tolerance"])
    ar_values = np.asarray(oracle["ar_values"], dtype=float)
    ar_phi = ar1_coefficient(ar_values)
    misspec_acf = centered_correlation(ar_values[:-1], ar_values[1:])
    if misspec_acf < float(oracle["misspecification_threshold"]):
        raise SystemExit("misspecified mean-only model was not rejected")
    ma_forecast, var_forecast, garch_variance = dynamic_oracles(oracle["dynamic_models"])
    kalman = kalman_update(**{name: float(value) for name, value in oracle["kalman"].items()})

    change_values = np.asarray(oracle["change_values"], dtype=float)
    true_change = int(oracle["true_change"])
    minimum_segment = int(oracle["minimum_segment"])
    change = evaluate_alarms(detect_alarms(change_values, minimum_segment, float(oracle["change_threshold"])), true_change)
    low = evaluate_alarms(detect_alarms(change_values, minimum_segment, float(oracle["sensitivity_thresholds"][0])), true_change)
    high = evaluate_alarms(detect_alarms(change_values, minimum_segment, float(oracle["sensitivity_thresholds"][1])), true_change)

    x = np.asarray(oracle["cointegration_x"], dtype=float)
    y = np.asarray(oracle["cointegration_y"], dtype=float)
    spread = y - float(oracle["cointegration_beta"]) * x
    spread_ar = centered_correlation(spread[:-1], spread[1:])
    pseudo = np.asarray(oracle["pseudo_spread"], dtype=float)
    pseudo_rejected = int(abs(float(pseudo[:3].mean() - pseudo[3:].mean())) > float(oracle["spread_shift_limit"]))

    validate_walk_forward(oracle["windows"])
    coefficients, errors, gross, net = rolling_ledger(oracle["rolling_windows"])
    failures = []
    validators = (
        lambda: validate_split(**oracle["bad_split"]),
        lambda: validate_scaler(**oracle["bad_scaler"]),
        lambda: validate_online_state(**oracle["bad_state"]),
    )
    for validator in validators:
        try:
            validator()
        except ValueError:
            failures.append(1)

    observed = {
        "ar_phi": ar_phi, "misspec_acf": misspec_acf,
        "ma_forecast": ma_forecast, "var_first": float(var_forecast[0]), "var_second": float(var_forecast[1]), "garch_variance": garch_variance,
        "kalman_gain": kalman[0], "kalman_mean": kalman[1], "kalman_variance": kalman[2],
        "change_index": change[1], "detection_delay": change[2], "false_alarms": change[3], "missed": change[4],
        "low_false_alarms": low[3], "high_missed": high[4],
        "spread_ar": spread_ar, "pseudo_rejected": pseudo_rejected,
        "rolling_windows": len(coefficients), "rolling_error_max": max(abs(value) for value in errors),
        "gross": gross, "net": net,
        "random_split_rejected": failures[0], "full_sample_rejected": failures[1], "future_state_rejected": failures[2],
    }
    for name, expected in oracle["expected"].items():
        value = observed[name]
        if isinstance(expected, int):
            if value != expected: raise SystemExit(f"{name} ledger failed: observed={value} expected={expected}")
        elif abs(float(value) - float(expected)) > tolerance:
            raise SystemExit(f"{name} ledger failed: observed={value} expected={expected}")

    print(
        "oracle=passed "
        f"ar_phi={ar_phi:.6f} misspec_acf={misspec_acf:.6f} "
        f"dynamic=({ma_forecast:.6f},{var_forecast[0]:.6f},{var_forecast[1]:.6f},{garch_variance:.6f}) "
        f"kalman=({kalman[0]:.6f},{kalman[1]:.6f},{kalman[2]:.6f}) "
        f"change=({change[1]},{change[2]},{change[3]},{change[4]}) sensitivity=({low[0]},{low[3]},{high[4]}) "
        f"cointegration=({spread_ar:.6f},{pseudo_rejected}) rolling=({len(coefficients)},{max(abs(value) for value in errors):.6f}) "
        f"returns=({gross:.6f},{net:.6f}) failures=({failures[0]},{failures[1]},{failures[2]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch02/oracle.json")))
