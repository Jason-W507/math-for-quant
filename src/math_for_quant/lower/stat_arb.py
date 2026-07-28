from __future__ import annotations

from datetime import date
import math
from pathlib import Path
import sys

import numpy as np

from math_for_quant.evidence import load_oracle_bundle


def ar1_coefficient(values: np.ndarray) -> float:
    lagged = values[:-1]
    current = values[1:]
    denominator = float(lagged @ lagged)
    if denominator == 0.0:
        raise ValueError("AR identification requires lag variation")
    return float(lagged @ current / denominator)


def lag_one_coefficient(residuals: np.ndarray) -> float:
    return ar1_coefficient(residuals)


def kalman_update(
    prior_mean: float, prior_variance: float, observation: float, observation_variance: float
) -> tuple[float, float, float]:
    gain = prior_variance / (prior_variance + observation_variance)
    posterior_mean = prior_mean + gain * (observation - prior_mean)
    posterior_variance = (1.0 - gain) * prior_variance
    return gain, posterior_mean, posterior_variance


def detect_change(values: np.ndarray, minimum_segment: int, threshold: float) -> tuple[int, int, int]:
    candidates = range(minimum_segment, values.size - minimum_segment + 1)
    scores = [abs(float(values[:index].mean() - values[index:].mean())) for index in candidates]
    best = list(candidates)[int(np.argmax(scores))]
    alarms = [index for index, score in zip(candidates, scores) if score >= threshold]
    false_alarms = sum(index < best for index in alarms)
    return best, 0, false_alarms


def validate_walk_forward(windows: list[dict[str, str]]) -> None:
    train, validation, trade = windows
    if not (
        date.fromisoformat(train["end"] + "-01")
        < date.fromisoformat(validation["start"] + "-01")
        <= date.fromisoformat(validation["end"] + "-01")
        < date.fromisoformat(trade["start"] + "-01")
    ):
        raise ValueError("walk-forward windows overlap or run backward")


def reject_research_shortcut(kind: str) -> None:
    if kind in {"random-split", "full-sample-standardization", "future-state"}:
        raise ValueError(f"research protocol rejects {kind}")


def main(oracle_path: Path = Path("evidence/lower-ch02/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    tolerance = float(oracle["absolute_tolerance"])

    ar_values = np.asarray(oracle["ar_values"], dtype=float)
    ar_phi = ar1_coefficient(ar_values)
    misspec_acf = lag_one_coefficient(ar_values)
    if misspec_acf < float(oracle["misspecification_threshold"]):
        raise SystemExit("misspecified mean-only model was not rejected")

    kalman = kalman_update(**{name: float(value) for name, value in oracle["kalman"].items()})
    change = detect_change(
        np.asarray(oracle["change_values"], dtype=float),
        int(oracle["minimum_segment"]),
        float(oracle["change_threshold"]),
    )

    x = np.asarray(oracle["cointegration_x"], dtype=float)
    y = np.asarray(oracle["cointegration_y"], dtype=float)
    spread = y - float(oracle["cointegration_beta"]) * x
    spread_mean = float(spread.mean())
    validate_walk_forward(oracle["windows"])

    failures = []
    for shortcut in ("random-split", "full-sample-standardization", "future-state"):
        try:
            reject_research_shortcut(shortcut)
        except ValueError:
            failures.append(1)

    gross = float(sum(float(value) for value in oracle["trade_returns"]))
    net = gross - float(oracle["total_cost"])
    observed = {
        "ar_phi": ar_phi,
        "misspec_acf": misspec_acf,
        "kalman_gain": kalman[0],
        "kalman_mean": kalman[1],
        "kalman_variance": kalman[2],
        "change_index": change[0],
        "detection_delay": change[1],
        "false_alarms": change[2],
        "cointegration_spread": spread_mean,
        "gross": gross,
        "net": net,
        "random_split_rejected": failures[0],
        "full_sample_rejected": failures[1],
        "future_state_rejected": failures[2],
    }
    for name, expected in oracle["expected"].items():
        value = observed[name]
        if isinstance(expected, int):
            if value != expected:
                raise SystemExit(f"{name} ledger failed: observed={value} expected={expected}")
        elif abs(float(value) - float(expected)) > tolerance:
            raise SystemExit(f"{name} ledger failed: observed={value} expected={expected}")

    print(
        "oracle=passed "
        f"ar_phi={ar_phi:.6f} misspec_acf={misspec_acf:.6f} "
        f"kalman=({kalman[0]:.6f},{kalman[1]:.6f},{kalman[2]:.6f}) "
        f"change=({change[0]},{change[1]},{change[2]}) "
        f"cointegration_spread={spread_mean:.6f} walk_forward=passed "
        f"returns=({gross:.6f},{net:.6f}) failures=({failures[0]},{failures[1]},{failures[2]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch02/oracle.json")))
