# %% [markdown]
# # 时间序列、预测与状态空间递推
#
# AR(1) 的平稳矩和预测误差方差、标量 Kalman 滤波手算值由解析 oracle
# 给出；固定种子模拟只用于交叉验证。独立随机游走和结构变化过程提供负例。

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def regression_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack((np.ones(x.size), x))
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficient
    centered = y - y.mean()
    return float(1.0 - residual @ residual / (centered @ centered))


def regression_mse(
    lag: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> float:
    design = np.column_stack((np.ones(train.size), lag[train]))
    coefficient = np.linalg.lstsq(design, target[train], rcond=None)[0]
    prediction = coefficient[0] + coefficient[1] * lag[test]
    return float(np.mean((target[test] - prediction) ** 2))


def main(oracle_path: Path = Path("evidence/ch12/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(oracle["seed"]))

    phi = float(oracle["ar1_phi"])
    innovation_variance = float(oracle["innovation_variance"])
    observations = int(oracle["ar1_observations"])
    burn_in = int(oracle["ar1_burn_in"])
    innovations = rng.normal(
        scale=np.sqrt(innovation_variance), size=observations + burn_in
    )
    series = np.zeros(observations + burn_in)
    for index in range(1, series.size):
        series[index] = phi * series[index - 1] + innovations[index]
    series = series[burn_in:]
    ar1_mean = float(series.mean())
    ar1_variance = float(series.var(ddof=1))
    ar1_autocorrelation = float(np.corrcoef(series[:-1], series[1:])[0, 1])

    horizon = int(oracle["forecast_horizon"])
    forecast = phi**horizon * float(oracle["forecast_origin"])
    forecast_error_variance = innovation_variance * sum(
        phi ** (2 * step) for step in range(horizon)
    )

    state_mean = float(oracle["kalman_initial_mean"])
    state_variance = float(oracle["kalman_initial_variance"])
    process_variance = float(oracle["kalman_process_variance"])
    measurement_variance = float(oracle["kalman_measurement_variance"])
    filtered_means: list[float] = []
    gains: list[float] = []
    for observation in oracle["kalman_observations"]:
        predicted_variance = state_variance + process_variance
        gain = predicted_variance / (predicted_variance + measurement_variance)
        state_mean = state_mean + gain * (float(observation) - state_mean)
        state_variance = (1.0 - gain) * predicted_variance
        filtered_means.append(state_mean)
        gains.append(gain)

    spurious_length = int(oracle["spurious_length"])
    first_walk = np.cumsum(rng.normal(size=spurious_length))
    second_walk = np.cumsum(rng.normal(size=spurious_length))
    levels_r_squared = regression_r_squared(first_walk, second_walk)
    differences_r_squared = regression_r_squared(
        np.diff(first_walk), np.diff(second_walk)
    )

    break_length = int(oracle["break_length"])
    break_point = int(oracle["break_point"])
    changing = np.zeros(break_length)
    break_innovations = rng.normal(size=break_length)
    for index in range(1, break_length):
        regime_phi = (
            float(oracle["phi_before_break"])
            if index < break_point
            else float(oracle["phi_after_break"])
        )
        changing[index] = regime_phi * changing[index - 1] + break_innovations[index]
    lag = changing[:-1]
    target = changing[1:]
    indices = np.arange(target.size)
    shuffled = rng.permutation(indices)
    cut = int(float(oracle["training_fraction"]) * target.size)
    random_mse = regression_mse(lag, target, shuffled[:cut], shuffled[cut:])
    chronological_mse = regression_mse(
        lag, target, indices[:cut], indices[cut:]
    )

    expected = oracle["expected"]
    absolute_tolerance = float(oracle["absolute_tolerance"])
    simulation_tolerance = float(oracle["simulation_tolerance"])
    analytic_variance = innovation_variance / (1.0 - phi**2)
    checks = [
        abs(ar1_mean - float(expected["ar1_mean"])) <= simulation_tolerance,
        abs(ar1_variance - float(expected["ar1_variance"]))
        <= simulation_tolerance,
        abs(ar1_autocorrelation - float(expected["ar1_autocorrelation"]))
        <= simulation_tolerance,
        abs(ar1_variance - analytic_variance)
        <= float(oracle["ar1_moment_tolerance"]),
        abs(ar1_autocorrelation - phi)
        <= float(oracle["ar1_moment_tolerance"]),
        abs(forecast - float(expected["forecast"])) <= absolute_tolerance,
        abs(forecast_error_variance - float(expected["forecast_error_variance"]))
        <= absolute_tolerance,
        np.max(
            np.abs(
                np.asarray(filtered_means)
                - np.asarray(expected["kalman_filtered_means"])
            )
        )
        <= absolute_tolerance,
        abs(gains[-1] - float(expected["kalman_final_gain"]))
        <= absolute_tolerance,
        abs(levels_r_squared - float(expected["levels_r_squared"]))
        <= simulation_tolerance,
        abs(differences_r_squared - float(expected["differences_r_squared"]))
        <= simulation_tolerance,
        levels_r_squared >= float(oracle["spurious_level_r2_minimum"]),
        differences_r_squared <= float(oracle["difference_r2_maximum"]),
        abs(random_mse - float(expected["random_split_mse"]))
        <= simulation_tolerance,
        abs(chronological_mse - float(expected["chronological_split_mse"]))
        <= simulation_tolerance,
        random_mse < chronological_mse,
    ]
    if not all(checks):
        raise SystemExit("time-series oracle or declared tolerance failed")

    print(
        "oracle=passed "
        f"ar1=({ar1_mean:.6f},{ar1_variance:.6f},{ar1_autocorrelation:.6f}) "
        f"forecast=({forecast:.6f},{forecast_error_variance:.6f}) "
        f"kalman=({filtered_means[0]:.6f},{filtered_means[1]:.6f},"
        f"{filtered_means[2]:.6f},{gains[-1]:.6f}) "
        f"spurious=({levels_r_squared:.6f},{differences_r_squared:.6f}) "
        f"split_mse=({random_mse:.6f},{chronological_mse:.6f})"
    )
    return 0


main()
