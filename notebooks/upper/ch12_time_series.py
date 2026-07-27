# %% [markdown]
# # 时间序列、预测与状态空间递推
#
# AR(1) 的平稳矩和预测误差方差、标量 Kalman 滤波手算值由解析 oracle
# 给出；固定种子模拟只用于交叉验证。独立随机游走和结构变化过程提供负例。

# %%
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


FIXED_NUMPY_VERSION = "2.5.1"
FIXED_LABELS = ["ARMA", "cointegration", "GARCH", "Kalman"]
FIXED_PROVENANCE = (
    "AR(1) stationary moments and forecast-error variance from the geometric "
    "series; scalar Kalman updates hand-calculated as exact fractions; "
    "fixed-seed independent random walks and a declared structural break "
    "provide reproducible negative controls"
)
FIXED_MARKERS = [
    "理论方差为 $2.777778$",
    "三步预测为 $0.512000$，预测误差方差为 $2.049600$",
    "三次滤波均值依次为 $0.555556$、$0.084615$ 与 $0.152494$",
    "水平回归 $R^2=0.344011$",
    "差分回归 $R^2=0.000216$",
    "随机切分 MSE 为 $1.436939$，时间切分 MSE 为 $2.226153$",
    "ARMA 解析方差为 $2.265625$",
    "GARCH 半衰期为 $6.578813$",
    "创新对数似然为 $-4.260729$",
]
FIXED_ARRAYS = {
    "kalman_observations": [1.0, -0.5, 0.25],
    "unit_root_horizons": [1, 10, 100],
}
FIXED_TOLERANCES = {
    "absolute_tolerance": 1e-12,
    "simulation_tolerance": 1e-10,
    "ar1_moment_tolerance": 0.04,
    "spurious_level_r2_minimum": 0.3,
    "difference_r2_maximum": 0.01,
}
FIXED_INTEGERS = {
    "seed": 20260727, "ar1_observations": 50000, "ar1_burn_in": 1000,
    "forecast_horizon": 3, "spurious_length": 400, "break_length": 600,
    "break_point": 360, "spread_seed": 20260728, "spread_length": 600,
    "rolling_window": 120,
}
FIXED_SCALARS = {
    "ar1_phi": 0.8, "innovation_variance": 1.0, "forecast_origin": 1.0,
    "arma_phi": 0.6, "arma_theta": 0.3, "arma_innovation_variance": 1.0,
    "cointegration_slope": 1.5, "spread_ar_phi": 0.4,
    "spread_innovation_variance": 0.36, "garch_omega": 0.1,
    "garch_alpha": 0.1, "garch_beta": 0.8, "kalman_initial_mean": 0.0,
    "kalman_initial_variance": 1.0, "kalman_process_variance": 0.25,
    "kalman_measurement_variance": 1.0, "phi_before_break": 0.85,
    "phi_after_break": -0.25, "training_fraction": 0.6,
}
REQUIRED_FIELDS = {
    "absolute_tolerance", "ar1_burn_in", "ar1_moment_tolerance",
    "ar1_observations", "ar1_phi", "arma_innovation_variance", "arma_phi",
    "arma_theta", "break_length", "break_point", "cointegration_slope",
    "difference_r2_maximum", "expected", "expected_arma_ar_root_modulus",
    "expected_arma_lag_one_autocorrelation", "expected_arma_ma_root_modulus",
    "expected_arma_variance", "expected_cointegration_spread_variance",
    "expected_ecm_adjustment", "expected_garch_half_life",
    "expected_garch_persistence", "expected_garch_unconditional_variance",
    "expected_kalman_innovation_variances", "expected_kalman_innovations",
    "expected_kalman_log_likelihood", "expected_kalman_smoothed_means",
    "expected_unit_root_variances", "forecast_horizon", "forecast_origin",
    "garch_alpha", "garch_beta", "garch_omega", "innovation_variance",
    "kalman_initial_mean", "kalman_initial_variance",
    "kalman_measurement_variance", "kalman_observations",
    "kalman_process_variance", "numpy_version", "phi_after_break",
    "phi_before_break", "provenance", "published_markers", "seed",
    "simulation_tolerance", "spread_ar_phi", "spread_innovation_variance",
    "spurious_length", "spurious_level_r2_minimum", "time_series_labels",
    "training_fraction", "unit_root_horizons", "spread_seed",
    "spread_length", "rolling_window",
}
EXPECTED_FIELDS = {
    "ar1_mean", "ar1_variance", "ar1_autocorrelation", "forecast",
    "forecast_error_variance", "kalman_filtered_means", "kalman_final_gain",
    "levels_r_squared", "differences_r_squared", "random_split_mse",
    "chronological_split_mse", "segment_phi_before_break",
    "segment_phi_after_break", "spread_rolling_phi_minimum",
    "spread_rolling_phi_median", "spread_rolling_phi_maximum",
}


def reject_nonfinite(value: object) -> None:
    if isinstance(value, bool):
        raise SystemExit("oracle numeric inputs must not contain booleans")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SystemExit("oracle numeric inputs must be finite")
    elif isinstance(value, list):
        for item in value:
            reject_nonfinite(item)
    elif isinstance(value, dict):
        for item in value.values():
            reject_nonfinite(item)


def validate_oracle(oracle: dict[str, object]) -> None:
    missing = sorted(REQUIRED_FIELDS - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")
    expected = oracle["expected"]
    if not isinstance(expected, dict):
        raise SystemExit("expected ledger must be an object")
    missing_expected = sorted(EXPECTED_FIELDS - expected.keys())
    if missing_expected:
        raise SystemExit(
            "expected ledger missing fields: " + ", ".join(missing_expected)
        )
    reject_nonfinite(oracle)
    if (
        oracle["numpy_version"] != FIXED_NUMPY_VERSION
        or np.__version__ != FIXED_NUMPY_VERSION
    ):
        raise SystemExit(f"NumPy version must equal {FIXED_NUMPY_VERSION}")
    if oracle["time_series_labels"] != FIXED_LABELS:
        raise SystemExit("time-series labels must match the published design")
    if oracle["provenance"] != FIXED_PROVENANCE:
        raise SystemExit("oracle provenance must match the published design")
    if oracle["published_markers"] != FIXED_MARKERS:
        raise SystemExit("published markers must match the chapter evidence")
    if any(oracle[name] != value for name, value in FIXED_ARRAYS.items()):
        raise SystemExit("canonical array design must not change")
    if any(oracle[name] != value for name, value in FIXED_TOLERANCES.items()):
        raise SystemExit("oracle tolerances must match the published design")
    if any(oracle[name] != value for name, value in FIXED_SCALARS.items()):
        raise SystemExit("canonical scalar design must not change")
    for name, expected in FIXED_INTEGERS.items():
        value = oracle[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"oracle {name} must be an integer")
        if value != expected:
            raise SystemExit(f"oracle {name} must match the published design")


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
    validate_oracle(oracle)
    absolute_tolerance = float(oracle["absolute_tolerance"])
    arma_phi = float(oracle["arma_phi"])
    arma_theta = float(oracle["arma_theta"])
    arma_sigma2 = float(oracle["arma_innovation_variance"])
    arma_ar_root = abs(1.0 / arma_phi)
    arma_ma_root = abs(1.0 / arma_theta)
    arma_variance = arma_sigma2 * (
        1.0 + arma_theta**2 + 2.0 * arma_phi * arma_theta
    ) / (1.0 - arma_phi**2)
    arma_lag_one_covariance = arma_sigma2 * (
        arma_phi + arma_theta
    ) * (1.0 + arma_phi * arma_theta) / (1.0 - arma_phi**2)
    arma_lag_one_autocorrelation = arma_lag_one_covariance / arma_variance
    arma_checks = (
        (arma_ar_root, "expected_arma_ar_root_modulus"),
        (arma_ma_root, "expected_arma_ma_root_modulus"),
        (arma_variance, "expected_arma_variance"),
        (arma_lag_one_autocorrelation, "expected_arma_lag_one_autocorrelation"),
    )
    if any(
        abs(actual - float(oracle[key])) > absolute_tolerance
        for actual, key in arma_checks
    ):
        raise SystemExit("ARMA root and moment ledger failed")

    unit_root_horizons = np.asarray(oracle["unit_root_horizons"], dtype=float)
    unit_root_variances = unit_root_horizons * float(
        oracle["innovation_variance"]
    )
    spread_phi = float(oracle["spread_ar_phi"])
    spread_innovation_variance = float(oracle["spread_innovation_variance"])
    spread_variance = spread_innovation_variance / (1.0 - spread_phi**2)
    ecm_adjustment = spread_phi - 1.0
    if (
        not np.allclose(
            unit_root_variances,
            np.asarray(oracle["expected_unit_root_variances"]),
            atol=absolute_tolerance,
            rtol=0.0,
        )
        or abs(
            spread_variance
            - float(oracle["expected_cointegration_spread_variance"])
        )
        > absolute_tolerance
        or abs(ecm_adjustment - float(oracle["expected_ecm_adjustment"]))
        > absolute_tolerance
    ):
        raise SystemExit("unit-root and cointegration ledger failed")

    garch_omega = float(oracle["garch_omega"])
    garch_alpha = float(oracle["garch_alpha"])
    garch_beta = float(oracle["garch_beta"])
    garch_persistence = garch_alpha + garch_beta
    garch_unconditional_variance = garch_omega / (1.0 - garch_persistence)
    garch_half_life = np.log(0.5) / np.log(garch_persistence)
    if (
        abs(garch_persistence - float(oracle["expected_garch_persistence"]))
        > absolute_tolerance
        or abs(
            garch_unconditional_variance
            - float(oracle["expected_garch_unconditional_variance"])
        )
        > absolute_tolerance
        or abs(garch_half_life - float(oracle["expected_garch_half_life"]))
        > absolute_tolerance
    ):
        raise SystemExit("GARCH persistence ledger failed")
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
    filtered_variances: list[float] = []
    innovations_list: list[float] = []
    innovation_variances: list[float] = []
    gains: list[float] = []
    likelihood_terms: list[float] = []
    kalman_log_likelihood = 0.0
    for observation in oracle["kalman_observations"]:
        predicted_variance = state_variance + process_variance
        innovation = float(observation) - state_mean
        observation_innovation_variance = (
            predicted_variance + measurement_variance
        )
        gain = predicted_variance / (predicted_variance + measurement_variance)
        likelihood_term = -0.5 * (
            math.log(2.0 * math.pi)
            + math.log(observation_innovation_variance)
            + innovation**2 / observation_innovation_variance
        )
        kalman_log_likelihood += likelihood_term
        state_mean = state_mean + gain * innovation
        state_variance = (1.0 - gain) * predicted_variance
        filtered_means.append(state_mean)
        filtered_variances.append(state_variance)
        innovations_list.append(innovation)
        innovation_variances.append(observation_innovation_variance)
        gains.append(gain)
        likelihood_terms.append(likelihood_term)

    smoothed_means = [filtered_means[-1]]
    smoothed_mean = filtered_means[-1]
    for index in range(len(filtered_means) - 2, -1, -1):
        smoother_gain = filtered_variances[index] / (
            filtered_variances[index] + process_variance
        )
        smoothed_mean = filtered_means[index] + smoother_gain * (
            smoothed_mean - filtered_means[index]
        )
        smoothed_means.append(smoothed_mean)
    smoothed_means.reverse()
    if (
        not np.allclose(
            innovations_list,
            oracle["expected_kalman_innovations"],
            atol=absolute_tolerance,
            rtol=0.0,
        )
        or not np.allclose(
            innovation_variances,
            oracle["expected_kalman_innovation_variances"],
            atol=absolute_tolerance,
            rtol=0.0,
        )
        or abs(
            kalman_log_likelihood
            - float(oracle["expected_kalman_log_likelihood"])
        )
        > absolute_tolerance
        or not np.allclose(
            smoothed_means,
            oracle["expected_kalman_smoothed_means"],
            atol=absolute_tolerance,
            rtol=0.0,
        )
    ):
        raise SystemExit("Kalman innovation and smoothing ledger failed")

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
    before_design = np.column_stack(
        (np.ones(break_point - 1), lag[: break_point - 1])
    )
    after_design = np.column_stack(
        (np.ones(target.size - break_point + 1), lag[break_point - 1 :])
    )
    segment_phi_before = float(
        np.linalg.lstsq(
            before_design, target[: break_point - 1], rcond=None
        )[0][1]
    )
    segment_phi_after = float(
        np.linalg.lstsq(
            after_design, target[break_point - 1 :], rcond=None
        )[0][1]
    )

    spread_rng = np.random.default_rng(int(oracle["spread_seed"]))
    spread_length = int(oracle["spread_length"])
    rolling_window = int(oracle["rolling_window"])
    spread_innovations = spread_rng.normal(
        scale=math.sqrt(spread_innovation_variance), size=spread_length
    )
    spread_path = np.zeros(spread_length)
    for index in range(1, spread_length):
        spread_path[index] = (
            spread_phi * spread_path[index - 1] + spread_innovations[index]
        )
    spread_lag = spread_path[:-1]
    spread_target = spread_path[1:]
    spread_rolling_phi: list[float] = []
    for end in range(rolling_window, spread_target.size + 1):
        start = end - rolling_window
        design = np.column_stack(
            (np.ones(rolling_window), spread_lag[start:end])
        )
        coefficient = np.linalg.lstsq(
            design, spread_target[start:end], rcond=None
        )[0]
        spread_rolling_phi.append(float(coefficient[1]))
    spread_rolling_summary = (
        min(spread_rolling_phi),
        float(np.median(spread_rolling_phi)),
        max(spread_rolling_phi),
    )

    expected = oracle["expected"]
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
        abs(segment_phi_before - float(expected["segment_phi_before_break"]))
        <= simulation_tolerance,
        abs(segment_phi_after - float(expected["segment_phi_after_break"]))
        <= simulation_tolerance,
        np.max(
            np.abs(
                np.asarray(spread_rolling_summary)
                - np.asarray(
                    [
                        expected["spread_rolling_phi_minimum"],
                        expected["spread_rolling_phi_median"],
                        expected["spread_rolling_phi_maximum"],
                    ]
                )
            )
        )
        <= simulation_tolerance,
    ]
    if not all(checks):
        failed = [str(index) for index, passed in enumerate(checks) if not passed]
        raise SystemExit(
            "time-series oracle or declared tolerance failed: " + ",".join(failed)
        )

    print(
        "oracle=passed "
        f"ar1=({ar1_mean:.6f},{ar1_variance:.6f},{ar1_autocorrelation:.6f}) "
        f"forecast=({forecast:.6f},{forecast_error_variance:.6f}) "
        f"kalman=({filtered_means[0]:.6f},{filtered_means[1]:.6f},"
        f"{filtered_means[2]:.6f},{gains[-1]:.6f}) "
        f"spurious=({levels_r_squared:.6f},{differences_r_squared:.6f}) "
        f"split_mse=({random_mse:.6f},{chronological_mse:.6f}) "
        f"segment_phi=({segment_phi_before:.6f},{segment_phi_after:.6f}) "
        f"spread_rolling_phi=({spread_rolling_summary[0]:.6f},"
        f"{spread_rolling_summary[1]:.6f},{spread_rolling_summary[2]:.6f}) "
        f"arma=({arma_ar_root:.6f},{arma_ma_root:.6f},"
        f"{arma_variance:.6f},{arma_lag_one_autocorrelation:.6f}) "
        f"unitroot=({unit_root_variances[0]:.1f},"
        f"{unit_root_variances[1]:.1f},{unit_root_variances[2]:.1f}) "
        f"cointegration=({spread_variance:.6f},{ecm_adjustment:.6f}) "
        f"garch=({garch_persistence:.6f},{garch_unconditional_variance:.6f},"
        f"{garch_half_life:.6f}) "
        f"kalman_ll={kalman_log_likelihood:.6f} "
        f"kalman_steps=nu({innovations_list[0]:.6f},{innovations_list[1]:.6f},"
        f"{innovations_list[2]:.6f});S({innovation_variances[0]:.6f},"
        f"{innovation_variances[1]:.6f},{innovation_variances[2]:.6f});"
        f"K({gains[0]:.6f},{gains[1]:.6f},{gains[2]:.6f});"
        f"ll({likelihood_terms[0]:.6f},{likelihood_terms[1]:.6f},"
        f"{likelihood_terms[2]:.6f}) boundary=(filter<=t,smooth<=T) "
        f"smooth=({smoothed_means[0]:.6f},{smoothed_means[1]:.6f},"
        f"{smoothed_means[2]:.6f})"
    )
    return 0


main(
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else Path("evidence/ch12/oracle.json")
)
