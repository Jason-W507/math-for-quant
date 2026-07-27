# %% [markdown]
# # Markov、鞅、Poisson 与 Brownian 过程
#
# 矩阵幂、稳态方程、Poisson 矩、Brownian 协方差与有限空间条件概率
# 均由解析 oracle 给出；模拟只做固定容差的交叉验证。

# %%
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


FIXED_NUMPY_VERSION = "2.5.1"
FIXED_PROCESS_LABELS = ["Markov", "martingale", "Poisson", "Brownian"]
FIXED_ARRAYS = {
    "transition_matrix": [[0.8, 0.2], [0.3, 0.7]],
    "initial_distribution": [1.0, 0.0],
    "brownian_times": [0.25, 0.5, 1.0],
    "doubling_horizons": [4, 8, 16],
    "variation_partitions": [16, 256],
}
FIXED_TOLERANCES = {
    "absolute_tolerance": 1e-12,
    "simulation_tolerance": 1e-10,
    "markov_tolerance": 0.015,
    "poisson_moment_tolerance": 0.12,
    "brownian_covariance_tolerance": 0.02,
    "quadratic_variation_mean_tolerance": 0.015,
    "quadratic_variation_variance_tolerance": 0.001,
}
INTEGER_FIELDS = {
    "seed": 20260726,
    "transition_horizon": 5,
    "simulation_horizon": 50,
    "simulation_paths": 30000,
    "quadratic_variation_paths": 10000,
    "quadratic_variation_partitions": 256,
    "gambler_ruin_upper_state": 4,
    "gambler_ruin_start_state": 2,
}


def reject_nonfinite_numbers(value: object) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SystemExit("oracle numeric inputs must be finite")
        return
    if isinstance(value, list):
        for item in value:
            reject_nonfinite_numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            reject_nonfinite_numbers(item)


def validate_oracle(oracle: dict[str, object]) -> None:
    required = {
        "provenance", "numpy_version", "process_labels", "published_markers",
        *FIXED_ARRAYS, *FIXED_TOLERANCES, *INTEGER_FIELDS,
        "expected_periodic_chain_period", "expected_periodic_even_distribution",
        "expected_periodic_odd_distribution", "expected_reducible_absorption_probability",
        "expected_reducible_absorption_time", "expected_doubling_wealth",
        "expected_doubling_loss_tail", "waiting_time_rate",
        "expected_waiting_time_mean", "expected_waiting_time_variance",
        "superposition_rates", "expected_superposition_rate", "thinning_base_rate",
        "thinning_keep_probability", "expected_thinned_rate", "expected_rejected_rate",
        "nonhomogeneous_interval_end", "expected_integrated_intensity",
        "reflection_level", "reflection_time", "expected_reflection_hitting_probability",
        "expected_brownian_total_variation",
    }
    missing = sorted(required - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")
    reject_nonfinite_numbers(oracle)
    if oracle["numpy_version"] != FIXED_NUMPY_VERSION or np.__version__ != FIXED_NUMPY_VERSION:
        raise SystemExit(f"NumPy version must equal {FIXED_NUMPY_VERSION}")
    if oracle["process_labels"] != FIXED_PROCESS_LABELS:
        raise SystemExit("process labels must match the published design")
    if any(oracle[name] != value for name, value in FIXED_ARRAYS.items()):
        raise SystemExit("canonical process design must not change")
    if any(oracle[name] != value for name, value in FIXED_TOLERANCES.items()):
        raise SystemExit("oracle tolerances must match the published design")
    for name, expected in INTEGER_FIELDS.items():
        value = oracle[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"oracle {name} must be an integer")
        if value != expected:
            raise SystemExit(f"oracle {name} must match the published design")


def main(oracle_path: Path = Path("evidence/ch11/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    validate_oracle(oracle)
    upper_state = int(oracle["gambler_ruin_upper_state"])
    start_state = int(oracle["gambler_ruin_start_state"])
    hitting_probability = start_state / upper_state
    hitting_time = float(start_state * (upper_state - start_state))
    absolute_tolerance = float(oracle["absolute_tolerance"])
    if (
        abs(
            hitting_probability
            - float(oracle["expected_gambler_ruin_probability"])
        )
        > absolute_tolerance
    ):
        raise SystemExit("hitting probability ledger failed")
    if (
        abs(hitting_time - float(oracle["expected_gambler_ruin_time"]))
        > absolute_tolerance
    ):
        raise SystemExit("expected hitting time ledger failed")

    periodic_transition = np.array([[0.0, 1.0], [1.0, 0.0]])
    periodic_initial = np.array([1.0, 0.0])
    periodic_even = periodic_initial @ np.linalg.matrix_power(
        periodic_transition, 2
    )
    periodic_odd = periodic_even @ periodic_transition
    periodic_period = 2
    if (
        periodic_period != oracle["expected_periodic_chain_period"]
        or not np.array_equal(
            periodic_even,
            np.asarray(oracle["expected_periodic_even_distribution"]),
        )
        or not np.array_equal(
            periodic_odd,
            np.asarray(oracle["expected_periodic_odd_distribution"]),
        )
    ):
        raise SystemExit("periodic chain ledger failed")

    reducible_transition = np.array([[1.0, 0.0], [0.5, 0.5]])
    reducible_absorption_probability = 1.0
    reducible_absorption_time = 1.0 / reducible_transition[1, 0]
    if (
        abs(
            reducible_absorption_probability
            - float(oracle["expected_reducible_absorption_probability"])
        )
        > absolute_tolerance
        or abs(
            reducible_absorption_time
            - float(oracle["expected_reducible_absorption_time"])
        )
        > absolute_tolerance
    ):
        raise SystemExit("reducible chain ledger failed")

    doubling_horizons = np.asarray(oracle["doubling_horizons"], dtype=int)
    doubling_wealth = np.array(
        [
            (1.0 - 2.0 ** (-horizon))
            - 2.0 ** (-horizon) * (2.0**horizon - 1.0)
            for horizon in doubling_horizons
        ]
    )
    doubling_loss_tail = np.array(
        [
            2.0 ** (-horizon) * (2.0**horizon - 1.0)
            for horizon in doubling_horizons
        ]
    )
    if (
        not np.allclose(
            doubling_wealth,
            np.asarray(oracle["expected_doubling_wealth"]),
            atol=absolute_tolerance,
            rtol=0.0,
        )
        or not np.allclose(
            doubling_loss_tail,
            np.asarray(oracle["expected_doubling_loss_tail"]),
            atol=absolute_tolerance,
            rtol=0.0,
        )
    ):
        raise SystemExit("doubling optional-stopping ledger failed")
    transition = np.asarray(oracle["transition_matrix"], dtype=float)
    initial = np.asarray(oracle["initial_distribution"], dtype=float)
    if transition.shape != (2, 2) or not np.allclose(transition.sum(axis=1), 1.0):
        raise SystemExit("transition matrix must be two stochastic rows")
    if np.any(transition < 0.0) or np.any(initial < 0.0):
        raise SystemExit("transition probabilities must be nonnegative")

    horizon_distribution = initial @ np.linalg.matrix_power(
        transition, int(oracle["transition_horizon"])
    )
    stationary_system = np.vstack((transition.T - np.eye(2), np.ones(2)))
    stationary_target = np.array([0.0, 0.0, 1.0])
    stationary = np.linalg.lstsq(
        stationary_system, stationary_target, rcond=None
    )[0]

    rng = np.random.default_rng(int(oracle["seed"]))
    paths = int(oracle["simulation_paths"])
    state = np.zeros(paths, dtype=np.int8)
    for _ in range(int(oracle["simulation_horizon"])):
        draws = rng.random(paths)
        state = np.where(
            state == 0,
            draws >= transition[0, 0],
            draws >= transition[1, 0],
        ).astype(np.int8)
    simulated_state_one = float(state.mean())

    poisson_parameter = float(oracle["poisson_rate"]) * float(
        oracle["poisson_time"]
    )
    counts = rng.poisson(poisson_parameter, size=paths)
    poisson_mean = float(counts.mean())
    poisson_variance = float(counts.var(ddof=1))

    waiting_rate = float(oracle["waiting_time_rate"])
    waiting_mean = 1.0 / waiting_rate
    waiting_variance = 1.0 / waiting_rate**2
    superposition_rate = sum(float(x) for x in oracle["superposition_rates"])
    thinning_base_rate = float(oracle["thinning_base_rate"])
    thinning_keep_probability = float(oracle["thinning_keep_probability"])
    thinned_rate = thinning_base_rate * thinning_keep_probability
    rejected_rate = thinning_base_rate * (1.0 - thinning_keep_probability)
    interval_end = float(oracle["nonhomogeneous_interval_end"])
    integrated_intensity = 2.0 * interval_end + 0.5 * interval_end**2
    poisson_transforms = [
        (waiting_mean, "expected_waiting_time_mean"),
        (waiting_variance, "expected_waiting_time_variance"),
        (superposition_rate, "expected_superposition_rate"),
        (thinned_rate, "expected_thinned_rate"),
        (rejected_rate, "expected_rejected_rate"),
        (integrated_intensity, "expected_integrated_intensity"),
    ]
    if any(
        abs(actual - float(oracle[key])) > absolute_tolerance
        for actual, key in poisson_transforms
    ):
        raise SystemExit("Poisson thinning ledger failed")

    times = np.asarray(oracle["brownian_times"], dtype=float)
    if not np.allclose(times, np.array([0.25, 0.5, 1.0])):
        raise SystemExit("Brownian grid must match the analytic covariance oracle")
    increments = rng.normal(scale=0.5, size=(paths, 4))
    brownian = np.cumsum(increments, axis=1)[:, [0, 1, 3]]
    empirical_covariance = np.cov(brownian, rowvar=False, ddof=1)
    theoretical_covariance = np.minimum.outer(times, times)
    covariance_error = float(
        np.max(np.abs(empirical_covariance - theoretical_covariance))
    )

    reflection_level = float(oracle["reflection_level"])
    reflection_time = float(oracle["reflection_time"])
    reflection_probability = math.erfc(
        reflection_level / math.sqrt(2.0 * reflection_time)
    )
    variation_partitions = np.asarray(oracle["variation_partitions"], dtype=int)
    expected_total_variation = np.sqrt(2.0 * variation_partitions / math.pi)
    if (
        abs(
            reflection_probability
            - float(oracle["expected_reflection_hitting_probability"])
        )
        > absolute_tolerance
        or not np.allclose(
            expected_total_variation,
            np.asarray(oracle["expected_brownian_total_variation"]),
            atol=absolute_tolerance,
            rtol=0.0,
        )
    ):
        raise SystemExit("Brownian reflection-principle ledger failed")

    qv_paths = int(oracle["quadratic_variation_paths"])
    partitions = int(oracle["quadratic_variation_partitions"])
    qv_increments = rng.normal(
        scale=math.sqrt(1.0 / partitions), size=(qv_paths, partitions)
    )
    quadratic_variation = np.sum(qv_increments**2, axis=1)
    qv_mean = float(quadratic_variation.mean())
    qv_variance = float(quadratic_variation.var(ddof=1))

    walk_outcomes = np.array(
        [(first, second) for first in (-1.0, 1.0) for second in (-1.0, 1.0)]
    )
    walk_s1 = walk_outcomes[:, 0]
    walk_s2 = walk_outcomes.sum(axis=1)
    martingale_conditional_means = np.array(
        [walk_s2[walk_s1 == state].mean() for state in (-1.0, 1.0)]
    )

    uv_outcomes = np.array(
        [(u, v) for u in (0.0, 1.0) for v in (0.0, 1.0)]
    )
    x0 = uv_outcomes[:, 0]
    x1 = uv_outcomes[:, 1]
    x2 = x0.copy()
    given_x1_zero = x1 == 0.0
    given_x1_zero_x0_one = given_x1_zero & (x0 == 1.0)
    nonmarkov_probabilities = np.array(
        [
            np.mean(x2[given_x1_zero] == 1.0),
            np.mean(x2[given_x1_zero_x0_one] == 1.0),
        ]
    )

    simulation_tolerance = float(oracle["simulation_tolerance"])
    expected_horizon = np.asarray(oracle["expected_horizon_distribution"])
    expected_stationary = np.asarray(oracle["expected_stationary_distribution"])
    checks = [
        np.max(np.abs(horizon_distribution - expected_horizon))
        <= float(oracle["absolute_tolerance"]),
        np.max(np.abs(stationary - expected_stationary))
        <= float(oracle["absolute_tolerance"]),
        abs(simulated_state_one - float(oracle["expected_simulated_state_one"]))
        <= simulation_tolerance,
        abs(simulated_state_one - expected_stationary[1])
        <= float(oracle["markov_tolerance"]),
        abs(poisson_mean - float(oracle["expected_poisson_mean"]))
        <= simulation_tolerance,
        abs(poisson_variance - float(oracle["expected_poisson_variance"]))
        <= simulation_tolerance,
        abs(poisson_mean - poisson_parameter)
        <= float(oracle["poisson_moment_tolerance"]),
        abs(poisson_variance - poisson_parameter)
        <= float(oracle["poisson_moment_tolerance"]),
        abs(
            covariance_error
            - float(oracle["expected_brownian_covariance_error"])
        )
        <= simulation_tolerance,
        covariance_error <= float(oracle["brownian_covariance_tolerance"]),
        abs(qv_mean - float(oracle["expected_quadratic_variation_mean"]))
        <= simulation_tolerance,
        abs(qv_variance - float(oracle["expected_quadratic_variation_variance"]))
        <= simulation_tolerance,
        abs(qv_mean - 1.0)
        <= float(oracle["quadratic_variation_mean_tolerance"]),
        abs(qv_variance - 2.0 / partitions)
        <= float(oracle["quadratic_variation_variance_tolerance"]),
        np.array_equal(
            martingale_conditional_means,
            np.asarray(oracle["expected_martingale_conditional_means"]),
        ),
        np.array_equal(
            nonmarkov_probabilities,
            np.asarray(oracle["expected_nonmarkov_probabilities"]),
        ),
    ]
    if not all(checks):
        raise SystemExit("stochastic-process oracle or declared tolerance failed")

    print(
        "oracle=passed "
        f"markov_p5=({horizon_distribution[0]:.6f},{horizon_distribution[1]:.6f}) "
        f"stationary=({stationary[0]:.6f},{stationary[1]:.6f}) "
        f"simulated_state1={simulated_state_one:.6f} "
        f"poisson=({poisson_mean:.6f},{poisson_variance:.6f}) "
        f"brownian_cov_error={covariance_error:.6f} "
        f"qv=({qv_mean:.6f},{qv_variance:.6f}) "
        f"martingale=({martingale_conditional_means[0]:.1f},"
        f"{martingale_conditional_means[1]:.1f}) "
        f"nonmarkov=({nonmarkov_probabilities[0]:.1f},"
        f"{nonmarkov_probabilities[1]:.1f}) "
        f"hitting=({hitting_probability:.6f},{hitting_time:.6f}) "
        f"periodic={periodic_period} "
        f"doubling_tail=({doubling_loss_tail[0]:.6f},"
        f"{doubling_loss_tail[1]:.6f},{doubling_loss_tail[2]:.6f}) "
        f"poisson_ops=({waiting_mean:.6f},{waiting_variance:.6f},"
        f"{superposition_rate:.6f},{thinned_rate:.6f},"
        f"{rejected_rate:.6f},{integrated_intensity:.6f}) "
        f"reflection={reflection_probability:.6f} "
        f"variation=({expected_total_variation[0]:.6f},"
        f"{expected_total_variation[1]:.6f})"
    )
    return 0


main(
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else Path("evidence/ch11/oracle.json")
)
