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

import numpy as np


def main(oracle_path: Path = Path("evidence/ch11/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
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
        f"{nonmarkov_probabilities[1]:.1f})"
    )
    return 0


main()
