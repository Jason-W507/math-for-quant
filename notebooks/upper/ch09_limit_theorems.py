# %% [markdown]
# # LLN、CLT、集中界与重尾失效
#
# 理论均值、标准误、精确二项尾概率和 Hoeffding 上界均独立于模拟给出。

# %%
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def main(oracle_path: Path = Path("evidence/ch09/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    p = float(oracle["bernoulli_p"])
    n = int(oracle["sample_size"])
    repetitions = int(oracle["repetitions"])
    deviation = float(oracle["tail_deviation"])
    if not 0.0 < p < 1.0 or n <= 0 or repetitions <= 0 or deviation <= 0.0:
        raise SystemExit("simulation design parameters are invalid")

    theory_sd = math.sqrt(p * (1.0 - p) / n)
    exact_tail = sum(
        math.comb(n, k) * p**k * (1.0 - p) ** (n - k)
        for k in range(n + 1)
        if abs(k / n - p) >= deviation - 1e-15
    )
    hoeffding = 2.0 * math.exp(-2.0 * n * deviation**2)

    rng = np.random.default_rng(int(oracle["seed"]))
    samples = rng.binomial(1, p, size=(repetitions, n))
    means = samples.mean(axis=1)
    observed_mean = float(means.mean())
    observed_sd = float(means.std(ddof=1))
    standardized = (means - p) / theory_sd
    coverage = float(
        np.mean(np.abs(standardized) <= float(oracle["normal_coverage_threshold"]))
    )
    simulated_tail = float(np.mean(np.abs(means - p) >= deviation - 1e-15))

    cauchy_medians = []
    for size in oracle["cauchy_sample_sizes"]:
        cauchy_means = rng.standard_cauchy(
            (int(oracle["cauchy_repetitions"]), int(size))
        ).mean(axis=1)
        cauchy_medians.append(float(np.median(np.abs(cauchy_means))))

    checks = [
        abs(theory_sd - float(oracle["expected_standard_error"])) <= float(oracle["absolute_tolerance"]),
        abs(exact_tail - float(oracle["expected_exact_binomial_tail"])) <= float(oracle["absolute_tolerance"]),
        abs(hoeffding - float(oracle["expected_hoeffding_bound"])) <= float(oracle["absolute_tolerance"]),
        abs(observed_mean - float(oracle["expected_mean"])) <= float(oracle["mean_tolerance"]),
        abs(observed_sd - theory_sd) <= float(oracle["standard_error_tolerance"]),
        abs(coverage - float(oracle["expected_normal_coverage"])) <= float(oracle["coverage_tolerance"]),
        abs(simulated_tail - exact_tail) <= float(oracle["tail_probability_tolerance"]),
    ]
    lower, upper = map(float, oracle["cauchy_median_band"])
    checks.extend(lower <= value <= upper for value in cauchy_medians)
    if not all(checks):
        raise SystemExit("limit-theorem oracle or declared simulation tolerance failed")

    print(
        "oracle=passed "
        f"mean={observed_mean:.6f} theory_sd={theory_sd:.6f} "
        f"observed_sd={observed_sd:.6f} clt_coverage={coverage:.6f} "
        f"exact_tail={exact_tail:.6f} simulated_tail={simulated_tail:.6f} "
        f"hoeffding={hoeffding:.6f} "
        f"cauchy_medians=({cauchy_medians[0]:.6f},{cauchy_medians[1]:.6f})"
    )
    return 0


main()
