# %% [markdown]
# # LLN、CLT、集中界与重尾失效
#
# 理论均值、标准误、精确二项尾概率和 Hoeffding 上界均独立于模拟给出。

# %%
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


FIXED_TOLERANCE = 1e-10
REQUIRED_FIELDS = {
    "provenance",
    "seed",
    "bernoulli_p",
    "sample_size",
    "repetitions",
    "expected_mean",
    "expected_standard_error",
    "normal_coverage_threshold",
    "expected_normal_coverage",
    "tail_deviation",
    "expected_exact_binomial_tail",
    "expected_chebyshev_bound",
    "expected_hoeffding_bound",
    "expected_bernstein_bound",
    "berry_esseen_constant",
    "expected_normal_cdf_distance",
    "expected_berry_esseen_bound",
    "rare_event_probability",
    "rare_event_sample_size",
    "expected_rare_event_zero_probability",
    "expected_naive_normal_interval",
    "expected_iid_mean_variance",
    "expected_dependent_mean_variance",
    "mean_tolerance",
    "standard_error_tolerance",
    "coverage_tolerance",
    "tail_probability_tolerance",
    "rare_spike_indices",
    "expected_rare_spike_l1",
    "cauchy_repetitions",
    "cauchy_sample_sizes",
    "cauchy_median_band",
    "published_markers",
    "expected",
    "absolute_tolerance",
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
    if isinstance(value, dict):
        for item in value.values():
            reject_nonfinite_numbers(item)


def main(oracle_path: Path = Path("evidence/ch09/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_FIELDS - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")
    reject_nonfinite_numbers(oracle)
    tolerance = float(oracle["absolute_tolerance"])
    if tolerance != FIXED_TOLERANCE:
        raise SystemExit("absolute tolerance must equal 1e-10")
    if (
        float(oracle["bernoulli_p"]) != 0.3
        or int(oracle["sample_size"]) != 200
        or int(oracle["repetitions"]) != 20000
        or int(oracle["seed"]) != 20260725
        or float(oracle["tail_deviation"]) != 0.1
    ):
        raise SystemExit("canonical Bernoulli experiment must not change")
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
    if (
        abs(exact_tail - float(oracle["expected_exact_binomial_tail"]))
        > tolerance
    ):
        raise SystemExit("exact binomial tail claim failed")
    variance = p * (1.0 - p)
    chebyshev = variance / (n * deviation**2)
    hoeffding = 2.0 * math.exp(-2.0 * n * deviation**2)
    bernstein = 2.0 * math.exp(
        -n * deviation**2 / (2.0 * variance + 2.0 * deviation / 3.0)
    )
    if (
        abs(bernstein - float(oracle["expected_bernstein_bound"]))
        > float(oracle["absolute_tolerance"])
    ):
        raise SystemExit("Bernstein bound claim failed")

    binomial_cdf = 0.0
    normal_cdf_distance = 0.0
    count_sd = math.sqrt(n * variance)
    for count in range(n + 1):
        previous_cdf = binomial_cdf
        binomial_cdf += (
            math.comb(n, count)
            * p**count
            * (1.0 - p) ** (n - count)
        )
        standardized_count = (count - n * p) / count_sd
        normal_cdf = 0.5 * (
            1.0 + math.erf(standardized_count / math.sqrt(2.0))
        )
        normal_cdf_distance = max(
            normal_cdf_distance,
            abs(previous_cdf - normal_cdf),
            abs(binomial_cdf - normal_cdf),
        )
    third_absolute_moment = (
        p * (1.0 - p) ** 3 + (1.0 - p) * p**3
    )
    berry_esseen = (
        float(oracle["berry_esseen_constant"])
        * third_absolute_moment
        / (variance**1.5 * math.sqrt(n))
    )
    if (
        abs(normal_cdf_distance - float(oracle["expected_normal_cdf_distance"]))
        > float(oracle["absolute_tolerance"])
    ):
        raise SystemExit("normal approximation distance claim failed")

    rare_p = float(oracle["rare_event_probability"])
    rare_n = int(oracle["rare_event_sample_size"])
    rare_zero_probability = (1.0 - rare_p) ** rare_n
    rare_standard_error = math.sqrt(rare_p * (1.0 - rare_p) / rare_n)
    rare_normal_interval = (
        rare_p - 1.96 * rare_standard_error,
        rare_p + 1.96 * rare_standard_error,
    )

    iid_mean_variance = variance / n
    dependent_mean_variance = variance
    if (
        abs(
            dependent_mean_variance
            - float(oracle["expected_dependent_mean_variance"])
        )
        > float(oracle["absolute_tolerance"])
    ):
        raise SystemExit("perfect-dependence variance claim failed")

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

    rare_spike_indices = [int(value) for value in oracle["rare_spike_indices"]]
    if rare_spike_indices != [10, 100, 1000]:
        raise SystemExit("rare-spike indices must match the published experiment")
    rare_spike_probabilities = [1.0 / value for value in rare_spike_indices]
    rare_spike_l1 = [value * probability for value, probability in zip(
        rare_spike_indices, rare_spike_probabilities, strict=True
    )]
    if any(
        abs(value - float(oracle["expected_rare_spike_l1"]))
        > float(oracle["absolute_tolerance"])
        for value in rare_spike_l1
    ):
        raise SystemExit("rare-spike L1 claim failed")

    cauchy_medians = []
    for size in oracle["cauchy_sample_sizes"]:
        cauchy_means = rng.standard_cauchy(
            (int(oracle["cauchy_repetitions"]), int(size))
        ).mean(axis=1)
        cauchy_medians.append(float(np.median(np.abs(cauchy_means))))

    checks = [
        abs(theory_sd - float(oracle["expected_standard_error"])) <= float(oracle["absolute_tolerance"]),
        abs(exact_tail - float(oracle["expected_exact_binomial_tail"])) <= float(oracle["absolute_tolerance"]),
        abs(chebyshev - float(oracle["expected_chebyshev_bound"])) <= float(oracle["absolute_tolerance"]),
        abs(hoeffding - float(oracle["expected_hoeffding_bound"])) <= float(oracle["absolute_tolerance"]),
        abs(observed_mean - float(oracle["expected_mean"])) <= float(oracle["mean_tolerance"]),
        abs(observed_sd - theory_sd) <= float(oracle["standard_error_tolerance"]),
        abs(coverage - float(oracle["expected_normal_coverage"])) <= float(oracle["coverage_tolerance"]),
        abs(simulated_tail - exact_tail) <= float(oracle["tail_probability_tolerance"]),
        abs(berry_esseen - float(oracle["expected_berry_esseen_bound"])) <= float(oracle["absolute_tolerance"]),
        normal_cdf_distance <= berry_esseen,
        abs(rare_zero_probability - float(oracle["expected_rare_event_zero_probability"])) <= float(oracle["absolute_tolerance"]),
        abs(iid_mean_variance - float(oracle["expected_iid_mean_variance"])) <= float(oracle["absolute_tolerance"]),
        all(
            abs(actual - float(expected)) <= float(oracle["absolute_tolerance"])
            for actual, expected in zip(
                rare_normal_interval,
                oracle["expected_naive_normal_interval"],
                strict=True,
            )
        ),
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
        f"bounds=({chebyshev:.6f},{hoeffding:.6f},{bernstein:.6f}) "
        f"normal=({normal_cdf_distance:.6f},{berry_esseen:.6f}) "
        f"rare_zero={rare_zero_probability:.6f} "
        f"variances=({iid_mean_variance:.6f},{dependent_mean_variance:.6f}) "
        f"rare_spike=({rare_spike_probabilities[0]:.3f},"
        f"{rare_spike_probabilities[1]:.3f},"
        f"{rare_spike_probabilities[2]:.3f};l1={rare_spike_l1[0]:.1f}) "
        f"cauchy_medians=({cauchy_medians[0]:.6f},{cauchy_medians[1]:.6f})"
    )
    return 0


main(
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else Path("evidence/ch09/oracle.json")
)
