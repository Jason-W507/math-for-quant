from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


FIXED_TOLERANCE = 1e-10
FIXED_VALID_IMPLICATIONS = [
    "almost_sure_implies_probability",
    "lp_implies_probability",
    "probability_implies_distribution",
    "lq_implies_lp_on_probability_space",
]
FIXED_COUNTEREXAMPLES = [
    "probability_not_almost_sure",
    "probability_not_l1",
    "l1_not_l2",
    "distribution_not_probability",
]
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
    "valid_implications",
    "counterexample_labels",
    "typewriter_levels",
    "expected_typewriter_probabilities",
    "expected_typewriter_layer_mass",
    "rare_spike_indices",
    "expected_rare_spike_l1",
    "l1_not_l2_indices",
    "expected_l1_not_l2_first_moments",
    "expected_l1_not_l2_second_moment",
    "expected_distribution_not_probability_gap",
    "expected_dominant_term_lindeberg_ratio",
    "cauchy_repetitions",
    "cauchy_sample_sizes",
    "cauchy_median_band",
    "published_markers",
    "expected",
    "absolute_tolerance",
}


from m4q.evidence import load_oracle_bundle

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


def require_integer(value: object, name: str, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"oracle {name} must be an integer")
    if value != expected:
        raise SystemExit(f"oracle {name} must equal {expected}")
    return value


def require_numeric_sequence(
    value: object, name: str, length: int
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SystemExit(f"oracle {name} must have length {length}")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise SystemExit(f"oracle {name} must be a numeric array")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise SystemExit("oracle numeric inputs must be finite")
    return result


def main(oracle_path: Path = Path("evidence/ch09/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    missing = sorted(REQUIRED_FIELDS - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")
    non_scalar_fields = {
        "provenance",
        "published_markers",
        "valid_implications",
        "counterexample_labels",
        "typewriter_levels",
        "expected_typewriter_probabilities",
        "rare_spike_indices",
        "l1_not_l2_indices",
        "expected_l1_not_l2_first_moments",
        "cauchy_sample_sizes",
        "cauchy_median_band",
        "expected_naive_normal_interval",
    }
    integer_fields = {
        "seed",
        "sample_size",
        "repetitions",
        "rare_event_sample_size",
        "cauchy_repetitions",
    }
    for name in sorted(REQUIRED_FIELDS - non_scalar_fields - integer_fields):
        value = oracle[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"oracle {name} must be numeric")
    reject_nonfinite_numbers(oracle)
    typewriter_expected = require_numeric_sequence(
        oracle["expected_typewriter_probabilities"],
        "expected_typewriter_probabilities",
        4,
    )
    l1_not_l2_expected = require_numeric_sequence(
        oracle["expected_l1_not_l2_first_moments"],
        "expected_l1_not_l2_first_moments",
        3,
    )
    rare_normal_expected = require_numeric_sequence(
        oracle["expected_naive_normal_interval"],
        "expected_naive_normal_interval",
        2,
    )
    cauchy_band = require_numeric_sequence(
        oracle["cauchy_median_band"], "cauchy_median_band", 2
    )
    tolerance = float(oracle["absolute_tolerance"])
    if tolerance != FIXED_TOLERANCE:
        raise SystemExit("absolute tolerance must equal 1e-10")
    if float(oracle["expected"]) != 0.3:
        raise SystemExit("published expected must equal 0.3")
    sample_size = require_integer(oracle["sample_size"], "sample_size", 200)
    repetitions = require_integer(oracle["repetitions"], "repetitions", 20000)
    seed = require_integer(oracle["seed"], "seed", 20260725)
    if (
        float(oracle["bernoulli_p"]) != 0.3
        or float(oracle["tail_deviation"]) != 0.1
    ):
        raise SystemExit("canonical Bernoulli experiment must not change")
    if (
        float(oracle["mean_tolerance"]) != 0.002
        or float(oracle["standard_error_tolerance"]) != 0.002
        or float(oracle["coverage_tolerance"]) != 0.015
        or float(oracle["tail_probability_tolerance"]) != 0.001
        or float(oracle["normal_coverage_threshold"]) != 1.96
        or float(oracle["expected_mean"]) != 0.3
        or float(oracle["expected_normal_coverage"]) != 0.95
    ):
        raise SystemExit("simulation gates must match the published design")
    if (
        float(oracle["berry_esseen_constant"]) != 0.56
        or float(oracle["rare_event_probability"]) != 0.01
        or require_integer(
            oracle["rare_event_sample_size"], "rare_event_sample_size", 10
        )
        != 10
    ):
        raise SystemExit("finite-sample approximation design must not change")
    if (
        require_integer(
            oracle["cauchy_repetitions"], "cauchy_repetitions", 8000
        )
        != 8000
        or cauchy_band != [0.7, 1.3]
    ):
        raise SystemExit("Cauchy design must match the published experiment")
    if oracle["valid_implications"] != FIXED_VALID_IMPLICATIONS:
        raise SystemExit("convergence implication labels must match the theorem graph")
    if oracle["counterexample_labels"] != FIXED_COUNTEREXAMPLES:
        raise SystemExit("convergence counterexample labels must match the published ledger")
    p = float(oracle["bernoulli_p"])
    n = sample_size
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
    rare_n = require_integer(
        oracle["rare_event_sample_size"], "rare_event_sample_size", 10
    )
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

    rng = np.random.default_rng(seed)
    samples = rng.binomial(1, p, size=(repetitions, n))
    means = samples.mean(axis=1)
    observed_mean = float(means.mean())
    observed_sd = float(means.std(ddof=1))
    standardized = (means - p) / theory_sd
    coverage = float(
        np.mean(np.abs(standardized) <= float(oracle["normal_coverage_threshold"]))
    )
    simulated_tail = float(np.mean(np.abs(means - p) >= deviation - 1e-15))

    typewriter_levels = oracle["typewriter_levels"]
    if typewriter_levels != [1, 2, 3, 4] or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in typewriter_levels
    ):
        raise SystemExit("typewriter levels must match the published counterexample")
    typewriter_probabilities = [2.0 ** (-level) for level in typewriter_levels]
    typewriter_layer_mass = [
        (2**level) * probability
        for level, probability in zip(
            typewriter_levels, typewriter_probabilities, strict=True
        )
    ]
    if (
        any(
            abs(actual - float(expected)) > tolerance
            for actual, expected in zip(
                typewriter_probabilities,
                typewriter_expected,
                strict=True,
            )
        )
        or any(
            abs(value - float(oracle["expected_typewriter_layer_mass"]))
            > tolerance
            for value in typewriter_layer_mass
        )
    ):
        raise SystemExit("typewriter counterexample ledger failed")

    rare_spike_indices = oracle["rare_spike_indices"]
    if rare_spike_indices != [10, 100, 1000] or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in rare_spike_indices
    ):
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

    l1_not_l2_indices = oracle["l1_not_l2_indices"]
    if l1_not_l2_indices != [10, 100, 1000] or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in l1_not_l2_indices
    ):
        raise SystemExit("L1-not-L2 indices must match the published experiment")
    l1_not_l2_first_moments = [
        1.0 / math.sqrt(value) for value in l1_not_l2_indices
    ]
    l1_not_l2_second_moments = [1.0] * len(l1_not_l2_indices)
    if (
        any(
            abs(actual - float(expected)) > tolerance
            for actual, expected in zip(
                l1_not_l2_first_moments,
                l1_not_l2_expected,
                strict=True,
            )
        )
        or any(
            abs(value - float(oracle["expected_l1_not_l2_second_moment"]))
            > tolerance
            for value in l1_not_l2_second_moments
        )
    ):
        raise SystemExit("L1-not-L2 counterexample ledger failed")

    distribution_not_probability_gap = 1.0
    if (
        abs(
            distribution_not_probability_gap
            - float(oracle["expected_distribution_not_probability_gap"])
        )
        > tolerance
    ):
        raise SystemExit("distribution-not-probability counterexample ledger failed")

    dominant_term_lindeberg_ratio = 1.0
    if (
        abs(
            dominant_term_lindeberg_ratio
            - float(oracle["expected_dominant_term_lindeberg_ratio"])
        )
        > tolerance
    ):
        raise SystemExit("dominant-term Lindeberg counterexample ledger failed")

    cauchy_medians = []
    cauchy_sample_sizes = oracle["cauchy_sample_sizes"]
    if cauchy_sample_sizes != [10, 1000] or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in cauchy_sample_sizes
    ):
        raise SystemExit("Cauchy design must match the published experiment")
    for size in cauchy_sample_sizes:
        cauchy_means = rng.standard_cauchy(
            (8000, size)
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
                rare_normal_expected,
                strict=True,
            )
        ),
    ]
    lower, upper = cauchy_band
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
        f"typewriter=({typewriter_probabilities[0]:.4f},"
        f"{typewriter_probabilities[-1]:.4f};mass={typewriter_layer_mass[0]:.1f}) "
        f"rare_spike=({rare_spike_probabilities[0]:.3f},"
        f"{rare_spike_probabilities[1]:.3f},"
        f"{rare_spike_probabilities[2]:.3f};l1={rare_spike_l1[0]:.1f}) "
        f"l1_l2=({l1_not_l2_first_moments[0]:.6f},"
        f"{l1_not_l2_first_moments[-1]:.6f};"
        f"l2={l1_not_l2_second_moments[0]:.1f}) "
        f"distribution_gap={distribution_not_probability_gap:.1f} "
        f"lindeberg={dominant_term_lindeberg_ratio:.1f} "
        f"cauchy_medians=({cauchy_medians[0]:.6f},{cauchy_medians[1]:.6f})"
    )
    return 0


if __name__ == "__main__":
    main(
        Path(sys.argv[1])
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
        else Path("evidence/ch09/oracle.json")
    )
