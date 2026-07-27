# %% [markdown]
# # 条件期望、嵌套信息、Bayes 与两类独立性反例
#
# 所有目标值来自有限状态手算账本；程序只复算定义积分、投影恒等式和概率分解。

# %%
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


FIXED_PROBABILITIES = [0.125] * 8
FIXED_COARSE_STATE = [0, 0, 0, 0, 1, 1, 1, 1]
FIXED_FINE_STATE = [0, 0, 1, 1, 2, 2, 3, 3]
FIXED_VALUES = [0.0, 2.0, 1.0, 5.0, 4.0, 6.0, 7.0, 9.0]
FIXED_KERNEL = [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
FIXED_COMMON = [0, 0, 0, 0, 1, 1, 1, 1]
FIXED_COLLIDER_A = [0, 0, 1, 1]
FIXED_COLLIDER_B = [0, 1, 0, 1]
FIXED_COLLIDER_SELECTED = [0, 1, 1, 0]
FIXED_TOLERANCE = 1e-10


def finite_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"oracle {name} must be a numeric array") from exc
    if array.shape != shape:
        raise SystemExit(f"oracle {name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise SystemExit("oracle numeric inputs must be finite")
    return array


def finite_scalar(value: object, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"oracle {name} must be numeric") from exc
    if not math.isfinite(scalar):
        raise SystemExit("oracle numeric inputs must be finite")
    return scalar


def conditional_summary(
    probabilities: np.ndarray, labels: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(labels)
    masses = np.empty(unique.shape, dtype=float)
    means = np.empty(unique.shape, dtype=float)
    variances = np.empty(unique.shape, dtype=float)
    lifted = np.empty(values.shape, dtype=float)
    for index, state in enumerate(unique):
        mask = labels == state
        mass = float(probabilities[mask].sum())
        if mass <= 0.0:
            raise SystemExit("conditioning state has zero probability")
        mean = float(probabilities[mask] @ values[mask] / mass)
        masses[index] = mass
        means[index] = mean
        variances[index] = float(
            probabilities[mask] @ ((values[mask] - mean) ** 2) / mass
        )
        lifted[mask] = mean
    return masses, means, variances, lifted


def max_binary_independence_gap(
    probabilities: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    joint = float(probabilities[(first == 1) & (second == 1)].sum())
    product = float(
        probabilities[first == 1].sum() * probabilities[second == 1].sum()
    )
    return abs(joint - product)


def main(oracle_path: Path = Path("evidence/ch08/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    required = {
        "probabilities",
        "conditioning_state",
        "fine_conditioning_state",
        "random_variable",
        "expected_conditional_means",
        "expected_fine_conditional_means",
        "expected_tower_conditional_means",
        "expected_unconditional_mean",
        "expected_conditioning_integrals",
        "expected_orthogonality_max",
        "expected_jensen_gaps",
        "expected_total_variance",
        "expected_conditional_variance_mean",
        "expected_variance_of_conditional_mean",
        "kernel_support",
        "kernel",
        "conditioning_probabilities",
        "expected_mixture",
        "common_cause_a",
        "common_cause_b",
        "expected_marginal_joint_one",
        "expected_marginal_product_one",
        "maximum_conditional_independence_error",
        "collider_probabilities",
        "collider_a",
        "collider_b",
        "collider_selected",
        "expected_collider_marginal_gap",
        "expected_collider_conditional_gap",
        "bayes_prior",
        "bayes_signal_likelihood",
        "expected_bayes_evidence",
        "expected_bayes_posterior",
        "expected",
        "absolute_tolerance",
    }
    missing = sorted(required - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")

    probabilities = finite_array(oracle["probabilities"], "probabilities", (8,))
    coarse_state = finite_array(
        oracle["conditioning_state"], "conditioning_state", (8,)
    )
    fine_state = finite_array(
        oracle["fine_conditioning_state"], "fine_conditioning_state", (8,)
    )
    values = finite_array(oracle["random_variable"], "random_variable", (8,))
    kernel = finite_array(oracle["kernel"], "kernel", (2, 3))
    kernel_support = finite_array(oracle["kernel_support"], "kernel_support", (3,))
    mixing = finite_array(
        oracle["conditioning_probabilities"], "conditioning_probabilities", (2,)
    )
    common_a = finite_array(oracle["common_cause_a"], "common_cause_a", (8,))
    common_b = finite_array(oracle["common_cause_b"], "common_cause_b", (8,))
    collider_probabilities = finite_array(
        oracle["collider_probabilities"], "collider_probabilities", (4,)
    )
    collider_a = finite_array(oracle["collider_a"], "collider_a", (4,))
    collider_b = finite_array(oracle["collider_b"], "collider_b", (4,))
    collider_selected = finite_array(
        oracle["collider_selected"], "collider_selected", (4,)
    )
    bayes_prior = finite_array(oracle["bayes_prior"], "bayes_prior", (2,))
    bayes_likelihood = finite_array(
        oracle["bayes_signal_likelihood"], "bayes_signal_likelihood", (2,)
    )
    tolerance = finite_scalar(oracle["absolute_tolerance"], "absolute_tolerance")
    published_expected = finite_scalar(oracle["expected"], "expected")

    probability_arrays = [probabilities, mixing, collider_probabilities, bayes_prior]
    if any(np.any(array < 0.0) for array in probability_arrays) or np.any(kernel < 0.0):
        if np.any(mixing < 0.0):
            raise SystemExit("kernel mixing weights must be nonnegative and sum to one")
        raise SystemExit("probability mass must be nonnegative")
    if abs(float(mixing.sum()) - 1.0) > tolerance:
        raise SystemExit("kernel mixing weights must be nonnegative and sum to one")
    if any(
        abs(float(array.sum()) - 1.0) > tolerance
        for array in (probabilities, collider_probabilities, bayes_prior)
    ):
        raise SystemExit("probability mass must sum to one")
    if np.any(bayes_likelihood < 0.0) or np.any(bayes_likelihood > 1.0):
        raise SystemExit("Bayes likelihoods must lie in [0,1]")
    if not np.allclose(kernel.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
        raise SystemExit("conditional kernel rows must sum to one")

    if (
        probabilities.tolist() != FIXED_PROBABILITIES
        or coarse_state.tolist() != FIXED_COARSE_STATE
        or fine_state.tolist() != FIXED_FINE_STATE
        or values.tolist() != FIXED_VALUES
        or kernel.tolist() != FIXED_KERNEL
        or kernel_support.tolist() != [0.0, 2.0, 4.0]
        or mixing.tolist() != [0.5, 0.5]
        or common_a.tolist() != FIXED_COMMON
        or common_b.tolist() != FIXED_COMMON
        or collider_probabilities.tolist() != [0.25, 0.25, 0.25, 0.25]
        or collider_a.tolist() != FIXED_COLLIDER_A
        or collider_b.tolist() != FIXED_COLLIDER_B
        or collider_selected.tolist() != FIXED_COLLIDER_SELECTED
        or bayes_prior.tolist() != [0.7, 0.3]
        or bayes_likelihood.tolist() != [0.1, 0.8]
    ):
        raise SystemExit("oracle must use the fixed conditioning ledger")
    if tolerance != FIXED_TOLERANCE:
        raise SystemExit("oracle numeric tolerance is fixed")
    if published_expected != 4.25:
        raise SystemExit("published expected must equal 17/4")
    maximum_conditional_error = finite_scalar(
        oracle["maximum_conditional_independence_error"],
        "maximum_conditional_independence_error",
    )
    if maximum_conditional_error != 1e-12:
        raise SystemExit("conditional-independence threshold is fixed")

    coarse_masses, coarse_means, coarse_variances, coarse_lifted = (
        conditional_summary(probabilities, coarse_state, values)
    )
    _, fine_means, _, fine_lifted = conditional_summary(
        probabilities, fine_state, values
    )
    tower_values = np.empty(coarse_means.shape, dtype=float)
    for index, state in enumerate(np.unique(coarse_state)):
        mask = coarse_state == state
        tower_values[index] = float(
            probabilities[mask] @ fine_lifted[mask] / coarse_masses[index]
        )
    unconditional_mean = float(probabilities @ values)
    conditioning_integrals = np.asarray(
        [
            float(probabilities[coarse_state == state] @ values[coarse_state == state])
            for state in np.unique(coarse_state)
        ]
    )
    conditional_integrals = np.asarray(
        [
            float(
                probabilities[coarse_state == state]
                @ coarse_lifted[coarse_state == state]
            )
            for state in np.unique(coarse_state)
        ]
    )
    residual = values - coarse_lifted
    orthogonality_max = max(
        abs(float(probabilities[coarse_state == state] @ residual[coarse_state == state]))
        for state in np.unique(coarse_state)
    )
    conditional_second_moments = np.asarray(
        [
            float(
                probabilities[coarse_state == state]
                @ (values[coarse_state == state] ** 2)
                / coarse_masses[index]
            )
            for index, state in enumerate(np.unique(coarse_state))
        ]
    )
    jensen_gaps = conditional_second_moments - coarse_means**2
    total_variance = float(probabilities @ ((values - unconditional_mean) ** 2))
    conditional_variance_mean = float(coarse_masses @ coarse_variances)
    variance_of_conditional_mean = float(
        coarse_masses @ ((coarse_means - unconditional_mean) ** 2)
    )

    expected_coarse = finite_array(
        oracle["expected_conditional_means"], "expected_conditional_means", (2,)
    )
    expected_fine = finite_array(
        oracle["expected_fine_conditional_means"],
        "expected_fine_conditional_means",
        (4,),
    )
    expected_tower = finite_array(
        oracle["expected_tower_conditional_means"],
        "expected_tower_conditional_means",
        (2,),
    )
    expected_integrals = finite_array(
        oracle["expected_conditioning_integrals"],
        "expected_conditioning_integrals",
        (2,),
    )
    expected_jensen = finite_array(
        oracle["expected_jensen_gaps"], "expected_jensen_gaps", (2,)
    )
    if max(
        float(np.max(np.abs(coarse_means - expected_coarse))),
        float(np.max(np.abs(fine_means - expected_fine))),
        float(np.max(np.abs(tower_values - expected_tower))),
        abs(unconditional_mean - finite_scalar(oracle["expected_unconditional_mean"], "expected_unconditional_mean")),
        float(np.max(np.abs(conditioning_integrals - expected_integrals))),
        float(np.max(np.abs(conditional_integrals - expected_integrals))),
        abs(orthogonality_max - finite_scalar(oracle["expected_orthogonality_max"], "expected_orthogonality_max")),
        float(np.max(np.abs(jensen_gaps - expected_jensen))),
    ) > tolerance:
        raise SystemExit("nested-conditioning ledger mismatch")

    expected_total_variance = finite_scalar(
        oracle["expected_total_variance"], "expected_total_variance"
    )
    expected_conditional_variance = finite_scalar(
        oracle["expected_conditional_variance_mean"],
        "expected_conditional_variance_mean",
    )
    expected_variance_of_mean = finite_scalar(
        oracle["expected_variance_of_conditional_mean"],
        "expected_variance_of_conditional_mean",
    )
    if max(
        abs(total_variance - expected_total_variance),
        abs(conditional_variance_mean - expected_conditional_variance),
        abs(variance_of_conditional_mean - expected_variance_of_mean),
        abs(total_variance - conditional_variance_mean - variance_of_conditional_mean),
    ) > tolerance:
        raise SystemExit("total-variance ledger mismatch")

    mixture = mixing @ kernel
    expected_mixture = finite_array(
        oracle["expected_mixture"], "expected_mixture", (3,)
    )
    if float(np.max(np.abs(mixture - expected_mixture))) > tolerance:
        raise SystemExit("kernel mixture mismatch")

    marginal_joint = float(probabilities[(common_a == 1) & (common_b == 1)].sum())
    marginal_product = float(
        probabilities[common_a == 1].sum() * probabilities[common_b == 1].sum()
    )
    conditional_errors = []
    for state in np.unique(coarse_state):
        mask = coarse_state == state
        mass = float(probabilities[mask].sum())
        conditional_errors.append(
            max_binary_independence_gap(
                probabilities[mask] / mass, common_a[mask], common_b[mask]
            )
        )
    conditional_error = max(conditional_errors)
    if max(
        abs(marginal_joint - finite_scalar(oracle["expected_marginal_joint_one"], "expected_marginal_joint_one")),
        abs(marginal_product - finite_scalar(oracle["expected_marginal_product_one"], "expected_marginal_product_one")),
    ) > tolerance or conditional_error > maximum_conditional_error:
        raise SystemExit("common-cause independence ledger mismatch")

    collider_marginal_gap = max_binary_independence_gap(
        collider_probabilities, collider_a, collider_b
    )
    selected = collider_selected == 1
    selected_mass = float(collider_probabilities[selected].sum())
    collider_conditional_gap = max_binary_independence_gap(
        collider_probabilities[selected] / selected_mass,
        collider_a[selected],
        collider_b[selected],
    )
    if max(
        abs(collider_marginal_gap - finite_scalar(oracle["expected_collider_marginal_gap"], "expected_collider_marginal_gap")),
        abs(collider_conditional_gap - finite_scalar(oracle["expected_collider_conditional_gap"], "expected_collider_conditional_gap")),
    ) > tolerance:
        raise SystemExit("collider-conditioning ledger mismatch")

    bayes_evidence = float(bayes_prior @ bayes_likelihood)
    bayes_posterior = float(
        bayes_prior[1] * bayes_likelihood[1] / bayes_evidence
    )
    if max(
        abs(bayes_evidence - finite_scalar(oracle["expected_bayes_evidence"], "expected_bayes_evidence")),
        abs(bayes_posterior - finite_scalar(oracle["expected_bayes_posterior"], "expected_bayes_posterior")),
    ) > tolerance:
        raise SystemExit("Bayes ledger mismatch")

    print(
        "oracle=passed "
        f"coarse=({coarse_means[0]:.6f},{coarse_means[1]:.6f}) "
        f"fine=({fine_means[0]:.6f},{fine_means[1]:.6f},{fine_means[2]:.6f},{fine_means[3]:.6f}) "
        f"tower=({tower_values[0]:.6f},{tower_values[1]:.6f}) "
        f"mean={unconditional_mean:.6f} orthogonality={orthogonality_max:.3e} "
        f"jensen=({jensen_gaps[0]:.6f},{jensen_gaps[1]:.6f}) "
        f"variance=({total_variance:.6f},{conditional_variance_mean:.6f},{variance_of_conditional_mean:.6f}) "
        f"kernel_rows=({kernel.sum(axis=1)[0]:.6f},{kernel.sum(axis=1)[1]:.6f}) "
        f"mixture=({mixture[0]:.6f},{mixture[1]:.6f},{mixture[2]:.6f}) "
        f"bayes=({bayes_evidence:.6f},{bayes_posterior:.6f}) "
        f"common=({conditional_error:.3e},{abs(marginal_joint - marginal_product):.6f}) "
        f"collider=({collider_marginal_gap:.3e},{collider_conditional_gap:.6f})"
    )
    return 0


oracle_path = Path("evidence/ch08/oracle.json")
if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    oracle_path = Path(sys.argv[1])
main(oracle_path)
