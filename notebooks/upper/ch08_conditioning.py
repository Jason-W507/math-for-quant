# %% [markdown]
# # 条件期望、概率核与共同原因反例
#
# 四状态账本的每个条件量都有手算结果；程序不通过抽样估计目标值。

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(oracle_path: Path = Path("evidence/ch08/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    tolerance = float(oracle["absolute_tolerance"])
    probabilities = np.asarray(oracle["probabilities"], dtype=float)
    states = np.asarray(oracle["conditioning_state"], dtype=int)
    values = np.asarray(oracle["random_variable"], dtype=float)
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or probabilities.shape != states.shape
        or probabilities.shape != values.shape
        or not np.all(np.isfinite(probabilities))
        or not np.all(np.isfinite(values))
        or np.any(probabilities < 0.0)
        or abs(float(probabilities.sum()) - 1.0) > tolerance
    ):
        raise SystemExit("finite conditioning ledger is not a probability space")

    unique_states = np.unique(states)
    conditional_means = []
    state_probabilities = []
    for state in unique_states:
        mask = states == state
        mass = float(probabilities[mask].sum())
        if mass <= 0.0:
            raise SystemExit("conditioning state has zero probability")
        state_probabilities.append(mass)
        conditional_means.append(float(probabilities[mask] @ values[mask] / mass))
    conditional_means = np.asarray(conditional_means)
    state_probabilities = np.asarray(state_probabilities)
    unconditional_mean = float(probabilities @ values)
    tower_mean = float(state_probabilities @ conditional_means)

    kernel = np.asarray(oracle["kernel"], dtype=float)
    kernel_support = np.asarray(oracle["kernel_support"], dtype=float)
    conditioning_probabilities = np.asarray(
        oracle["conditioning_probabilities"], dtype=float
    )
    if (
        kernel.ndim != 2
        or kernel_support.ndim != 1
        or kernel.shape[1] != kernel_support.size
        or kernel.shape[0] != conditioning_probabilities.size
        or not np.all(np.isfinite(kernel))
        or not np.all(np.isfinite(kernel_support))
        or np.any(kernel < 0.0)
    ):
        raise SystemExit("conditional kernel must be a finite nonnegative matrix")
    if (
        conditioning_probabilities.ndim != 1
        or not np.all(np.isfinite(conditioning_probabilities))
        or np.any(conditioning_probabilities < 0.0)
        or abs(float(conditioning_probabilities.sum()) - 1.0) > tolerance
        or not np.allclose(
            conditioning_probabilities,
            state_probabilities,
            atol=tolerance,
            rtol=0.0,
        )
    ):
        raise SystemExit("kernel mixing weights must match nonnegative conditioning-state probabilities")
    kernel_rows = kernel.sum(axis=1)
    if not np.allclose(kernel_rows, 1.0, atol=tolerance, rtol=0.0):
        raise SystemExit("conditional kernel rows must sum to one")
    mixture = conditioning_probabilities @ kernel
    expected_mixture = np.asarray(oracle["expected_mixture"], dtype=float)
    if expected_mixture.shape != kernel_support.shape:
        raise SystemExit("expected mixture shape must match kernel support")

    common_a = np.asarray(oracle["common_cause_a"], dtype=int)
    common_b = np.asarray(oracle["common_cause_b"], dtype=int)
    if common_a.shape != probabilities.shape or common_b.shape != probabilities.shape:
        raise SystemExit("common-cause variables must share the probability-space shape")
    marginal_joint = float(probabilities[(common_a == 1) & (common_b == 1)].sum())
    marginal_product = float(probabilities[common_a == 1].sum() * probabilities[common_b == 1].sum())
    conditional_errors = []
    for state in unique_states:
        mask = states == state
        mass = float(probabilities[mask].sum())
        joint = float(probabilities[mask & (common_a == 1) & (common_b == 1)].sum() / mass)
        first = float(probabilities[mask & (common_a == 1)].sum() / mass)
        second = float(probabilities[mask & (common_b == 1)].sum() / mass)
        conditional_errors.append(abs(joint - first * second))
    conditional_error = max(conditional_errors)

    if not np.allclose(
        conditional_means,
        oracle["expected_conditional_means"],
        atol=tolerance,
        rtol=0.0,
    ):
        raise SystemExit("conditional expectation mismatch")
    if max(abs(unconditional_mean - float(oracle["expected_unconditional_mean"])), abs(tower_mean - unconditional_mean)) > tolerance:
        raise SystemExit("tower-property mismatch")
    if not np.allclose(mixture, oracle["expected_mixture"], atol=tolerance, rtol=0.0):
        raise SystemExit("kernel mixture mismatch")
    if abs(marginal_joint - float(oracle["expected_marginal_joint_one"])) > tolerance:
        raise SystemExit("marginal joint mismatch")
    if abs(marginal_product - float(oracle["expected_marginal_product_one"])) > tolerance:
        raise SystemExit("marginal product mismatch")
    if conditional_error > float(oracle["maximum_conditional_independence_error"]):
        raise SystemExit("conditional-independence mismatch")

    print(
        "oracle=passed "
        f"conditional=({conditional_means[0]:.6f},{conditional_means[1]:.6f}) "
        f"tower={tower_mean:.6f} "
        f"kernel_rows=({kernel_rows[0]:.6f},{kernel_rows[1]:.6f}) "
        f"mixture=({mixture[0]:.6f},{mixture[1]:.6f},{mixture[2]:.6f}) "
        f"marginal_joint={marginal_joint:.6f} marginal_product={marginal_product:.6f} "
        f"conditional_error={conditional_error:.3e}"
    )
    return 0


oracle_path = Path("evidence/ch08/oracle.json")
if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    oracle_path = Path(sys.argv[1])
main(oracle_path)
