# %% [markdown]
# # 投影、最小二乘、SVD 与近奇异放大
#
# 正规方程、残差和扰动解先由手算给出；NumPy 负责独立分解与重构核验。

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main(oracle_path: Path = Path("evidence/ch06/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    design = np.asarray(oracle["design"], dtype=float)
    response = np.asarray(oracle["response"], dtype=float)

    beta = np.linalg.solve(design.T @ design, design.T @ response)
    residual = response - design @ beta
    orthogonality_error = float(np.max(np.abs(design.T @ residual)))
    sse = float(residual @ residual)

    left, singular_values, right_transpose = np.linalg.svd(design, full_matrices=False)
    reconstruction = left @ np.diag(singular_values) @ right_transpose
    reconstruction_error = float(np.max(np.abs(reconstruction - design)))
    projection = design @ np.linalg.inv(design.T @ design) @ design.T
    projection_error = max(
        float(np.max(np.abs(projection - projection.T))),
        float(np.max(np.abs(projection @ projection - projection))),
    )

    epsilon = float(oracle["near_singular_epsilon"])
    perturbation = float(oracle["rhs_perturbation"])
    near_singular = np.asarray([[1.0, 1.0], [1.0, 1.0 + epsilon]])
    base_rhs = np.asarray([0.0, -epsilon])
    perturbed_rhs = base_rhs + np.asarray([perturbation, 0.0])
    base_solution = np.linalg.solve(near_singular, base_rhs)
    perturbed_solution = np.linalg.solve(near_singular, perturbed_rhs)
    component_amplification = float(
        np.max(np.abs(perturbed_solution - base_solution)) / perturbation
    )
    condition_number = float(np.linalg.cond(near_singular))

    tolerance = float(oracle["absolute_tolerance"])
    comparisons = [
        (beta, oracle["expected_beta"]),
        (residual, oracle["expected_residual"]),
        (singular_values, oracle["expected_singular_values"]),
        (base_solution, oracle["expected_base_solution"]),
        (perturbed_solution, oracle["expected_perturbed_solution"]),
    ]
    if any(
        float(np.max(np.abs(observed - np.asarray(expected)))) > tolerance
        for observed, expected in comparisons
    ):
        raise SystemExit("oracle mismatch in decomposition or perturbation ledger")
    if abs(sse - float(oracle["expected_sse"])) > tolerance:
        raise SystemExit(f"SSE mismatch: {sse}")
    if abs(component_amplification - float(oracle["expected_component_amplification"])) > 1.0:
        raise SystemExit(f"amplification mismatch: {component_amplification}")
    if orthogonality_error > float(oracle["maximum_orthogonality_error"]):
        raise SystemExit(f"orthogonality mismatch: {orthogonality_error}")
    if max(reconstruction_error, projection_error) > float(
        oracle["maximum_reconstruction_error"]
    ):
        raise SystemExit("decomposition reconstruction mismatch")

    print(
        "oracle=passed "
        f"beta=({beta[0]:.6f},{beta[1]:.6f}) sse={sse:.6f} "
        f"orth_error={orthogonality_error:.3e} recon_error={reconstruction_error:.3e} "
        f"sigma=({singular_values[0]:.6f},{singular_values[1]:.6f}) "
        f"condition={condition_number:.3e} amplification={component_amplification:.0f}"
    )
    return 0


main()
