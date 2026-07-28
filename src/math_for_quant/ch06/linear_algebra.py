from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


FIXED_DESIGN = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
FIXED_RESPONSE = [1.0, 2.0, 2.0]
FIXED_EPSILON = 1e-6
FIXED_PERTURBATION = 1e-8
FIXED_ABSOLUTE_TOLERANCE = 1e-10
FIXED_ERROR_GATE = 1e-12


from math_for_quant.evidence import load_oracle_bundle

def _finite_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"oracle {name} must be a numeric array") from exc
    if array.shape != shape:
        raise SystemExit(f"oracle {name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise SystemExit("oracle numeric inputs must be finite")
    return array


def _finite_scalar(value: object, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"oracle {name} must be numeric") from exc
    if not math.isfinite(scalar):
        raise SystemExit("oracle numeric inputs must be finite")
    return scalar


def main(oracle_path: Path = Path("evidence/ch06/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    required = {
        "design",
        "response",
        "near_singular_epsilon",
        "rhs_perturbation",
        "absolute_tolerance",
        "maximum_orthogonality_error",
        "maximum_reconstruction_error",
        "expected_condition_number",
        "expected_worst_direction_amplification",
        "expected_beta",
        "expected_residual",
        "expected_sse",
        "expected_singular_values",
        "expected_base_solution",
        "expected_perturbed_solution",
        "expected_component_amplification",
        "expected_relative_amplification",
        "expected_covariance_eigenvalues",
        "expected",
        "maximum_solver_error",
    }
    missing = sorted(required - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")

    design = _finite_array(oracle["design"], "design", (3, 2))
    response = _finite_array(oracle["response"], "response", (3,))
    epsilon = _finite_scalar(oracle["near_singular_epsilon"], "epsilon")
    perturbation = _finite_scalar(oracle["rhs_perturbation"], "perturbation")
    tolerance = _finite_scalar(oracle["absolute_tolerance"], "absolute_tolerance")
    orthogonality_gate = _finite_scalar(
        oracle["maximum_orthogonality_error"], "maximum_orthogonality_error"
    )
    reconstruction_gate = _finite_scalar(
        oracle["maximum_reconstruction_error"], "maximum_reconstruction_error"
    )
    solver_gate = _finite_scalar(oracle["maximum_solver_error"], "maximum_solver_error")
    expected_beta = _finite_array(oracle["expected_beta"], "expected_beta", (2,))
    expected_residual = _finite_array(
        oracle["expected_residual"], "expected_residual", (3,)
    )
    expected_sse = _finite_scalar(oracle["expected_sse"], "expected_sse")
    expected_singular_values = _finite_array(
        oracle["expected_singular_values"], "expected_singular_values", (2,)
    )
    expected_base_solution = _finite_array(
        oracle["expected_base_solution"], "expected_base_solution", (2,)
    )
    expected_perturbed_solution = _finite_array(
        oracle["expected_perturbed_solution"], "expected_perturbed_solution", (2,)
    )
    expected_component_amplification = _finite_scalar(
        oracle["expected_component_amplification"],
        "expected_component_amplification",
    )
    expected_relative_amplification = _finite_scalar(
        oracle["expected_relative_amplification"],
        "expected_relative_amplification",
    )
    expected_covariance_eigenvalues = _finite_array(
        oracle["expected_covariance_eigenvalues"],
        "expected_covariance_eigenvalues",
        (2,),
    )
    expected_condition = _finite_scalar(
        oracle["expected_condition_number"], "expected_condition_number"
    )
    expected_worst = _finite_scalar(
        oracle["expected_worst_direction_amplification"],
        "expected_worst_direction_amplification",
    )
    published_expected = _finite_scalar(oracle["expected"], "expected")
    if (
        design.tolist() != FIXED_DESIGN
        or response.tolist() != FIXED_RESPONSE
        or epsilon != FIXED_EPSILON
        or perturbation != FIXED_PERTURBATION
    ):
        raise SystemExit("oracle must use the fixed linear-algebra ledger")
    if (
        tolerance != FIXED_ABSOLUTE_TOLERANCE
        or orthogonality_gate != FIXED_ERROR_GATE
        or reconstruction_gate != FIXED_ERROR_GATE
        or solver_gate != FIXED_ERROR_GATE
    ):
        raise SystemExit("oracle numeric tolerances are fixed")
    if published_expected != 7.0 / 6.0:
        raise SystemExit("published expected must equal 7/6")

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

    q_matrix, r_matrix = np.linalg.qr(design, mode="reduced")
    qr_error = max(
        float(np.max(np.abs(q_matrix.T @ q_matrix - np.eye(2)))),
        float(np.max(np.abs(q_matrix @ r_matrix - design))),
    )
    qr_beta = np.linalg.solve(r_matrix, q_matrix.T @ response)
    lstsq_beta = np.linalg.lstsq(design, response, rcond=None)[0]
    pseudoinverse_beta = np.linalg.pinv(design) @ response
    solver_error = max(
        float(np.max(np.abs(qr_beta - expected_beta))),
        float(np.max(np.abs(lstsq_beta - expected_beta))),
        float(np.max(np.abs(pseudoinverse_beta - expected_beta))),
        float(np.max(np.abs(beta - expected_beta))),
    )
    rank_one = singular_values[0] * np.outer(left[:, 0], right_transpose[0, :])
    rank_one_spectral_error = float(np.linalg.norm(design - rank_one, ord=2))
    design_condition = float(np.linalg.cond(design))
    gram_condition = float(np.linalg.cond(design.T @ design))
    gram_condition_ratio = gram_condition / (design_condition**2)

    covariance = np.asarray([[2.0, 1.0], [1.0, 2.0]])
    eigenvalues = np.linalg.eigvalsh(covariance)
    cholesky = np.linalg.cholesky(covariance)
    cholesky_error = float(np.max(np.abs(cholesky @ cholesky.T - covariance)))

    near_singular = np.asarray([[1.0, 1.0], [1.0, 1.0 + epsilon]])
    base_rhs = np.asarray([0.0, -epsilon])
    perturbed_rhs = base_rhs + np.asarray([perturbation, 0.0])
    base_solution = np.linalg.solve(near_singular, base_rhs)
    perturbed_solution = np.linalg.solve(near_singular, perturbed_rhs)
    component_amplification = float(
        np.max(np.abs(perturbed_solution - base_solution)) / perturbation
    )
    relative_amplification = float(
        (np.linalg.norm(perturbed_solution - base_solution) / np.linalg.norm(base_solution))
        / (np.linalg.norm(perturbed_rhs - base_rhs) / np.linalg.norm(base_rhs))
    )
    condition_number = float(np.linalg.cond(near_singular))

    near_left, near_sigma, near_right_transpose = np.linalg.svd(near_singular)
    worst_base_rhs = near_sigma[0] * near_left[:, 0]
    worst_delta_rhs = perturbation * near_sigma[0] * near_left[:, 1]
    worst_base_solution = np.linalg.solve(near_singular, worst_base_rhs)
    worst_perturbed_solution = np.linalg.solve(
        near_singular, worst_base_rhs + worst_delta_rhs
    )
    worst_direction_amplification = float(
        (
            np.linalg.norm(worst_perturbed_solution - worst_base_solution)
            / np.linalg.norm(worst_base_solution)
        )
        / (np.linalg.norm(worst_delta_rhs) / np.linalg.norm(worst_base_rhs))
    )

    comparisons = [
        (beta, expected_beta),
        (residual, expected_residual),
        (singular_values, expected_singular_values),
        (base_solution, expected_base_solution),
        (perturbed_solution, expected_perturbed_solution),
        (eigenvalues, expected_covariance_eigenvalues),
    ]
    if any(
        float(np.max(np.abs(observed - np.asarray(expected)))) > tolerance
        for observed, expected in comparisons
    ):
        raise SystemExit("oracle mismatch in decomposition or perturbation ledger")
    if abs(sse - expected_sse) > tolerance:
        raise SystemExit(f"SSE mismatch: {sse}")
    if abs(component_amplification - expected_component_amplification) > 1.0:
        raise SystemExit(f"amplification mismatch: {component_amplification}")
    if abs(relative_amplification - expected_relative_amplification) > 1e-9:
        raise SystemExit(f"relative amplification mismatch: {relative_amplification}")
    if orthogonality_error > orthogonality_gate:
        raise SystemExit(f"orthogonality mismatch: {orthogonality_error}")
    if max(
        reconstruction_error,
        projection_error,
        qr_error,
        cholesky_error,
    ) > reconstruction_gate:
        raise SystemExit("decomposition reconstruction mismatch")
    if solver_error > solver_gate:
        raise SystemExit("least-squares solver mismatch")
    if abs(rank_one_spectral_error - singular_values[1]) > tolerance:
        raise SystemExit("rank-one approximation ledger mismatch")
    if abs(gram_condition_ratio - 1.0) > tolerance:
        raise SystemExit("normal-equation condition-squaring mismatch")
    if abs(condition_number - expected_condition) > 0.1:
        raise SystemExit("condition-number ledger mismatch")
    if abs(worst_direction_amplification - expected_worst) > 0.1:
        raise SystemExit("worst-direction amplification ledger mismatch")

    print(
        "oracle=passed "
        f"beta=({beta[0]:.6f},{beta[1]:.6f}) sse={sse:.6f} "
        f"orth_error={orthogonality_error:.3e} recon_error={reconstruction_error:.3e} "
        f"sigma=({singular_values[0]:.6f},{singular_values[1]:.6f}) "
        f"condition={condition_number:.3e} amplification={component_amplification:.0f} "
        f"relative={relative_amplification:.6f} qr_error={qr_error:.3e} "
        f"solve_error={solver_error:.3e} "
        f"rank1_error={rank_one_spectral_error:.6f} gram_ratio={gram_condition_ratio:.6f} "
        f"eigen=({eigenvalues[0]:.1f},{eigenvalues[1]:.1f}) "
        f"worst={worst_direction_amplification:.3e}"
    )
    return 0


if __name__ == "__main__":
    oracle_path = Path("evidence/ch06/oracle.json")
    if Path(sys.argv[0]).stem == "ch06_linear_algebra" and len(sys.argv) > 1:
        oracle_path = Path(sys.argv[1])
    main(oracle_path)
