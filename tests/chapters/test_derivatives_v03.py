from __future__ import annotations

import unittest
from pathlib import Path
import subprocess
import sys

import numpy as np

from math_for_quant.lower.derivatives_numerics import (
    SurfaceNode,
    black_scholes_call,
    fit_parametric_total_variance,
    implicit_fd_call,
    point_implied_volatilities,
)
from math_for_quant.lower.derivatives_stochastic import (
    nested_quadratic_variation,
    terminal_singular_theta_energy,
    validate_novikov_exponential_moment,
)
from math_for_quant.lower.derivatives_hedging import simulate_hedging_distribution


class DerivativesV03Tests(unittest.TestCase):
    def test_quadratic_variation_uses_nested_partitions_and_real_failure_witness(self) -> None:
        increments = np.random.default_rng(17).normal(size=4096) / np.sqrt(4096.0)
        levels = nested_quadratic_variation(increments, block_sizes=(64, 16, 4, 1))
        self.assertEqual(tuple(levels), (64, 16, 4, 1))
        self.assertLess(abs(levels[1] - 1.0), abs(levels[64] - 1.0))
        self.assertGreater(validate_novikov_exponential_moment([100.0], [1.0]), 0.0)
        with self.assertRaisesRegex(ValueError, "Novikov condition"):
            terminal_singular_theta_energy(1.0, 0.0)

    def test_closed_form_and_pde_have_independently_bounded_error(self) -> None:
        closed = black_scholes_call(100.0, 100.0, 0.02, 0.2, 1.0)
        pde = implicit_fd_call(
            100.0, 100.0, 0.02, 0.2, 1.0,
            space_steps=240, time_steps=1200, spot_max=400.0,
        )
        self.assertLess(abs(pde - closed), 0.03)

    def test_point_inversion_is_separate_from_parametric_surface_calibration(self) -> None:
        spot, rate = 100.0, 0.02
        coefficients = np.asarray([0.035, 0.08, 0.006])
        nodes: list[SurfaceNode] = []
        for maturity in (0.5, 1.0, 1.5):
            forward = spot * np.exp(rate * maturity)
            for strike in (82.0, 93.0, 100.0, 112.0, 131.0):
                log_moneyness = np.log(strike / forward)
                total_variance = (
                    coefficients[0] * maturity
                    + coefficients[1] * log_moneyness**2
                    + coefficients[2] * maturity**2
                )
                sigma = np.sqrt(total_variance / maturity)
                nodes.append(
                    SurfaceNode(
                        strike=strike,
                        maturity=maturity,
                        price=black_scholes_call(spot, strike, rate, sigma, maturity),
                        weight=1.0,
                    )
                )
        point_vols = point_implied_volatilities(spot, rate, nodes)
        fit = fit_parametric_total_variance(spot, rate, nodes)
        self.assertEqual(len(point_vols), len(nodes))
        np.testing.assert_allclose(fit.coefficients, coefficients, atol=1e-9)
        self.assertLess(fit.maximum_price_error, 1e-9)
        self.assertLess(fit.library_coefficient_gap, 1e-12)

    def test_hedging_reports_multi_path_error_and_cost_distribution(self) -> None:
        result = simulate_hedging_distribution(
            spot=100.0, strike=100.0, rate=0.02, sigma=0.2, maturity=1.0,
            paths=256, steps=24, cost_rate=0.0005, seed=29,
        )
        self.assertEqual(result.paths, 256)
        self.assertGreater(result.no_cost_rmse, 0.0)
        self.assertGreater(result.mean_cost, 0.0)
        self.assertLessEqual(result.error_q05, result.error_q50)
        self.assertLessEqual(result.error_q50, result.error_q95)
        self.assertLess(result.summary_gap, 1e-12)

    def test_three_teaching_notebooks_and_real_data_protocol_pass(self) -> None:
        root = Path(__file__).resolve().parents[2]
        commands = (
            ("notebooks/lower/derivatives_stochastic.py", "evidence/derivatives-stochastic/oracle.json"),
            ("notebooks/lower/derivatives_numerics.py", "evidence/derivatives-numerics/oracle.json"),
            ("notebooks/lower/derivatives_hedging.py", "evidence/derivatives-hedging/oracle.json"),
            ("evidence/derivatives/validate_real_data.py", "evidence/derivatives/real-data-oracle.json"),
        )
        for source, oracle in commands:
            result = subprocess.run(
                [sys.executable, source, oracle], cwd=root, text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("passed", result.stdout)

    def test_route_report_is_rebuilt_exactly(self) -> None:
        from tools.build_derivatives_report import build_report

        root = Path(__file__).resolve().parents[2]
        expected = (root / "reports/derivatives-v03-summary.md").read_text(encoding="utf-8")
        self.assertEqual(build_report(), expected)


if __name__ == "__main__":
    unittest.main()
