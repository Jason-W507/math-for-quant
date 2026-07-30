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
from math_for_quant.lower.derivatives import validate_surface_constraints
from math_for_quant.lower.derivatives_numerics_library import (
    library_black_scholes_call,
    library_binomial_call,
    library_implicit_fd_call,
    library_quadrature_call,
)
from math_for_quant.lower.derivatives_stochastic import (
    nested_quadratic_variation,
    terminal_singular_theta_energy,
    validate_novikov_exponential_moment,
)
from math_for_quant.lower.derivatives_hedging import (
    greek_convergence,
    simulate_hedging_distribution,
)
from math_for_quant.lower.derivatives_stochastic_library import (
    measure_change_density_gap,
)
from math_for_quant.lower.derivatives_route import run_numerics


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

    def test_mature_library_pricing_paths_agree_with_transparent_paths(self) -> None:
        closed = black_scholes_call(100.0, 100.0, 0.02, 0.2, 1.0)
        self.assertAlmostEqual(
            library_black_scholes_call(100.0, 100.0, 0.02, 0.2, 1.0),
            closed,
            places=12,
        )
        self.assertLess(
            abs(library_binomial_call(100.0, 100.0, 0.02, 0.2, 1.0, 256) - closed),
            0.02,
        )
        self.assertLess(
            abs(
                library_implicit_fd_call(
                    100.0, 100.0, 0.02, 0.2, 1.0,
                    space_steps=240, time_steps=1200, spot_max=400.0,
                )
                - closed
            ),
            0.03,
        )
        price, quadrature_error = library_quadrature_call(
            100.0, 100.0, 0.02, 0.2, 1.0
        )
        self.assertLess(abs(price - closed), 1e-8)
        self.assertLess(quadrature_error, 1e-8)

    def test_forward_normalized_calendar_gate_supports_dividends(self) -> None:
        nodes = []
        for maturity, forward, discount, normalized_prices in (
            (0.5, 98.0, 0.99, (0.24, 0.12, 0.04)),
            (1.0, 96.0, 0.97, (0.26, 0.14, 0.06)),
        ):
            for moneyness, normalized_price in zip((0.9, 1.0, 1.1), normalized_prices):
                nodes.append(
                    {
                        "maturity": maturity,
                        "strike": moneyness * forward,
                        "price": normalized_price * discount * forward,
                        "forward": forward,
                        "discount_factor": discount,
                    }
                )
        validate_surface_constraints(
            nodes,
            rate=-0.01,
            dividend_yield=0.03,
            calendar_mode="forward-normalized",
        )

    def test_route_splits_pde_errors_and_checks_greeks(self) -> None:
        root = Path(__file__).resolve().parents[2]
        fixture = __import__("json").loads(
            (root / "data/fixtures/derivatives-numerics.json").read_text(encoding="utf-8")
        )
        observed = run_numerics(fixture)
        for key in (
            "closed_library_gap", "tree_library_gap", "pde_library_gap",
            "mc_library_gap", "pde_space_gap", "pde_time_gap", "pde_boundary_gap",
            "forward_calendar_passed",
        ):
            self.assertIn(key, observed)
        self.assertEqual(observed["forward_calendar_passed"], 1)
        self.assertLessEqual(
            observed["mc_library_gap"], 2.0 * observed["mc_standard_error"]
        )
        greeks = greek_convergence(
            100.0, 100.0, 0.02, 0.2, 1.0, steps=(1.0, 0.5, 0.25, 0.125)
        )
        self.assertLess(greeks.delta_gap, 1e-5)
        self.assertLess(greeks.gamma_gap, 1e-5)
        self.assertLess(greeks.vega_gap, 1e-3)
        self.assertEqual(greeks.steps, (1.0, 0.5, 0.25, 0.125))
        self.assertLess(greeks.delta_errors[-1], greeks.delta_errors[0])
        self.assertLess(greeks.gamma_errors[-1], greeks.gamma_errors[0])
        self.assertLess(greeks.vega_errors[-1], greeks.vega_errors[0])
        self.assertLess(measure_change_density_gap(theta=0.3, observation=0.7), 1e-12)

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
