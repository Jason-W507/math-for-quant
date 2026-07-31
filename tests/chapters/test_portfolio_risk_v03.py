from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

from math_for_quant.lower.portfolio_estimation import (
    bootstrap_portfolio_volatility,
    factor_covariance,
    risk_contributions,
    risk_parity_weights,
    shrink_covariance,
)
from math_for_quant.lower.portfolio_optimization import (
    black_litterman_posterior,
    cvar_optimize,
    robust_cost_aware_rebalance,
)
from math_for_quant.lower.portfolio_optimization_library import enumerate_two_asset_cvar
from math_for_quant.lower.portfolio_real_data import real_tail_gate_status, run_portfolio_real_data
from math_for_quant.lower.portfolio_route import render_route_report, run_estimation, run_optimization, run_tail
from math_for_quant.lower.portfolio_tail import (
    empirical_tail_risk,
    nonlinear_portfolio_loss,
    reverse_stress_scale,
)
from tools.build_portfolio_risk_report import build_report


ROOT = Path(__file__).resolve().parents[2]


class PortfolioRiskV03Tests(unittest.TestCase):
    def test_three_teaching_notebooks_and_real_data_oracle_execute(self) -> None:
        commands = [
            [sys.executable, "notebooks/lower/portfolio_risk_estimation.py", "evidence/portfolio-risk-estimation/oracle.json"],
            [sys.executable, "notebooks/lower/portfolio_risk_optimization.py", "evidence/portfolio-risk-optimization/oracle.json"],
            [sys.executable, "notebooks/lower/portfolio_risk_tail.py", "evidence/portfolio-risk-tail/oracle.json"],
            [sys.executable, "evidence/portfolio-risk-real/validate_real_data.py", "evidence/portfolio-risk-real/real-data-oracle.json"],
        ]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_route_report_is_rebuilt_exactly(self) -> None:
        expected = (ROOT / "reports" / "portfolio-risk-v03-summary.md").read_text(encoding="utf-8")
        self.assertEqual(build_report(), expected)

    def test_risk_contributions_sum_and_risk_parity_equalizes_them(self) -> None:
        covariance = np.diag([0.04, 0.09, 0.16])
        weights = risk_parity_weights(covariance)
        contributions = risk_contributions(weights, covariance)
        self.assertAlmostEqual(float(contributions.sum()), float(weights @ covariance @ weights))
        np.testing.assert_allclose(
            contributions / contributions.sum(), np.full(3, 1.0 / 3.0), atol=1e-7
        )
        negatively_correlated = np.array(
            [[1.0, -0.1664, -0.4966], [-0.1664, 1.0, 0.0466], [-0.4966, 0.0466, 1.0]]
        )
        weights = risk_parity_weights(negatively_correlated)
        contributions = risk_contributions(weights, negatively_correlated)
        np.testing.assert_allclose(
            contributions / contributions.sum(), np.full(3, 1.0 / 3.0), atol=1e-7
        )

    def test_shrinkage_factor_model_and_bootstrap_have_independent_oracles(self) -> None:
        returns = np.array(
            [[0.01, 0.00], [0.02, 0.01], [-0.01, 0.02], [0.00, -0.01], [0.03, 0.01]]
        )
        sample = np.cov(returns, rowvar=False, ddof=1)
        target = np.diag(np.diag(sample))
        np.testing.assert_allclose(shrink_covariance(sample, target, 0.25), 0.75 * sample + 0.25 * target)
        loadings = np.array([[1.0], [0.5]])
        modeled = factor_covariance(loadings, np.array([[0.04]]), np.array([0.01, 0.02]))
        np.testing.assert_allclose(modeled, np.array([[0.05, 0.02], [0.02, 0.03]]))
        interval = bootstrap_portfolio_volatility(
            returns, np.array([0.6, 0.4]), samples=200, seed=17, confidence=0.90
        )
        self.assertLess(interval.lower, interval.point)
        self.assertLess(interval.point, interval.upper)

    def test_black_litterman_and_cvar_lp_are_executable(self) -> None:
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        mean, posterior_covariance = black_litterman_posterior(
            prior_mean=np.array([0.05, 0.04]),
            prior_covariance=covariance,
            views=np.array([[1.0, -1.0]]),
            view_returns=np.array([0.03]),
            view_covariance=np.array([[0.0025]]),
            tau=0.05,
        )
        self.assertGreater(mean[0] - mean[1], 0.01)
        self.assertTrue(np.all(np.linalg.eigvalsh(posterior_covariance) > 0.0))
        scenarios = np.array([[0.04, 0.01], [0.02, 0.00], [-0.03, 0.01], [-0.08, 0.02]])
        result = cvar_optimize(scenarios, confidence=0.75, maximum_weight=0.8)
        self.assertAlmostEqual(float(result.weights.sum()), 1.0)
        self.assertLessEqual(float(result.weights.max()), 0.8 + 1e-10)
        self.assertAlmostEqual(result.objective, result.recomputed_cvar, places=9)

    def test_robust_rebalance_validates_contract_and_tradability(self) -> None:
        result = robust_cost_aware_rebalance(
            expected_returns=np.array([0.08, 0.04]),
            covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
            current_weights=np.array([0.4, 0.6]),
            return_uncertainty=np.array([0.02, 0.01]),
            risk_aversion=1.0,
            uncertainty_penalty=0.5,
            cost_rate=0.001,
            capital=100_000.0,
            maximum_weight=0.7,
            tradable=np.array([1, 1]),
            grid_step=0.1,
        )
        self.assertAlmostEqual(float(result.weights.sum()), 1.0)
        self.assertAlmostEqual(result.cash_cost, 100_000.0 * 0.001 * result.turnover)
        with self.assertRaisesRegex(ValueError, "risk aversion"):
            robust_cost_aware_rebalance(
                np.array([0.08, 0.04]), np.eye(2), np.array([0.5, 0.5]),
                np.array([0.01, 0.01]), risk_aversion=0.0,
                uncertainty_penalty=1.0, cost_rate=0.0, capital=1.0,
                maximum_weight=1.0, tradable=np.array([1, 1]), grid_step=0.1,
            )

    def test_tail_gate_reports_pass_warn_reject_and_uncertainty(self) -> None:
        losses = np.linspace(-2.0, 8.0, 500)
        warned = empirical_tail_risk(
            losses, 0.95, minimum_tail_observations=20, warning_tail_observations=30,
            bootstrap_samples=200, seed=23,
        )
        self.assertEqual(warned.status, "warn")
        self.assertAlmostEqual(warned.effective_tail_observations, 25.0)
        self.assertGreater(warned.quantile_resolution, 0.0)
        self.assertLessEqual(warned.es_interval[0], warned.expected_shortfall)
        self.assertGreaterEqual(warned.es_interval[1], warned.expected_shortfall)
        passed = empirical_tail_risk(
            np.linspace(-2.0, 8.0, 1_000), 0.95,
            minimum_tail_observations=20, warning_tail_observations=30,
            bootstrap_samples=100, seed=29,
        )
        self.assertEqual(passed.status, "pass")
        self.assertAlmostEqual(passed.effective_tail_observations, 50.0)
        with self.assertRaisesRegex(ValueError, "effective tail observations"):
            empirical_tail_risk(losses[:100], 0.99, minimum_tail_observations=20)

    def test_reverse_stress_uses_nonlinear_repricing(self) -> None:
        shocks = np.array([-0.10, 0.04])
        linear = np.array([60.0, 40.0])
        gamma = np.array([80.0, -20.0])
        unit_loss = nonlinear_portfolio_loss(shocks, linear, gamma)
        self.assertNotAlmostEqual(unit_loss, float(-(linear @ shocks)))
        scale = reverse_stress_scale(
            shocks, linear, gamma, loss_threshold=10.0, maximum_scale=10.0
        )
        self.assertAlmostEqual(
            nonlinear_portfolio_loss(scale * shocks, linear, gamma), 10.0, places=8
        )
        peak = 0.5005
        curvature = 100_000.0
        narrow_threshold = curvature * peak * peak - 0.01
        first_crossing = reverse_stress_scale(
            np.array([1.0]), np.array([-2.0 * curvature * peak]),
            np.array([2.0 * curvature]), loss_threshold=narrow_threshold,
            maximum_scale=1.0,
        )
        self.assertLess(first_crossing, peak)
        self.assertAlmostEqual(
            nonlinear_portfolio_loss(
                np.array([first_crossing]), np.array([-2.0 * curvature * peak]),
                np.array([2.0 * curvature]),
            ),
            narrow_threshold,
            places=7,
        )

    def test_all_key_algorithms_have_independent_comparisons(self) -> None:
        estimation = run_estimation(
            __import__("json").loads((ROOT / "data/fixtures/portfolio-risk-estimation.json").read_text(encoding="utf-8"))
        )
        optimization = run_optimization(
            __import__("json").loads((ROOT / "data/fixtures/portfolio-risk-optimization.json").read_text(encoding="utf-8"))
        )
        tail = run_tail(
            __import__("json").loads((ROOT / "data/fixtures/portfolio-risk-tail.json").read_text(encoding="utf-8"))
        )
        for key in (
            "factor_library_gap", "bootstrap_library_point_gap",
            "rebalance_library_gap", "nonlinear_library_gap", "reverse_stress_library_gap",
        ):
            source = estimation if key in estimation else optimization if key in optimization else tail
            self.assertLess(float(source[key]), 1e-8, key)
        self.assertLess(float(estimation["bootstrap_library_interval_gap"]), 0.002)

    def test_real_data_track_is_consumed_by_portfolio_route(self) -> None:
        observed = run_portfolio_real_data(
            ROOT / "data/real/stat-arb-us-macro-1999q4-2009q3.json"
        )
        self.assertEqual(observed["rows"], 40)
        self.assertEqual(observed["growth_rows"], 39)
        self.assertGreater(observed["covariance_trace"], 0.0)
        self.assertAlmostEqual(
            observed["risk_parity_weight_1"] + observed["risk_parity_weight_2"], 1.0
        )
        self.assertAlmostEqual(observed["cvar_weight_1"] + observed["cvar_weight_2"], 1.0)
        self.assertEqual(observed["tail_status"], "reject")
        self.assertEqual(real_tail_gate_status(np.linspace(-1.0, 1.0, 100), 0.75), "warn")
        self.assertEqual(real_tail_gate_status(np.linspace(-1.0, 1.0, 160), 0.75), "pass")

    def test_report_uses_observed_tail_contract_not_hardcoded_labels(self) -> None:
        import json
        estimation = run_estimation(json.loads((ROOT / "data/fixtures/portfolio-risk-estimation.json").read_text(encoding="utf-8")))
        optimization = run_optimization(json.loads((ROOT / "data/fixtures/portfolio-risk-optimization.json").read_text(encoding="utf-8")))
        fixture = json.loads((ROOT / "data/fixtures/portfolio-risk-tail.json").read_text(encoding="utf-8"))
        fixture.update({"confidence": 0.90, "warning_tail_observations": 40, "loss_threshold": 5.0})
        tail = run_tail(fixture)
        real_data = run_portfolio_real_data(
            ROOT / "data/real/stat-arb-us-macro-1999q4-2009q3.json"
        )
        report = render_route_report(estimation, optimization, tail, real_data)
        self.assertIn("90% VaR/ES", report)
        self.assertIn(f"状态 {tail['tail_status']}", report)
        self.assertIn("损失达到 5", report)

    def test_enumerated_cvar_rejects_invalid_or_infeasible_contracts(self) -> None:
        scenarios = np.array([[0.01, -0.01], [-0.02, 0.02]])
        with self.assertRaisesRegex(ValueError, "confidence"):
            enumerate_two_asset_cvar(scenarios, confidence=1.0, grid_step=0.1, maximum_weight=0.8)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            enumerate_two_asset_cvar(scenarios, confidence=0.75, grid_step=0.1, maximum_weight=0.4)


if __name__ == "__main__":
    unittest.main()
