from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from math_for_quant.lower.portfolio_risk import (
    cost_aware_rebalance,
    empirical_var_es,
    factor_covariance,
    minimum_variance_two_asset,
    render_experiment_contract,
    validate_covariance,
)


ROOT = Path(__file__).resolve().parents[2]


class LowerPortfolioRiskTests(unittest.TestCase):
    def test_portfolio_risk_pipeline_matches_independent_oracle(self) -> None:
        result = subprocess.run(
            [sys.executable, "notebooks/lower/ch05_portfolio_risk.py", "evidence/lower-ch05/oracle.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "oracle=passed covariance=(0.040000,0.010000,0.090000) "
            "weights=(0.727273,0.272727) stability=(0.500000,0.250000,0.493506) "
            "rebalance=(0.600000,0.400000,0.400000,40.000000) "
            "risk=(2.000000,4.000000,7.000000) failures=(1,1,1)\n",
        )

    def test_two_asset_minimum_variance_has_hand_solution(self) -> None:
        weights = minimum_variance_two_asset(np.array([[0.04, 0.01], [0.01, 0.09]]))
        np.testing.assert_allclose(weights, np.array([8.0 / 11.0, 3.0 / 11.0]), atol=1e-12)

    def test_var_and_es_use_loss_tail_not_return_tail(self) -> None:
        var, es = empirical_var_es(
            np.array([-1.0, 0.0, 2.0, 4.0]), 0.75, minimum_tail_observations=1
        )
        self.assertEqual(var, 2.0)
        self.assertEqual(es, 4.0)
        repeated_var, repeated_es = empirical_var_es(
            np.array([0.0, 2.0, 2.0, 4.0]), 0.5, minimum_tail_observations=1
        )
        self.assertEqual(repeated_var, 2.0)
        self.assertEqual(repeated_es, 3.0)

    def test_rebalance_has_independent_turnover_and_cost_ledger(self) -> None:
        weights, turnover, cost = cost_aware_rebalance(
            np.array([0.08, 0.04]),
            np.array([[0.04, 0.01], [0.01, 0.09]]),
            np.array([0.4, 0.6]),
            risk_aversion=1.0,
            cost_rate=0.001,
            capital=100000.0,
            maximum_weight=0.6,
            tradable=np.array([1, 1]),
        )
        np.testing.assert_allclose(weights, np.array([0.6, 0.4]), atol=1e-12)
        self.assertAlmostEqual(turnover, abs(0.6 - 0.4) + abs(0.4 - 0.6), places=12)
        self.assertAlmostEqual(cost, 100000.0 * 0.001 * 0.4, places=12)

    def test_failures_have_distinct_stable_categories(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            validate_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))
        with self.assertRaisesRegex(ValueError, "no feasible portfolio"):
            cost_aware_rebalance(
                np.array([0.08, 0.04]), np.array([[0.04, 0.01], [0.01, 0.09]]),
                np.array([0.7, 0.3]), risk_aversion=1.0, cost_rate=0.001,
                capital=100000.0, maximum_weight=0.6, tradable=np.array([0, 1]),
            )
        with self.assertRaisesRegex(ValueError, "effective tail observations"):
            empirical_var_es(np.array([1.0, 2.0, 3.0]), 0.95)

    def test_tail_gate_uses_confidence_and_effective_tail_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "effective tail observations"):
            empirical_var_es(np.arange(100.0), 0.95)
        var, es = empirical_var_es(
            np.arange(100.0), 0.95, minimum_tail_observations=5
        )
        self.assertEqual(var, 94.0)
        self.assertAlmostEqual(es, 97.0)

    def test_rebalance_rejects_invalid_research_contract(self) -> None:
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        common = dict(
            expected_returns=np.array([0.08, 0.04]),
            covariance=covariance,
            current_weights=np.array([0.4, 0.6]),
            risk_aversion=1.0,
            cost_rate=0.001,
            capital=100000.0,
            maximum_weight=0.6,
            tradable=np.array([1, 1]),
        )
        for update, diagnostic in (
            ({"risk_aversion": 0.0}, "risk aversion"),
            ({"capital": 0.0}, "capital"),
            ({"cost_rate": -0.1}, "cost rate"),
            ({"grid_step": 0.3}, "divide one"),
        ):
            with self.subTest(diagnostic=diagnostic), self.assertRaisesRegex(ValueError, diagnostic):
                cost_aware_rebalance(**(common | update))

    def test_factor_model_rejects_illegal_variance_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            factor_covariance(np.ones((2, 1)), np.array([[-1.0]]), np.array([2.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            factor_covariance(np.array([[10.0], [1.0]]), np.array([[1.0]]), np.array([-0.01, 1.0]))

    def test_report_contract_is_derived_from_parameters(self) -> None:
        text = render_experiment_contract(
            {"confidence": 0.9, "ridge": 0.02, "risk_aversion": 3.0, "cost_rate": 0.004,
             "maximum_weight": 0.55, "grid_step": 0.05, "tradable_assets": 1}
        )
        for marker in ("0.9000", "0.020000", "3.0000", "0.004000", "0.5500", "0.0500", "tradable_assets=1"):
            self.assertIn(marker, text)

    def test_clean_copy_reproduces_the_research_package(self) -> None:
        clean = ROOT / "build" / "test-packages" / "lower-ch05-clean"
        if clean.exists():
            shutil.rmtree(clean)
        try:
            for relative in (
                "data/fixtures/lower-ch05.json",
                "evidence/lower-ch05/oracle.json",
                "reports/lower-ch05-summary.md",
                "src/math_for_quant/__init__.py",
                "src/math_for_quant/evidence.py",
                "src/math_for_quant/lower/__init__.py",
                "src/math_for_quant/lower/portfolio_risk.py",
            ):
                destination = clean / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(clean / "src")
            loaded = subprocess.run(
                [sys.executable, "-c", "import math_for_quant.lower.portfolio_risk as module; print(module.__file__)"],
                cwd=clean, env=environment, text=True, capture_output=True, check=False,
            )
            self.assertEqual(loaded.returncode, 0, loaded.stderr)
            self.assertTrue(Path(loaded.stdout.strip()).resolve().is_relative_to(clean.resolve()))
            result = subprocess.run(
                [sys.executable, "-m", "math_for_quant.lower.portfolio_risk", "evidence/lower-ch05/oracle.json"],
                cwd=clean, env=environment, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith("oracle=passed "))
        finally:
            if clean.exists():
                shutil.rmtree(clean)


if __name__ == "__main__":
    unittest.main()
