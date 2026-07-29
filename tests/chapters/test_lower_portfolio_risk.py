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
    minimum_variance_two_asset,
    reject_untradable_change,
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
            "weights=(0.727273,0.272727,0.700000,0.300000) "
            "rebalance=(0.600000,0.400000,0.400000,40.000000) "
            "risk=(2.000000,4.000000,7.000000) failures=(1,1,1)\n",
        )

    def test_two_asset_minimum_variance_has_hand_solution(self) -> None:
        weights = minimum_variance_two_asset(np.array([[0.04, 0.01], [0.01, 0.09]]))
        np.testing.assert_allclose(weights, np.array([8.0 / 11.0, 3.0 / 11.0]), atol=1e-12)

    def test_var_and_es_use_loss_tail_not_return_tail(self) -> None:
        var, es = empirical_var_es(np.array([-1.0, 0.0, 2.0, 4.0]), 0.75)
        self.assertEqual(var, 2.0)
        self.assertEqual(es, 4.0)

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
        with self.assertRaisesRegex(ValueError, "untradable"):
            reject_untradable_change(np.array([0.4, 0.6]), np.array([0.5, 0.5]), np.array([0, 1]))
        with self.assertRaisesRegex(ValueError, "at least four"):
            empirical_var_es(np.array([1.0, 2.0, 3.0]), 0.95)

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
