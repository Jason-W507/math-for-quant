from __future__ import annotations

import os
import subprocess
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

from math_for_quant.lower.derivatives import (
    call_delta,
    delta_hedge,
    implied_volatility,
    render_experiment_budget,
    validate_market_price_risk_energy,
    validate_novikov_exponential_moment,
    validate_risk_neutral_drift,
    validate_surface_constraints,
)


ROOT = Path(__file__).resolve().parents[2]


class LowerDerivativesTests(unittest.TestCase):
    def run_oracle(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "notebooks/lower/ch04_derivatives.py", "evidence/lower-ch04/oracle.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_pricing_calibration_and_hedging_pipeline_matches_oracle(self) -> None:
        result = self.run_oracle()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "oracle=passed ito=(0.860000,0.020000,0.020000) gbm=(109.229859,109.512790) "
            "prices=(8.916037,8.911086,8.909574,0.085601) calibration=(0.200000) "
            "greeks=(0.579260,0.579260,39.104269) surface=(0.000000000000,0.000000000000) "
            "hedge=(1.476783,1.391342,0.085442,0.084354) failures=(1,1,1,1,1,1)\n",
        )

    def test_measure_change_and_quote_failures_have_distinct_diagnostics(self) -> None:
        with self.assertRaisesRegex(ValueError, "wrong risk-neutral drift"):
            validate_risk_neutral_drift(0.08, 0.02, 0.2, -0.3)
        with self.assertRaisesRegex(ValueError, "arbitrage bounds"):
            implied_volatility(101.0, 100.0, 100.0, 0.02, 1.0)
        with self.assertRaisesRegex(ValueError, "not bracketed"):
            implied_volatility(100.0, 100.0, 100.0, 0.02, 1.0)

    def test_capstone_report_is_bound_to_the_pipeline(self) -> None:
        report = ROOT / "reports/lower-ch04-summary.md"
        self.assertTrue(report.is_file())
        text = report.read_text(encoding="utf-8")
        for marker in ("二次变差", "Black--Scholes", "二叉树", "Monte Carlo", "隐含波动率", "离散对冲", "不可声称"):
            self.assertIn(marker, text)

    def test_clean_copy_reproduces_the_research_package(self) -> None:
        clean = ROOT / "build" / "test-packages" / "lower-ch04-clean"
        if clean.exists():
            shutil.rmtree(clean)
        try:
            for relative in (
                "data/fixtures/lower-ch04.json",
                "evidence/lower-ch04/oracle.json",
                "reports/lower-ch04-summary.md",
                "src/math_for_quant/__init__.py",
                "src/math_for_quant/evidence.py",
                "src/math_for_quant/lower/__init__.py",
                "src/math_for_quant/lower/derivatives.py",
            ):
                destination = clean / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(clean / "src")
            loaded = subprocess.run(
                [sys.executable, "-c", "import math_for_quant.lower.derivatives as module; print(module.__file__)"],
                cwd=clean,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(loaded.returncode, 0, loaded.stderr)
            module_path = Path(loaded.stdout.strip()).resolve()
            self.assertTrue(module_path.is_relative_to(clean.resolve()), module_path)
            result = subprocess.run(
                [sys.executable, "-m", "math_for_quant.lower.derivatives", "evidence/lower-ch04/oracle.json"],
                cwd=clean,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith("oracle=passed "))
        finally:
            if clean.exists():
                shutil.rmtree(clean)

    def test_report_budget_is_derived_from_experiment_parameters(self) -> None:
        line = render_experiment_budget(
            {"tree_steps": 8, "tree_dt": 0.125, "mc_samples": 16, "seed": 7, "surface_nodes": 4}
        )
        self.assertIn("8 步", line)
        self.assertIn("0.125000", line)
        self.assertIn("16 路径", line)
        self.assertIn("seed=7", line)
        self.assertIn("4 个执行价/期限节点", line)

    def test_one_step_hedge_cost_has_an_independent_hand_ledger(self) -> None:
        no_cost, after_cost, drag, raw_cost = delta_hedge(100.0, 100.0, 0.02, 0.2, 1.0, np.array([0.0]), 0.0005)
        initial_cost = 0.0005 * call_delta(100.0, 100.0, 0.02, 0.2, 1.0) * 100.0
        self.assertAlmostEqual(raw_cost, initial_cost, places=12)
        self.assertAlmostEqual(drag, initial_cost * np.exp(0.02), places=12)
        self.assertAlmostEqual(no_cost - after_cost, drag, places=12)

    def test_nonuniform_strike_convexity_compares_neighboring_slopes(self) -> None:
        valid = [
            {"maturity": 1.0, "strike": 90.0, "price": 15.0},
            {"maturity": 1.0, "strike": 100.0, "price": 9.0},
            {"maturity": 1.0, "strike": 130.0, "price": 3.0},
        ]
        validate_surface_constraints(valid, rate=0.02, dividend_yield=0.0)
        invalid = [dict(node) for node in valid]
        invalid[1]["price"] = 13.0
        with self.assertRaisesRegex(ValueError, "butterfly convexity"):
            validate_surface_constraints(invalid, rate=0.02, dividend_yield=0.0)

    def test_novikov_condition_is_not_an_arbitrary_energy_threshold(self) -> None:
        self.assertTrue(validate_novikov_exponential_moment([100.0], [1.0]) > 0.0)
        with self.assertRaisesRegex(ValueError, "Novikov condition"):
            validate_novikov_exponential_moment([float("inf")], [1.0])
        with self.assertRaisesRegex(ValueError, "risk energy budget"):
            validate_market_price_risk_energy(2.0, 1.0)


if __name__ == "__main__":
    unittest.main()
