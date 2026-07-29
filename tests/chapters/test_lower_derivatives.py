from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from math_for_quant.lower.derivatives import implied_volatility, validate_risk_neutral_drift


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
            "greeks=(0.579260,0.579260,39.104269) hedge=(1.417254,0.059026) failures=(1,1,1,1,1)\n",
        )

    def test_measure_change_and_quote_failures_have_distinct_diagnostics(self) -> None:
        with self.assertRaisesRegex(ValueError, "wrong risk-neutral drift"):
            validate_risk_neutral_drift(0.08, 0.02, 0.2, -0.3)
        with self.assertRaisesRegex(ValueError, "arbitrage bounds"):
            implied_volatility(101.0, 100.0, 100.0, 0.02, 1.0)

    def test_capstone_report_is_bound_to_the_pipeline(self) -> None:
        report = ROOT / "reports/lower-ch04-summary.md"
        self.assertTrue(report.is_file())
        text = report.read_text(encoding="utf-8")
        for marker in ("二次变差", "Black--Scholes", "二叉树", "Monte Carlo", "隐含波动率", "离散对冲", "不可声称"):
            self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
