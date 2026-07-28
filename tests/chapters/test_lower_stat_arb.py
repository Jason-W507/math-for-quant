from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LowerStatArbTests(unittest.TestCase):
    def run_oracle(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "notebooks/lower/ch02_stat_arb.py",
                "evidence/lower-ch02/oracle.json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_known_dynamic_models_produce_the_independent_ledger(self) -> None:
        result = self.run_oracle()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "oracle=passed ar_phi=0.600000 misspec_acf=0.600000 "
            "kalman=(0.800000,0.960000,0.200000) change=(4,0,0) "
            "cointegration_spread=0.000000 walk_forward=passed "
            "returns=(0.050000,0.040000) failures=(1,1,1)\n",
        )

    def test_notebook_is_only_a_guarded_module_wrapper(self) -> None:
        source = (ROOT / "notebooks/lower/ch02_stat_arb.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('runpy.run_module("math_for_quant.lower.stat_arb"', source)
        self.assertIn('if __name__ == "__main__":', source)
        self.assertNotIn("sys.path", source)

    def test_report_preserves_rolling_boundaries_and_failure_analysis(self) -> None:
        report = (ROOT / "reports/lower-ch02-summary.md").read_text(encoding="utf-8")
        for marker in (
            "训练窗口",
            "验证窗口",
            "交易窗口",
            "检测延迟：0",
            "误报：0",
            "随机切分：拒绝",
            "全样本标准化：拒绝",
            "历史相关性不等于稳定套利关系",
        ):
            self.assertIn(marker, report)


if __name__ == "__main__":
    unittest.main()
