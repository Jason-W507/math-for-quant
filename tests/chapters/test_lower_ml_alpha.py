from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LowerMlAlphaTests(unittest.TestCase):
    def run_oracle(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "notebooks/lower/ch03_ml_alpha.py", "evidence/lower-ch03/oracle.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_pipeline_produces_the_frozen_generalization_ledger(self) -> None:
        result = self.run_oracle()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "oracle=passed risks=(0.000000,0.040000,0.010000) "
            "models=(0.090000,0.040000,0.010000) sequence=(2,3,2,1,0.090000,0.010000) "
            "drift=(0.800000,1) calibration=(0.025000,0.160000) "
            "explanation=(0.333333,0) returns=(0.030000,0.024000) "
            "failures=(1,1,1,1,1,1)\n",
        )

    def test_notebook_is_only_a_guarded_module_wrapper(self) -> None:
        source = (ROOT / "notebooks/lower/ch03_ml_alpha.py").read_text(encoding="utf-8")
        self.assertIn('runpy.run_module("math_for_quant.lower.ml_alpha"', source)
        self.assertIn('if __name__ == "__main__":', source)
        self.assertNotIn("sys.path", source)

    def test_report_records_split_audit_drift_and_costs(self) -> None:
        report = (ROOT / "reports/lower-ch03-summary.md").read_text(encoding="utf-8")
        for marker in (
            "训练窗口",
            "验证窗口",
            "固定推理窗口",
            "随机切分：拒绝",
            "全样本预处理：拒绝",
            "漂移响应",
            "特征重要性不等于因果",
            "成本后收益：0.024",
        ):
            self.assertIn(marker, report)


if __name__ == "__main__":
    unittest.main()
