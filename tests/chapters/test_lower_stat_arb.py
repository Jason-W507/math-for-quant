from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from math_for_quant.lower.stat_arb import (
    detect_alarms,
    evaluate_alarms,
    validate_online_state,
    validate_scaler,
    validate_split,
    validate_walk_forward,
)


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
            "oracle=passed ar_phi=0.600000 misspec_acf=1.000000 "
            "dynamic=(0.400000,0.700000,0.800000,0.500000) "
            "kalman=(0.800000,0.960000,0.200000) change=(4,0,0,0) sensitivity=(2,2,1) "
            "cointegration=(-0.597614,1) rolling=(2,0.000000) "
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

    def test_change_metrics_use_truth_and_expose_threshold_sensitivity(self) -> None:
        values = np.asarray([0.0, 0.1, -0.1, 0.0, 3.0, 3.1, 2.9, 3.0])
        base = detect_alarms(values, 2, 2.5)
        low = detect_alarms(values, 2, 0.05)
        high = detect_alarms(values, 2, 3.5)
        self.assertEqual(evaluate_alarms(base, 4), (4, 4, 0, 0, 0))
        self.assertEqual(evaluate_alarms(low, 4), (2, 4, 0, 2, 0))
        self.assertEqual(evaluate_alarms(high, 4), (-1, -1, -1, 0, 1))
        self.assertEqual(detect_alarms(values[:4], 2, 0.05), [2, 3])
        self.assertEqual([alarm for alarm in low if alarm < 4], [2, 3])

    def test_protocol_validators_reject_actual_contaminated_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "random or overlapping"):
            validate_split([0, 2, 4], [1, 3], [5, 6])
        with self.assertRaisesRegex(ValueError, "after the training"):
            validate_scaler("2024-12", "2019-12")
        with self.assertRaisesRegex(ValueError, "future observations"):
            validate_online_state("2022-02-28", "2022-01-31")
        with self.assertRaisesRegex(ValueError, "runs backward"):
            validate_walk_forward(
                [
                    {"name": "train", "start": "2020-01", "end": "2019-12"},
                    {"name": "validation", "start": "2021-01", "end": "2021-12"},
                    {"name": "trade", "start": "2022-01", "end": "2021-12"},
                ]
            )


if __name__ == "__main__":
    unittest.main()
