from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from math_for_quant.lower.multifactor import (
    null_search_p_values,
    protocol_metrics,
    ranks,
)


ROOT = Path(__file__).resolve().parents[2]


class LowerMultifactorTests(unittest.TestCase):
    def run_oracle(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "notebooks/lower/ch01_multifactor.py",
                "evidence/lower-ch01/oracle.json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_known_signal_panel_produces_the_independent_research_ledger(self) -> None:
        result = self.run_oracle()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "oracle=passed fm_mean=0.020000 fm_se=0.002000 "
            "ic=1.000000 rank_ic=1.000000 neutral_max=0.000000 "
            "tests=(20,2,0) returns=(0.071000,0.067000,0.061000) "
            "split_ic=(0.583807,0.711369,0.726531) "
            "decay_ic=(0.646657,0.445136,0.186461) "
            "scope=global+a-share-boundary "
            "alignment_rejected=1 unstable_rejected=1\n",
        )

    def test_average_ranks_make_tied_rank_ic_permutation_invariant(self) -> None:
        values = np.asarray([2.0, 1.0, 1.0, 3.0])
        permutation = np.asarray([2, 0, 3, 1])
        expected = np.asarray([3.0, 1.5, 1.5, 4.0])
        np.testing.assert_array_equal(ranks(values), expected)
        np.testing.assert_array_equal(ranks(values[permutation]), expected[permutation])

    def test_null_search_generates_the_frozen_false_positive_example(self) -> None:
        p_values = null_search_p_values(seed=4, observations=60, attempts=20)
        self.assertEqual(len(p_values), 20)
        self.assertEqual(sum(value < 0.05 for value in p_values), 2)
        self.assertAlmostEqual(p_values[2], 0.0276906432, places=9)
        self.assertAlmostEqual(sorted(p_values)[0], 0.0276906432, places=9)
        self.assertAlmostEqual(sorted(p_values)[1], 0.0378466680, places=9)

    def test_protocol_panel_computes_split_and_decay_evidence(self) -> None:
        split_ic, decay_ic = protocol_metrics(
            seed=41,
            window_observations=[48, 24, 36],
            decay_strengths=[0.5, 0.3, 0.1],
            noise_scale=0.5,
        )
        self.assertEqual(len(split_ic), 3)
        self.assertEqual(len(decay_ic), 3)
        self.assertGreater(split_ic[0], 0.5)
        self.assertGreater(split_ic[1], 0.5)
        self.assertGreater(split_ic[2], 0.5)
        self.assertGreater(decay_ic[0], decay_ic[1])
        self.assertGreater(decay_ic[1], decay_ic[2])

    def test_research_report_records_splits_decay_and_market_scope(self) -> None:
        report = (ROOT / "reports/lower-ch01-summary.md").read_text(encoding="utf-8")
        for marker in (
            "训练窗口",
            "选择窗口",
            "最终测试窗口",
            "衰减滞后：1、3、6 个月",
            "尝试次数：20",
            "全球核心协议",
            "A 股制度边界",
        ):
            self.assertIn(marker, report)

    def test_multifactor_notebook_is_only_a_guarded_module_wrapper(self) -> None:
        source = (ROOT / "notebooks/lower/ch01_multifactor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('runpy.run_module("math_for_quant.lower.multifactor"', source)
        self.assertIn('if __name__ == "__main__":', source)
        self.assertNotIn("sys.path", source)


if __name__ == "__main__":
    unittest.main()
