from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


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
            "alignment_rejected=1 unstable_rejected=1\n",
        )

    def test_multifactor_notebook_is_only_a_guarded_module_wrapper(self) -> None:
        source = (ROOT / "notebooks/lower/ch01_multifactor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('runpy.run_module("math_for_quant.lower.multifactor"', source)
        self.assertIn('if __name__ == "__main__":', source)
        self.assertNotIn("sys.path", source)


if __name__ == "__main__":
    unittest.main()
