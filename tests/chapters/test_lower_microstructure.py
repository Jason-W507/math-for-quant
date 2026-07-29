from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from math_for_quant.lower.microstructure import (
    optimal_execution,
    poisson_mle,
    replay,
    twap_schedule,
    validate_constant_intensity,
    validate_hawkes_stability,
    validate_information_time,
)


ROOT = Path(__file__).resolve().parents[2]


class LowerMicrostructureTests(unittest.TestCase):
    def test_event_replay_and_execution_pipeline_matches_oracle(self) -> None:
        result = subprocess.run(
            [sys.executable, "notebooks/lower/ch06_microstructure.py", "evidence/lower-ch06/oracle.json"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "oracle=passed point=(1.000000,-4.000000,1) "
            "book=(100.000000,1,101.000000,2,11) "
            "execution=(2,2,2,3,2,1,12.000000,17.000000) failures=(1,1,1,1,1,1)\n",
        )

    def test_poisson_mle_has_hand_calculated_oracle(self) -> None:
        intensity, log_likelihood = poisson_mle([0.5, 1.0, 1.5, 1.0])
        self.assertEqual(intensity, 1.0)
        self.assertEqual(log_likelihood, -4.0)

    def test_twap_schedule_preserves_inventory_exactly(self) -> None:
        self.assertEqual(twap_schedule(6, 3), [2, 2, 2])
        self.assertEqual(sum(twap_schedule(7, 3)), 7)

    def test_fifo_partial_fill_and_cancel_match_manual_ledger(self) -> None:
        events = [
            {"type": "add", "order_id": "b1", "side": "buy", "price": 100, "quantity": 5},
            {"type": "add", "order_id": "b2", "side": "buy", "price": 100, "quantity": 4},
            {"type": "add", "order_id": "a1", "side": "sell", "price": 101, "quantity": 6},
            {"type": "market", "side": "sell", "quantity": 7},
            {"type": "cancel", "order_id": "b2", "quantity": 1},
            {"type": "market", "side": "buy", "quantity": 4},
        ]
        book, fills = replay(events)
        self.assertEqual(fills, [("b1", 5, 100.0), ("b2", 2, 100.0), ("a1", 4, 101.0)])
        self.assertEqual(book.best("buy"), (100.0, 1))
        self.assertEqual(book.best("sell"), (101.0, 2))
        self.assertEqual(book.traded_quantity, 11)

    def test_dynamic_program_matches_independent_enumeration(self) -> None:
        schedule, cost = optimal_execution(6, 3, 1.0, 0.3, 3, 3)
        candidates = []
        for first in range(4):
            for second in range(4):
                third = 6 - first - second
                if 0 <= third <= 3:
                    objective = first**2 + second**2 + third**2 + 0.3 * ((6 - first) ** 2 + (6 - first - second) ** 2)
                    candidates.append((objective, [first, second, third]))
        expected_cost, expected_schedule = min(candidates)
        self.assertEqual(schedule, expected_schedule)
        self.assertEqual(cost, expected_cost)

    def test_failure_categories_are_distinct(self) -> None:
        with self.assertRaisesRegex(ValueError, "intraday seasonality"):
            validate_constant_intensity([8, 2], 2.0)
        with self.assertRaisesRegex(ValueError, "unstable Hawkes"):
            validate_hawkes_stability(1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "finite displayed liquidity"):
            optimal_execution(6, 3, 1.0, 0.3, 7, 3)
        with self.assertRaisesRegex(ValueError, "future price"):
            validate_information_time(10.0, 11.0)

    def test_clean_copy_reproduces_the_research_package(self) -> None:
        clean = ROOT / "build" / "test-packages" / "lower-ch06-clean"
        if clean.exists():
            shutil.rmtree(clean)
        try:
            for relative in (
                "data/fixtures/lower-ch06.json",
                "evidence/lower-ch06/oracle.json",
                "reports/lower-ch06-summary.md",
                "src/math_for_quant/__init__.py",
                "src/math_for_quant/evidence.py",
                "src/math_for_quant/lower/__init__.py",
                "src/math_for_quant/lower/microstructure.py",
            ):
                destination = clean / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(clean / "src")
            loaded = subprocess.run(
                [sys.executable, "-c", "import math_for_quant.lower.microstructure as module; print(module.__file__)"],
                cwd=clean, env=environment, text=True, capture_output=True, check=False,
            )
            self.assertEqual(loaded.returncode, 0, loaded.stderr)
            self.assertTrue(Path(loaded.stdout.strip()).resolve().is_relative_to(clean.resolve()))
            result = subprocess.run(
                [sys.executable, "-m", "math_for_quant.lower.microstructure", "evidence/lower-ch06/oracle.json"],
                cwd=clean, env=environment, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith("oracle=passed "))
        finally:
            if clean.exists():
                shutil.rmtree(clean)


if __name__ == "__main__":
    unittest.main()
