from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from math_for_quant.lower.microstructure_control import (
    execution_path,
    inventory_quotes,
    market_making_cycle,
    optimal_execution,
)
from math_for_quant.lower.microstructure_events import (
    OrderBook,
    hawkes_log_likelihood,
    joint_order_price_beta,
    poisson_mle,
    queue_fill_probability,
    seasonally_adjusted_poisson_mle,
)
from math_for_quant.lower.microstructure_route import build_route_report
from math_for_quant.lower.notebook_evidence import assert_expected
from math_for_quant.lower.microstructure_simulation import (
    analyze_sec_order_placement,
    paired_event_simulation,
)
from math_for_quant.lower.microstructure_control_library import enumerate_execution
from math_for_quant.lower.microstructure_events_library import (
    library_hawkes_log_likelihood,
    library_joint_beta,
    library_poisson_mle,
    library_queue_fill_probability,
    library_seasonal_poisson_mle,
)
from math_for_quant.lower.microstructure_simulation_library import (
    vectorized_feedback_ledger,
)


ROOT = Path(__file__).resolve().parents[2]


class MicrostructureV03Tests(unittest.TestCase):
    def test_poisson_queue_and_joint_price_oracles(self) -> None:
        intensity, log_likelihood = poisson_mle([0.5, 1.0, 1.5, 1.0])
        self.assertEqual((intensity, log_likelihood), (1.0, -4.0))
        probability = queue_fill_probability(
            queue_ahead=2, own_quantity=1, depletion_intensity=3.0, horizon=1.0
        )
        self.assertAlmostEqual(probability, 1.0 - np.exp(-3.0) * (1.0 + 3.0 + 4.5))
        beta = joint_order_price_beta(
            np.array([-1.0, -1.0, 1.0, 1.0]), np.array([-0.2, -0.1, 0.1, 0.2])
        )
        self.assertAlmostEqual(beta, 0.15)
        self.assertAlmostEqual(library_poisson_mle([0.5, 1.0, 1.5, 1.0]), intensity, places=6)
        self.assertAlmostEqual(library_queue_fill_probability(3, 3.0), probability)
        self.assertAlmostEqual(
            library_joint_beta(np.array([-1.0, -1.0, 1.0, 1.0]), np.array([-0.2, -0.1, 0.1, 0.2])),
            beta,
        )
        self.assertAlmostEqual(library_poisson_mle([0.001, 0.001]), 1000.0)
        self.assertAlmostEqual(library_poisson_mle([1e308]), 1e-308)
        seasonal_rate, _ = library_seasonal_poisson_mle(
            [0.001, 0.001], [1.0, 1.0]
        )
        self.assertAlmostEqual(seasonal_rate, 1000.0)
        seasonal_low, _ = library_seasonal_poisson_mle([1e308], [1.0])
        self.assertAlmostEqual(seasonal_low, 1e-308)

    def test_hawkes_and_seasonality_have_frozen_numerical_evidence(self) -> None:
        times = np.array([0.2, 0.7, 1.4, 2.0])
        value = hawkes_log_likelihood(
            times, 0.8, 0.3, 1.2, horizon=2.5, initial_excitation=0.1
        )
        self.assertAlmostEqual(value, -2.9403405029867584)
        self.assertAlmostEqual(
            value,
            library_hawkes_log_likelihood(times, 0.8, 0.3, 1.2, 2.5, 0.1),
        )
        intensity, residuals = seasonally_adjusted_poisson_mle(
            [0.5, 1.0, 0.5, 1.0], [1.0, 0.5, 2.0, 1.0]
        )
        self.assertAlmostEqual(intensity, 4.0 / 3.0)
        self.assertAlmostEqual(float(np.mean(residuals)), 1.0)
        with self.assertRaisesRegex(ValueError, "horizon"):
            hawkes_log_likelihood(times, 0.8, 0.3, 1.2, horizon=1.9)
        with self.assertRaisesRegex(ValueError, "finite"):
            hawkes_log_likelihood(
                np.array([0.2, math.nan]), 0.8, 0.3, 1.2, horizon=2.5
            )
        with self.assertRaisesRegex(ValueError, "window"):
            hawkes_log_likelihood(
                np.array([-0.2, 0.7]), 0.8, 0.3, 1.2, horizon=1.0
            )

    def test_nonfinite_oracles_and_library_inputs_are_rejected(self) -> None:
        oracle = {"absolute_tolerance": 1e-8, "expected": {"x": 123.0}}
        with self.assertRaisesRegex(SystemExit, "nonfinite"):
            assert_expected({"x": math.nan}, oracle)
        for kwargs in (
            dict(inventory=1, steps=-1, temporary_impact=1.0, permanent_impact=0.0,
                 inventory_risk=0.0, maximum_slice=1),
            dict(inventory=1, steps=1, temporary_impact=math.nan, permanent_impact=0.0,
                 inventory_risk=0.0, maximum_slice=1),
        ):
            with self.assertRaisesRegex(ValueError, "invalid"):
                enumerate_execution(**kwargs)

    def test_fifo_queue_position_partial_fill_cancel_and_no_fill(self) -> None:
        book = OrderBook()
        book.add("b1", "buy", 100.0, 3)
        book.add("b2", "buy", 100.0, 2)
        self.assertEqual(book.queue_ahead("b2"), 3)
        fills, unfilled = book.market("sell", 4, allow_partial=True)
        self.assertEqual(fills, [("b1", 3, 100.0), ("b2", 1, 100.0)])
        self.assertEqual(unfilled, 0)
        book.cancel("b2", 1)
        fills, unfilled = book.market("sell", 1, allow_partial=True)
        self.assertEqual(fills, [])
        self.assertEqual(unfilled, 1)
        book.assert_invariants()

    def test_execution_has_temporary_permanent_impact_random_completion_and_stop(self) -> None:
        result = execution_path(
            schedule=[3, 3, 3], fill_rates=[1.0, 0.5, 0.0], initial_inventory=9,
            arrival_price=100.0, temporary_impact=0.2, permanent_impact=0.1,
            stop_after=3,
        )
        self.assertEqual(result.filled, [3, 2, 0])
        self.assertEqual(result.remaining, 4)
        self.assertTrue(result.stopped)
        self.assertGreater(result.implementation_shortfall, 0.0)
        schedule, cost = optimal_execution(
            inventory=6, steps=3, temporary_impact=1.0, permanent_impact=0.2,
            inventory_risk=0.3, maximum_slice=3,
        )
        self.assertEqual(sum(schedule), 6)
        self.assertGreater(cost, 0.0)
        library_schedule, library_cost = enumerate_execution(
            inventory=6, steps=3, temporary_impact=1.0, permanent_impact=0.2,
            inventory_risk=0.3, maximum_slice=3,
        )
        self.assertEqual(library_schedule, schedule)
        self.assertAlmostEqual(library_cost, cost)

    def test_market_making_inventory_updates_feed_back_to_quotes(self) -> None:
        bid0, ask0 = inventory_quotes(100.0, 0, 0.5, 0.1)
        path = market_making_cycle(
            midprices=[100.0, 100.0], fills=["bid", "none"], initial_inventory=0,
            half_spread=0.5, inventory_skew=0.1,
        )
        self.assertEqual(path.inventories, [0, 1, 1])
        self.assertLess(path.bids[1], bid0)
        self.assertLess(path.asks[1], ask0)

    def test_paired_simulation_preserves_random_source_and_compares_controls(self) -> None:
        result = paired_event_simulation(seed=59, events=500, base_intensity=2.0)
        self.assertEqual(result.events, 500)
        self.assertEqual(result.baseline_signs_hash, result.control_signs_hash)
        self.assertEqual(result.baseline_random_hash, result.control_random_hash)
        self.assertTrue(np.isfinite(result.baseline_pnl))
        self.assertTrue(np.isfinite(result.control_pnl))
        self.assertNotEqual(result.baseline_pnl, result.control_pnl)
        self.assertNotEqual(result.baseline_fills, result.control_fills)
        self.assertNotEqual(
            result.baseline_ending_inventory, result.control_ending_inventory
        )
        for kwargs in (
            dict(seed=1, events=10, base_intensity=math.nan),
            dict(seed=1, events=10, base_intensity=math.inf),
            dict(seed=1, events=10.5, base_intensity=1.0),
        ):
            with self.assertRaisesRegex(ValueError, "invalid"):
                paired_event_simulation(**kwargs)

    def test_vectorized_feedback_ledger_reconstructs_and_validates_trace(self) -> None:
        observed = vectorized_feedback_ledger(
            np.array([1, -1, 1]),
            np.array([101.0, 99.0, 102.0]),
            np.array([True, True, False]),
            100.0,
        )
        self.assertEqual(observed, (2.0, 2, 0, 1))
        invalid_cases = (
            (np.array([1, 0]), np.array([101.0, 99.0]), np.array([True, False]), 100.0),
            (np.array([1, -1]), np.array([math.nan, 99.0]), np.array([True, False]), 100.0),
            (np.array([1, -1]), np.array([101.0, 99.0]), np.array([1, 0]), 100.0),
            (np.array([1, -1]), np.array([101.0, 99.0]), np.array([True, False]), math.inf),
        )
        for arguments in invalid_cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "feedback ledger"):
                vectorized_feedback_ledger(*arguments)

    def test_public_domain_sec_snapshot_is_consumed(self) -> None:
        observed = analyze_sec_order_placement(
            ROOT / "data/real/sec-order-placement-2014.json"
        )
        self.assertEqual(observed["categories"], 5)
        self.assertAlmostEqual(observed["event_share_sum"], 1.0)
        self.assertGreater(observed["weighted_cancel_to_trade"], 0.0)
        self.assertTrue(0.0 < observed["implied_execution_probability"] < 1.0)
        with self.assertRaisesRegex(ValueError, "declared domains"):
            analyze_sec_order_placement(
                ROOT / "tests/fixtures/sec-order-placement-invalid.json"
            )

    def test_three_notebooks_and_exact_route_report(self) -> None:
        for name in ("events", "control", "simulation"):
            result = subprocess.run(
                [sys.executable, f"notebooks/lower/microstructure_{name}.py"],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        expected = (ROOT / "reports/microstructure-v03-summary.md").read_text(encoding="utf-8")
        self.assertEqual(build_route_report(), expected)

    def test_public_route_command_runs_end_to_end(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/validate_microstructure_route.py"],
            cwd=ROOT, text=True, capture_output=True, check=False, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("microstructure-route=passed publications=lower+shared-solutions", result.stdout)


if __name__ == "__main__":
    unittest.main()
