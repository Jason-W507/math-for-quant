from __future__ import annotations

import json
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
    joint_order_price_beta,
    poisson_mle,
    queue_fill_probability,
)
from math_for_quant.lower.microstructure_route import build_route_report
from math_for_quant.lower.microstructure_simulation import (
    analyze_coinbase_trades,
    paired_event_simulation,
)
from math_for_quant.lower.microstructure_control_library import enumerate_execution
from math_for_quant.lower.microstructure_events_library import (
    library_joint_beta,
    library_poisson_mle,
    library_queue_fill_probability,
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
        self.assertTrue(np.isfinite(result.baseline_pnl))
        self.assertTrue(np.isfinite(result.control_pnl))

    def test_real_trade_snapshot_is_consumed(self) -> None:
        observed = analyze_coinbase_trades(
            ROOT / "data/real/coinbase-btc-usd-trades-2026-07-31.json"
        )
        self.assertEqual(observed["trades"], 20)
        self.assertEqual(observed["first_trade_id"], 1064791075)
        self.assertEqual(observed["last_trade_id"], 1064791094)
        self.assertGreater(observed["duration_seconds"], 0.0)
        self.assertTrue(np.isfinite(observed["order_price_beta"]))

    def test_three_notebooks_and_exact_route_report(self) -> None:
        for name in ("events", "control", "simulation"):
            result = subprocess.run(
                [sys.executable, f"notebooks/lower/microstructure_{name}.py"],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        expected = (ROOT / "reports/microstructure-v03-summary.md").read_text(encoding="utf-8")
        self.assertEqual(build_route_report(), expected)

    def test_public_route_command_build_scope_is_explicit(self) -> None:
        text = (ROOT / "tools/validate_microstructure_route.py").read_text(encoding="utf-8")
        self.assertIn('validate_route("microstructure")', text)


if __name__ == "__main__":
    unittest.main()
