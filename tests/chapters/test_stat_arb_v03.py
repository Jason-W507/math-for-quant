from __future__ import annotations

import unittest
from unittest import mock
import hashlib
import json
from pathlib import Path

import numpy as np

from math_for_quant.lower.stat_arb_estimation import (
    kalman_filter_and_smooth,
    regime_filter,
)
from math_for_quant.lower.stat_arb_models import (
    engle_granger,
    fit_ecm,
    johansen_rank,
    ou_diagnostics,
)
from math_for_quant.lower.stat_arb_research import (
    ExecutionPolicy,
    build_forecast_ledger,
    validate_purged_walk_forward,
)
from math_for_quant.lower.stat_arb_library import cross_check_long_run
from tools import validate_stat_arb_route


ROOT = Path(__file__).resolve().parents[2]


class StatArbV03Tests(unittest.TestCase):
    def test_long_run_relation_feeds_ecm_and_ou_diagnostics(self) -> None:
        x = np.cumsum(np.resize(np.asarray([1.0, 2.0, 1.5]), 30))
        residual = np.asarray([0.55**i for i in range(x.size)])
        y = 1.5 + 2.0 * x + residual
        relation = engle_granger(y, x)
        self.assertAlmostEqual(relation.slope, 2.0, delta=0.02)
        self.assertLess(relation.residual_adf_statistic, -2.0)
        self.assertEqual(johansen_rank(np.column_stack([y, x])), 1)

        ecm = fit_ecm(y, x, relation)
        self.assertLess(ecm.adjustment_speed, 0.0)
        diagnostics = ou_diagnostics(relation.residuals, step=1.0)
        self.assertGreater(diagnostics.half_life, 0.0)
        self.assertGreater(diagnostics.expected_first_passage, 0.0)

    def test_filter_and_smoother_have_different_information_sets(self) -> None:
        observations = np.asarray([0.0, 0.2, 2.8, 3.1])
        states = kalman_filter_and_smooth(
            observations,
            transition=1.0,
            observation_loading=1.0,
            process_variance=0.1,
            observation_variance=0.2,
            initial_mean=0.0,
            initial_variance=1.0,
        )
        self.assertEqual(states.filtered.shape, observations.shape)
        self.assertNotAlmostEqual(states.filtered[1], states.smoothed[1])

        probabilities = regime_filter(
            observations,
            transition=np.asarray([[0.95, 0.05], [0.05, 0.95]]),
            means=np.asarray([0.0, 3.0]),
            variances=np.asarray([0.2, 0.2]),
            initial=np.asarray([0.5, 0.5]),
        )
        self.assertGreater(probabilities[-1, 1], 0.99)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_purge_and_embargo_are_enforced_at_the_label_boundary(self) -> None:
        validate_purged_walk_forward(
            train_indices=range(0, 6),
            validation_indices=range(8, 10),
            trade_indices=range(12, 14),
            label_horizon=2,
            embargo=2,
        )
        with self.assertRaisesRegex(ValueError, "purge"):
            validate_purged_walk_forward(
                train_indices=range(0, 7),
                validation_indices=range(8, 10),
                trade_indices=range(12, 14),
                label_horizon=2,
                embargo=2,
            )

    def test_forecasts_determine_positions_fills_and_net_returns(self) -> None:
        ledger = build_forecast_ledger(
            forecasts=np.asarray([0.8, -0.7, 0.1, 0.9]),
            realized_returns=np.asarray([0.02, -0.01, 0.03, 0.04]),
            fill_fractions=np.asarray([1.0, 0.5, 1.0, 0.0]),
            policy=ExecutionPolicy(
                entry_threshold=0.5,
                position_limit=1.0,
                holding_period=1,
                rebalance_every=1,
                cost_per_unit_turnover=0.002,
            ),
        )
        np.testing.assert_allclose(ledger.target_positions, [1.0, -1.0, 0.0, 1.0])
        np.testing.assert_allclose(ledger.filled_positions, [1.0, -0.5, 0.0, 0.0])
        self.assertAlmostEqual(ledger.gross_return, 0.025)
        self.assertAlmostEqual(ledger.turnover, 3.0)
        self.assertAlmostEqual(ledger.net_return, 0.019)

    def test_mature_library_cross_checks_the_public_snapshot(self) -> None:
        snapshot = ROOT / "data/real/stat-arb-us-macro-1999q4-2009q3.json"
        oracle = json.loads(
            (ROOT / "evidence/stat-arb/real-data-oracle.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            oracle["fixture"]["sha256"],
        )
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        y = np.log([row["realgdp"] for row in data["rows"]])
        x = np.log([row["realcons"] for row in data["rows"]])
        check = cross_check_long_run(y, x)
        self.assertAlmostEqual(check.transparent_slope, check.library_slope)
        self.assertAlmostEqual(check.transparent_adf, check.library_adf)
        self.assertEqual(check.johansen_rank, 1)

    def test_public_route_command_builds_only_lower_and_shared_solutions(self) -> None:
        completed = __import__("subprocess").CompletedProcess([], 0)
        with mock.patch.object(
            validate_stat_arb_route.subprocess,
            "run",
            side_effect=[completed, completed],
        ) as run:
            self.assertEqual(validate_stat_arb_route.main(), 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][-1], "stat-arb")
        self.assertEqual(commands[1][-2:], ["--volume", "lower"])
        self.assertNotIn("upper", commands[1])


if __name__ == "__main__":
    unittest.main()
