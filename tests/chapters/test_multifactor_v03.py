from __future__ import annotations

import hashlib
import json
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from math_for_quant.lower.multifactor_estimation import (
    classic_two_pass_fama_macbeth,
    lasso_coordinate_descent,
    predictive_fama_macbeth,
    ridge_closed_form,
)
from math_for_quant.lower.multifactor_research import (
    PortfolioInputs,
    PortfolioPolicy,
    Weighting,
    build_group_portfolio_ledger,
    load_real_cross_section,
)
from math_for_quant.lower.multifactor_library import (
    cross_check_estimators,
    cross_check_route_statistics,
)
from tools import validate_multifactor_route


ROOT = Path(__file__).resolve().parents[2]


class MultifactorV03Tests(unittest.TestCase):
    def test_predictive_and_classic_fama_macbeth_have_distinct_outputs(self) -> None:
        signals = np.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
        future_returns = 0.01 + np.asarray([[0.02], [0.03]]) * signals
        predictive = predictive_fama_macbeth(signals, future_returns)
        self.assertAlmostEqual(predictive.mean_coefficient, 0.025)

        factor_returns = np.asarray([[-0.02], [0.01], [0.03], [-0.01], [0.02]])
        betas = np.asarray([0.5, 1.0, 1.5])
        asset_returns = 0.004 + factor_returns @ betas[None, :]
        classic = classic_two_pass_fama_macbeth(asset_returns, factor_returns)
        np.testing.assert_allclose(classic.betas[:, 0], betas, atol=1e-12)
        self.assertAlmostEqual(classic.risk_prices[0], float(factor_returns.mean()))
        self.assertGreaterEqual(classic.shanken_multiplier, 1.0)

    def test_regularized_estimators_match_independent_closed_forms(self) -> None:
        design = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        target = np.asarray([0.0, 1.0, 2.0, 2.5])
        ridge = ridge_closed_form(design, target, penalty=0.5)
        expected_ridge = np.linalg.solve(
            design.T @ design + 0.5 * np.eye(2), design.T @ target
        )
        np.testing.assert_allclose(ridge, expected_ridge, atol=1e-12)
        lasso = lasso_coordinate_descent(
            design[:, 1:], target - target.mean(), penalty=0.1, iterations=10_000
        )
        self.assertLess(abs(lasso[0]), abs(np.linalg.lstsq(design[:, 1:], target - target.mean(), rcond=None)[0][0]))

    def test_mature_libraries_reproduce_transparent_estimators(self) -> None:
        signals = np.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
        future_returns = 0.01 + np.asarray([[0.02], [0.03]]) * signals
        factor_returns = np.asarray([[-0.02], [0.01], [0.03], [-0.01], [0.02]])
        betas = np.asarray([0.5, 1.0, 1.5])
        asset_returns = 0.004 + factor_returns @ betas[None, :]
        design = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        target = np.asarray([0.5, 1.0, 1.5, 1.8])
        library = cross_check_estimators(
            signals=signals,
            future_returns=future_returns,
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            regularization_design=design,
            regularization_target=target,
            ridge_penalty=0.5,
            lasso_penalty=0.1,
        )
        transparent_predictive = predictive_fama_macbeth(signals, future_returns)
        transparent_classic = classic_two_pass_fama_macbeth(asset_returns, factor_returns)
        np.testing.assert_allclose(library.predictive_mean, transparent_predictive.mean_coefficient)
        np.testing.assert_allclose(library.classic_betas, transparent_classic.betas, atol=1e-12)
        np.testing.assert_allclose(library.classic_risk_prices, transparent_classic.risk_prices, atol=1e-12)
        np.testing.assert_allclose(
            library.ridge_coefficients,
            ridge_closed_form(design, target, penalty=0.5),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            library.lasso_coefficients,
            lasso_coordinate_descent(design, target, penalty=0.1, iterations=100_000),
            atol=1e-10,
        )

    def test_group_portfolio_ledger_connects_signal_to_net_return(self) -> None:
        signals = np.asarray(
            [[-2.0, -1.0, 1.0, 2.0], [-1.0, 2.0, -2.0, 1.0]]
        )
        realized = np.asarray(
            [[-0.02, -0.01, 0.01, 0.03], [-0.01, 0.02, -0.02, 0.01]]
        )
        caps = np.asarray([[1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 2.0, 3.0]])
        ledger = build_group_portfolio_ledger(
            inputs=PortfolioInputs(signals, realized, caps),
            policy=PortfolioPolicy(
                quantiles=2,
                weighting=Weighting.EQUAL,
                holding_periods=1,
                cost_per_unit_turnover=0.001,
                capacity_impact=0.0005,
            ),
        )
        self.assertEqual(ledger.weights.shape, signals.shape)
        self.assertAlmostEqual(ledger.gross_return, 0.0325)
        self.assertGreater(ledger.turnover, 0.0)
        self.assertAlmostEqual(
            ledger.net_return,
            ledger.gross_return - ledger.cost - ledger.capacity_impact,
        )

    def test_mature_libraries_reproduce_panel_and_route_statistics(self) -> None:
        panel_x = np.asarray([[0.0], [1.0], [0.0], [1.0], [1.0], [2.0]])
        entities = np.asarray([0, 0, 1, 1, 2, 2])
        entity_intercepts = np.asarray([1.0, 1.0, -2.0, -2.0, 0.5, 0.5])
        panel_y = entity_intercepts + 0.4 * panel_x[:, 0]
        signal = np.asarray([-2.0, -1.0, 1.0, 2.0])
        future = np.asarray([-0.03, -0.01, 0.02, 0.04])
        horizons = np.asarray([future, 0.5 * future + np.asarray([0.0, 0.001, -0.001, 0.0])])
        check = cross_check_route_statistics(
            panel_design=panel_x,
            panel_target=panel_y,
            entities=entities,
            signal=signal,
            size=np.asarray([-1.0, -1.0, 1.0, 1.0]),
            industry=np.asarray([0.0, 1.0, 0.0, 1.0]),
            future_returns=future,
            horizon_returns=horizons,
            p_values=[0.001, 0.02, 0.3, 0.8],
            alpha=0.05,
        )
        self.assertLess(check.panel_slope_gap, 1e-12)
        self.assertLess(check.neutralization_gap, 1e-12)
        self.assertLess(check.ic_gap, 1e-12)
        self.assertLess(check.rank_ic_gap, 1e-12)
        self.assertLess(check.decay_gap, 1e-12)
        self.assertEqual(check.bh_count_gap, 0)

    def test_real_snapshot_has_frozen_hash_and_strict_time_order(self) -> None:
        snapshot = ROOT / "data/real/multifactor-wdi-2013-2014.json"
        provenance = json.loads(
            (ROOT / "evidence/multifactor/real-data.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            hashlib.sha256(snapshot.read_bytes()).hexdigest(), provenance["sha256"]
        )
        cross_section = load_real_cross_section(snapshot)
        self.assertEqual(len(cross_section.countries), 7)
        self.assertLess(cross_section.signal_year, cross_section.outcome_year)
        self.assertTrue(np.isfinite(cross_section.correlation))

    def test_public_route_command_builds_only_lower_and_shared_solutions(self) -> None:
        completed = __import__("subprocess").CompletedProcess([], 0)
        with mock.patch.object(
            validate_multifactor_route.subprocess,
            "run",
            side_effect=[completed, completed],
        ) as run:
            self.assertEqual(validate_multifactor_route.main(), 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("--track", commands[0])
        self.assertEqual(commands[0][-1], "multifactor")
        self.assertEqual(commands[1][-2:], ["--volume", "lower"])
        self.assertNotIn("upper", commands[1])


if __name__ == "__main__":
    unittest.main()
