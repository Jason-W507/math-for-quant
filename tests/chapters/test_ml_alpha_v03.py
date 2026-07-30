from __future__ import annotations

import unittest

import numpy as np

from math_for_quant.lower.ml_alpha_models import (
    TorchTrainingConfig,
    restore_tiny_mlp_predictions,
    sequence_order_sensitivity,
    train_tiny_mlp,
)
from math_for_quant.lower.ml_alpha_library import cross_check_classical_models
from math_for_quant.lower.ml_alpha_research import (
    AlphaLedgerInputs,
    PortfolioPolicy,
    build_alpha_ledger,
    pairwise_ranking_loss,
    return_weighted_loss,
)
from math_for_quant.lower.ml_alpha_text import (
    audit_text_timestamps,
    compare_text_adaptation,
)
from math_for_quant.lower.ml_alpha_validation import (
    PurgedNestedSplit,
    cross_fitted_ridge_predictions,
    platt_calibrate,
    validate_model_selection,
    validate_nested_time_split,
    validate_preprocessing_cutoff,
    validate_target_alignment,
)
from math_for_quant.lower.ml_alpha_validation_library import (
    library_cross_fitted_ridge_predictions,
    maximum_prediction_gap,
)
from math_for_quant.lower.ml_alpha_execution_library import library_alpha_ledger


class MlAlphaV03Tests(unittest.TestCase):
    def test_pytorch_training_is_seeded_and_checkpointed(self) -> None:
        features = np.asarray([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
        target = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        config = TorchTrainingConfig(seed=17, epochs=80, learning_rate=0.03)
        left = train_tiny_mlp(features, target, config=config)
        right = train_tiny_mlp(features, target, config=config)
        np.testing.assert_allclose(left.predictions, right.predictions, atol=0.0)
        self.assertEqual(left.checkpoint_sha256, right.checkpoint_sha256)
        restored = restore_tiny_mlp_predictions(left.checkpoint, features)
        np.testing.assert_allclose(restored, left.predictions, atol=0.0)
        self.assertLess(left.loss, 0.03)
        self.assertEqual(left.batch_count, 1)

    def test_classical_route_exercises_linear_tree_and_boosting(self) -> None:
        result = cross_check_classical_models(
            np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
            np.asarray([-0.4, -0.2, 0.0, 1.0, 1.2]),
            np.asarray([-1.5, 0.5, 1.5]),
            boosting_rounds=2,
        )
        self.assertLess(result.linear_max_gap, 1e-12)
        self.assertLess(result.stump_max_gap, 1e-12)
        self.assertLess(result.boosting_max_gap, 1e-12)

    def test_sequence_models_retain_order(self) -> None:
        sequence = np.asarray([[[1.0], [2.0], [4.0], [8.0]]], dtype=np.float32)
        sensitivity = sequence_order_sensitivity(sequence, seed=23)
        self.assertAlmostEqual(sensitivity["mean_pool"], 0.0)
        for model in ("causal_conv", "rnn", "attention", "transformer"):
            self.assertGreater(sensitivity[model], 1e-5)

    def test_nested_split_and_cross_fitting_keep_time_order(self) -> None:
        split = PurgedNestedSplit(
            train=range(0, 8), validation=range(10, 12), test=range(14, 16),
            label_horizon=2, embargo=2,
        )
        validate_nested_time_split(split)
        with self.assertRaisesRegex(ValueError, "purge"):
            validate_nested_time_split(
                PurgedNestedSplit(range(0, 9), range(10, 12), range(14, 16), 2, 2)
            )
        features = np.arange(12.0)[:, None]
        target = 0.5 * features[:, 0] + 1.0
        predictions = cross_fitted_ridge_predictions(
            features, target, folds=[(range(0, 4), range(4, 6)), (range(0, 6), range(6, 8))], alpha=0.1
        )
        self.assertEqual(sorted(predictions), [4, 5, 6, 7])
        library = library_cross_fitted_ridge_predictions(
            features, target, folds=[(range(0, 4), range(4, 6)), (range(0, 6), range(6, 8))], alpha=0.1
        )
        self.assertLess(maximum_prediction_gap(predictions, library), 1e-12)

    def test_validation_rejects_future_information_and_test_reselection(self) -> None:
        validate_preprocessing_cutoff(fitted_through=7, evaluation_starts=10)
        validate_target_alignment(feature_time=7, target_time=9, horizon=2)
        validate_model_selection(attempts=3, budget=3, test_reused=False)

        with self.assertRaisesRegex(ValueError, "future preprocessing"):
            validate_preprocessing_cutoff(fitted_through=10, evaluation_starts=10)
        with self.assertRaisesRegex(ValueError, "target misalignment"):
            validate_target_alignment(feature_time=7, target_time=10, horizon=2)
        with self.assertRaisesRegex(ValueError, "selection budget"):
            validate_model_selection(attempts=4, budget=3, test_reused=False)
        with self.assertRaisesRegex(ValueError, "test reselection"):
            validate_model_selection(attempts=3, budget=3, test_reused=True)

    def test_probability_calibration_is_evaluated_after_its_fit_window(self) -> None:
        calibrated = platt_calibrate(
            np.asarray([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]),
            np.asarray([0, 0, 0, 1, 1, 1]),
            np.asarray([-1.5, -0.25, 0.25, 1.5]),
        )
        self.assertEqual(calibrated.shape, (4,))
        with self.assertRaisesRegex(ValueError, "later than calibration fit"):
            platt_calibrate(
                np.asarray([-2.0, 2.0]),
                np.asarray([0, 1]),
                np.asarray([-1.0, 1.0]),
                fit_ends=8,
                evaluation_starts=8,
            )

    def test_text_adaptation_and_revision_audit_are_explicit(self) -> None:
        audit_text_timestamps(
            publication_dates=["2024-01-01", "2024-01-02"],
            revision_dates=["2024-01-01", "2024-01-02"],
            decision_date="2024-01-03",
        )
        with self.assertRaisesRegex(ValueError, "revision leakage"):
            audit_text_timestamps(
                publication_dates=["2024-01-01"], revision_dates=["2024-01-05"], decision_date="2024-01-03"
            )
        result = compare_text_adaptation(
            train_texts=["profit growth", "loss warning", "profit upgrade", "loss decline"],
            train_labels=np.asarray([1, 0, 1, 0]),
            inference_texts=["profit warning", "loss upgrade"],
            seed=29,
        )
        self.assertEqual(result.token_count, 6)
        self.assertEqual(result.encoder_id, "mfq-tiny-text-encoder")
        self.assertEqual(result.encoder_version, "1.0.0")
        self.assertEqual(result.encoder_license, "CC0-1.0")
        self.assertLess(result.lora_trainable_parameters, result.full_trainable_parameters)
        self.assertEqual(result.zero_shot_scores.shape, (2,))
        self.assertEqual(result.few_shot_scores.shape, (2,))
        self.assertEqual(result.full_finetune_scores.shape, (2,))
        self.assertEqual(result.lora_scores.shape, (2,))

    def test_predictions_drive_fills_turnover_costs_and_net_return(self) -> None:
        scores = np.asarray([[0.9, 0.2, -0.8], [-0.7, 0.8, 0.1]])
        realized = np.asarray([[0.02, 0.00, -0.01], [-0.02, 0.03, 0.01]])
        fills = np.asarray([[1.0, 1.0, 1.0], [0.5, 1.0, 0.0]])
        inputs = AlphaLedgerInputs(
            scores, realized, fills,
            PortfolioPolicy(long_count=1, short_count=1, gross_limit=2.0, cost_per_unit_turnover=0.001),
        )
        ledger = build_alpha_ledger(inputs=inputs)
        np.testing.assert_allclose(ledger.target_positions[0], [1.0, 0.0, -1.0])
        np.testing.assert_allclose(ledger.filled_positions[1], [0.0, 1.0, -1.0])
        self.assertAlmostEqual(ledger.gross_return, 0.05)
        self.assertAlmostEqual(ledger.turnover, 4.0)
        self.assertAlmostEqual(ledger.net_return, 0.046)
        library_ledger = library_alpha_ledger(inputs=inputs)
        np.testing.assert_allclose(ledger.filled_positions, library_ledger.filled_positions)
        self.assertGreater(pairwise_ranking_loss(np.asarray([0.8, 0.1]), np.asarray([0.02, -0.01])), 0.0)
        self.assertLess(return_weighted_loss(np.asarray([0.8, -0.5]), np.asarray([0.02, -0.01])), 0.0)


if __name__ == "__main__":
    unittest.main()
