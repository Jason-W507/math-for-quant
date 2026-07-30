from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedNestedSplit:
    train: range
    validation: range
    test: range
    label_horizon: int
    embargo: int


def validate_preprocessing_cutoff(*, fitted_through: int, evaluation_starts: int) -> None:
    if fitted_through >= evaluation_starts:
        raise ValueError("future preprocessing reaches the evaluation window")


def validate_target_alignment(
    *, feature_time: int, target_time: int, horizon: int
) -> None:
    if horizon <= 0 or target_time - feature_time != horizon:
        raise ValueError("target misalignment violates the frozen prediction horizon")


def validate_model_selection(
    *, attempts: int, budget: int, test_reused: bool
) -> None:
    if budget < 1 or attempts < 1 or attempts > budget:
        raise ValueError("selection budget was exceeded")
    if test_reused:
        raise ValueError("test reselection is forbidden")


def validate_nested_time_split(split: PurgedNestedSplit) -> None:
    train, validation, test = list(split.train), list(split.validation), list(split.test)
    if not train or not validation or not test:
        raise ValueError("nested time split contains an empty window")
    if split.label_horizon < 0 or split.embargo < 0:
        raise ValueError("purge and embargo must be non-negative")
    if max(train) + split.label_horizon >= min(validation):
        raise ValueError("purge gap does not cover the label horizon")
    if max(validation) + split.embargo >= min(test):
        raise ValueError("embargo gap does not separate validation and test")


def cross_fitted_ridge_predictions(
    features: np.ndarray,
    target: np.ndarray,
    *,
    folds: list[tuple[range, range]],
    alpha: float,
) -> dict[int, float]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],) or alpha < 0.0:
        raise ValueError("cross-fitting inputs are invalid")
    predictions: dict[int, float] = {}
    for train_range, validation_range in folds:
        train, validation = list(train_range), list(validation_range)
        if not train or not validation or max(train) >= min(validation):
            raise ValueError("cross-fitting folds must move forward in time")
        overlap = predictions.keys() & set(validation)
        if overlap:
            raise ValueError("cross-fitting validation observations overlap")
        design = np.column_stack([np.ones(len(train)), x[train]])
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ y[train],
        )
        validation_design = np.column_stack(
            [np.ones(len(validation)), x[validation]]
        )
        predictions.update(
            zip(validation, (validation_design @ coefficients).tolist(), strict=True)
        )
    return predictions


def platt_calibrate(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    values = np.asarray(scores, dtype=float)
    target = np.asarray(labels, dtype=int)
    if values.ndim != 1 or target.shape != values.shape or np.unique(target).size != 2:
        raise ValueError("calibration requires aligned scores and both classes")
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=0)
    return model.fit(values[:, None], target).predict_proba(values[:, None])[:, 1]


def importance_jaccard(left: np.ndarray, right: np.ndarray, *, top_k: int) -> float:
    if top_k < 1 or top_k > left.size or left.shape != right.shape:
        raise ValueError("importance stability inputs are invalid")
    left_set = set(np.argsort(left)[-top_k:].tolist())
    right_set = set(np.argsort(right)[-top_k:].tolist())
    return len(left_set & right_set) / len(left_set | right_set)
