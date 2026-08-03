from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from m4q.lower.ml_alpha import (
    fit_boosting,
    fit_linear,
    fit_stump,
    predict_boosting,
    predict_linear,
    predict_stump,
)


@dataclass(frozen=True)
class ClassicalLibraryCrossCheck:
    linear_max_gap: float
    stump_max_gap: float
    boosting_max_gap: float
    stump_library_threshold: float


def cross_check_classical_models(
    train_features: np.ndarray,
    train_target: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    boosting_rounds: int,
) -> ClassicalLibraryCrossCheck:
    x_train = np.asarray(train_features, dtype=float)
    y_train = np.asarray(train_target, dtype=float)
    x_evaluation = np.asarray(evaluation_features, dtype=float)
    if x_train.ndim != 1 or y_train.shape != x_train.shape or x_evaluation.ndim != 1:
        raise ValueError("classical-model inputs must be vectors")
    transparent_linear = fit_linear(x_train[:, None], y_train)
    library_linear = LinearRegression().fit(x_train[:, None], y_train)
    transparent_linear_prediction = predict_linear(
        x_evaluation[:, None], transparent_linear
    )
    library_linear_prediction = library_linear.predict(x_evaluation[:, None])
    transparent_stump = fit_stump(x_train, y_train)
    library_stump = DecisionTreeRegressor(max_depth=1, random_state=0).fit(
        x_train[:, None], y_train
    )
    transparent_stump_prediction = predict_stump(x_evaluation, transparent_stump)
    library_stump_prediction = library_stump.predict(x_evaluation[:, None])
    transparent_boosting = fit_boosting(x_train, y_train, boosting_rounds)
    library_boosting = GradientBoostingRegressor(
        n_estimators=boosting_rounds,
        learning_rate=1.0,
        max_depth=1,
        random_state=0,
        loss="squared_error",
    ).fit(x_train[:, None], y_train)
    transparent_boosting_prediction = predict_boosting(
        x_evaluation, transparent_boosting
    )
    library_boosting_prediction = library_boosting.predict(x_evaluation[:, None])
    return ClassicalLibraryCrossCheck(
        linear_max_gap=float(
            np.max(np.abs(transparent_linear_prediction - library_linear_prediction))
        ),
        stump_max_gap=float(
            np.max(np.abs(transparent_stump_prediction - library_stump_prediction))
        ),
        boosting_max_gap=float(
            np.max(
                np.abs(
                    transparent_boosting_prediction - library_boosting_prediction
                )
            )
        ),
        stump_library_threshold=float(library_stump.tree_.threshold[0]),
    )
