from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def library_cross_fitted_ridge_predictions(
    features: np.ndarray,
    target: np.ndarray,
    *,
    folds: list[tuple[range, range]],
    alpha: float,
) -> dict[int, float]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    predictions: dict[int, float] = {}
    for train_range, validation_range in folds:
        train, validation = list(train_range), list(validation_range)
        model = Ridge(alpha=alpha).fit(x[train], y[train])
        predictions.update(
            zip(validation, model.predict(x[validation]).tolist(), strict=True)
        )
    return predictions


def maximum_prediction_gap(
    transparent: dict[int, float], library: dict[int, float]
) -> float:
    if transparent.keys() != library.keys():
        raise ValueError("cross-fitting implementations cover different rows")
    return max(abs(transparent[key] - library[key]) for key in transparent)
