from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def library_total_variance_coefficients(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    model = LinearRegression(fit_intercept=False).fit(
        np.asarray(design, dtype=float),
        np.asarray(target, dtype=float),
        sample_weight=np.asarray(weights, dtype=float),
    )
    return np.asarray(model.coef_, dtype=float)


__all__ = ["library_total_variance_coefficients"]
