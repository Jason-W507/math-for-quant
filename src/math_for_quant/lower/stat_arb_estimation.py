from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StateEstimates:
    filtered: np.ndarray
    smoothed: np.ndarray
    filtered_variances: np.ndarray


@dataclass(frozen=True)
class ScalarStateSpaceSpec:
    transition: float
    observation_loading: float
    process_variance: float
    observation_variance: float
    initial_mean: float
    initial_variance: float

    def validate(self) -> None:
        values = (
            self.transition,
            self.observation_loading,
            self.process_variance,
            self.observation_variance,
            self.initial_mean,
            self.initial_variance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("state-space parameters must be finite")
        if self.process_variance < 0.0 or self.observation_variance <= 0.0:
            raise ValueError("state-space variances are invalid")
        if self.initial_variance <= 0.0:
            raise ValueError("initial variance must be positive")


def kalman_filter_and_smooth(
    observations: np.ndarray,
    *,
    spec: ScalarStateSpaceSpec,
) -> StateEstimates:
    values = np.asarray(observations, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("observations must be a finite series")
    spec.validate()
    count = values.size
    predicted_means = np.empty(count)
    predicted_variances = np.empty(count)
    filtered = np.empty(count)
    variances = np.empty(count)
    mean, variance = spec.initial_mean, spec.initial_variance
    for index, observation in enumerate(values):
        predicted_means[index] = spec.transition * mean
        predicted_variances[index] = (
            spec.transition**2 * variance + spec.process_variance
        )
        innovation_variance = (
            spec.observation_loading**2 * predicted_variances[index]
            + spec.observation_variance
        )
        gain = (
            predicted_variances[index]
            * spec.observation_loading
            / innovation_variance
        )
        mean = predicted_means[index] + gain * (
            observation - spec.observation_loading * predicted_means[index]
        )
        variance = (1.0 - gain * spec.observation_loading) * predicted_variances[index]
        filtered[index], variances[index] = mean, variance
    smoothed = filtered.copy()
    for index in range(count - 2, -1, -1):
        gain = (
            variances[index]
            * spec.transition
            / predicted_variances[index + 1]
        )
        smoothed[index] += gain * (smoothed[index + 1] - predicted_means[index + 1])
    return StateEstimates(filtered, smoothed, variances)


def regime_filter(
    observations: np.ndarray,
    *,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    values = np.asarray(observations, dtype=float)
    matrix = np.asarray(transition, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    probabilities = np.asarray(initial, dtype=float)
    states = means.size
    if matrix.shape != (states, states) or variances.shape != (states,) or probabilities.shape != (states,):
        raise ValueError("regime dimensions do not agree")
    if np.any(variances <= 0.0) or np.any(matrix < 0.0):
        raise ValueError("regime probabilities or variances are invalid")
    if not np.allclose(matrix.sum(axis=1), 1.0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("regime probabilities must sum to one")
    output = np.empty((values.size, states))
    for index, value in enumerate(values):
        prior = probabilities @ matrix
        likelihood = np.exp(-0.5 * (value - means) ** 2 / variances) / np.sqrt(2.0 * math.pi * variances)
        probabilities = prior * likelihood
        normalizer = float(probabilities.sum())
        if normalizer == 0.0:
            raise ValueError("regime likelihood underflow")
        probabilities /= normalizer
        output[index] = probabilities
    return output
