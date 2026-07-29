from __future__ import annotations

from datetime import date


def validate_chronological_split(train: list[int], validation: list[int], holdout: list[int]) -> None:
    if not train or not validation or not holdout:
        raise ValueError("time split must contain train, validation, and holdout observations")
    if not (max(train) < min(validation) and max(validation) < min(holdout)):
        raise ValueError("random or overlapping time split rejected")


def validate_fit_cutoff(fit_end: str, train_end: str) -> None:
    if date.fromisoformat(fit_end + "-01") > date.fromisoformat(train_end + "-01"):
        raise ValueError("preprocessor used observations after the training window")
