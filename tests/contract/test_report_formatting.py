from __future__ import annotations

from math_for_quant.reporting import REPORT_ZERO_TOLERANCE, stable_gap


def test_stable_gap_normalizes_only_sub_tolerance_noise() -> None:
    assert stable_gap(REPORT_ZERO_TOLERANCE / 2) == 0.0
    assert stable_gap(-REPORT_ZERO_TOLERANCE / 2) == 0.0
    assert stable_gap(REPORT_ZERO_TOLERANCE) == REPORT_ZERO_TOLERANCE
    assert stable_gap(-2) == -2.0
