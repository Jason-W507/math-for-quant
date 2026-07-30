from __future__ import annotations

import json
from pathlib import Path

from math_for_quant.lower.derivatives_route import (
    render_route_report,
    run_hedging,
    run_numerics,
    run_stochastic,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, object]:
    return json.loads(
        (ROOT / "data" / "fixtures" / name).read_text(encoding="utf-8")
    )


def build_report() -> str:
    return render_route_report(
        run_stochastic(_load("derivatives-stochastic.json")),
        run_numerics(_load("derivatives-numerics.json")),
        run_hedging(_load("derivatives-hedging.json")),
    )


if __name__ == "__main__":
    print(build_report(), end="")
