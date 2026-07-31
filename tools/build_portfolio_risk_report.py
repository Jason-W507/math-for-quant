from __future__ import annotations

import json
from pathlib import Path

from math_for_quant.lower.portfolio_route import (
    render_route_report,
    run_estimation,
    run_optimization,
    run_tail,
)


ROOT = Path(__file__).resolve().parents[1]


def load_fixture(name: str) -> dict[str, object]:
    return json.loads(
        (ROOT / "data" / "fixtures" / name).read_text(encoding="utf-8")
    )


def build_report() -> str:
    return render_route_report(
        run_estimation(load_fixture("portfolio-risk-estimation.json")),
        run_optimization(load_fixture("portfolio-risk-optimization.json")),
        run_tail(load_fixture("portfolio-risk-tail.json")),
    )


def main() -> int:
    print(build_report(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
