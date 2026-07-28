# %% [markdown]
# # 凸组合界与反例
#
# Jupytext 文本源是权威版本；期望值来自独立手算记录。

# %%
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from math_for_quant.ch01.convex_bound import evaluate_contract, run_contract


def main(input_path: Path | None = None) -> int:
    oracle_path = Path("evidence/ch01/oracle.json")
    if input_path is None:
        fixture_path = Path("data/fixtures/ch01.json")
        result = run_contract(fixture_path, oracle_path)
    else:
        # Compatibility path for a deliberately mutated one-file test fixture.
        import json

        document = json.loads(input_path.read_text(encoding="utf-8"))
        input_fields = {"returns", "weights", "counterexample_weights"}
        if input_fields.intersection(document):
            fixture = json.loads(
                Path(document["fixture"]["path"]).read_text(encoding="utf-8")
            )
            fixture.update(
                {field: document[field] for field in input_fields if field in document}
            )
            result = evaluate_contract(fixture, document)
        else:
            result = run_contract(Path(document["fixture"]["path"]), input_path)
    print(
        "oracle=passed "
        f"weighted_return={result.weighted_return:.6f} lower={result.lower:.6f} "
        f"upper={result.upper:.6f} counterexample={result.counterexample:.6f}"
    )
    return 0


if __name__ == "__main__":
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        raise SystemExit(main(input_path))
    except ValueError as error:
        raise SystemExit(str(error)) from error
