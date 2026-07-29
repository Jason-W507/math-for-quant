# %% [markdown]
# # 组合与风险：可复现实验入口

# %%
from __future__ import annotations

import sys
from pathlib import Path

from math_for_quant.lower.portfolio_risk import main


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch05/oracle.json")))
