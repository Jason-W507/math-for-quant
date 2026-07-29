# %% [markdown]
# # 高频、微观结构与执行：可复现实验入口

# %%
from __future__ import annotations

import sys
from pathlib import Path

from math_for_quant.lower.microstructure import main


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch06/oracle.json")))
