# %% [markdown]
# # 独立 oracle 最小实验
#
# 文本源是权威版本；`.ipynb` 是由 Jupytext 生成的出版产物。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (
    Path(__file__).resolve().parents[2]
    if "__file__" in globals()
    else Path.cwd()
)
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    runpy.run_module(
        "math_for_quant.foundation.independent_oracle", run_name="__main__"
    )
