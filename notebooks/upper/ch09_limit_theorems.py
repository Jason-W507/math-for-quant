# %% [markdown]
# # LLN、CLT、集中界与重尾失效
#
# 理论均值、标准误、精确二项尾概率和 Hoeffding 上界均独立于模拟给出。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch09.limit_theorems", run_name="__main__")
