# %% [markdown]
# # 压缩迭代与非一致收敛反例
#
# Jupytext 文本源只复现独立手算账本；一般收敛结论仍由正文证明。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch02.analysis_foundations", run_name="__main__")
