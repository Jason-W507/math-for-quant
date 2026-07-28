# %% [markdown]
# # 二次型、Jacobian 链式法则与不可微反例
#
# 解析梯度先由微分形式手算；中心差分只负责独立交叉核验。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch05.matrix_calculus", run_name="__main__")
