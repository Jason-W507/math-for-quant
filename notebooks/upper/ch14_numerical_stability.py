# %% [markdown]
# # 浮点误差、稳定算法与病态线性系统
#
# Decimal 与解析公式提供独立 oracle；NumPy float64 是被测实现。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch14.numerical_stability", run_name="__main__")
