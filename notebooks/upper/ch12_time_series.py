# %% [markdown]
# # 时间序列、预测与状态空间递推
#
# AR(1) 的平稳矩和预测误差方差、标量 Kalman 滤波手算值由解析 oracle
# 给出；固定种子模拟只用于交叉验证。独立随机游走和结构变化过程提供负例。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch12.time_series", run_name="__main__")
