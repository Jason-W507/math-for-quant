# %% [markdown]
# # 简单函数积分与极限交换反例
#
# 解析结果来自独立手算；程序只复现有限求和和固定见证值。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch03.measure_integration", run_name="__main__")
