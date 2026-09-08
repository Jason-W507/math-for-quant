# 量化研究数学

《量化研究数学》面向经济学与金融学读者。读者应会使用一元与多元微积分、计算矩阵乘法和求解线性方程组，理解随机变量、期望与方差。证明方法、抽象向量空间、测度论和随机过程由本书讲授。实验需要 Python 的变量、数组与绘图基础；基础计量经济学不是入学要求。

当前源码包含后续教学修订；下方 Release 下载为已发布材料，可能早于当前源码，不能据此核对本轮修改。课程清单包含 35 个正式学习单元。

上册建立量化研究共同使用的数学基础，包括分析、测度与概率、统计推断、随机过程、时间序列、优化和数值计算。下册将这些工具用于多因子、统计套利、机器学习、衍生品、组合风险和市场微观结构。

全书持续追问三个问题：公式在什么条件下成立，计算结果怎样得到独立核对，经验结论能否经受样本外检验和交易摩擦。正文以问题、定义、推导、算例和研究连接为主；配套实验负责重复计算与观察有限样本现象。

## 获取成品

本轮修订的本地 PDF 位于 `output/pdf/`，教学 Notebook 位于 `notebooks/`。GitHub Release 资产只有在另行发布后才会更新。

- [最新 GitHub Release](https://github.com/Jason-W507/math-for-quant/releases/latest)
- [上册 PDF](https://github.com/Jason-W507/math-for-quant/releases/latest)
- [下册 PDF](https://github.com/Jason-W507/math-for-quant/releases/latest)
- [共享答案册](https://github.com/Jason-W507/math-for-quant/releases/latest)
- 已确认勘误：[ERRATA.md](ERRATA.md)

## 阅读路线

- **应用主线**：第 1、2、5、6、7、8、9、10、12、13、14、15 章；第 8 章先读有限状态与预测部分，一般存在性与 L² 理论回读第 3、4 章。第 16、17 章作为研究实践阅读。
- **理论增强线**：第 1—9、11 章，再进入动态模型、优化和研究实践。

完整依赖图、下册路线诊断和桥接地图由 `curriculum/manifest.json` 生成，见书中的课程地图。

## 项目与复现

数学正文在 `tex/`，可执行教材在 `notebooks/`，可复用计算内核在 `src/m4q/`（导入名为 `m4q`），图源与出版资产在 `figures/` 和 `tex/figures/`。项目约定、构建命令和发布流程见 [`docs/development.md`](docs/development.md)；正文与维护者约定的边界见 [`docs/evidence-contract.md`](docs/evidence-contract.md)。

维护者可用下面的入口直接编译书稿或执行教学 notebook：

```powershell
uv sync
$env:JUPYTER_ALLOW_INSECURE_WRITES = "true"
$env:IPYTHONDIR = "build/ipython"
$env:JUPYTER_RUNTIME_DIR = "build/jupyter-runtime"
New-Item -ItemType Directory -Force build/ipython, build/jupyter-runtime | Out-Null
# 交互式检查：
uv run jupyter notebook notebooks/foundation/independent_oracle.ipynb
# 无界面时检查单个 notebook：
uv run jupyter nbconvert --to notebook --execute --stdout `
  --ExecutePreprocessor.timeout=60 `
  --ExecutePreprocessor.allow_error_names=SystemExit `
  notebooks/foundation/independent_oracle.ipynb
pwsh tools/publish.ps1 -Volume upper
pwsh tools/publish.ps1 -Volume lower
pwsh tools/publish.ps1 -Volume all
pwsh tools/publish.ps1 -Release
```

`latexmk` 的终端输出就是 LaTeX 编译日志；排版检查直接打开或截图
`output/pdf/` 中的 PDF；Notebook 检查直接在 Jupyter 中运行对应的
`notebooks/**/*.ipynb`。仓库不再维护一套平行的 Python 测试编排层。

图形规格改变时另行运行 `uv run python tools/build_figures.py`；上册或下册的局部改动不需要先构建全书。

## 权威来源与许可

- 数学正文：`tex/`
- 可执行教材：`notebooks/` 下受 Git 跟踪的 `.ipynb`（读者直接打开的教学界面）
- 发布打包副本：`output/notebooks/`（由发布命令复制，属于忽略的构建产物）
- 课程图、路线与证据路径：`curriculum/manifest.json`
- 符号与术语：`curriculum/notation.json`、`curriculum/glossary.json`
- 来源映射题台账：`curriculum/interview-problem-ledger.json`
- 贡献规范：[CONTRIBUTING.md](CONTRIBUTING.md)

代码采用 MIT License；原创书稿与图形采用 CC BY-NC-SA 4.0；ElegantBook 类文件沿用 LPPL；真实数据在引入时分别记录来源与许可。
