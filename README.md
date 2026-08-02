# 量化研究数学

《量化研究数学》面向已经学过微积分、线性代数、概率统计并会使用 Python 的经济学与金融学读者。

当前版本为 v0.4.0，课程清单包含 35 个正式学习单元。

上册建立量化研究共同使用的数学基础，包括分析、测度与概率、统计推断、随机过程、时间序列、优化和数值计算。下册将这些工具用于多因子、统计套利、机器学习、衍生品、组合风险和市场微观结构。

全书持续追问三个问题：公式在什么条件下成立，计算结果怎样得到独立核对，经验结论能否经受样本外检验和交易摩擦。正文以问题、定义、推导、算例和研究连接为主；配套实验负责重复计算与观察有限样本现象。

## 获取成品

- [最新 GitHub Release](https://github.com/Jason-W507/math-for-quant/releases/latest)
- [上册 PDF](https://github.com/Jason-W507/math-for-quant/releases/latest)
- [下册 PDF](https://github.com/Jason-W507/math-for-quant/releases/latest)
- [共享答案册](https://github.com/Jason-W507/math-for-quant/releases/latest)
- 已确认勘误：[ERRATA.md](ERRATA.md)

## 阅读路线

- **应用主线**：第 1、5、6、7、9、10、12、13、14、15、16、17 章；遇到条件化或极限交换时回看测度论桥接。
- **理论增强线**：第 1—4、7—9、11 章，再进入动态模型、优化和研究审计。

完整依赖图、下册路线诊断和桥接地图由 `curriculum/manifest.json` 生成，见书中的课程地图。

## 项目与复现

数学正文在 `tex/`，可执行实验在 `notebooks/`，图源与出版资产在 `figures/` 和 `tex/figures/`。项目约定、构建命令和发布流程见 [`docs/development.md`](docs/development.md)；正文与维护者约定的边界见 [`docs/evidence-contract.md`](docs/evidence-contract.md)。

维护者可用下面的入口完成完整验证；按改动范围选择相应册或路线即可：

```powershell
uv sync
uv run python tools/render_shared_registries.py --check
uv run python -m unittest discover -s tests
uv run python tools/check_pdf_visual_regression.py
uv run python tools/check_learning_unit.py --manifest curriculum/manifest.json --volume all
uv run python tools/build_books.py --volume upper
uv run python tools/build_books.py --volume lower
uv run python tools/build_books.py --volume all
uv run python tools/validate_multifactor_route.py
uv run python tools/validate_stat_arb_route.py
uv run python tools/validate_ml_alpha_route.py
uv run python tools/validate_derivatives_route.py
uv run python tools/validate_portfolio_risk_route.py
uv run python tools/validate_microstructure_route.py
uv run python tools/build_release.py
```

图形规格改变时另行运行 `uv run python tools/build_figures.py`；上册或下册的局部改动不需要先构建全书。

## 权威来源与许可

- 数学正文：`tex/`
- 可执行教材：`notebooks/` 下的 Jupytext 文本源
- 课程图、路线与证据路径：`curriculum/manifest.json`
- 符号与术语：`curriculum/notation.json`、`curriculum/glossary.json`
- 贡献规范：[CONTRIBUTING.md](CONTRIBUTING.md)

代码采用 MIT License；原创书稿与图形采用 CC BY-NC-SA 4.0；ElegantBook 类文件沿用 LPPL；真实数据在引入时分别记录来源与许可。
