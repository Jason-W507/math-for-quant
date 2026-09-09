# 量化研究数学

《量化研究数学》面向经济学与金融学读者。读者应会使用一元与多元微积分、计算矩阵乘法和求解线性方程组，理解随机变量、期望与方差。证明方法、抽象向量空间、测度论和随机过程由本书讲授。实验需要 Python 的变量、数组与绘图基础；基础计量经济学不是入学要求。

当前发行版为 **v0.6.1**，包含全书叙述、例题位置与题解修订。课程清单包含 35 个正式学习单元；教学实验按 39 个正文单元组织。

上册建立量化研究共同使用的数学基础，包括分析、测度与概率、统计推断、随机过程、时间序列、优化和数值计算。下册将这些工具用于多因子、统计套利、机器学习、衍生品、组合风险和市场微观结构。

本书从具体问题引入数学概念，通过推导与例题解释结论，并用练习帮助读者独立使用这些工具。配套 notebook 展示计算过程、图形和可修改参数的实验，联系解析结果与有限样本现象。

## 获取成品

以下下载对应 v0.6.1，已包含本轮教材叙述修订。

- [最新 GitHub Release](https://github.com/Jason-W507/math-for-quant/releases/latest)
- [上册 PDF](https://github.com/Jason-W507/math-for-quant/releases/download/v0.6.1/math-for-quant-upper.pdf)
- [下册 PDF](https://github.com/Jason-W507/math-for-quant/releases/download/v0.6.1/math-for-quant-lower.pdf)
- [共享答案册](https://github.com/Jason-W507/math-for-quant/releases/download/v0.6.1/math-for-quant-solutions.pdf)
- [Notebook 压缩包](https://github.com/Jason-W507/math-for-quant/releases/download/v0.6.1/math-for-quant-notebooks.zip)
- 已确认勘误：[ERRATA.md](ERRATA.md)

## 阅读路线

- **应用主线**：第 1、2、5、6、7、8、9、10、12、13、14、15 章；第 8 章先读有限状态与预测部分，一般存在性与 L² 理论回读第 3、4 章。第 16、17 章作为研究实践阅读。
- **理论增强线**：第 1—9、11 章，再进入动态模型、优化和研究实践。

各章之间的联系及下册各方向所需的基础，见书中的课程地图与先修表。

## 配套实验

从仓库根目录运行 `uv sync`，然后运行 `uv run --with notebook jupyter notebook`，打开 `notebooks/` 中相应章节的文件。正文和答案册可以独立阅读，实验用于观察中间计算、修改参数并解释图形。

首次使用、运行顺序和常见问题见[实验指南](docs/reader-experiments.md)。编译书稿、维护图形和发布版本的方法见[开发说明](docs/development.md)。

## 源文件与许可

- 数学正文：`tex/`
- 可执行教材：`notebooks/` 下受 Git 跟踪的 `.ipynb`（读者直接打开的教学界面）
- 课程目录与先修关系：`curriculum/manifest.json`
- 符号与术语：`curriculum/notation.json`、`curriculum/glossary.json`
- 补充习题题源：`curriculum/interview-problem-ledger.json`
- 贡献规范：[CONTRIBUTING.md](CONTRIBUTING.md)

代码采用 MIT License；原创书稿与图形采用 CC BY-NC-SA 4.0；ElegantBook 类文件沿用 LPPL；真实数据在引入时分别记录来源与许可。
