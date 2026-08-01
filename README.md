# 量化研究数学

《量化研究数学》是一套面向具备经济学或金融学基础、已经会 Python 的读者的双册自学教程：

- 上册《通用工具与研究证据》包含 17 个学习单元、四级习题、可执行实验与独立 oracle；
- 全书保留 35 个正式学习单元：上册 17 个，下册 18 个；下册把这些学习单元按知识耦合度组织为 9 个方向章节，并将机器学习与 AI 编排为“树模型—现代与前沿 AI—研究设计”三章主线；
- v0.4.0 系统重写下册方向章节，补充全书可视化体系，并将《量化绿皮书》来源映射题扩展到冻结总题量的 50%。

## 获取成品

- [最新 GitHub Release](https://github.com/Jason-W507/math-for-quant/releases/latest)
- 本地构建产物：`output/pdf/math-for-quant-upper.pdf`
- 上下册共享答案：`output/pdf/math-for-quant-solutions.pdf`
- 下册方向路线：`output/pdf/math-for-quant-lower.pdf`
- 已确认勘误：[ERRATA.md](ERRATA.md)

生成 PDF 不提交到源码历史；正式版本由 tag 构建并上传到 Release。

## 阅读路线

- **应用主线**：先建立研究语言，再进入线性代数、概率统计、时间序列、优化、数值计算和研究审计；遇到条件化或极限交换时按课程地图回看测度论桥接。
- **理论增强线**：完整学习分析、测度、条件期望、概率极限与随机过程。

权威先修图与两条路线保存在 `curriculum/manifest.json`，构建时生成到上册课程地图；章首先修不手工维护第二份副本。

## 验证

```powershell
uv sync
uv run python tools/render_shared_registries.py --check
uv run python -m unittest discover -s tests
uv run python tools/check_learning_unit.py `
  --manifest curriculum/manifest.json `
  --volume upper
```

## 构建受影响的册

图形采用独立图源与缓存矢量资产。修改图形规格后运行
`uv run python tools/build_figures.py`；普通书稿构建不会重跑绘图或数值实验。完整约定见
[`docs/figures.md`](docs/figures.md)。

本机使用 MiKTeX/XeLaTeX。上册改动只构建上册：

```powershell
uv run python tools/build_books.py --volume upper
```

下册改动改用 `--volume lower`；只有共享源或正式全书发布才使用 `--volume all`。原模板目录保持不变，项目内模板来源见 `docs/template-provenance.md`。

多因子路线的公共验收命令会验证本路线登记的学习单元，并且只重建下册与共享答案册：

```powershell
uv run python tools/validate_multifactor_route.py
```

时间序列与统计套利路线采用相同边界，只构建下册与共享答案册：

```powershell
uv run python tools/validate_stat_arb_route.py
```

机器学习路线从树模型出发，依次覆盖深度学习与 CNN、RNN/Transformer/LLM、GNN/DRL 与多模态前沿，再贯通嵌套时序验证、漂移监控以及分数到成本后净收益：

```powershell
uv run python tools/validate_ml_alpha_route.py
```

衍生品路线把随机分析、定价与校准、离散对冲拆成三个学习单元，并以真实的 Novikov 失败见证、非等距执行价约束和多路径对冲分布作为验收边界：

```powershell
uv run python tools/validate_derivatives_route.py
```

组合风险路线贯通风险估计、稳健实施与尾部/压力风险：

```powershell
uv run python tools/validate_portfolio_risk_route.py
```

高频、微观结构与执行路线贯通事件/队列、执行/做市控制和共同随机数仿真：

```powershell
uv run python tools/validate_microstructure_route.py
```

正式发布使用统一入口；它要求干净工作树，执行全部 Jupytext 源、构建三份 PDF、核对学习单元/Capstone/模板基线，并生成带来源、版本、校验值和许可证文件的发行清单。Notebook 压缩包同时携带代码与书稿许可文本：

```powershell
uv run python tools/build_release.py
```

## 权威来源

- 数学正文：`tex/`
- 可执行教材：`notebooks/` 下的 Jupytext 文本源
- 课程图、路线与证据路径：`curriculum/manifest.json`
- 符号与术语：`curriculum/notation.json`、`curriculum/glossary.json`
- 贡献规范：[CONTRIBUTING.md](CONTRIBUTING.md)

## 许可

- 代码：MIT；
- 原创书稿与图形：CC BY-NC-SA 4.0；
- ElegantBook：LPPL；
- 数据：逐数据集记录自身许可。
