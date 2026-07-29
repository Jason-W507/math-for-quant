# 量化研究数学

《量化研究数学》是一套面向具备经济学或金融学基础、已经会 Python 的读者的双册自学教程：

- 上册《通用工具与研究证据》初稿已完成，包含 17 个学习单元、四级习题、可执行实验与独立 oracle；完整答案单独成册，便于先作答再核对；
- 下册《方向模型与研究项目》已完成路线诊断、统一 Capstone 契约，以及多因子、统计套利、机器学习 Alpha、衍生品定价、组合风险和微观结构六个完整学习单元。

## 获取成品

- [最新 GitHub Release](https://github.com/Jason-W507/math-for-quant/releases/latest)
- 本地构建产物：`output/pdf/math-for-quant-upper.pdf`
- 配套完整答案：`output/pdf/math-for-quant-upper-solutions.pdf`
- 下册路线导论：`output/pdf/math-for-quant-lower.pdf`
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

本机使用 MiKTeX/XeLaTeX。上册改动只构建上册：

```powershell
uv run python tools/build_books.py --volume upper
```

下册改动改用 `--volume lower`；只有共享源或正式全书发布才使用 `--volume all`。原模板目录保持不变，项目内模板来源见 `docs/template-provenance.md`。

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
