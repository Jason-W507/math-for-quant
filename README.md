# 量化研究数学

《量化研究数学》是一套面向具备经济学或金融学基础、已经会 Python 的读者的双册自学教程：

- 上册《通用工具与研究证据》建立共享数学与研究有效性基础；
- 下册《方向模型与实战项目》提供六条独立方向路线与 Capstone。

当前仓库完成了最小出版与学习单元证据链，尚未开始正式章节写作。

## 验证最小学习单元

```powershell
uv sync
uv run python tools/check_learning_unit.py `
  --manifest curriculum/manifest.json `
  --unit foundation.oracle-smoke
uv run python -m unittest discover -s tests -v
```

期望看到 `evidence=4/4`，并由手算的 `1/150` 独立 oracle 核对 Python 输出。

## 生成 notebook 出版产物

```powershell
uv run python tools/build_notebook.py `
  --source notebooks/foundation/independent_oracle.py `
  --output build/notebooks/foundation/independent_oracle.ipynb
```

Jupytext 文本源是权威版本；生成的 `.ipynb` 不手工维护。

## 构建两册 PDF

本机使用 MiKTeX 管理 XeLaTeX：

```powershell
uv run python tools/build_books.py --volume all
```

成品写入 `output/pdf/math-for-quant-upper.pdf` 与 `output/pdf/math-for-quant-lower.pdf`。原模板目录保持不变；项目内模板来源与兼容调整见 `docs/template-provenance.md`。

## 许可

- 代码：MIT；
- 原创书稿与图形：CC BY-NC-SA 4.0；
- ElegantBook：LPPL；
- 数据：逐数据集记录自身许可。
