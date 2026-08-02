# Git 跟踪的 notebook 是教学权威来源

教学 notebook 直接以可审查、可合并的 `.ipynb` 文件存放在 `notebooks/` 并纳入 Git。读者打开的就是这份文件，因此教学内容不再藏在 Python 文本源或被忽略的 `output/` 目录中。

发布时，`tools/publish.ps1 -Release` 将权威 notebook 复制到 `output/notebooks/` 并打包；它不会为了发布无条件执行整套课程。修改 notebook 后由维护者直接在 Jupyter 中运行受影响文件，执行输出、PDF 和压缩包均是可删除、可重建的构建产物。这样保留单一教学接口，同时把源文件和出版产物边界说清楚。
