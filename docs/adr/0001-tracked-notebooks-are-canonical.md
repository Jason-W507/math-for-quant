# Git 跟踪的 notebook 是教学权威来源

教学 notebook 直接以可审查、可合并的 `.ipynb` 文件存放在 `notebooks/` 并纳入 Git。读者打开的就是这份文件，因此教学内容不再藏在 Python 文本源或被忽略的 `output/` 目录中。

发布时，`jupyter nbconvert` 在不改写权威 notebook 的前提下执行它们，并把带输出的副本写入 `output/notebooks/` 供发行包使用。执行副本、PDF 和压缩包均是可删除、可重建的构建产物。这样保留单一教学接口，同时把运行时证据和源文件边界说清楚。
