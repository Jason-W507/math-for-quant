# 本轮课程评审逐条核对（2026-09-08）

核对对象：引用对话中最新一轮、以 6b45919 为评审基准的完整回复。按其二至六节的每项改进要求核对；第一节是对已完成工作的肯定，末段是前述事项的优先级重述，不另计新要求。此表位于维护文档，不进入教材。

结论：本轮所有具体内容改进要求已落实，发行项由 v0.6.0 完成。此结论不等于对全书所有定理或所有段落作无缺陷保证。

| 评审段落/要求 | 对应来源 | 核对结果 |
|---|---|---|
| 二/估计方法：观测固定、似然是参数函数 | [ch10.tex](../tex/upper/chapters/ch10.tex#L42) | 已明确与样本分布及参数概率的区别。 |
| 二/Bernoulli 推导 | [ch10.tex](../tex/upper/chapters/ch10.tex#L49) | 完整似然、一阶二阶导数、内点解、全零全一边界和十点算例。 |
| 二/正态线性模型与 OLS | [ch10.tex](../tex/upper/chapters/ch10.tex#L70) | 联合密度、梯度、Hessian、方差 MLE 与无偏分母、三点算例均已展开。 |
| 二/条件密度衔接时间序列 | [ch10.tex](../tex/upper/chapters/ch10.tex#L101) | 条件与精确似然、初始密度、AR/GARCH/Kalman 与准似然边界；第十二章回指。 |
| 二/线性代数基础过快 | [ch06.tex](../tex/upper/chapters/ch06.tex#L17) | 实际算列空间、零空间、行空间及左零空间，解释观测与参数空间及维数。 |
| 二/Gram–Schmidt 缺步骤 | [ch06.tex](../tex/upper/chapters/ch06.tex#L96) | 逐项构造 q1、u2、q2、Q、R，再三角回代；说明重复列时归一化失败的原因。 |
| 三.1/病态与绝对尺度混淆 | [ch06.tex](../tex/upper/solutions/ch06.tex#L15) | 使用最大/最小奇异值比，并比较 1e-12 I 与 diag(1e12,1)。 |
| 三.2/原始输入修改后秩亏 | [ch06_linear_algebra.ipynb](../notebooks/upper/ch06_linear_algebra.ipynb#L77) | 按数值秩分支，SVD 截断、伪逆及 lstsq 使用同一相对阈值；默认、重复列、近共线均已执行。 |
| 三.3/分位数标签统一禁令 | [ml-alpha-model.tex](../tex/lower/chapters/ml-alpha-model.tex#L54) | 区分历史固定阈值与逐期横截面目标；固定资产集合和并列规则，分别说明三个时点；对应题解已补。 |
| 四/置信区间算例前移 | [ch10.tex](../tex/upper/chapters/ch10.tex#L117) | 位于渐近理论之前；原章末版本已移除，Notebook 展示固定与选择最大项区间图。 |
| 四/标量网络梯度算例前移 | [ml-alpha-deep.tex](../tex/lower/chapters/ml-alpha-deep.tex#L10) | 位于矩阵反向传播之前，明确使用旧参数梯度同时更新；原单层冗余算例删除。 |
| 四/离散 ES 边界质量算例前移 | [portfolio-risk-tail.tex](../tex/lower/chapters/portfolio-risk-tail.tex#L24) | 紧接 VaR/ES 定义，综合项目之后不再重复该算例；保留 0.6 水平结果 7 及错误条件平均 6 的解释。 |
| 四/第十章固定输出删除 | [ch10.tex](../tex/upper/chapters/ch10.tex#L26) | 保留由已知方差推得的 MSE，删除 mse 字符串、inconsistent_var 字符串及未给数据的模拟小数。 |
| 四/ridge 与 lasso 数值来源 | [multifactor-estimation.tex](../tex/lower/chapters/multifactor-estimation.tex#L52) | 删除 0.465574、0.470000；用 x=(1,2)、y=(1,1)、lambda=1 比较 OLS、ridge、lasso，解释损失缩放。 |
| 五/无协整对照 | [ch05_stat_arb_model.ipynb](../notebooks/lower/ch05_stat_arb_model.ipynb#L136) | 独立随机游走作为原假设；共同趋势模型作为备择；代码以 scenarios 中 False/True 选择两种生成机制。 |
| 五/模拟标准误 | [ch05_stat_arb_model.ipynb](../notebooks/lower/ch05_stat_arb_model.ipynb#L144) | 显示插件标准误与 Wilson 区间，比较 60/300 次精度，解释频率边界和几百分点差异。 |
| 五/相同创新下尺度比较 | [ch05_stat_arb_model.ipynb](../notebooks/lower/ch05_stat_arb_model.ipynb#L182) | 固定 x 与标准化创新，仅改变尺度 0.5/2；核对系数误差和残差四倍缩放与统计量不变。 |
| 五/投影解释尺度不变性 | [stat-arb-model.tex](../tex/lower/chapters/stat-arb-model.tex#L106) | 正文给投影恒等式和 DF 方差抵消推导，并说明持续性与额外测量噪声不属于纯缩放。 |
| 六/发布资产与源码一致 | [README.md](../README.md#L5) | v0.6.0 统一版本号、日期、三份 PDF 与所有教学 Notebook；下载链接指向明确标签。 |

此前已通过的三个改动 Notebook 及两个矩阵变体，其已执行单元源码与当前源码逐格一致，且无执行错误；本次复核了这五个情形，未把旧结果套用于不同源码。数学推导与段落位置另作人工核对，不以关键词命中替代判断。

第六章投影存在性、PCA/白化的训练样本与总体区别、第八章有限状态开篇、随机积分构造等上一轮已完成内容仍保留。具体运行观察与已知非阻断提示见[本轮修改记录](course-review-followup-2026-09-08.md)。

## 第四节的段落位置复核

| 算例 | 修改前 | 修改后 |
|---|---|---|
| 置信区间例子 | 6b45919 的第 227 行，位于此前理论/综合项目之后 | 当前第 117 行，已移至渐近近似、检验与经济效应之前，仅保留一处 |
| 两层网络标量例子 | 6b45919 的第 106 行，位于此前理论/综合项目之后 | 当前第 10 行，已移至下面把同样的链式法则写成矩阵形式之前，仅保留一处 |
| 离散 ES 例子 | 6b45919 的第 90 行，位于此前理论/综合项目之后 | 当前第 24 行，已移至本路线的综合项目之前，仅保留一处 |

另外删除了被两层网络例子替代的单层梯度小例、重复的三点 OLS 例及尾部风险的重复段落。正则化的谱方向推导已放回 ridge 处，给定数据的三种估计比较紧随 lasso。上述核对确认第四节具体建议已落实；并不把全书语言风格视为可一次性清零的检查项。
