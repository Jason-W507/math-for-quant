# 全书逐章教学改写记录（2026-09-07）

本轮承接首轮课程评审，完成其余正文单元的教学改写，并复核此前重点修订内容。下册沿用六条路线组织方式，表中按 22 个教学单元列出，不表示 PDF 新增为 22 章。

正文以问题、定义、推导、算例与适用条件组织；补充基础练习和对应反馈，保留已有数学证明与来源映射题。Notebook 展示关键计算步骤，输入可修改；新增深度学习、序列模型与前沿选读三本 Notebook。综合案例改用三笔交易贯穿收益口径、成本、显著性与成交假设。

根据进一步反馈，日志、哈希、退出码、验收清单和维护契约从教程正文、习题、答案、教学 Notebook、符号表和术语索引移出。原维护段落独立归档于 docs/maintenance，不输入出版物。ADR 0010 记录此边界。数值误差分析、样本外评价与统计假设保留，因为它们属于教学内容。

## 覆盖清单

| 单元 | 正文 | Notebook | 状态 |
|---|---|---|---|
| 上 01 | [数学语言、证明、反例与量纲](../tex/upper/chapters/ch01.tex) | [ch01_convex_bound](../notebooks/upper/ch01_convex_bound.ipynb) | 已修订；执行通过 |
| 上 02 | [极限、连续、完备性与度量空间](../tex/upper/chapters/ch02.tex) | [ch02_analysis_foundations](../notebooks/upper/ch02_analysis_foundations.ipynb) | 已修订；执行通过 |
| 上 03 | [测度、可测函数与 Lebesgue 积分](../tex/upper/chapters/ch03.tex) | [ch03_measure_integration](../notebooks/upper/ch03_measure_integration.ipynb) | 已修订；执行通过 |
| 上 04 | [$L^p$ 空间、收敛方式与乘积测度](../tex/upper/chapters/ch04.tex) | [ch04_lp_product_measure](../notebooks/upper/ch04_lp_product_measure.ipynb) | 已修订；执行通过 |
| 上 05 | [多元微积分与矩阵微分](../tex/upper/chapters/ch05.tex) | [ch05_matrix_calculus](../notebooks/upper/ch05_matrix_calculus.ipynb) | 已修订；执行通过 |
| 上 06 | [线性代数、分解、投影与条件数](../tex/upper/chapters/ch06.tex) | [ch06_linear_algebra](../notebooks/upper/ch06_linear_algebra.ipynb) | 已修订；执行通过 |
| 上 07 | [概率空间、随机变量与分布](../tex/upper/chapters/ch07.tex) | [ch07_probability_distributions](../notebooks/upper/ch07_probability_distributions.ipynb) | 已修订；执行通过 |
| 上 08 | [条件期望、独立性与概率核](../tex/upper/chapters/ch08.tex) | [ch08_conditioning](../notebooks/upper/ch08_conditioning.ipynb) | 已修订；执行通过 |
| 上 09 | [概率收敛、极限定理与集中不等式](../tex/upper/chapters/ch09.tex) | [ch09_limit_theorems](../notebooks/upper/ch09_limit_theorems.ipynb) | 已修订；执行通过 |
| 上 10 | [估计、检验、渐近、回归与多重检验](../tex/upper/chapters/ch10.tex) | [ch10_statistical_inference](../notebooks/upper/ch10_statistical_inference.ipynb) | 已修订；执行通过 |
| 上 11 | [随机过程、Markov、鞅、Poisson 与 Brownian](../tex/upper/chapters/ch11.tex) | [ch11_stochastic_processes](../notebooks/upper/ch11_stochastic_processes.ipynb) | 已修订；执行通过 |
| 上 12 | [时间序列、预测与状态空间基础](../tex/upper/chapters/ch12.tex) | [ch12_time_series](../notebooks/upper/ch12_time_series.ipynb) | 已修订；执行通过 |
| 上 13 | [凸优化、KKT 与对偶](../tex/upper/chapters/ch13.tex) | [ch13_convex_optimization](../notebooks/upper/ch13_convex_optimization.ipynb) | 已修订；执行通过 |
| 上 14 | [浮点数、数值线性代数与病态问题](../tex/upper/chapters/ch14.tex) | [ch14_numerical_stability](../notebooks/upper/ch14_numerical_stability.ipynb) | 已修订；执行通过 |
| 上 15 | [Monte Carlo、bootstrap 与方差缩减](../tex/upper/chapters/ch15.tex) | [ch15_monte_carlo](../notebooks/upper/ch15_monte_carlo.ipynb) | 已修订；执行通过 |
| 上 16 | [研究有效性、样本外与交易摩擦](../tex/upper/chapters/ch16.tex) | [ch16_research_validity](../notebooks/upper/ch16_research_validity.ipynb) | 已修订；执行通过 |
| 上 17 | [Capstone：从三笔交易检验研究主张](../tex/upper/chapters/ch17.tex) | [ch17_research_audit](../notebooks/upper/ch17_research_audit.ipynb) | 已修订；执行通过 |
| 下 01 | [Brainteaser：把陌生题压缩成可证明结构](../tex/lower/chapters/brainteasers.tex) | [ch01_brainteasers](../notebooks/lower/ch01_brainteasers.ipynb) | 已修订；执行通过 |
| 下 02 | [多因子模型与稳健估计：从预测回归到风险价格](../tex/lower/chapters/multifactor-model.tex) | [ch02_multifactor_model](../notebooks/lower/ch02_multifactor_model.ipynb) | 已修订；执行通过 |
| 下 03 | [正则化、数值实现与估计不确定性](../tex/lower/chapters/multifactor-estimation.tex) | [ch03_multifactor_estimation](../notebooks/lower/ch03_multifactor_estimation.ipynb) | 已修订；执行通过 |
| 下 04 | [组合构建、摩擦与可复现研究材料](../tex/lower/chapters/multifactor-research.tex) | [ch04_multifactor_research](../notebooks/lower/ch04_multifactor_research.ipynb) | 已修订；执行通过 |
| 下 05 | [时间序列、协整与状态估计：从单位根到 OU](../tex/lower/chapters/stat-arb-model.tex) | [ch05_stat_arb_model](../notebooks/lower/ch05_stat_arb_model.ipynb) | 已修订；执行通过 |
| 下 06 | [状态估计、切换与变点](../tex/lower/chapters/stat-arb-estimation.tex) | [ch06_stat_arb_estimation](../notebooks/lower/ch06_stat_arb_estimation.ipynb) | 已修订；执行通过 |
| 下 07 | [从预测到净收益的统计套利研究](../tex/lower/chapters/stat-arb-research.tex) | [ch07_stat_arb_research](../notebooks/lower/ch07_stat_arb_research.ipynb) | 已修订；执行通过 |
| 下 08 | [表格统计学习：从决策树到梯度提升](../tex/lower/chapters/ml-alpha-model.tex) | [ch08_ml_alpha_model](../notebooks/lower/ch08_ml_alpha_model.ipynb) | 已修订；执行通过 |
| 下 09 | [现代人工智能方法：表示学习、关系建模与序贯决策](../tex/lower/chapters/ml-alpha-deep.tex) | [ch09_ml_alpha_deep](../notebooks/lower/ch09_ml_alpha_deep.ipynb) | 已修订；执行通过 |
| 下 10 | [序列模型与大语言模型：从 RNN 到 LLM](../tex/lower/chapters/ml-alpha-sequence.tex) | [ch10_ml_alpha_sequence](../notebooks/lower/ch10_ml_alpha_sequence.ipynb) | 已修订；执行通过 |
| 下 11 | [选读：图神经网络、强化学习与多模态研究](../tex/lower/chapters/ml-alpha-frontiers.tex) | [ch11_ml_alpha_frontiers](../notebooks/lower/ch11_ml_alpha_frontiers.ipynb) | 已修订；执行通过 |
| 下 12 | [机器学习研究设计：验证、监控与 Alpha 决策](../tex/lower/chapters/ml-alpha-validation.tex) | [ch12_ml_alpha_validation](../notebooks/lower/ch12_ml_alpha_validation.ipynb) | 已修订；执行通过 |
| 下 13 | [Alpha 决策与研究材料：从损失到净收益](../tex/lower/chapters/ml-alpha-research.tex) | [ch13_ml_alpha_research](../notebooks/lower/ch13_ml_alpha_research.ipynb) | 已修订；执行通过 |
| 下 14 | [随机分析与无套利：从二次变差到风险中性测度](../tex/lower/chapters/derivatives-stochastic.tex) | [ch14_derivatives_stochastic](../notebooks/lower/ch14_derivatives_stochastic.ipynb) | 已修订；执行通过 |
| 下 15 | [定价、校准与对冲：从数值方法到误差分布](../tex/lower/chapters/derivatives-numerics.tex) | [ch15_derivatives_numerics](../notebooks/lower/ch15_derivatives_numerics.ipynb) | 已修订；执行通过 |
| 下 16 | [离散对冲与模型失效](../tex/lower/chapters/derivatives-hedging.tex) | [ch16_derivatives_hedging](../notebooks/lower/ch16_derivatives_hedging.ipynb) | 已修订；执行通过 |
| 下 17 | [投资组合与风险管理：从协方差估计到压力测试](../tex/lower/chapters/portfolio-risk-estimation.tex) | [ch17_portfolio_risk_estimation](../notebooks/lower/ch17_portfolio_risk_estimation.ipynb) | 已修订；执行通过 |
| 下 18 | [观点后验、尾部目标与稳健实施](../tex/lower/chapters/portfolio-risk-optimization.tex) | [ch18_portfolio_risk_optimization](../notebooks/lower/ch18_portfolio_risk_optimization.ipynb) | 已修订；执行通过 |
| 下 19 | [尾部、压力测试与模型风险治理](../tex/lower/chapters/portfolio-risk-tail.tex) | [ch19_portfolio_risk_tail](../notebooks/lower/ch19_portfolio_risk_tail.ipynb) | 已修订；执行通过 |
| 下 20 | [市场微观结构与执行：从订单簿到事件仿真](../tex/lower/chapters/microstructure-events.tex) | [ch20_microstructure_events](../notebooks/lower/ch20_microstructure_events.ipynb) | 已修订；执行通过 |
| 下 21 | [执行、做市与库存控制](../tex/lower/chapters/microstructure-control.tex) | [ch21_microstructure_control](../notebooks/lower/ch21_microstructure_control.ipynb) | 已修订；执行通过 |
| 下 22 | [事件时间仿真与研究项目](../tex/lower/chapters/microstructure-simulation.tex) | [ch22_microstructure_simulation](../notebooks/lower/ch22_microstructure_simulation.ipynb) | 已修订；执行通过 |

## 验证

39 本 Notebook 全部在独立内核执行成功。课程导航、符号与索引生成检查通过；模板文件完整性检查通过。所有数值练习并未因此被视为自动证明，执行成功也不代替数学审阅。新增及改动算例、答案已同步核对，修正尾部风险重复答案、ReLU 二阶矩表述等问题。

正文风格检查仍给出 7 条非阻断提示，均涉及部分章节段首标题数量；没有为清零提示而机械删除有效结构。PDF 由出版脚本生成，最终页数与版式抽查结果见本记录末尾。

最终出版验证：上册 142 页、下册 129 页、答案册 60 页。三册成功编译；最终编译无缺字或未定义引用，术语表有一处约 1.72pt 的轻微超宽提示。抽查线性代数推导、机器学习收益阈值与答案页面，未见正文或公式裁切。PDF 文本扫描未发现日志、哈希、退出码、校验值、审计轨迹或验收清单；答案册旧“恢复路径”整段移除。最终 `git diff --check` 通过。
