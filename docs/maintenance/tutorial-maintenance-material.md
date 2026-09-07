# 移出教程的维护材料

以下为逐章教学修订时移出的旧计算验收与打包说明，仅供维护历史核对，不作为读者练习。

## tex/upper/chapters/ch01-questions.tex

```tex
\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 用给定的 $r=(0.02,-0.01,0.03)$、$w=(0.5,0.3,0.2)$ 运行本章程序，核对组合收益与区间证据行，并说明输入单位。
  \item 修改参照数据，依次令正常权重和不为 1、正常权重含负值、反例权重和不为 1、反例权重全非负、收益与权重长度不同、输入包含非有限值；记录六条完整稳定诊断，并解释每个检查保护的数学定义。
  \item 写一个不调用被测矩阵乘法的独立参考实现：对整数基点收益和有理权重使用分数运算，验证至少五组凸组合，并输出任何越界见证。
\end{enumerate}


```

## tex/upper/solutions/ch01.tex

```tex
\section*{数值编程}
本章独立参照标记为 $0.013$ 与 $0.035$。
\begin{enumerate}[leftmargin=*]
  \item 三项贡献分别为 $0.010,-0.003,0.006$，所以 $R_p=0.013$；分量最小值为 $-0.01$、最大值为 $0.03$。运行
  \begin{center}
  \path{uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.allow_error_names=SystemExit notebooks/upper/ch01_convex_bound.ipynb}
  \end{center}
  应逐字包含以下字段：
  \begin{lstlisting}
oracle=passed weighted_return=0.013000
lower=-0.010000 upper=0.030000
counterexample=0.035000
  \end{lstlisting}
  \item 六种输入依次给出以下完整稳定诊断：
  \begin{lstlisting}
assumption gate failed: weights must sum to one
assumption gate failed: weights must be nonnegative
counterexample gate failed: weights must sum to one
counterexample gate failed: deleted nonnegativity assumption is absent
assumption gate failed: returns and weights must have equal length
assumption gate failed: returns and weights must be finite
  \end{lstlisting}
  它们依次保护正常案例的归一化与非负性、受控反例只删除一条假设、点积定义域和有限数值域。
  \item 可用 \texttt{Fraction} 保存整数基点收益与有理权重。对每组输入先用独立参照累加 $\sum_iw_ir_i$，再取最小值、最大值并比较。期望结果必须来自分数算术或解析记录，而不是调用待验收的 NumPy 点积。若权重非负且和为 1，任何越界都应作为程序或输入错误报告；若主动加入负权重，则记录权重、收益与越界方向作为失败见证。
\end{enumerate}


```

## tex/upper/chapters/ch02-questions.tex

```tex
\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 复现 $T(x)=0.02+0.8x$ 的完整五步记录，核对 $x_5=0.067232$、真实误差与先验/后验界均为 $0.032768$。
  \item 修改参照数据，依次令 $q=1$、声明点不满足 $T(x)=x$、见证下标为 0；记录三条完整诊断并解释对应的定理检查。
  \item 对 $f_n(x)=x^n$ 比较固定均匀网格最大误差与解析移动见证；改变网格密度和 $n$，展示为什么固定网格不能定义一致收敛 独立基准。
\end{enumerate}


```

## tex/upper/solutions/ch02.tex

```tex
\section*{数值编程}
本章独立标记为 $0.067232$、$0.032768$ 与 $1/2$。
\begin{enumerate}[leftmargin=*]
  \item 迭代依次为 $0.02,0.036,0.0488,0.05904,0.067232$。固定点是 $0.1$，所以真实误差为 $0.032768$。先验界
  \[
   \frac{0.8^5}{1-0.8}(0.02)=0.032768,
  \]
  后验界 $0.8/(1-0.8)\times(0.067232-0.05904)$ 也等于 $0.032768$。运行本章脚本后，证据行还应包含两个 \texttt{0.500000} 见证误差。
  \item 三条完整诊断依次为：
  \begin{lstlisting}
contraction gate failed: factor must satisfy 0 <= q < 1
fixed-point gate failed: declared point does not satisfy T(x)=x
witness gate failed: indices must be positive integers
  \end{lstlisting}
  它们分别防止把非压缩迭代套入 Banach 定理、用错误目标点定义误差，以及在 $2^{-1/n}$ 的定义域外制造见证。
  \item 对每个固定网格 $G$ 可计算 $\max_{x\in G}|x^n-f(x)|$，但该值只控制 $G$。解析见证 $x_n=2^{-1/n}$ 随 $n$ 接近 1，并始终给出 $1/2$。若网格不含 $x_n$，加大 $n$ 会把失败区推入最后两个网格点之间。只有解析上界、区间算术、可证明 Lipschitz 网格误差或直接求连续域上确界，才能把网格结果提升为一致误差证据。
\end{enumerate}


```

## tex/upper/chapters/ch03-questions.tex

```tex
\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 复现 $m=2,4,8$ 的有限和以及 $n=10,100$ 的尖峰面积、固定点值与交换差额，并逐项对照预先给定的解析参照；expected 不得由同一循环生成。
  \item 修改临时参照数据，分别注入 `NaN`、删去一个期望值、改变简单函数层级、尖峰下标和固定点；记录五类稳定诊断并解释它们保护的证据含义。
  \item 实现一个固定均匀网格积分器研究尖峰列；改变 $n$ 与网格规模，报告何时漏采质量，并与解析面积 $1$ 比较。
\end{enumerate}


```

## tex/upper/solutions/ch03.tex

```tex
\section*{数值编程}

\begin{enumerate}[leftmargin=*]
  \item 独立解析参照给出 $m=2,4,8$ 时 $0.375$、$0.46875$、$0.498046875$。尖峰面积恒为 $n(1/n)=1$；$x=0.2$ 对 $n=10,100$ 都在 $(0,1/n]$ 外，点值为零，因此极限交换差额为 1。运行本章契约后，stdout 必须逐项复现这些独立常数。
  \item `NaN` 应触发 `numeric gate failed`；期望数量错位触发 `expected counts must match input counts`；层级、尖峰下标和固定点的改写分别触发固定账本诊断。恢复时只还原临时 JSON；不能删测试、放宽 tolerance 或用被测结果重写 expected。
  \item 设网格点为 $j/M$。若最小正网格点 $1/M>1/n$，且未特别采样零点右侧，则所有采样值都可能为零，数值面积误报为零。即使 $M\geq n$，矩形规则的端点约定也会影响结果。报告 $M,n$、采样位置与误差，并把解析面积 $1$ 作为独立 oracle。
\end{enumerate}


```

## tex/upper/chapters/ch04-questions.tex

```tex
\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 复现 $p=1,2,4$ 的 $\lVert x\rVert_p$、$N=64$ 的中点积分、实际误差和解析上界；expected 必须独立写出。
  \item 复制参照数据后分别注入 `NaN`、改变 $p$ 值、网格数、截断尺寸和期望数组长度；记录五类稳定诊断。
  \item 编写 typewriter 序列实验：对每层输出单项 $L^p$ 范数，并跟踪一个非二进点的命中序列，验证范数趋零而点值不收敛。
\end{enumerate}


```

## tex/upper/solutions/ch04.tex

```tex
\section*{数值编程}

\begin{enumerate}[leftmargin=*]
  \item 独立解析值为
  \[
   0.5,\quad 3^{-1/2}=0.577350269\ldots,\quad
   5^{-1/4}=0.668740305\ldots.
  \]
  $N=64$ 时中点值为 $0.33331298828125$，解析积分为 $1/3$，差为 $2/(24\cdot64^2)=0.000020345052\ldots$，恰好达到二次函数的余项上界。程序输出必须逐项与这些独立解析常数比较。
  \item `NaN` 触发 `numeric gate failed: oracle scalars must be finite`；改写 $p$ 值、网格数、截断尺寸分别触发对应的固定账本诊断；删去一个期望数组元素触发 `expected counts must match input counts`。恢复方法是还原临时 JSON，不得放宽 tolerance 或用 observed 重写 expected。
  \item 第 $k$ 层每个示性函数支撑集长度 $2^{-k}$，所以 $L^p$ 范数为 $2^{-k/p}$。选择非二进点如 $x=1/3$，每层恰有一个区间包含它，因此逐层完整输出中总会出现一个 1；同层其余项为 0。随着 $k$ 增大范数趋零，但点值序列在 0 与 1 间反复，验证没有逐点收敛。
\end{enumerate}


```

## tex/upper/chapters/ch05-questions.tex

```tex
\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 复现固定二次型与复合函数的解析梯度，扫描多个差分步长，验证过小步长导致误差回升；解释为什么二次函数不能展示一般误差 U 形的截断侧。
  \item 从临时参照数据注入 `NaN`、错误点、错误步长、梯度长度错配、宽容差和错误发布标签，记录六类稳定诊断。
  \item 为二维均值--方差目标实现解析梯度、中心差分和一个故意漏掉对称化的错误梯度；用非对称矩阵证明测试能区分它们。
\end{enumerate}


```

## tex/upper/solutions/ch05.tex

```tex
\section*{数值编程}
\begin{enumerate}[leftmargin=*]
  \item 解析参照给出二次型值 $-1$、梯度 $(0,-0.5)^T$ 和链式梯度 $(4,-6.5)^T$。固定 $h=10^{-5}$ 时最大差异为 $2.62\times10^{-11}$，最大差异小于 $10^{-9}$。对 $|t|$，左差商为 $-1$、右差商为 $1$，因此零点不可微。固定扫描中，$h=10^{-12}$ 时误差超过 $10^{-6}$，明确展示消去误差回升。二次函数的三阶导数为零，中心差分没有通常的 $O(h^2)$ 截断主项，所以本实验不声称给出完整 U 形；最低点仍依赖机器和数据类型。
  \item `NaN` 触发稳定 numeric gate；错误点触发 analytic-ledger 诊断；错误步长触发 fixed-step 诊断；长度错配触发 fixed-dimensions；放宽容差触发 numeric-tolerances；错误发布标签触发 expected=-1。恢复时只还原临时 JSON，不得用 observed 改写 expected。
  \item 具体组合账本中 $\det\Sigma=1.75$，$\lambda\mu=(0.4,0.2)^T$，故最优权重为 $(0.171429,0.114286)^T$，目标值为 $-0.0457142857\ldots$，且驻点梯度为零。另取非对称 $A$，正确梯度为 $((A+A^T)/2)w-\lambda\mu$，错误实现为 $Aw-\lambda\mu$；中心差分只匹配正确的对称化公式。若只用对称矩阵，错误实现会被测试掩盖。
\end{enumerate}


```

## tex/upper/chapters/ch07-questions.tex

```tex
\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 复现固定分布记录，并让程序在不抽样的情况下核验 CDF、平方推前律和联合协方差。
  \item 直接运行程序核验 Bernoulli MGF、Poisson PGF、正态分布 MGF/CDF、Uniform/Exponential/Lognormal 矩；逐项读取解析参照并标注误差门，不要求先手工复算复杂数值。
  \item 从临时参照数据注入 NaN expected、缺字段、错误形状、负质量、宽容差和伪造 expected，记录六类稳定诊断。
\end{enumerate}


```

## tex/upper/solutions/ch07.tex

```tex
\section*{数值编程}
\begin{enumerate}[leftmargin=*]
  \item oracle 不抽样，直接用固定质量向量求和。它核验
  \[
    (F(-1),F(0),F(1),F(2))=(0.25,0.75,0.75,1),
  \]
  平方推前在 $(0,1,4)$ 上的质量为 $(0.5,0.25,0.25)$；联合表独立性最大差为 $0.125$、协方差为 $0.5$。这些 expected 来自分数账本而非 NumPy 输出。
  \item 解析账本给出 $M_B(\log 2)=5/4$、$G_N(1/2)=e^{-1}$、$M_Z(1/2)=e^{1/8}$ 和 $\Phi(1)=0.841344746\ldots$；$U(-1,3)$ 的均值/方差为 $1,4/3$；$\mathrm{Exp}(2)$ 的均值和生存概率为 $1/2,e^{-2}$；Lognormal$(0,0.5^2)$ 的均值/方差为 $1.133148,0.364696$。Pareto 截断到 $10$ 与 $1000$ 时分别为 $2.302585$、$6.907755$。全部使用固定 $10^{-10}$ 绝对误差门。
  \item NaN expected 触发有限值门，删除正态分布 MGF 触发缺字段门，单元素边缘触发固定形状门，负质量触发非负门，更换支持值触发固定账本门，宽容差触发 tolerance 门，错误顶层 expected 触发 $1/4$ 发布标签门。恢复时只改临时 JSON，不得把 observed 写回 expected。
\end{enumerate}


```

## tex/upper/chapters/ch09-questions.tex

```tex
\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 实现打字机、稀有尖峰与 $L^1$ 非 $L^2$ 三个反例记录，逐项报告偏差概率和矩。
  \item 精确枚举二项尾概率与标准化 CDF 距离；独立计算 Chebyshev、Hoeffding、Bernstein 和 Berry--Esseen 界。
  \item 用固定种子复现 Bernoulli 与 Cauchy 实验，并注入伪造尾概率、放宽容差、改变 canonical 设计和完全依赖方差。
\end{enumerate}


```

## tex/upper/solutions/ch09.tex

```tex
\section*{数值编程}
\begin{enumerate}[leftmargin=*]
  \item 打字机第 $m$ 层每个区间概率 $2^{-m}$，但每点每层被覆盖一次；稀有尖峰在 $n=(10,100,1000)$ 的偏差概率为 $(0.1,0.01,0.001)$、$L^1$ 距离均为 $1$；$\sqrt n$ 尖峰的 $L^1$ 距离为 $n^{-1/2}$、$L^2$ 距离平方恒为 $1$。这些解析账本分别阻断 $\mathbb P\Rightarrow a.s.$、$\mathbb P\Rightarrow L^1$ 与 $L^1\Rightarrow L^2$。
  \item 精确二项尾概率对满足 $|k/200-0.3|\ge0.1$ 的质量求和，得到 $0.00256512$。标准化 CDF 在每个跳点同时比较左极限与右值，sup 距离为 $0.03480918$，小于 Berry--Esseen 界 $0.05011773$。三种集中界为 $(0.105000,0.036631,0.032829)$；它们均不小于精确尾概率。
  \item 固定种子实验输出均值 $0.299794$、标准差 $0.032672$、覆盖率 $0.943100$ 和模拟尾概率 $0.002850$。Cauchy 的中位绝对均值为 $(0.980865,1.029577)$。测试分别篡改精确尾概率、正态距离、Bernstein 界、完全依赖方差、canonical $p$、容差和有限性；每次都必须非零退出，不能通过放宽门限修复。
\end{enumerate}


```

## tex/upper/chapters/ch10-questions.tex

```tex
  \item \textbf{数值编程：}对同一均值比较 iid、AR(1) 与簇相关的解析方差；说明为什么不能用 iid bootstrap 为后两者背书。

```

## tex/upper/chapters/ch10-questions.tex

```tex
  \item \textbf{数值编程：}实现 Bonferroni、Holm 与 BH，复现固定 $p$ 值序列的拒绝数 $(2,2,4)$，并为排序或停止规则错误写负例。

```

## tex/upper/chapters/ch10-questions.tex

```tex
  \item \textbf{数值编程：}扩展 独立基准，使其验证 plug-in $\hat p^2$ 的有限样本偏差与 Delta 方差；用独立公式写死期望值。

```

## tex/upper/solutions/ch11.tex

```tex
\section*{数值编程}
\begin{enumerate}[leftmargin=*,start=7]
  \item 对给定链，$P^5$ 给出 $(0.612500,0.387500)$，稳态分布为 $(0.600000,0.400000)$。交替链偶数步为 $(1,0)$、奇数步为 $(0,1)$，周期 2，直接反驳“有稳态便收敛”。条件结构反例中的条件概率分别为 $0.5$ 与 $1$。
  \item 率 2 的等待均值和方差为 $0.5,0.25$；率 2 与 3 叠加为 5；率 5 按 $0.3$ 保留后两流为 $1.5,3.5$；$\int_0^3(2+t)dt=10.5$。模拟中 Poisson 计数的均值与方差为 $5.980833$ 与 $5.988466$。
  \item $c=4,t=1/4$ 时缩放两边方差为 1；$n=100$ 的归一化随机游走方差为 1，Donsker 四阶矩基线为 $2.980000$，但函数空间弱收敛不等于有限样本逐路径相等。Brownian 协方差最大误差为 $0.006948$；二次变差均值为 $1.000731$，总变差基线从 $3.191538$ 增至 $12.766153$。
\end{enumerate}


```

## tex/upper/solutions/ch12.tex

```tex
\section*{数值编程}
\begin{enumerate}[leftmargin=*,start=7]
\item 完整实现位于 \path{notebooks/upper/ch12_time_series.ipynb}。程序先按 $\sigma^2/(1-\phi^2)$ 与系数和计算解析矩，再用 \texttt{default\_rng(20260727)} 生成独立路径；模拟矩为 $(-0.013814,2.764404,0.799071)$，只在预先冻结的容差内交叉验证解析 $(0,2.777778,0.8)$。ARMA 部分独立计算根、$\sum\psi_j^2$ 与 $\sum\psi_j\psi_{j+1}$，不得从同一模拟结果反算理论量。运行命令为：
\begin{lstlisting}[language=bash]
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.allow_error_names=SystemExit notebooks/upper/ch12_time_series.ipynb
\end{lstlisting}

\item 程序直接检查随机游走时长 $1,10,100$ 的方差 $1,10,100$，并按 $0.36/(1-0.4^2)$ 与 $0.4-1$ 得到协整价差方差 $0.428571$、调整系数 $-0.6$。对固定种子的平稳价差使用长度 120 的真实滚动窗口，480 个 AR 斜率的最小值、中位数和最大值为 $(0.299681,0.459715,0.558824)$。结构断点过程另做两次分段 OLS，断点前后斜率为 $0.824923$ 与 $-0.215817$；随机切分和时间切分 MSE 为 $1.436939$ 与 $2.226153$。水平回归 $R^2=0.344011$、差分回归 $R^2=0.000216$；这些输出共同区分单位根、均值回复与制度变化。

\item 实现按“预测方差 $\rightarrow$ 创新 $\rightarrow$ 创新方差 $\rightarrow$ 增益 $\rightarrow$ 更新”循环，并保存
\[
\begin{array}{c|ccc}
t&1&2&3\\\hline
\nu_t&1&-1.055556&0.165385\\
S_t&2.25&1.805556&1.696154\\
K_t&0.555556&0.446154&0.410431
\end{array}
\]
\par\noindent{\footnotesize 三次滤波均值依次为 $0.555556$、$0.084615$ 与 $0.152494$。}\par
反向 RTS 循环产生平滑均值
\[
(0.260771,0.128118,0.152494).
\]
前两项不同正是未来观测进入平滑器的证据。命令行逐步打印 $\nu_t,S_t,K_t,\ell_t$，并另行打印信息边界：
\begin{center}
\texttt{boundary=(filter<=t,smooth<=T)}
\end{center}
测试 \path{test_chapter_twelve_notebook_reproduces_time_series_oracles} 逐字锁定账本、信息边界与空标准错误流。
\end{enumerate}


```

## tex/upper/solutions/ch13.tex

```tex
\section*{数值编程}
\begin{enumerate}[leftmargin=*,start=7]
\item \path{notebooks/upper/ch13_convex_optimization.ipynb} 从 $(2,0)$ 以步长 $0.2$ 做 500 次投影梯度，得到数值解为 $(0.500000,0.500000)$。程序从数值候选而非预先写入的解析目标值重新计算证书：驻点残差为 $2.220\times10^{-16}$，原可行、对偶可行、互补残差与原始--对偶间隙为零。命令 \texttt{--audit-candidate 0.6 0.4} 会稳定拒绝被扰动候选。负例还确认驻点 $x=0$ 的目标值为 1，并确认资格失败问题的 KKT 驻点残差恒为 1。
\item 程序分别计算 $p^*(1.2)=0.36$、解析乘子 $0.6$，以及步长 $10^{-5}$ 的中心差分 $\{p^*(b+\delta)-p^*(b-\delta)\}/(2\delta)=0.6$。oracle 冻结右端项、差分步长与容差；同步修改输入和预期仍会被 canonical design 拒绝。
\item 对给定 $\Sigma$ 解线性方程而不显式求逆，再按预算归一化，得到权重 $(0.711864,0.288136)$、方差 $0.030203$、条件数 $2.308723$。把非对角元从 $0.006$ 改为 $0.02$ 后，权重为 $(0.777778,0.222222)$、方差 $0.035556$、条件数 $2.941260$，权重 $L^1$ 变化相对输入改变量的放大比为 $9.416196$。两矩阵都正定且约束未改变，因此差异来自输入敏感度，而不是求解失败。
\end{enumerate}


```

## tex/upper/solutions/ch14.tex

```tex
\section*{数值编程}
\begin{enumerate}[nosep,leftmargin=*,start=7]
\item 对含 10 个小量的固定序列，朴素、成对、补偿与 Decimal 求和分别得到 $0$、$8$、$10$、$10$。成对树必须冻结；它改善误差增长阶数，但本例仍损失两个单位。
\item 用构造真值 $(1,1)$ 比较三条路径。正规方程、显式 QR 与显式 SVD 的相对系数误差分别约为 $2.221\times10^{-4}$、$1.000\times10^{-4}$ 与 $1.000\times10^{-4}$；相对残差分别约为 $4.984\times10^{-11}$、$0$ 与 $2.220\times10^{-16}$，三者对该矩阵均报告秩 2。独立秩亏矩阵的第二奇异值约 $7.32\times10^{-16}$，应报告不可识别方向而非强行解释系数。
\item 对 $(1000,1000)$，朴素指数上溢；减去最大值后，稳定 log-sum-exp 为 $1000.693147$。比较器使用 $|a-b|\le\mathrm{atol}+\mathrm{rtol}\max(|a|,|b|)$，并先拒绝 NaN/Inf。固定缩放实验中，列缩放把条件数从 $1.000\times10^9$ 降到 $1.407\times10^1$。
\end{enumerate}


```

## tex/upper/chapters/ch15-questions.tex

```tex
\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 用解析积分、两个样本量和多个独立 seeds 核对 $N^{-1/2}$ 误差率，并提交误差预算。
  \item 实现控制变量实验；故意把已知均值改为 $0.6$，解释为何 VRF 不变而答案错误。
  \item 穷举 $(1,2,4)$ 的 27 个 bootstrap 均值，再用随机重抽样逼近精确分布并比较误差。
\end{enumerate}


```

## tex/upper/solutions/ch15.tex

```tex
\section*{数值编程}
证据摘要为：标准误按 $N^{-1/2}$ 缩小；解析最优系数为 $1$；理论方差缩减倍数为 $16$；穷举 27 个重抽样均值时，bootstrap 方差为 $14/27$。
\begin{enumerate}[leftmargin=*]
  \item 独立 oracle 是 $\mathbb E[U^2]=1/3$ 与 $\operatorname{Var}(U^2)=4/45$。对每个 $N$ 保存独立重复的均方根误差，检查大样本与小样本 RMSE 比接近样本量比的平方根；单次结果只要求落在预先声明的多个标准误内。
  \item 正确调整量为 $U^2-(U-1/2)$。把中心改为 $0.6$ 后，调整量整体增加 $0.1$；加常数不改变方差，因此 VRF 仍约 16，但期望从 $1/3$ 变为 $1/3+0.1$。这证明方差缩减验收必须同时检查目标均值。
  \item 先用有理数笛卡尔积生成全部 27 个状态并保存精确频数，再用 $B$ 次随机重抽样比较每个支持点概率或均值、方差。随机误差应随 $B^{-1/2}$ 缩小，但精确答案不能由随机版本生成。
\end{enumerate}


```

## tex/upper/chapters/ch16-questions.tex

```tex
\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 为每行实现 $t_e\le t_a\le t_d<t_y$ 检查；构造一个只改可得时间的泄漏 fixture，验证非零退出与稳定诊断。
  \item 给滚动标签区间实现 overlap 检查，自动生成 purge 集合；用一个反例证明随机切分会共享未来价格。
  \item 扩展本章基准：加入未成交比例和规模相关冲击，分别报告成交条件下净收益与包含机会成本的实现差额。
\end{enumerate}


```

## tex/upper/solutions/ch16.tex

```tex
\section*{数值编程}
\begin{enumerate}[leftmargin=*]
  \item 解析 ISO 时间后逐行检查两个复合条件。正常行必须满足 $t_e\le t_a\le t_d$ 且 $t_d<t_y$；把第一行 $t_a$ 改为 9:31，而 $t_d$ 保持 9:25，应稳定失败为“row 1 was not available before the decision”。不要把当前日期或机器时区作为 oracle。
  \item 把每个样本标签表示为半开区间 $[s_i,e_i)$。验证折为 $V$ 时，从训练候选中删除所有满足
  \[
    [s_i,e_i)\cap[s_j,e_j)\ne\varnothing,\qquad j\in V
  \]
  的样本。反例可取相邻两日信号都使用未来五日收益：随机分到两折后仍共享四个未来价格。
  \item 为每个订单记录决策基准价、可成交数量、实际成交均价和期末基准价。成交部分扣佣金、价差与规模冲击；未成交部分的机会成本按预先声明的基准计算。分别报告 executed-only P\&L 与 implementation shortfall，不能把未成交订单当作零成本成交。
\end{enumerate}


```

## tex/upper/chapters/ch17-questions.tex

```tex
\section*{章末练习}
\begingroup\small
\subsection*{口述概念}
\begin{enumerate}[nosep,leftmargin=*]
  \item 区分字节完整性、机械复现、统计有效性与可交易性，并各举一个“前者通过、后者失败”的例子。
  \item 为什么“限制报告有六条”不等于“限制报告没有被篡改”？
  \item 审计者为什么要把独立参考账本与被测实现分开？
\end{enumerate}

\subsection*{笔试推导}
\begin{enumerate}[nosep,leftmargin=*]
  \item 从假设族和成本分项完整推导阈值 $0.0025$ 与净贡献 $0.034$，写明每个等式的假设。
  \item 给出 $t_e\le t_a\le t_d<t_y$ 与版本化日历的审计谓词，并说明哈希为何不能推出该谓词。
  \item 对 $(10^{16},1,-10^{16})$ 模拟 IEEE 双精度左到右累加，解释为何稳定和应为 $1$。
\end{enumerate}

\subsection*{数值编程}
\begin{enumerate}[nosep,leftmargin=*]
  \item 从空目录复制声明资产并运行 Capstone；证明删掉原仓库数据后临时包仍能独立执行。
  \item 修改第一行可得时间并更新 CSV 哈希，确认审计因时间泄漏而不是哈希失败。
  \item 保持限制条数不变，把“不能建立泛化”改成“证明广泛泛化”，确认报告哈希检查拒绝。
\end{enumerate}

\subsection*{研究判断}
\begin{enumerate}[nosep,leftmargin=*]
  \item 当前包通过全部检查。写出三条可以声称、三条不可声称的结论，并为每条定位证据或缺口。
  \item 如果研究者说实际尝试了 200 个配置而非 20 个，你如何更新阈值、报告与样本外流程？
  \item 若把合成数据换成真实分钟数据，现有证据矩阵还缺哪些数据、成交、许可和数值审计？
\end{enumerate}
\endgroup

\subsection*{基础衔接：独立计算}
在本章固定成交假设下，把成本压力系数改为 2，再求总净贡献。该计算能否估计实际可部署资金上限？

```

## tex/upper/solutions/ch17.tex

```tex
\chapter{第 17 章 Capstone 分层答案}
\label{app:ch17-solutions}

\section*{口述概念}
\begin{enumerate}[leftmargin=*]
  \item 字节完整性说明文件与冻结哈希一致；机械复现说明同一包得到同一输出；统计有效性说明错误率与估计在假设下可信；可交易性说明订单在市场约束和成本下可实现。哈希正确的数据仍可泄漏；可复现的 $p$ 值仍可漏算尝试；统计显著信号仍可能无法成交。
  \item 条数只约束结构。攻击者可把一条限制改成夸大声明而保持六个 bullet；对报告字节做校验值，才能检测这种保持计数的篡改。哈希仍不判断限制是否充分，内容审查不可省略。
  \item 独立参考账本把期望值与被测实现分离。若程序先算净收益，再让测试读取该输出作为 oracle，符号、单位或成本遗漏会同时污染结果与预期；解析记录、第二算法或固定 fixture 则能与实现真正发生分歧。
\end{enumerate}

\section*{笔试推导}
\begin{enumerate}[leftmargin=*]
  \item 假设族完整且目标为 Bonferroni FWER 时，阈值 $0.05/20=0.0025$；$0.002$ 通过。三项是同一资本基数的可加收益贡献，故毛贡献 $0.04$。三笔都发生且固定单笔成本 $0.002$，总成本 $0.006$，净贡献 $0.034$。若复利、权重或成交状态改变，等式须重写。
  \item 对每行解析带时区时间并要求 $t_e\le t_a\le t_d<t_y$，再要求 $t_d$ 的本地日期属于冻结日历且不早于测试起点。哈希只比较文件字节，无法理解时间语义；泄漏 CSV 可以拥有完全合法的新哈希。
  \item 双精度在 $10^{16}$ 附近的相邻可表示数间隔大于 $1$，所以第一步 $10^{16}+1$ 舍入回 $10^{16}$，再加 $-10^{16}$ 得 $0$。精确和为 $1$；稳定归约保留低位贡献并返回 $1$。
\end{enumerate}

\section*{数值编程}
\begin{enumerate}[leftmargin=*]
  \item 测试复制 \path{data/ch17} 与 \path{evidence/ch17} 到独立 root，再显式传入 oracle 与 package root。成功输出仍为 \texttt{package=upper-capstone-v1} 和 \texttt{net=0.034000}，证明不依赖原输入目录。
  \item 把 8:00 改为 9:31 后重算 SHA-256，完整性门禁通过，时间门禁以非零状态报告第一行在决策时不可得。若不更新哈希，只能测到 checksum mismatch，不能证明泄漏检查有效。
  \item 保持六个 bullet，只替换泛化措辞。程序应报告由下列两段以一个空格连接而成的稳定诊断：
  \begin{quote}\ttfamily
  report gate failed:\\
  limitation report checksum mismatch
  \end{quote}
  恢复原字节后再运行，确认不是路径或编码问题。
\end{enumerate}

\section*{研究判断}
\begin{enumerate}[leftmargin=*]
  \item 可以声称：当前合成 CSV 与冻结哈希一致；三行满足声明时间门禁；给定固定账本可复现净贡献 $0.034$。不可声称：真实市场有该收益；样本足以泛化；十倍资金仍保持净绩效。前三项分别由哈希、timeline 和 ledger 输出证明；后三项被合成数据、小样本与无容量模型直接限制。
  \item 若完整族为 200，Bonferroni 阈值改为 $0.05/200=0.00025$，原 $p=0.002$ 不通过。应更新 append-only 尝试账本与 oracle，承认先前报告低估搜索；若最终测试已被反复观察，应退回开发证据并另设未见时间区间，而不是只改阈值。
  \item 真实分钟数据还需：供应商与字段许可、原始文件校验、时区与交易日历、历史资产池和公司行动、报价/成交 schema、延迟和未成交、价差与冲击校准、容量压力情景、缺失/异常处理、聚合与浮点稳定、数据版本和下载时间。真实数据不能成为自身正确性的唯一 oracle。
\end{enumerate}

\section*{代码恢复路径与提交模板}
依次保存三条命令结果：正常 notebook 构建；独立 package root 的直接审计；时间泄漏或报告篡改负例。最终证据摘要逐行保留以下四个标记：
\begin{quote}
\small\ttfamily
package=upper-capstone-v1\\
data=research\_rows.csv rows=3\\
gross=0.040000,cost=0.006000,net=0.034000\\
numeric=passed license=CC0-1.0 licenses=4 limitations=6
\end{quote}
% published-marker: data=research_rows.csv rows=3

提交模板分四段：\textbf{声明}只写当前包能证明的结果；\textbf{证据}列文件、哈希、命令与输出；\textbf{失败}列注入方式、非零状态和稳定诊断；\textbf{限制}逐条保留合成数据、小样本、加法收益、固定成本、教学日历和非盈利证明。任何删除限制换取更漂亮叙事的修改，都必须被审计拒绝。

\section*{逐题逐步核对}
\begin{enumerate}[leftmargin=*]
  \item 四层证据要逐层验收：先比较 package 字节 hash，再在独立 root 运行得到同一 stdout，再核对统计门禁/错误率，最后按点时数据、成交约束和成本检查可交易性。任何一层失败都不能被下一层的漂亮数字覆盖。
  \item 报告完整性不仅是六个限制条目数量，还要冻结每条文本的字节序列和 hash。把一条限制改成夸大表述、保持条数不变时，hash/报告 gate 应失败；因此必须同时保留内容审查和机械校验。
  \item 先在独立文件中写 gross、每类 cost、net、时间门禁、许可和 limitations 的 expected，再运行被测实现；程序只能生成 observed。若把 observed 写回 ledger，符号、单位或成本遗漏会同时污染两边，测试失去区分力。
  \item 假设族完整时 Bonferroni 阈值是 .05/20=.0025，$p=.002$ 通过。收益账本逐笔相加 $0.02+0.01+0.01=.04$，成本 $3\times.002=.006$，净值 .034；若有复利、权重或未成交状态，必须改写对应公式并在限制中说明。
  \item 每行先解析时区，再检查 $t_e\le t_a\le t_d<t_y$ 和冻结日历；修改 \texttt{available\_at} 到 \texttt{decision\_at} 之后，应报告第一行不可用。即使重新计算 hash，时间门禁仍必须失败，证明 hash 不是泄漏检测的替代品。
  \item 独立 package root 的复现要显式传入 data/evidence/oracle 路径，确认不依赖当前工作目录、缓存或未声明环境变量。时间负例、报告文字负例和 numeric expected 负例分别验证三类 gate，恢复后逐项说明为什么增加样本或换 seed 无法修复。
  \item 研究判断只允许声称“当前合成包、三行数据、固定账本和哈希证明的内容”；不得外推市场收益、泛化或容量。真实数据迁移还需补许可证、历史资产池、公司行动、时区、延迟、未成交、冲击、版本和独立 oracle。
  \item 最终提交按声明/证据/失败/限制四段组织，并逐行保留命令、输入 hash、stdout、stderr、退出码和版本。这样读者可以从一条限制追到失败 fixture，也能区分“计算成功”“统计证据足够”和“结论可行动”。
\end{enumerate}

\section*{基础衔接：独立计算}
$0.04-2(0.006)=0.028$。不能据此求容量，因为没有规模到参与率、成交、冲击和毛收益的映射。

```

## tex/upper/main.tex

```tex
\chapter{实验结果的整理与复核}
\input{tex/upper/guides/reproducibility}
```

## tex/upper/guides/reproducibility.tex

```tex
\section{复现、数值与证据边界}

研究有效性也会被数值计算破坏。病态协方差矩阵可让优化权重对微小数据改动剧烈跳变；长序列收益连乘可能溢出或下溢；并行归约可能改变末位；过紧的浮点相等测试会制造假失败。第 14 章的条件数、后向误差、稳定算法和尺度化容差必须进入研究报告。数值稳定不能修复统计泄漏，但统计设计也不能替代数值检查。

为了让另一位研究者重算，至少要保留：
\begin{itemize}[leftmargin=*]
  \item 数据快照、schema、校验值、时区、日历与许可；
  \item 代码提交、依赖锁、随机源与硬件/软件环境；
  \item 假设族与尝试账本、训练/验证/测试边界；
  \item 独立 oracle、预期输出、容差、负例和失败消息；
  \item 毛收益、逐项成本、未成交、容量假设和限制声明。
\end{itemize}
复现只说明“相同输入和环境可以得到相同输出”，不证明模型真实、统计显著或可交易。它是研究可信度的一部分，而不是结论本身。

\MFQLead{把研究设计变成可观察结果}

配套计算把上述问题变成可观察的证据：逐行检查
$t_e\le t_a\le t_d<t_y$，核对交易日历和时间切分，计算多重检验阈值，并重算 $0.04-0.006=0.034$。把一行数据的可得时间改到决策之后，或把 $p$ 值改成 $0.003$，都应得到与原假设相对应的失败结果；正文关注的是这些变化为什么会改变结论，而不是某个工具的调用方式。
```

## tex/lower/templates/capstone-evidence.tex

```tex
\section{统一研究项目}

六条路线共享一份研究项目模板。模型名称可以变化，但读者始终需要知道研究对象、信息边界和结论的限度。

\begin{longtable}{p{0.20\textwidth}p{0.70\textwidth}}
\toprule
研究项目的组成 & 需要回答的问题 \\
\midrule
研究问题 & 可证伪的目标是什么？评价指标与决策用途是什么？\\
数据与时间协议 & 数据在何时可得？训练、验证、最终测试如何按时间隔离？数据来源与许可是什么？\\
独立基线 & 哪个手算、简单模型或外部实现独立约束主模型？\\
主模型 & 假设、目标函数、估计或校准步骤及数值容差是什么？\\
样本外设计 & 选择权在哪里消耗？哪些窗口只用于最终一次评估？\\
成本与容量 & 手续费、价差、冲击、换手、借券或未成交怎样进入净结果？\\
故障注入 & 一个时间泄漏、输入损坏、数值病态或制度摩擦的反例怎样被构造并识别？\\
限制报告 & 哪些结论只在当前数据、市场制度、频率和模型假设下成立？\\
一键复现 & 另一位读者怎样从相同输入生成表格和图形？\\
\bottomrule
\end{longtable}

\MFQLead{四类证据}

每个方向模块和 Capstone 都必须同时给出：数学推导证据、独立计算证据、失败边界证据、研究有效性证据。只展示成功回测不构成完整交付。

\MFQLead{陈述边界}

学习项目不得声称生产部署、实盘业绩或团队经验，除非作者能提供可核查且获准公开的外部证据。可诚实陈述的是：完成了什么可复现实验、哪些测试通过、哪些假设尚未验证，以及如果进入真实组织还需补哪些控制。

\MFQLead{导论四级练习}

\begin{enumerate}
  \item[口述题] 用一分钟说明“高样本外收益”为什么不能单独证明研究有效。
  \item[推导题] 若每题得分为 $s_i\in\{0,1,2\}$，写出总分 $S=\sum_{i=1}^{10}s_i$，并说明为什么总分相同的两名读者可能需要不同桥接单元。
  \item[计算题] 某路线十题得分为 $(2,2,1,0,2,1,2,0,1,2)$。计算总分、判定入口，并列出 0 分题对应的回看动作。
  \item[研究判断题] 一份 Capstone 只有主模型、回测曲线和最终测试集表现。指出至少四项缺失证据，并给出最小修复顺序。
\end{enumerate}

\MFQLead{分级反馈}

口述题应同时提到选择偏差、时间协议与摩擦；推导题的关键不是求和，而是诊断向量不能被总分充分概括；计算题总分为 13，应先完成桥接，并优先补两道 0 分题；研究判断题至少应补数据与时间协议、独立基线、样本外选择记录、成本容量、故障注入和限制报告。若答案只说“多做测试”，仍未形成可审计的修复计划。
```

## tex/upper/solutions/ch10.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item \textbf{口述。}无偏要求每个 $n$ 下 $\mathbb E\hat\theta_n=\theta$；一致要求误差超过任意固定阈值的概率趋零；渐近正态描述适当缩放后的极限分布；有效性只在同一目标和正则估计量类中比较渐近方差。若 $X_i\sim\mathcal N(\theta,1)$ 且 $T_n=X_1$，则 $T_n$ 无偏但方差恒为 $1$，故不一致。
  \item \textbf{口述。}大小是零假设集合上的最坏误拒概率，功效是在给定备择下拒绝的概率；$p$ 值是零假设下尾概率；覆盖率是随机区间在重复抽样中覆盖固定真值的频率；经济效应是目标量本身的实际尺度。前四项都不能替代成本后效应。
  \item \textbf{口述。}经典公式要求条件同方差且不相关；HC 允许异方差；HAC 允许声明带宽内的序列相关；簇稳健允许簇内任意相关而依赖簇间独立及足够簇数。它们都不修复遗漏变量、泄漏、反向因果、错误函数形式或无效识别。
  \item \textbf{推导。}写 $\hat\theta-\theta=(\hat\theta-\mathbb E\hat\theta)+b$。平方取期望，交叉项为 $2b\mathbb E(\hat\theta-\mathbb E\hat\theta)=0$，故 MSE 等于方差加 $b^2$。收缩估计偏差 $-0.2$、方差 $0.64\times0.25=0.16$、MSE $0.20$，低于无偏估计的 $0.25$。
  \item \textbf{推导。}$Q(b)=(y-Xb)^T(y-Xb)$，所以 $\nabla Q=-2X^Ty+2X^TXb$。满秩时令其为零得 $\hat\beta=(X^TX)^{-1}X^Ty$；于是 $X^T\hat\varepsilon=X^T(y-X\hat\beta)=0$，即残差与每个回归量样本列正交。
  \item \textbf{推导。}将 $z=\pi x+v$ 且 $\pi=\operatorname{Cov}(x,z)/\operatorname{Var}(x)$ 代入 $y=\beta_xx+\beta_zz+\varepsilon$，短回归极限斜率为 $\beta_x+\beta_z\pi$。固定参数给出 $1+2(0.5)/1=2$，偏差为 $1$。稳健协方差只包围这个极限，不会改变它。
  \item \textbf{编程。}对 $\hat p$，$\mathbb E\hat p^2=p^2+p(1-p)/n$，所以 $p=0.4,n=250$ 时 plug-in 偏差为 $0.000960$。$g'(p)=0.8$，故 Delta 方差为 $0.8^2\times0.4(0.6)/250=0.0006144$。答案应从公式写死两项期望，再篡改任一字段验证程序非零退出。
  \item \textbf{编程。}排序序列为 $(.001,.009,.021,.040,.200)$。Bonferroni 阈值 $.01$，拒绝前两项；Holm 阈值依次为 $.01,.0125,.016\overline6$，在第三项停止，仍为两项；BH 阶梯为 $.01,.02,.03,.04,.05$，最大满足索引为 $4$。负例应覆盖未排序、Holm 不停止和 BH 取首个而非最大索引。
  \item \textbf{编程。}iid 均值方差为 $1/100=0.01$。AR(1) 为 $[100+2\sum_{h=1}^{99}(100-h)0.6^h]/100^2=0.03925$。五个一簇、相关系数 $.4$ 的设计效应为 $1+4(.4)=2.6$，故方差 $0.026$。逐点 iid bootstrap 会打散相关结构，因此不能作为后两项的独立证据。
  \item \textbf{研究判断。}先把每日查看次数、统计量和最长试验期登记；选择一个 alpha-spending/序贯检验方案，或禁止中途停止；探索结束后冻结策略，只在未查看的后移确认区间运行一次。报告所有查看、退出规则、效应区间及未通过结果。十次朴素查看的误拒概率为 $0.401263$。
  \item \textbf{研究判断。}HC3 只处理异方差。还需排查行业变量是否是碰撞点或处理后变量、遗漏质量/规模暴露、测量误差、非线性、反向因果、幸存者偏差和时间泄漏。因果声明需要明确处理、潜在结果/结构模型、识别假设、安慰剂和对假设的敏感性分析。
  \item \textbf{研究判断。}先冻结 500 项检验族、BH 水平与依赖假设，保存全部结果；对入选项在未参与选择的时间后移样本中重估效应与依赖稳健区间；再做数据可得性、衰减、换手、佣金、价差、冲击、借券与容量压力测试；最后报告失败项和限制。BH 控制特定条件下的错误发现比例，不保证方向稳定或净收益为正。
\end{enumerate}

\section*{独立 oracle 核对表}
收缩例的 MSE 是 $0.20$，小于无偏估计的 $0.25$；它与“无偏也不保证一致”的反例共同划清有限样本和渐近性质。plug-in 一致不等于有限样本无偏。Bernoulli 主实验的 plug-in 偏差为 $0.000960$，Delta 方差为 $0.0006144$。异方差下朴素区间覆盖率为 $0.868900$，稳健区间覆盖率为 $0.938900$。相关误差实验使用 AR(1) 相关系数 $0.6$；固定得分下 HAC 与簇稳健标准误分别为 $0.559017$ 与 $0.707107$。遗漏变量账本中，遗漏后的斜率是 $2$，偏差是 $1$。固定排序序列上 Bonferroni、Holm、BH 分别拒绝 $(2,2,4)$ 项。全局零假设下未经校正的 FWER 为 $0.639975$，Bonferroni 后为 $0.047400$。两个独立零效应正态估计中选择较大者，其期望为 $1/\sqrt\pi=0.564190$。
\section*{逐题逐步核对}
```

## tex/upper/solutions/ch11.tex

```tex
\section*{逐题逐步核对}
\begin{enumerate}[leftmargin=*]
  \item 有限维分布只给 $(X_{t_1},\ldots,X_{t_k})$ 的联合律；版本是每个固定 $t$ 几乎处处相等；不可区分则要求除同一个零测集外整条路径相等。连续修改需要额外的矩估计/正则性，单点边缘不能推出跨时依赖或路径粗糙度。
  \item 适应性是 $X_t$ 对 $\mathcal F_t$ 可测，可预测性要求动作在区间开始前由过去决定，停止时刻要求 $\{\tau\le t\}\in\mathcal F_t$。把当日最高价或未来收盘决定当日成交，直接违反信息集条件。
  \item 有限状态链先按沟通类分解；闭类内常返，进入闭类后不能回出的状态暂留。求稳态解 $\pi P=\pi$ 还要检查归一化、不可约和周期；详细平衡是充分条件而非必要条件，周期链可有稳态却不从任意初值收敛。
  \item 赌徒破产的命中概率满足 $h_i=\frac12h_{i-1}+\frac12h_{i+1}$，边界 $h_0=0,h_N=1$，解为 $i/N$；期望时间满足 $m_i=1+\frac12m_{i-1}+\frac12m_{i+1}$，边界为零，解为 $i(N-i)$。先写差分方程再代 $N=4,i=2$，得到 .5 与 4。
  \item 展开 $S_n^2-S_{n-1}^2=2S_{n-1}\Delta S_n+(\Delta S_n)^2$，条件期望给 $\mathbb E[(\Delta S_n)^2\mid\mathcal F_{n-1}]=1$，故 $S_n^2-n$ 是鞅；Poisson 增量条件均值为 $\lambda\Delta t$，所以 $N_t-\lambda t$ 是鞅。加倍下注失败时财富尾部不一致可积，不能直接套可选停止。
  \item 反射原理把首次达到 $a$ 且终点低于 $a$ 的路径映到终点高于 $a$ 的路径，得到 $2[1-\Phi(a/\sqrt t)]$。离散监控只看网格点，漏掉网格间越界，因此命中概率通常向下偏；报告时要分开连续 oracle 和离散估计。
  \item 对链先计算 $\mu_0P^5$，再解 $\pi P=\pi$；交替链的 $P^n$ 在两个状态间来回，说明“有稳态”不等于“收敛”。条件结构反例要逐项写出条件概率，而不是只报一个差值。
  \item 指数等待率 $\lambda$ 的均值/方差为 $1/\lambda,1/\lambda^2$；独立 Poisson 流叠加率相加；按 .3 保留率 5 的流拆为 1.5 与 3.5；积分 $\int_0^3(2+t)dt=10.5$。模拟输出必须同时给固定输入、样本均值、样本方差和标准误。
  \item 随机游走归一化为 $S_{\lfloor nt\rfloor}/\sqrt n$，方差基线为 $t$；Donsker 是函数空间弱收敛，不是每条模拟路径逐点相等。Brownian 二次变差随细化趋 $T$，总变差则增长，需用不同统计量检查两者。
  \item 事件率研究先估计日内 $\lambda(t)$，检查计数均值/方差、等待时间和增量自相关；若季节解释后仍聚集，再比较 Hawkes/Cox/状态模型的滚动样本外似然。每一步都固定时间戳、合并事件规则和数据版本。
  \item 滚动估计转移矩阵并看置信区间和漂移，再把库存/波动/成交量加入扩充状态；若扩充后仍有预测力，原状态不是充分统计量；若矩阵随时间漂移，则需时变参数或再训练协议。全样本最大收益时刻依赖未来，应用有界停止或截断并检查一致可积。
\end{enumerate}
```

## tex/upper/solutions/ch13.tex

```tex
\section*{逐题逐步核对}
\begin{enumerate}[leftmargin=*]
  \item 凸性定义给弦不低于函数图像；若二阶可导，先沿任意方向 $d$ 看 $\frac{d^2}{dt^2}f(x+td)=d^T\nabla^2f(x+td)d\ge0$，再由积分得到一阶下界。强凸时多出 $\frac m2\|y-x\|^2$，在最优点代入便得目标差控制距离；不可微时用次梯度替代梯度。
  \item KKT 四条件来自原可行、对偶可行、驻点和互补松弛。局部最优到 KKT 需要 LICQ/MFCQ 等资格；凸问题中 Slater 给强对偶和乘子存在，KKT 点再由拉格朗日下界推出全局最优。缺资格时“求解器有输出”不证明必要性。
  \item 对偶函数 $q(\lambda,\nu)=\inf_x L(x,\lambda,\nu)$ 对任意原可行点给下界，故弱对偶；Slater 下最优间隙为零，原/对偶可行加 KKT 残差就是证书。乘子解释的是约束右端改变时最优值的一阶导，只有固定单位和活跃集附近有效。
  \item 约束 $x+y\ge1$ 下，若不活跃则无约束最小点 $(0,0)$ 不可行，所以它必须活跃。$L=\frac12(x^2+y^2)+\lambda(1-x-y)$ 的驻点给 $x=y=\lambda$，结合 $x+y=1$ 得 $(.5,.5),\lambda=.5$；代回 $q(\lambda)=\lambda-\lambda^2$，最大值 .25 与原目标一致。
  \item 右端为 $b$ 时同样得 $x=y=b/2$，$p^*(b)=b^2/4$，所以 $dp^*/db=b/2=\lambda^*$。在 $b=1.2$，目标 .36、乘子 .6；中心差分用同一解析函数在 $b\pm10^{-5}$ 计算，独立核对斜率 .6。
  \item 最小方差问题的拉格朗日驻点是 $2\Sigma w-\nu\mathbf1=0$，预算归一化得 $w=\Sigma^{-1}\mathbf1/(\mathbf1^T\Sigma^{-1}\mathbf1)$。若某分量为负，long-only 的乘子和互补松弛必须加入，不能把负数裁成零后仍声称满足原问题 KKT。
  \item 投影梯度的更新为 $x_{k+1}=\Pi_C(x_k-\eta\nabla f(x_k))$；若 $f$ 的梯度 $L$-Lipschitz，取 $0<\eta<2/L$ 才能使用下降不等式。停止同时检查投影梯度映射、约束残差和目标变化；候选 $(.6,.4)$ 需重新计算证书而不是与解析目标逐字相等。
  \item 右端灵敏度实验固定输入和差分步长，比较解析乘子、有限差分和 KKT 残差；改变一个协方差非对角元时同时报告权重、目标、条件数和扰动放大，不把“当前问题最优”误写成“研究结论稳定”。
  \item KKT 小残差只保证当前输入的局部证书。对协方差、预期收益和成本做有符号扰动，若活跃集或权重翻转，应报告区间/稳健解；固定费用、最小手数等离散约束还需要混合整数或明确的最优性缺口。
  \item 约束与乘子题的完整数值答案应把原始/对偶可行性、互补残差、间隙、条件数和输入 hash 放在同一表中；缺任何一项，不能从一个“solver success”字符串推出经济解释。
\end{enumerate}
```

## tex/upper/solutions/ch14.tex

```tex
\section*{逐题逐步核对}
\begin{enumerate}[leftmargin=*]
  \item binary64 把数写成 $(-1)^s(1.f)_2 2^e$，有效尾数 53 位；$\varepsilon_{mach}=2^{-52}$，最近舍入的单位误差为 $2^{-53}$。在指数改变处 ulp 也改变，次正规数/NaN/Inf 不适用普通相对误差模型，答案要把输入类别先分开。
  \item 前向误差是输出差，后向误差是让计算输出成为精确解所需的最小输入扰动，残差是方程不满足程度；条件数属于问题，稳定性属于算法。典型界是前向误差 $\lesssim\kappa(A)\times$ 后向误差，故小残差在病态问题上仍可对应大答案误差。
  \item 共轭化 $\sqrt{x+1}-\sqrt x=1/(\sqrt{x+1}+\sqrt x)$ 避免相消；$x=10^{16}$ 时分子两项均舍入成相同数而得 0，稳定式与高精度值约 $5\times10^{-9}$。这个中间量必须和原式并列报告，不能只打印最终相等。
  \item $X=U\Sigma V^T$ 给 $X^TX=V\Sigma^2V^T$，所以条件数从 $\sigma_{max}/\sigma_{min}$ 变成平方。正规方程残差小不代表系数可靠；QR 保持奇异值尺度，SVD 还能暴露秩亏方向。
  \item 写 $(A+\Delta A)(x+\Delta x)=b+\Delta b$，展开并忽略 $\Delta A\Delta x$ 得 $A\Delta x\approx\Delta b-\Delta Ax$；左乘 $A^{-1}$、取范数并用 $\|A^{-1}\|\|A\|=\kappa(A)$，得到前向误差界。近奇异例中 $\kappa\approx4\times10^8$，残差 $10^{-15}$ 仍可能对应 $10^{-2}$ 的答案误差。
  \item 对十个小量分别用 naive、pairwise、Kahan 和 Decimal；逐步保存每次部分和，才能解释 0、8、10、10 的差异。pairwise 改善误差阶数但不保证该具体数据精确；Decimal 也需说明精度和舍入模式。
  \item 构造真值 $(1,1)$ 后，正规方程/QR/SVD 都要列系数、残差、相对误差和秩。若第二奇异值 $7.32\times10^{-16}$，应判为数值不可识别并报告最小范数/区间；不能用一个任意小 ridge 把方向伪装成精确经济系数。
  \item log-sum-exp 先取 $m=\max_i x_i$，再算 $m+\log\sum_i e^{x_i-m}$；对 $(1000,1000)$ 得 $1000+\log2=1000.693147$，避免指数上溢。比较器先拒绝 NaN/Inf，再用 atol+rtol 规则，列缩放后还要重新报告条件数。
  \item 研究判断先冻结协方差、单位和估计窗，列特征谱、有效秩、条件数、权重和换手；再在相关项/窗口/缩放扰动下重算。集中权重只有在谱稳定、样本外风险和交易约束都稳定时才可解释为经济信号。
  \item 两算法都可有很小残差，因为它们可能精确求解了不同的邻近问题。必须比较后向误差、条件数、最坏扰动方向、系数前向差和留出预测；更多小数位不等于更多信息。
  \item 数值题的独立 oracle 应放在每道题旁边：二进制/ulp 公式、精确求和、高精度矩阵解和稳定 log-sum-exp；答案册不能只在章末给一个总表，让读者猜中间量对应哪一题。
\end{enumerate}
```

## tex/upper/solutions/ch15.tex

```tex
\section*{逐题逐步核对}
\begin{enumerate}[leftmargin=*]
  \item seed 只保证伪随机序列重放；先检查路径分布、支付函数、单位和参数，再把模拟结果与闭式/枚举 oracle 比较。稳定重放的错误实现仍然是错误，答案必须列失败 fixture。
  \item 把总误差拆成模型错设、路径/网格离散偏差、有限样本抽样误差、浮点数值误差和实现错误；增加 $N$ 只直接降低抽样项，不能修复前四项中的其他错误。报告要给每项的独立诊断或界。
  \item Monte Carlo 估计 $\hat\mu=N^{-1}\sum Y_i$，IID 时方差为 $\sigma^2/N$；bootstrap 则条件于观测经验分布重抽样，近似统计量的抽样分布。时间依赖需要 block/stationary bootstrap 或模拟覆盖率，交换样本本身不保证独立。
  \item 方差展开 $N^{-2}\sum_{i,j}\operatorname{Cov}(Y_i,Y_j)$，相关误差的非对角项不能删。对控制变量 $g-\beta h$ 展开二次式，求导得到 $\beta^*=\operatorname{Cov}(g,h)/\operatorname{Var}(h)$，最小方差为 $\operatorname{Var}(g)(1-\rho^2)$；先检查 $\operatorname{Var}(h)>0$。
  \item 三次抽样的 27 个重抽样状态先用整数频数生成，再除以 27；频数和为 27，均值/方差由频数加权。随机 bootstrap 只能近似这个精确账本，差异应落在 $B^{-1/2}$ 误差内。
  \item 对 $U^2$，独立 oracle 为 $E[U^2]=1/3$、$\operatorname{Var}(U^2)=4/45$；$N$ 增至 $4N$ 时标准误约减半。每个 $N$ 保存多次重复 RMSE、标准误和区间，不能拿一条更接近的路径宣称收敛。
  \item 控制变量调整为 $U^2-(U-1/2)$，中心若改成 .6，均值增加 .1 而方差不变；这说明方差减少必须同时检查目标均值。antithetic、分层和重要性抽样也都要给权重/支撑条件。
  \item 逐日 IID bootstrap 会打散自相关和波动聚集；移动块/平稳 bootstrap 要说明块长、边界和有效样本。重要性抽样还要报告权重最大值、有效样本量、二阶矩和支撑覆盖，否则少数路径可能支配结果。
  \item 从 500 候选中先筛选再 bootstrap，筛选步骤本身也是随机量；若每次重抽样不重做筛选，区间漏掉选择不确定性。修复为重做完整流程、嵌套选择或样本分割，并另留时间外验证。
\end{enumerate}
```

## tex/upper/solutions/ch16.tex

```tex
\section*{逐题逐步核对}
\begin{enumerate}[leftmargin=*]
  \item 事件时间 $t_e$、首次可得 $t_a$、决策时间 $t_d$、标签起点 $t_y$ 必须满足 $t_e\le t_a\le t_d<t_y$。逐行解析时区和日历，任何一条违反都拒绝；哈希只能证明字节一致，不能证明时间语义正确。
  \item 测试集只能评估冻结流程；若根据测试分数改特征、阈值、窗口或叙事，测试已进入选择集。保留未见最终区间，或把结果明确降级为开发/探索证据；不能继续称一次性样本外。
  \item 复现、统计有效性和可交易性分别检查字节/命令、误差控制、订单/成本约束。泄漏回测可以完全可复现，显著结果也可能无法成交；答案要把三种结论分栏。
  \item 十次独立零假设查看的 FWER 为 $1-(1-.05)^{10}=.401263$；Bonferroni 用 $\alpha/20=.0025$，所以 $p=.002$ 通过而 .003 不通过。先把检验族、alpha-spending 和停止规则写进协议，再读取数据。
  \item 训练/验证/测试区间用标签半开区间 $[s_i,e_i)$ 表示；验证样本 $j$ 存在时，删除所有与 $[s_j,e_j)$ 相交的训练样本，并在验证末端加 embargo。逐行打印保留索引，才能确认 purge 不是只改了标题。
  \item 三笔交易总毛收益 .04；每笔成本 .002，$C=.006$，净收益 .034。未成交部分不能当作零成本成交，应分 executed-only PnL 和 implementation shortfall，并记录 arrival price、成交量、价差和冲击。
  \item 多重检验审计必须保留完整候选、p 值、选择规则、校正方法和未通过项；A 股还需逐研究期检查 T+1、涨跌停、停牌、复权可得性、融券资格和成交完成率。当前市场规则不能自动外推到历史期。
  \item 容量报告要把资金规模、参与率、波动、执行时长和冲击模型放入函数，画净绩效/容量曲线并报告未成交和压力场景；“资金增十倍仍相同表现”没有订单级证据时应拒绝。
\end{enumerate}
```

## tex/lower/chapters/brainteasers-solutions.tex

```tex
以下答案强调可迁移方法；题面是来源题目的独立改写，不复刻原书参数。

独立计算账本的发布标记为 \texttt{reachable=1} 与 \texttt{missing=8,15}；它们分别核验相邻翻转不变量和和/平方和恢复公式。

\begin{enumerate}[leftmargin=*]
  \item \textbf{GB-2-01。}设 $V_n$ 为还剩 $n$ 人时的唯一向后归纳结果。提案者先计算若自己退出，$V_{n-1}$ 中每人的收益；在需要的票数中购买成本最低者，每票只需比其下一轮收益多一单位，剩余归己。严格过半、至少一半以及“平票是否通过”改变所需票数，因此必须从基例重算。常见错误是直接套某个海盗人数的分配表。追问：若奖金不足以买票，均衡可能不唯一，需增加偏好细则。
  \item \textbf{GB-2-02。}状态为 $(g,i_1,i_2,t)$：守卫、两闯入者的位置与行动方。把被捕或成功闯入标成终止胜负，再反向标记：当前方只要存在一条边进入己方胜态即为胜；所有边均进入对方胜态才为负。有限状态反复迭代至不动点即可。只描述一条追逐路线不能证明对所有对手动作安全。
  \item \textbf{GB-2-03。}记模型、密钥、原始数据为 $M,K,D$。一条最短序列是 $K$ 过、人空返、$M$ 过、$K$ 回、$D$ 过、人空返、$K$ 过，共七次。把状态写成 $(r,m,k,d)$ 的四个左岸指示位，删去两条安全约束不满足的节点，再以“研究员与至多一件物品同时换岸”连边。BFS 首次在第七层到达 $(0,0,0,0)$，因而这条路径不仅可行，而且最短。追问：禁止空船返回时问题无解。
  \item \textbf{GB-2-04。}平年同一日期星期前进一日；若区间跨过二月二十九则前进两日。观测增量 $+1,+1,+2$（模 $7$）说明最后一个区间跨闰日。若不知道生日在二月二十九之前还是之后，只能定位跨越区间，不能唯一确定公历年份。
  \item \textbf{GB-2-05。}若高牌在任何信念下加注的边际收益不低于低牌，则策略满足单交叉，可先搜索“牌面不低于 $k$ 即加注”的阈值族。但没有底池、加注额、弃牌收益、对手先验与行动顺序，就没有数值阈值。常见错误是把合理的策略形状误当成完整均衡。
  \item \textbf{GB-2-06。}时刻零同时点燃甲绳两端、乙绳一端。甲绳在三十分钟后烧尽；此刻点燃乙绳另一端，乙绳剩余部分再烧十五分钟，总计四十五分钟。只需假设每根绳从任一端烧完整根的总时间固定；不假设局部燃速均匀。
  \item \textbf{GB-2-07。}异常物品有 $12\times2=24$ 个“编号--轻重”状态，三次称量提供 $3^3=27$ 个结果串，故少于三次不可能。构造时每次左右盘数量相等，并让每个状态对应唯一结果码；还要保留“未上秤”与轻重翻转的对称。仅给信息量下界而不给可实现编码，不算完整答案。
  \item \textbf{GB-2-08。}$125!$ 中 $5$ 的指数为 $\lfloor125/5\rfloor+\lfloor125/25\rfloor+\lfloor125/125\rfloor=31$；因 $2$ 更多，十进制尾零为 $31$。一般进制 $b=\prod p_i^{e_i}$ 的尾零数为 $\min_i\lfloor v_{p_i}(n!)/e_i\rfloor$。只数 $5$ 仅适用于十进制中 $2$ 不构成瓶颈的情形。
  \item \textbf{GB-2-09。}先赛五组，再赛五个组冠军。设冠军顺序为 $A_1>B_1>C_1>D_1>E_1$。仍可能前三者只来自 $A_1,A_2,A_3,B_1,B_2,C_1$；$A_1$ 已确定第一，再赛其余五个，前两名给全局第二、三，合计七轮。候选表同时构成下界：第六轮后这五个候选之间尚无充分比较。
  \item \textbf{GB-2-10。}特征方程 $r^2-3r+2=0$，故 $a_n=c_1+c_2 2^n$，由 $a_0,a_1$ 解系数。若要求有限极限，必须有 $c_2=0$。数值递推中任何沿根 $2$ 的微小误差都会指数放大，因此即使理论解为常数，浮点初值偏差也可能主导。
  \item \textbf{GB-2-11。}不能。设单位立方体与箱体的 $0.9$ 边方向对应的单位向量为 $u=(u_1,u_2,u_3)$。旋转后立方体在该方向的投影宽度为 $|u_1|+|u_2|+|u_3|$。由 $\|u\|_1\ge\|u\|_2=1$，该宽度不可能小于或等于 $0.9$，所以任何旋转都放不进去。体积和空间对角线只是标量摘要，不控制最窄方向的投影宽度。
  \item \textbf{GB-2-12。}日期 $01$--$09$ 迫使 $0,1,2$ 都能与多个数字配对，因此两枚方块都需含 $0,1,2$。一组可行标记为 $\{0,1,2,3,4,5\}$ 与 $\{0,1,2,6,7,8\}$，把 $6$ 旋转作为 $9$；交换左右即可显示全部日期。若字体不允许 $6/9$ 共用，则十二个面不足，说明排版约定是模型条件。
  \item \textbf{GB-2-13。}一次二元回答最多区分两个叶节点，无法保证识别三个互斥结果；信息下界为 $\lceil\log_2 3\rceil=2$。两问协议：先问“结果是否为录用”；若是则结束，若否再问“结果是否为加面”，否则即为拒绝。该三叶决策树达到下界，故最优。
  \item \textbf{GB-2-14。}若消息或确认可能永久丢失，有限轮协议无法让双方同时获得“对方知道我知道”的共同知识：最后一条确认总可能丢失。工程协议应使用单调序号、幂等处理、确认、超时和有界重试；它能提供去重与最终失败报告，但在不可靠网络上不能同时保证有限终止和必达。常见错误是把重试次数增大当成数学上的必达证明。
  \item \textbf{GB-2-15。}记当前蓝球数和总球数为 $B_t,N_t$。同色中，取两蓝再放一蓝使 $(B_t,N_t)$ 各减一，取两红再放一蓝使二者分别加一、减一；异色操作使二者也各减一。因而 $B_t+N_t\pmod2$ 不变。最后 $N_T=1$，若初始 $B+N$ 为奇数，则 $B_T=0$ 而最后一球为红；若为偶数，则 $B_T=1$ 而最后一球为蓝。
  \item \textbf{GB-2-16。}把状态写成 $x\in\{0,1\}^n$，操作为加向量 $e_i+e_{i+1}$（模 $2$）。每次翻转两个，故总奇偶性不变；事实上这些向量张成所有偶校验向量，所以目标可达当且仅当亮灯数为偶数。从左向右若当前位置与目标不同就翻转它和右邻，最后一位给出一致性检查，时间 $O(n)$。
  \item \textbf{GB-2-17。}期望奖金只给一阶矩。至少还需奖金方差与下行分位数、延期或递延支付、离职 forfeiture、奖金与公司/策略收益的相关性、税务与流动性、固定工资及个人风险厌恶。两个 offer 可有相同期望却有完全不同的确定性等价；若未给效用函数与约束，就只能比较期望现金流，不能宣称哪个更优。
  \item \textbf{GB-2-18。}任取 $k$ 枚为第一堆，其余为第二堆。若第一堆原有 $h$ 枚正面，则第二堆正面数为已知总数 $k-h$；把第一堆全部翻面后，其正面数也变为 $k-h$。证明依赖整堆正面总数 $k$ 已知，不依赖观察单枚。
  \item \textbf{GB-2-19。}从标“混合”的袋中抽一次；因标签全错，该袋必为纯袋，样本直接确定它是哪一种。剩余两个标签再用“都错”约束唯一分配。若只说“至少一个标签错”，一次抽样不再足够，这是常见追问。
  \item \textbf{GB-2-20。}主持人的声明把“全白”世界删掉。第一人说不知道，说明他看见的不是两白；第二人结合该回答仍不知道，进一步删去他能据此确定的世界。第三人对剩余可行世界取交集，若自己的颜色一致即可推断。核心是每个回答成为新的公共知识；忽略这一层会误算。
  \item \textbf{GB-2-21。}四块数字和若相等，每块应为 $78/4=19.5$，但每块包含完整数字，和必为整数，因此不可能。这个整除性下界已经足够，无需继续枚举切线。若题目允许切穿数字或按面积分配，模型改变，整数论证失效。
  \item \textbf{GB-2-22。}设缺失数为 $x,y$。由总和得 $s=x+y$，由平方和得 $q=x^2+y^2$，于是 $xy=(s^2-q)/2$；$x,y$ 是 $t^2-st+xy=0$ 的根。判定需检查判别式为非负完全平方、两根为 $[1,n]$ 内不同整数，且重建统计量一致。
  \item \textbf{GB-2-23。}给第 $i$ 组取 $2^i$ 枚（或使用相应二进制权重），总重量相对全真币基准的缺口除以单枚差额，就是假币组指标的权重和，其二进制展开恢复集合。量程需容纳总取样，测量误差必须小于相邻编码间距的一半；否则数学可辨识不等于实验可辨识。
  \item \textbf{GB-2-24。}三色中取四只必有两只同色，三只各异给出下界。保证两双且颜色不同：最坏可先取得某色大量而另两色各一；若每色数量不设上限，仅靠总抽取数无法保证不同色两双，必须知道每色库存或规定抽样上限。这个追问考察隐含假设。
  \item \textbf{GB-2-25。}把人作为顶点、握手作为边。每条边给两个端点各贡献一度，故 $\sum_v d(v)=2|E|$。模 $2$ 后左侧奇数度顶点的个数必须为偶数。常见错误是只举例而不说明每条边恰计两次。
  \item \textbf{GB-2-26。}固定顶点 $v$，它与其余五人的关系至少有三条同类。若其中三人里有一对认识，则该对与 $v$ 构成三人互识；若没有任何一对认识，则这三人两两不认识。将“认识/不认识”互换，论证完全对称。
  \item \textbf{GB-2-27。}两只同速蚂蚁相撞掉头，与它们保持方向穿过、仅交换身份标签具有相同的无标签位置集合。设第 $i$ 只蚂蚁沿其初始方向到首个顶点出口的距离为 $d_i\in[0,1]$，则全部离开时间为 $\max_i d_i/v$。碰撞只改变“哪个名字沿哪条轨迹”，不改变无标签轨迹何时到达出口。若速度不同、碰撞有停顿或顶点不是出口，等价就不再给出该答案。
  \item \textbf{GB-2-28。}给每枚物品一个非零三进制码 $c_i\in\{-1,0,1\}^3$，三个分量表示三次称量放左、放外、放右。偏重产生码 $c_i$，偏轻产生 $-c_i$；所以不同物品不能用相同或相反码。每次左右盘还需基准平衡。三次最多提供 $(3^3-1)/2=13$ 对非零相反码。
  \item \textbf{GB-2-29。}第一人观察其余人的 $0/1$ 状态，公布它们之和的奇偶位。他自己的状态可能无法恢复；之后每人结合该公开位、自己看到的状态以及前人的公开回答，在模 $2$ 下反解自己的位。协议保存的是一个全局奇偶校验，而不是模 $m$ 的多值符号。若状态有 $m>2$ 种，一个比特通常不足以携带所需校验量。
  \item \textbf{GB-2-30。}若 $N=\sum_k d_k10^k$，则 $10\equiv1\pmod9$ 给出 $N\equiv\sum_kd_k\pmod9$。单个数字从 $d$ 改为 $d'$ 且 $d'-d$ 不是 $9$ 的倍数时可检出；但交换两个数字、多个误差相互抵消或 $0\leftrightarrow9$ 可能保持模 $9$ 校验。它是错误检测，不是纠错码。
  \item \textbf{GB-2-31。}一次不同颜色相遇使两个颜色计数各减 $1$，第三色加 $2$；模 $3$ 看等价于三个计数同时减 $1$，因而任意两两差模 $3$ 不变。初态 $(8,4,3)$ 有 $8-4\equiv1\pmod3$。若最终全部同色，由于总数 $15\equiv0\pmod3$，目标状态的两两差均为 $0\pmod3$，与初态矛盾，故不可能。这一题只要求用不变量证明不可达，不需构造充分性。
  \item \textbf{GB-2-32。}任取 $n$ 枚为第一堆，其余为第二堆。若第一堆原有 $h$ 枚正面，因总正面数为 $n$，第二堆有 $n-h$ 枚正面；把第一堆全部翻面后也恰有 $n-h$ 枚正面。构造只依赖总正面数，不依赖未知排列。
  \item \textbf{GB-2-33。}初始一块，目标 $mn$ 块；每次只把一块变成两块，碎片总数恰增加一，所以至少需要 $mn-1$ 次。沿任意格线不断掰开都可每次增加一块并在 $mn-1$ 次达到目标。若允许一次叠放多块共同切割，操作模型不同。
  \item \textbf{GB-2-34。}同向时相对位移为 $(u-v)t$，第 $k$ 次追及满足 $(u-v)t_k=kL$，故 $t_k=kL/(u-v)$，位置为 $ut_k\pmod L$。反向时相对速度为 $u+v$，第 $k$ 次相遇时刻为 $kL/(u+v)$。需要明确“第零次同点”是否计数。
  \item \textbf{GB-2-35。}令 $x=\sqrt2^{\sqrt2}$。若 $x$ 有理，取 $a=b=\sqrt2$ 即可；若 $x$ 无理，取 $a=x,b=\sqrt2$，则 $a^b=(\sqrt2^{\sqrt2})^{\sqrt2}=2$。分情况证明无需知道 $x$ 究竟属于哪一类，这正是构造性存在证明的关键。
  \item \textbf{GB-2-36。}把 $q$ 种颜色编码为 $\mathbb Z_q$。第一位看见前方颜色和并公开其模 $q$ 值；他可能答错。之后每人用该校验、已听到的正确答案和仍看得见的颜色反解自己的颜色，所以至多第一人错误。协议要求固定顺序、所有人知道编码且能听见此前回答；若回答不能承载 $q$ 个符号或存在噪声，需要额外纠错冗余。
\end{enumerate}

\section*{逐题面试官追问}
```

## tex/lower/chapters/derivatives-numerics-solutions.tex

```tex
\begin{enumerate}
  \item 树主要有时间离散；PDE 有空间、时间和边界截断；Monte Carlo 有采样误差；三者都共享参数和模型错设。
  \item $m$ 个报价各配一个 $\sigma_j$ 时有 $m$ 个自由参数，逐点残差接近零不说明低维曲面解释了报价。参数曲面用少量共享参数同时重定价全部节点。
  \item 对内部节点 $i$，$a_i=(\sigma^2i^2-ri)/2,b_i=-\sigma^2i^2-r,c_i=(\sigma^2i^2+ri)/2$；隐式层使用下、主、上对角 $-\Delta ta_i,1-\Delta tb_i,-\Delta tc_i$。
  \item 离散凸函数的割线斜率非降：$(C_i-C_{i-1})/(K_i-K_{i-1})\le(C_{i+1}-C_i)/(K_{i+1}-K_i)$。只有步长相等时才化为常见二阶差分。
  \item 固定示例闭式为 $8.916037$，树和 PDE 为 $8.908302,8.908382$。实验应分别固定另外两个维度，记录误差是否进入平台区；平台可能来自未加密的维度。
  \item 透明与 sklearn 恢复 $(0.035,0.080,0.006)$，系数差为 0，最大价格误差约 $1.42\times10^{-14}$。真实数据不应期待这种由生成模型保证的零残差。
  \item 分红、负利率或远期变化会破坏未经归一化的固定执行价期限比较。应改用与适用假设一致的远期/贴现表示，或显式跳过该检查。
\end{enumerate}
% MFQ source replacements removed=1 unit=DerivativesNumerics

\input{tex/common/source-mapped-exercises}
\MFQMappedDerivativesNumericsSolutions

\section*{逐题推导与核对}
```

## tex/lower/chapters/derivatives-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item Brownian 增量量级为 $\sqrt{\mathrm dt}$，平方后为 $\mathrm dt$，所以二阶项累积为有限量。
  \item $\mathbb P$ 描述真实发生频率；在满足等价测度条件时，$\mathbb Q$ 使贴现可交易资产成为鞅，用于无套利定价。
  \item $\mathrm d\log S_t=(\mu-\sigma^2/2)\mathrm dt+\sigma\mathrm dW_t$，积分并指数化即得解析解。
  \item Itô 展开 $V$，持有 $V_S$ 股标的消去 $\mathrm dW$ 项，再由无套利令剩余组合按 $r$ 增长，得到 PDE。
  \item 记录步数、价格和相对闭式误差；不能只报告“更接近”。
  \item 路径扩大四倍时，若方差稳定，置信区间半宽约减半；应以实际样本标准差核对。
  \item 越界报价没有无套利隐波；Vega 接近零时，小报价误差会变成大波动率误差。
  \item 建立无成本基线，再加入相同路径上的成本；另用错误波动率重复，对三类误差分别归因。
\end{enumerate}

\section*{独立 oracle 核对表}
离散二次变差为 $0.86$，Itô 离散恒等式两侧为 $0.02$。闭式、二叉树与 Monte Carlo 价格分别为 $8.916037$、$8.911086$ 与 $8.909574$，Monte Carlo 95\% 半宽为 $0.085601$。解析 Delta 与差分 Delta 分别为 $0.579260$ 与 $0.579260$。无成本复制误差为 $1.476783$，成本后误差为 $1.391342$。

\section*{逐题推导与核对}
```

## tex/lower/chapters/derivatives-stochastic-solutions.tex

```tex
\section*{逐题推导与核对}
```

## tex/lower/chapters/microstructure-events-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 条件强度随当前历史变化；无条件率是对历史分布积分后的平均。两者只有在齐次 Poisson 等特殊模型中相同。
  \item $\ell=n\log\lambda-\lambda T$，故 $\ell'=n/\lambda-T$；令其为零得 $\widehat\lambda=n/T$，且 $\ell''=-n/\lambda^2<0$。
  \item 点过程似然是“事件点处强度乘积”乘“整个区间无额外事件的生存概率”。补偿子正是后者的负对数；删掉后不存在有限最优强度。
  \item 响应为 $\alpha e^{-\beta u}$。分枝比越接近 1，簇集持续越久且方差越大；达到或超过 1 时平稳分枝解释失效。
  \item 对每个市价单按最佳价格再按序号消费数量；若允许部分成交则报告剩余，否则在改变状态前拒绝。撤单不得超过活动数量。
  \item maker sell 表示主动买。误当主动卖会令所有方向反号，联合回归斜率也反号，经济解释随之颠倒。
  \item 应补盘中季节基线、时间单位与窗口、$\alpha/\beta<1$、初始历史处理、完整补偿子、时间变换残差和样本外参数稳定性。
  \item 合格协议包含逐档增删改、事件序号、交易所时间与接收时间、订单编号、隐藏量边界、重放校验和缺包/乱序/重复事件处理。
\end{enumerate}

事先确定 oracle：Poisson MLE 为 1、队列完全成交概率为 $0.576810$、联合回归斜率为 $0.15$；FIFO 账本得到 $3+1$ 手成交。

\section*{逐题推导与核对}
```

## tex/lower/chapters/ml-alpha-model-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 树无需标准化、能表示阈值和交互、原生支持多种缺失策略，适合中等样本表格；相对线性模型失去简单全局系数，相对神经网络不擅长端到端学习序列、文本和图像表示。
  \item CART 逐节点最小化不纯度；随机森林平均 bootstrap 树并随机抽特征以降相关；gradient boosting 在函数空间拟合负梯度；XGBoost 再加入二阶曲率、叶权重和结构惩罚。
  \item $s=-1.5,0,2$ 的 SSE 分别为 $17.33,10,10.67$（按各侧均值计算），所以 $s=0$ 最优；不分裂 SSE 为 46。
  \item 叶目标为 $G_jw+\frac12(H_j+\lambda)w^2+\gamma$。一阶条件 $G_j+(H_j+\lambda)w=0$ 给出 $w_j^*=-G_j/(H_j+\lambda)$。
  \item 使用相同外层 walk-forward，内层选择复杂度；报告模型、时间、验证损失、事先确定仓位映射后的净收益与换手。若 MLP 不能稳定胜过提升树，应保留提升树。
  \item 随机拆行会让同一资产相邻日期、同一市场冲击和重叠标签同时进入训练测试；AUC 衡量的是污染后的区分能力。应按决策时间切分并 purge 标签区间。
\end{enumerate}

\section*{逐题推导与核对}
```

## tex/lower/chapters/ml-alpha-modern-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item shape 绑定 batch/time/feature 语义；dtype 与 device 限定数值和算子；\texttt{requires\_grad} 决定 autograd 是否追踪。
  \item CNN 共享局部核，RNN 用递归状态压缩前缀，attention 对可见 token 内容寻址，Transformer 叠加多头注意力、前馈、残差和归一化；四者都必须显式限制未来。
  \item GNN 增加随时间变化的节点、边和消息传递；DRL 增加动作影响的状态转移、奖励和长期价值，不能再只评价静态预测误差。
  \item $h=\phi(W_1x+b_1)$、$\hat y=W_2h+b_2$；梯度从损失经 $W_2$ 与 $\phi'$ 回到 $W_1$。残差 $h+F(h)$ 使梯度含恒等项 1，降低深层连乘完全消失的风险。
  \item 按当前减前一期得到 $(2,-1,3)$。若对称 padding 的卷积在 $t$ 使用 $x_{t+1}$，历史预测已读取未来。
  \item 同步置换 token 会同步置换 $Q,K,V$，注意力输出只重排；$\operatorname{softmax}(0,1)=(1/(1+e),e/(1+e))$。位置编码或因果结构才赋予先后。
  \item GCN 用归一化邻接聚合 $H'=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW)$；全图含未来边即泄漏。Bellman 方程连接当前奖励和下一状态价值；若模拟器不让动作改变成交或价格，环境反事实失真。
  \item 均值池化对反转为零差；因果 CNN、RNN、带位置和 mask 的 attention/Transformer 应有正差。Padding mask 屏蔽不存在 token，causal mask 屏蔽未来 token。
  \item 保存模型/优化器状态、随机种子、特征 schema、预处理和版本；恢复后逐项预测一致。交换特征顺序应由 schema 哈希或显式列名检查拒绝。
  \item 每个决策日只使用此前已生效的边构图。静态全图若包含后续供应链公告，预测差异本身就是泄漏证据，不是性能提升。
  \item 时间均值先删除顺序；没有位置、因果 mask、可训练序列层和置换负例时，只能称随机表示基线。
  \item 全量微调表达力和成本最高，任务头最稳但表示固定，LoRA 只学习低秩更新。三者都要绑定基础权重、tokenizer、语料截止和 prompt；权重可能记住历史事件后续信息。
  \item 建立两期、有限库存和已知冲击成本的小 MDP，用穷举或动态规划得最优动作，再检查 RL。固定价格回放没有动作反事实，无法评价另一订单对成交和价格的影响。
  \item 合格项目应包含合法数据截止、简单基线、外层样本外、独立局部 oracle、预测到仓位、成本与容量、失败注入和限制；模型名称不是验收项。
\end{enumerate}

\section*{逐题推导与核对}
```

## tex/lower/chapters/ml-alpha-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 经验风险衡量训练拟合，泛化风险面向未来分布，正则化目标在训练风险上增加复杂度代价。
  \item 低损失不保证概率刻度正确；正确校准不保证交易盈利；稳定重要性也不构成因果识别。因果主张由单独的研究设计检查拒绝，不能由稳定性分数代替。
  \item 训练风险、样本外风险与正则化目标依次为 $0.4256$、$1.5154$、$0.4356$。固定推理段中基线、树和 boosting 的 MSE 依次为 $1.5154$、$0.2050$、$0.1625$。
  \item 校准前后 Brier 分数为 $0.16$ 与 $0.025$；它只核对概率误差。
  \item 新 tensor 为 $(B,4,F)$，mask 为 $(B,4)$，标签时间必须严格晚于第四步。
  \item 先写独立 ledger，再更新 fixture、哈希和 oracle；不能从实现输出反抄期望。
  \item 随机切分混合未来状态，全样本 scaler 使用未来统计量，推理段 early stopping 又把最终窗口变成选择数据。
  \item 先确定阈值与响应等级；轻微校准漂移可重校准，结构漂移应停用并重新训练，固定推理段仍不可回写。
\end{enumerate}

\section*{独立 oracle 核对表}
训练风险、样本外风险与正则化目标依次为 $0.4256$、$1.5154$、$0.4356$。序列任务中最后一步线性基线与随机特征表示基线 MSE 为 $6.398979$ 与 $1.051576$；后者仍不学习时间顺序。毛收益为 $0.030$，成本后收益为 $0.024$。其中，校准后两笔分数映射为多、空各一个单位仓位，总换手为 $2$，单位换手成本为 $0.003$。校准前后 Brier 分数为 $0.16$ 与 $0.025$，Top-2 解释集合 Jaccard 为 $1/3$。

\section*{逐题推导与核对}
```

## tex/lower/chapters/ml-alpha-validation-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item purge 删除标签结果跨入验证区的训练样本；embargo 在选择区后留空，阻断持仓或相邻依赖跨入测试。
  \item 内层用于模型/超参数选择，外层评价包含选择在内的完整程序；反复查看外层会消耗其独立性。
  \item $\max I_{\rm train}+h=7+2=9<10$；$\max I_{\rm val}+e=11+2=13<14$。
  \item $\mathrm{BS}=n^{-1}\sum(p_i-y_i)^2$；它衡量概率刻度，不含仓位、收益、换手或成本。
  \item 第一折用 0--3 预测 4--5，第二折用 0--5 预测 6--7；任何验证行不得参与本折拟合。
  \item 计算每个 $k$ 的 $|A_k\cap B_k|/|A_k\cup B_k|$；结论若随 $k$ 剧烈改变，应报告敏感性。
  \item 测试集参与 early stopping 后就是验证集；必须另留未见数据或明确降级证据强度。
  \item 输入漂移触发数据审查，标签漂移触发先验/校准更新，概念漂移触发重训审批，执行漂移触发降仓或停机；阈值事先确定。
\end{enumerate}

\section*{逐题推导与核对}
```

## tex/lower/chapters/multifactor-model-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 预测系数描述 $x_t$ 与 $r_{t+1}$ 的样本关系；风险价格属于 beta 定价模型；因果效应还需处理与识别假设。三者不能靠显著性互换。
  \item Pearson IC 保留距离并受极端值影响；Rank IC 保留次序。单调非线性、异常值或并列秩会令二者分离。
  \item 均值为 $0.025$。两点离均差为 $\pm0.005$，样本标准差为 $0.007071$，均值标准误为 $0.005$。
  \item 第一遍对每个资产沿时间回归因子收益，第二遍对平均收益沿资产回归估计 beta。第二遍解释变量有估计误差；修正依赖具体定价模型与抽样条件。
  \item 共同 alpha 进入第二遍截距，不改变精确风险价格；资产特异 alpha 会污染 beta 与平均收益的横截面关系。
  \item 相关候选会改变 FDR 保证所需条件。应保存相关结构、完整尝试族，并用适用的校正或重采样协议。
  \item 缺失项包括因子可交易性、beta 的时间稳定性、定价误差结构、生成回归量误差与样本外检验。
  \item 最终期一旦参与尺度、窗口或检验族选择，就成为选择集的一部分；必须另留从未查看的数据。
\end{enumerate}

\section*{独立 oracle 核对}
预测型 Fama--MacBeth 的两期斜率为 $0.020$ 与 $0.030$，均值为 $0.025$。
经典两遍法的因子风险价格为因子收益均值 $0.006$。

\section*{逐题推导与核对}
```

## tex/lower/chapters/multifactor-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 横截面系数描述指定样本中的线性预测关系；风险价格还需资产定价模型与暴露解释；因果效应还需处理定义和识别假设。三者不能由同一个显著性替代。
  \item IC 保留数值距离并受极端值影响；Rank IC 只保留次序。非线性单调关系、异常值或大量并列值会使二者明显不同。
  \item 均值为 $(0.018+0.022)/2=0.020$。两点离均差为 $\pm0.002$，样本方差为 $8\times10^{-6}$，均值方差为 $4\times10^{-6}$，标准误为 $0.002$。
  \item $P_Z=Z(Z^TZ)^{-1}Z^T$，故 $Z^T(I-P_Z)=Z^T-Z^TZ(Z^TZ)^{-1}Z^T=0$。秩亏时逆不存在；数值实现应通过分解和条件数检查拒绝，而非悄悄给出任意解。
  \item 第三期的斜率、IC、价差收益与新均值应来自 fixture 外的解析/枚举参考账本，再更新哈希和 oracle；若先运行实现再抄输出，就失去独立性。
  \item 日期相等触发 time-alignment failure；共线暴露触发 ill-conditioned neutralization；篡改期望值触发 ledger failure。三者分别保护研究时间、数值识别和计算正确性。
  \item 事先确定 500 项检验族和 BH 协议，在完全未参与选择的后移样本复核；再报告 IC 衰减、换手、佣金、价差、冲击、借券与容量。入选不等于可交易。
  \item 先检查公式和估计目标是否变化；再检查发布日期、复权、成分和缺失；最后检查涨跌停、停牌、T+1、融券与成交。只有第三类属于制度边界。
\end{enumerate}

\section*{独立 oracle 核对表}
Fama--MacBeth 横截面预测斜率均值为 $0.020$，时间序列标准误为 $0.002$。毛收益 $0.071$、基础成本后净收益 $0.067$、大容量冲击后 $0.061$。20 次零信号搜索产生 2 个朴素显著项，而 BH 拒绝数为 0。

事先确定协议的结果为
\[
\begin{aligned}
(\mathrm{IC}_{\mathrm{train}},\mathrm{IC}_{\mathrm{select}},\mathrm{IC}_{\mathrm{test}})
  &=(0.583807,0.711369,0.726531),\\
(\mathrm{IC}_{1},\mathrm{IC}_{3},\mathrm{IC}_{6})
  &=(0.646657,0.445136,0.186461).
\end{aligned}
\]
这些是合成协议的可复核结果，不应外推成市场规律。

\section*{逐题推导与核对}
```

## tex/lower/chapters/portfolio-risk-estimation-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 半正定只保证 $w^\top\Sigma w\ge0$；很小的正特征值仍会造成巨大条件数，使逆矩阵和权重对采样误差极敏感。
  \item $\sum_i\mathrm{RC}_i=\sum_iw_i(\Sigma w)_i=w^\top\Sigma w$。若使用波动率贡献，再整体除以组合波动率。
  \item 对角矩阵下 $\mathrm{RC}_i=w_i^2\sigma_i^2$。令贡献相等得 $w_i\sigma_i=k$，归一化后 $w_i=\sigma_i^{-1}/\sum_j\sigma_j^{-1}$。
  \item $\lambda$ 从 0 到 1 时，矩阵向目标移动。图中应同时报告最小特征值、条件数和最终风险贡献；不能只挑权重最平滑的强度。
  \item 对起点 $b$ 抽取 $(r_b,r_{b+1},r_{b+2},r_{b+3})$，循环或只取合法区块，再拼到原样本长度。区块长度也要做敏感性分析。
  \item 可构造三资产、三日期且每一对资产只共享不同两日的数据。逐对相关都可能看似合理，但拼接矩阵的某个特征值为负。
  \item 因子模型在 $n/T$ 大时用结构降低方差；若遗漏流动性、行业或非线性因子，低维稳定性会以系统性偏差为代价。
  \item 合格协议列出资产进入/退出规则、交易日交集、时区、公司行动、收益公式、缺失值、异常值、估计窗口与信息可用时点。
\end{enumerate}

事先确定 oracle：风险平价权重为 $(6/13,4/13,3/13)$，风险预算差为 0；透明收缩与 sklearn 差为 0；组合波动率及 90\% bootstrap 区间为 $0.007950$ 和 $[0.005984,0.009304]$。

\section*{逐题推导与核对}
```

## tex/lower/chapters/stat-arb-model-solutions.tex

```tex
\section*{逐题推导与核对}
```

## tex/lower/chapters/stat-arb-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 弱白噪声只要求零均值、常方差和跨期不相关；创新是相对当前信息集不可预测的误差；错设残差可能保留显著自相关，因此不能把三者混用。
  \item 高相关只描述样本共同变化；协整要求非平稳序列存在平稳线性组合；可交易还需时点正确、关系稳定、成本后为正且有可执行容量。
  \item 因 $X_t=0.6X_{t-1}$，分子等于 $0.6\sum X_{t-1}^2$，故斜率为 $0.6$。均值模型残差仍遵循同一动态，残差一阶系数超过阈值即拒绝。
  \item $K=1/(1+0.25)=0.8$，$m^+=0.8\times1.2=0.96$，$P^+=(1-0.8)\times1=0.2$。
  \item 应对每个阈值保存检测索引、真实变点、延迟、提前误报和是否漏检；只报最佳阈值属于选择后汇报。
  \item 新交易必须先写独立解析/枚举/第二实现参考账本，再修改 fixture、哈希和 oracle；毛收益与每类成本分别列示。
  \item 随机切分破坏因果时间方向；全样本标准化把未来均值与方差送入过去。二者都使验证集不再模拟当时可获得的信息。
  \item 例如连续三期残差越界或在线状态概率低于阈值即停用，阈值与恢复条件在交易前事先确定；平滑状态只能用于事后诊断。
\end{enumerate}

\section*{独立 oracle 核对表}
AR(1) 系数为 $0.6$，错设残差的一阶相关为 $1$。

MA 预测为 $0.4$；VAR 两维预测为 $(0.7,0.8)$；GARCH 方差为 $0.5$。Kalman 增益、后验均值和后验方差为 $0.8,0.96,0.2$。

已知变点索引为 4，检测延迟和误报均为 0；低阈值有 2 个提前误报，高阈值漏检。协整价差一阶相关约为 $-0.597614$，漂移伪价差被拒绝。毛收益为 $0.05$，成本后净收益为 $0.04$。

\section*{逐题推导与核对}
```

## tex/upper/solutions/ch10.tex

```tex
\item iid 均值方差为 $1/100=.01$；相关样本方差是 $n^{-2}[n\gamma(0)+2\sum_{h=1}^{n-1}(n-h)\gamma(h)]$，代入 AR(1) $\rho=.6$ 得 $0.03925$。五个一簇、簇内相关 .4 的设计效应 $1+4(.4)=2.6$，方差为 .026；逐点 bootstrap 打散相关结构，不能核对后两者。
```

## tex/upper/solutions/ch10.tex

```tex
\item 先排序 p 值，再计算三种阈值：
  \[
    \tau_{\mathrm B}=\alpha/m,\qquad
    \tau_{\mathrm H,j}=\alpha/(m-j+1),\qquad
    \tau_{\mathrm{BH},j}=j\alpha/m.
  \]
  对 $(.001,.009,.021,.040,.200)$，三种方法的拒绝数为 $(2,2,4)$。BH 必须取满足条件的最大 $j$，不能取第一个通过项；未排序、门槛错位和只取首项都应作为失败 fixture。
```

## tex/upper/solutions/ch10.tex

```tex
\item 对 Bernoulli 样本 $\hat p=n^{-1}\sum X_i$，$\mathbb E\hat p^2=p^2+\operatorname{Var}(\hat p)=p^2+p(1-p)/n$。$g(p)=p^2$ 的 Delta 方差为 $(2p)^2p(1-p)/n$；代入 $p=.4,n=250$ 得偏差 $0.000960$、方差 $0.0006144$。expected 必须由公式独立写入，再运行程序。
```

## tex/lower/chapters/ml-alpha-modern-questions.tex

```tex
\item \textbf{数值编程：}保存并恢复 checkpoint，验证预测与哈希不漂移；随后故意更换特征顺序并证明契约会拒绝。
```

## tex/lower/chapters/ml-alpha-modern-solutions.tex

```tex
\item checkpoint 必须包含模型、优化器、随机种子、特征 schema、预处理参数、基础权重和版本；恢复后用同一输入逐元素比较预测和哈希。交换特征顺序应由 schema 哈希或列名断言拒绝，不能只比较最终收益。
```

## tex/lower/chapters/derivatives-hedging-questions.tex

```tex
\item \textbf{开放 Capstone：}交付从报价协议到多路径对冲限制报告的一键研究材料。
```

## tex/lower/chapters/derivatives-hedging-solutions.tex

```tex
\item 合格 Capstone 包含合约与报价时间、曲线/股息、模型假设、四种定价误差、曲面约束、Greek 验证、多路径对冲、摩擦压力、故障注入、限制与一键命令。
```

## tex/lower/chapters/derivatives-hedging-solutions.tex

```tex
\item Capstone 应按顺序提交：报价快照与单位；模型和参数假设；解析/树/PDE/Monte Carlo 定价互核；Delta/Gamma/Vega 检查；多路径对冲分布；成本、跳跃和波动率错设压力；失败 fixture；最后给限制和一键执行入口。每一步都保存输入哈希和独立 oracle，才能从报价追到最终意见。
```

## tex/lower/chapters/derivatives-hedging.tex

```tex
透明实现逐路径执行该账本，第二个独立向量实现以 SciPy \texttt{ndtr} 计算批量 Delta，并用 NumPy 同时更新 512 条路径。它不是外部对冲引擎，课程 manifest 因而把它明确标成 \texttt{independent-vectorized}，不冒充成熟库实现。两者最大逐路径误差差约为 $5.7\times10^{-14}$；它们共享模型假设，不能互相证明 GBM 合理。
```

## tex/lower/chapters/derivatives-numerics-solutions.tex

```tex
\item 做误差表时一次只加密一个维度：固定 $(M,S_{\max})$ 改 $N_t$，固定 $(N_t,S_{\max})$ 改 $M$，固定 $(M,N_t)$ 改边界。每一行同时记录价格、绝对误差、相邻细化差和运行时间；若相邻差不再下降，应检查未加密维度、边界污染或线性求解容差。闭式值 $8.916037$ 是独立参照，不能从 PDE 输出反推。
```

## tex/lower/chapters/derivatives-numerics.tex

```tex
它们与透明实现共享合约和输入，却采用不同数值原语。闭式、树与 PDE 的事先确定差分别为 $0$、$6.8\times10^{-14}$ 与 $0$；抽样 Monte Carlo 与独立积分参照相差 $0.136405$，约为 $1.41$ 个 Monte Carlo 标准误，因而没有被伪装成机器精度一致。成熟库对照能发现实现差异，仍不能证明模型适合市场。
```

## tex/lower/chapters/derivatives-numerics.tex

```tex
合成节点负责正确性 oracle。真实轨只事先确定美国财政部 2024-12-02 的 3 个月、6 个月和 1 年期国债收益率，并保存官方 URL、观察日、访问日、许可说明和 SHA-256。它只演示如何把决策日前的公开利率转换为贴现因子；国债 par yield 也不等于期权定价所需的完整零息曲线。该快照不能验证隐波、校准或交易收益。
```

## tex/lower/chapters/derivatives-stochastic-solutions.tex

```tex
\item 从 4096 个独立 $N(0,T/4096)$ 增量出发，分别将每 $64,16,4,1$ 个增量相加，再计算平方和；步数为 $64,256,1024,4096$。先聚合再平方与先平方再聚合不同，后者不会产生粗网格的二次变差。每条路径数值不同，不要求固定输出；用多路径均值接近 $T$、方差接近 $2T^2/m$ 来理解定理。由正态四阶矩，$\operatorname{Var}((\Delta W)^2)=2(T/m)^2$，相加即得方差公式。
```

## tex/lower/chapters/microstructure-control.tex

```tex
完成率不是预测误差的装饰项：它决定剩余库存、后续冲击和最终违约状态。停止规则可以是固定期限、累计损失阈值、市场状态或风险限制；一旦触发，应输出 stopped、剩余数量和采用的处置规则。
```

## tex/lower/chapters/microstructure-simulation-solutions.tex

```tex
\item 合格 Capstone 应附确定性 oracle、真实/合成双轨、独立实现比较、事先确定配置、故障注入、逐字报告、一键命令和明确限制；收益领先不是验收前提。
```

## tex/lower/chapters/microstructure-simulation-solutions.tex

```tex
\item Capstone 从冻结事件输入开始，依次重放队列、生成成交、更新库存、生成报价，再比较控制器。提交双轨数据、独立账本、故障 fixture、共同随机数、逐字输出和限制报告；若静态策略收益更高，也必须同时报告风险和统计不确定性。
```

## tex/lower/chapters/ml-alpha-research-questions.tex

```tex
\item \textbf{开放 Capstone：}交付模型、验证、文本协议、组合、摩擦、漂移与限制的一键研究材料。
```

## tex/lower/chapters/ml-alpha-research-solutions.tex

```tex
\item 首次发布时间晚于决策、抓取时使用未来版本、修订日晚于决策都应分别拒绝并给出稳定诊断。
```

## tex/lower/chapters/ml-alpha-research-solutions.tex

```tex
\item 合格 Capstone 包含许可与版本、基线、checkpoint、嵌套时序、校准/漂移、文本审计、仓位成交、成本容量、故障注入、限制和一键命令。
```

## tex/lower/chapters/ml-alpha-research-solutions.tex

```tex
\item Capstone 依次固定数据许可与截止、基线和 checkpoint、嵌套时间协议、文本/模型版本、预测到订单与成交、成本容量、漂移、失败注入和限制。最终包应能从原始输入重建每条 \texttt{target}、\texttt{order}、\texttt{fill}、\texttt{position} 与 \texttt{net\_return} 记录，并保存独立 oracle 和命令输出。
```

## tex/lower/chapters/ml-alpha-research.tex

```tex
账本必须保存分数、rank、目标、旧仓位、订单、成交比例、实际仓位、收益日期和成本原因码。透明实现逐期递推“旧仓位 + 成交增量”；成熟库对照把每个资产的递推写成下三角线性系统并用 SciPy 求解，两者在事先确定账本上逐项一致。透明路径适合解释状态变化；稠密三角求解只适合小型批量核验，大规模稀疏系统应利用结构。两条路径都没有模拟队列、市场冲击、借券、涨跌停或停牌，不能把数值一致包装成成交模型充分。用目标仓位计算收益、删除未成交样本或把成交失败当成零仓都会产生虚假回测。
```

## tex/lower/chapters/multifactor-estimation-solutions.tex

```tex
\item 惩罚路径先冻结标准化、截距和目标，再扫描一组 $\lambda$。表中同时列每个 $\lambda$ 的 ridge/lasso 系数、非零数、训练/验证损失和条件数；透明实现与库实现只在同一目标、同一缩放下比较，否则差异可能来自契约而非算法。
```

## tex/lower/chapters/multifactor-estimation-solutions.tex

```tex
\item 全零列的 $x_j^Tx_j=0$，坐标更新没有分母，应在迭代前以 schema/方差门禁拒绝或明确删除；近共线列则报告最小奇异值和条件数，并比较解对微小扰动的变化。不能让求解器静默返回一个看似有限的系数。
```

## tex/lower/chapters/multifactor-estimation.tex

```tex
透明实现负责：展示目标函数、单位、截距规则、更新公式和失败条件。成熟库负责：提供独立实现、更多诊断和工业级数值路径。交叉检查覆盖横截面与面板回归、中性化残差、IC、Rank IC、衰减、BH、ridge 与 lasso；至少比较系数、预测、残差、目标函数与退出状态。只比较“程序跑完”没有证据强度。
```

## tex/lower/chapters/multifactor-estimation.tex

```tex
外部快照使用 2013 年资本形成增速作信号、2014 年人均 GDP 增速作结果，覆盖 7 个国家，严格满足信号年早于结果年。文件绑定 SHA-256，并记录 World Bank WDI 来源与 CC BY 4.0 许可。事先确定相关系数约为 $-0.336654$。
```

## tex/lower/chapters/multifactor-estimation.tex

```tex
这个数不能被称为股票因子收益：样本只有 7 个国家，变量是宏观增速，没有可交易资产、发布日期滞后、交易成本或因果识别。它的唯一用途是让读者练习“外部数据必须同时带来源、许可、哈希、字段解释、时间协议和限制声明”。合成面板仍是唯一正确性 oracle。
```

## tex/lower/chapters/multifactor-estimation.tex

```tex
Capstone 必须保存训练期变换参数、惩罚路径、选择规则、库版本、条件数和实现差异。若透明实现与库实现不一致，先确定输入和目标函数，不允许挑选看起来更好的一份输出。
```

## tex/lower/chapters/multifactor-research-solutions.tex

```tex
\item 每期断言 gross 等于仓位点乘下一期收益；外生收益字段应删除。错位必须非零退出。
```

## tex/lower/chapters/multifactor-research-solutions.tex

```tex
\item 开放 Capstone 至少包含双轨数据、双实现、滚动样本外、组合映射、成本容量、故障注入、限制报告与一键重建。
```

## tex/lower/chapters/multifactor-research-solutions.tex

```tex
\item 删除外生 \texttt{trade\_return} 后，逐期断言 $\mathrm{gross}_t=\sum_i\mathrm{position}_{i,t}\,r_{i,t+1}$，再由实际成交增量算成本。把收益字段换成错位或伪造值应非零退出，并指出是 position/return alignment 还是 ledger failure。
```

## tex/lower/chapters/multifactor-research-solutions.tex

```tex
\item Capstone 需从双轨数据和独立实现开始，经过滚动样本外、信号到组合映射、成本容量、失败注入和限制报告，最终生成可重建的 target/order/fill/position/net-return 台账。每个结果旁边给输入哈希、版本和 oracle。
```

## tex/lower/chapters/portfolio-risk-estimation-solutions.tex

```tex
\item 收缩矩阵写成 $\Sigma_\lambda=(1-\lambda)\widehat\Sigma+\lambda T$。对每个 $\lambda$ 计算最小特征值、条件数、解权重、组合波动和预算差；若权重更平滑但条件数仍高，不能只凭图选择强度。oracle 的透明实现与库实现应在同一矩阵上逐元素比较。
```

## tex/lower/chapters/portfolio-risk-estimation.tex

```tex
该公式看似简单，却依赖收益口径、时区、公司行动、缺失值和资产池协议。成对删除会让不同矩阵元素来自不同日期集合，拼出的矩阵甚至可能不再半正定。课程实现先检查矩阵有限、对称且半正定，再允许它进入优化器。
```

## tex/lower/chapters/portfolio-risk-estimation.tex

```tex
真实轨复用美国宏观公共快照，检查哈希、季度顺序和截至日期，把 40 个 GDP/消费水平转换为 39 个相邻对数增长率，再实际计算协方差与两列等风险权重。协方差迹为 $7.415213\times10^{-5}$，权重为 $(0.434184,0.565816)$。这演示 provenance、时点协议和真实输入适配；正文明确禁止把两列宏观增长率冒充可交易资产收益或风险模型有效性证据。
```

## tex/lower/chapters/stat-arb-estimation-solutions.tex

```tex
\item 负例把估计截止设为决策日之后，调用时间验证器必须非零退出；修复是只保存当时滤波状态并从下一滚动窗重新估计。
```

## tex/lower/chapters/stat-arb-estimation-solutions.tex

```tex
\item 负例把估计截止设在决策日之后，时间验证器应检查 \texttt{fit\_end <= decision\_time} 并非零退出。修复是只保存当时滤波状态，下一滚动窗重新估计；平滑结果可用于事后诊断但不能进入历史交易。
```

## tex/lower/chapters/stat-arb-research-solutions.tex

```tex
\item 拟合截止晚于训练结束必须失败；成交比例为 0 时，仓位必须等于旧仓位。再逐字重建毛收益、换手、成本与净收益。
```

## tex/lower/chapters/stat-arb-research-solutions.tex

```tex
\item 合格项目至少含数据许可和发布时间、独立解析/枚举/第二实现参考账本、透明/库对照、滚动窗口、模型预测、仓位/成交/成本账本、零仓基线、故障注入、失效门槛、一键命令和不夸大的限制说明。
```

## tex/lower/chapters/stat-arb-research-solutions.tex

```tex
\item 机器拒绝条件包括 \texttt{fit\_end > decision\_time}、\texttt{fill\_ratio} 超出 $[0,1]$、成交量超过订单、停牌日出现成交、仓位与账本不一致。通过后再逐字重建毛收益、换手、成本和净收益。
```

## tex/lower/chapters/stat-arb-research.tex

```tex
透明实现逐期暴露订单、成交与旧仓位的语义；成熟库对照把同一递推写成有限维下三角线性系统并交给 SciPy 求解。后者适合独立核对小型事先确定账本，但稠密矩阵会随期数平方增长，也没有队列位置、撮合优先级、冲击或停牌状态，因此不能冒充真实执行引擎。长时间轴应利用双对角稀疏结构或直接递推，市场微观结构则需要另建状态模型。
```

## tex/lower/chapters/stat-arb-research.tex

```tex
开放 Capstone 选择一组有经济联系的价格或利差，先声明数据许可与发布时间，再交付：研究问题、单位根/协整规格、透明与成熟库对照、状态估计、walk-forward 切分、独立零仓基线、成交与成本、故障注入、失效规则、限制报告和一键复现命令。若数据不支持交易时点，项目应停在方法演示，不得包装成回测。
```

## tex/upper/chapters/ch03.tex

```tex
固定网格还可能完全漏掉窄区间 $(0,1/n]$，把面积误报为零。增加若干网格点不是 DCT 证明；本例的独立基准是“高度乘宽度”，不是被测网格的输出。
```

## tex/upper/chapters/ch07.tex

```tex
固定离散例取 $Y=X^2$。三个原值推到 $0,1,4$，质量分别为 $1/2,1/4,1/4$；$-1$ 与 $1$ 若同时出现，质量必须合并，不能把输出当成逐行不重复表。连续例若 $Z\sim\mathcal N(0,\sigma^2)$、$Y=e^Z$，则 $Y$ 为对数正态，正值域和右偏尾部都来自变换，不应再用对称正态误差解释。
```

## tex/upper/chapters/ch10.tex

```tex
$0.16+(-0.2)^2=0.20$，小于无偏估计的 $0.25$。本章基准 固定输出
```

## tex/upper/chapters/ch12-questions.tex

```tex
\item 独立复现 AR(1)/ARMA 解析矩，并用固定种子模拟只做容差交叉验证。
```

## tex/upper/chapters/ch12.tex

```tex
当 $\phi=0.8,\sigma^2=1$ 时，理论方差为 $2.777778$；从 $X_t=1$ 出发的三步预测为 $0.512000$，预测误差方差为 $2.049600$。固定种子长路径得到均值 $-0.013814$、方差 $2.764404$、一阶自相关 $0.799071$，只在声明容差内交叉验证解析式。
```

## tex/upper/chapters/ch14-hints.tex

```tex
\item 固定输入顺序并保留所有算法输出；高精度路径不能复用 float64 累加。
```

## tex/upper/chapters/ch14.tex

```tex
\begin{implementationnote}
并行归约改变分块与合并顺序，所以线程数、设备和调度可能改变末位。研究包应冻结归约树或声明允许的误差带，并同时保留一个高精度、小规模 oracle。若逐位可复现是协议要求，需要使用确定性归约或精确累加器，而不是把 \texttt{rtol} 调大。
\end{implementationnote}
```

## tex/upper/chapters/ch14.tex

```tex
相邻浮点数的距离可以用 $\operatorname{ulp}$ 读出：在 $[1,2)$ 内相邻数间隔为 $2^{-52}$，在 $[2,4)$ 内间隔变为 $2^{-51}$。因此固定的绝对容差在不同数量级上含义不同；一个适合 $10^{-8}$ 的阈值可能完全不适合 $10^8$。
```

## tex/upper/chapters/ch14.tex

```tex
$\mathrm{atol}$ 管理接近零的绝对尺度，$\mathrm{rtol}$ 管理远离零的相对尺度。固定实验中，相对策略在 $10^{-8}$ 与 $10^8$ 两个尺度保持一致；纯绝对容差会接受小尺度 100\% 误差，却拒绝大尺度的微小相对误差。数组比较还必须拒绝意外 NaN/Inf，避免 NaN 传播被“都不相等”或特殊选项掩盖。
```

## tex/upper/chapters/ch17.tex

```tex
可复现材料让别人使用同一输入重建这些数值。文件哈希只能检验字节是否改变，不能判断价格真实、信息及时、检验有效或交易可行。环境、命令、文件完整性和许可记录的操作方法见附录“实验结果的整理与复核”；在本章，读者要解释的是计算与主张之间的联系。
```

## tex/upper/solutions/ch01.tex

```tex
\begin{lstlisting}[language=bash]
uv run jupyter nbconvert --to notebook --execute --stdout \
  --ExecutePreprocessor.timeout=60 \
  --ExecutePreprocessor.allow_error_names=SystemExit \
  notebooks/upper/ch01_convex_bound.ipynb
\end{lstlisting}
```

## tex/upper/solutions/ch02.tex

```tex
\begin{lstlisting}[language=bash]
uv run jupyter nbconvert --to notebook --execute --stdout \
  --ExecutePreprocessor.timeout=60 \
  --ExecutePreprocessor.allow_error_names=SystemExit \
  notebooks/upper/ch02_analysis_foundations.ipynb
\end{lstlisting}
```

## tex/upper/solutions/ch03.tex

```tex
\begin{lstlisting}[language=bash]
uv run jupyter nbconvert --to notebook --execute --stdout \
  --ExecutePreprocessor.timeout=60 \
  --ExecutePreprocessor.allow_error_names=SystemExit \
  notebooks/upper/ch03_measure_integration.ipynb
\end{lstlisting}
```

## tex/upper/solutions/ch03.tex

```tex
再复制 `evidence/ch03/oracle.json` 到临时路径逐项注入坏输入。每次只改变一个门禁，记录退出码与完整 stderr，恢复后要求原始固定账本重新通过。若数值网格与解析面积冲突，先审查采样位置、区间宽度和尾部质量，不要修改独立 expected 迁就程序。
```

## tex/upper/solutions/ch04.tex

```tex
\begin{lstlisting}[language=bash]
uv run jupyter nbconvert --to notebook --execute --stdout \
  --ExecutePreprocessor.timeout=60 \
  --ExecutePreprocessor.allow_error_names=SystemExit \
  notebooks/upper/ch04_lp_product_measure.ipynb
\end{lstlisting}
```

## tex/upper/solutions/ch04.tex

```tex
再复制 `evidence/ch04/oracle.json` 到临时文件，逐项注入一个坏值并从公开 CLI 运行脚本。每次记录退出码、完整 stderr 和被保护的数学账本。若两个积分次序不同，先保留有限矩形与绝对值和证据，再判断 Tonelli 或 Fubini 哪条假设缺失；不要用更高浮点精度掩盖一个本来就不合法的换序。
```

## tex/upper/solutions/ch05.tex

```tex
\begin{lstlisting}[language=bash]
uv run jupyter nbconvert --to notebook --execute --stdout \
  --ExecutePreprocessor.timeout=60 \
  --ExecutePreprocessor.allow_error_names=SystemExit \
  notebooks/upper/ch05_matrix_calculus.ipynb
\end{lstlisting}
```

## tex/upper/solutions/ch05.tex

```tex
\item 先固定最小标量目标、参数点和 dtype；列出每个中间量形状及 Jacobian 约定；读取独立解析闭式结果。再比较自动微分与解析参照，检查 detach、原地修改、广播、转置和不可微算子；最后扫描中心差分步长并比较左右差商。每次只改变一个因素，保留退出状态与完整输出，不通过随意转置或放宽容差隐藏错误。
```

## tex/upper/solutions/ch05.tex

```tex
再复制 `evidence/ch05/oracle.json` 到临时文件逐项注入坏值，从公开 CLI 记录稳定诊断。若自动微分、差分和手推不一致，先回到固定非对称二次型与形状账本；只有该闭式案例一致后，才把复杂模型逐层接回。
```

## tex/upper/solutions/ch07.tex

```tex
\begin{lstlisting}[language=bash]
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.allow_error_names=SystemExit notebooks/upper/ch07_probability_distributions.ipynb
uv run jupyter nbconvert --to notebook --execute --stdout \
  --ExecutePreprocessor.timeout=60 \
  --ExecutePreprocessor.allow_error_names=SystemExit \
  notebooks/upper/ch07_probability_distributions.ipynb
\end{lstlisting}
```

## tex/upper/solutions/ch15.tex

```tex
\begin{center}
  \path{uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.allow_error_names=SystemExit notebooks/upper/ch15_monte_carlo.ipynb}
\end{center}
```

## tex/upper/solutions/ch16.tex

```tex
\begin{center}
\path{uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.allow_error_names=SystemExit notebooks/upper/ch16_research_validity.ipynb}
\end{center}
```

## tex/lower/chapters/multifactor-estimation.tex

```tex
\MFQTransition{双实现协议}

透明实现负责：展示目标函数、单位、截距规则、更新公式和失败条件。成熟库负责：提供独立实现、更多诊断和工业级数值路径。交叉检查覆盖横截面与面板回归、中性化残差、IC、Rank IC、衰减、BH、ridge 与 lasso；至少比较系数、预测、残差、目标函数与退出状态。只比较“程序跑完”没有证据强度。

\begin{longtable}{p{0.18\textwidth}p{0.34\textwidth}p{0.34\textwidth}}
\toprule
任务 & 透明路径 & 成熟库路径 \\
\midrule
横截面/两遍回归 & NumPy 最小二乘与手算 解析参照 & \texttt{statsmodels.OLS} \\
Ridge & 闭式线性方程 & \texttt{sklearn.linear\_model.Ridge} \\
Lasso & 坐标下降与软阈值 & \texttt{sklearn.linear\_model.Lasso} \\
\bottomrule
\end{longtable}

\MFQTransition{WDI 事先确定截面：只演示研究设计}

外部快照使用 2013 年资本形成增速作信号、2014 年人均 GDP 增速作结果，覆盖 7 个国家，严格满足信号年早于结果年。事先确定相关系数约为 $-0.336654$。

这个数不能被称为股票因子收益：样本只有 7 个国家，变量是宏观增速，没有可交易资产、发布日期滞后、交易成本或因果识别。合成面板仍是唯一正确性 解析参照。

Capstone 必须保存训练期变换参数、惩罚路径、选择规则、库版本、条件数和实现差异。
```

## tex/lower/chapters/portfolio-risk-optimization.tex

```tex
真实数据轨把事先确定宏观快照的 39 个相邻增长率向量作为情景输入同一 LP；在 75\% 水平和 0.8 权重上限下，得到权重 $(0.2,0.8)$ 与 CVaR $0.001212$。这证明真实快照确实进入优化问题，但 GDP 与消费增长并非可交易资产，结果只理解数据适配、时间边界和约束台账。
```

## D:\Latex\math-for-quant\tex\upper\solutions\ch15.tex

```tex
\section*{代码恢复路径}
在仓库根目录运行

正常输出只给稳定证据类别：Monte Carlo 误差位于四标准误内、误差率检查通过、控制变量均值与方差条件通过、bootstrap 精确枚举为 27 个状态。随后分别把 解析参照 中的 \texttt{control\_known\_mean} 改成 $0.6$、把 \texttt{bootstrap\_observations\_iid} 改成 false；前者应报告目标均值被改变，后者应拒绝 IID 重抽样。完成恢复时应能解释每个失败为什么不能靠增加样本量或重复次数修复。



```

## D:\Latex\math-for-quant\tex\upper\solutions\ch16.tex

```tex
\section*{代码恢复路径}
在仓库根目录运行

正常路径必须同时给出 timeline、timezone、calendar、split、multiplicity、performance 和 friction 七类通过证据。随后复制 解析参照：先把第一行 \path{available_at} 改到 \path{decision_at} 之后，确认时间条件失败；再恢复时间并把 \path{selected_raw_p_value} 改为 $0.003$，确认多重检验条件失败。把任一时间改成错误 UTC offset 或把决策日移出 \path{2024-07-v1} 日历，也必须稳定失败。恢复完成的标准不是“改回能跑”，而是能解释负例分别破坏了哪个研究假设，以及为什么增加样本或更换随机 seed 都无法修复。


```

## D:\Latex\math-for-quant\tex\upper\solutions\ch16.tex

```tex
\section*{解析或独立计算参照 证据对照}
本章的发布证据固定为：\textbf{事件时间、可得时间、决策时间与收益起点}满足先后约束；Bonferroni 阈值为 $0.0025$；三笔交易的\textbf{总成本为 $0.006$}；\textbf{成本后收益为 $0.034$}。市场条件协议要求逐研究期核验\textbf{T+1、涨跌停成交、停牌、复权可得性与融券资格}，而不是把当前规则参数当作跨时期常数。



```

## D:\Latex\math-for-quant\tex\lower\chapters\derivatives-hedging-solutions.tex

```tex
\begin{enumerate}
  \item 还剩 Gamma、Vega、Theta、高阶项、参数误差、跳跃、离散调仓、成本和成交误差。Delta 只是当前模型下的一阶局部导数。
  \item 单路径无法估计 bias、方差或尾部；研究者还可能事后挑路径。至少应固定路径生成协议并报告全体误差分布。
  \item 令现金 $B_i$ 先增长为 $B_{i-1}e^{r\Delta t}$，再令 $q_i=\Delta_i-\Delta_{i-1}$，则 $B_i=B_{i-1}e^{r\Delta t}-q_iS_i-c|q_i|S_i$。
  \item bias 揭示平均方向，RMSE 同时惩罚偏差与波动，分位数不假设对称并揭示尾部。三者不能互相替代。
  \item 事先确定 24 步结果无成本 RMSE 为 $1.416360$，成本后为 $1.434142$；12 与 52 步应在同一随机路径和成本规则下重算，不能只比较其中最优者。
  \item 在某一步把 $S$ 乘以跳跃因子并保持旧 Delta 到下一个调仓点。报告跳跃方向、时点、大小、成本以及误差分位数变化。
  \item 错误报告只给路径编号、没有全体路径和预先选择规则。修复应事先确定种子与路径数，提交所有统计或可重建计算记录，而不是再挑一条看似典型的路径。
  \item 结合本章模型，解释预测、成交和成本如何共同决定结果，并说明所用假设。
\end{enumerate}

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\microstructure-control-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 临时冲击 $\eta x_k$ 只进入第 $k$ 期价格；永久冲击 $\gamma X_{k-1}$ 随累计成交进入所有后续期。两者对“快做还是慢做”给出不同边际成本。
  \item 第一期价格 $100+0.2\times3=100.6$，成本 $1.8$；第二期价格 $100+0.1\times3+0.2\times2=100.7$，成本 $1.4$，合计 $3.2$，剩余 4 手另行报告。
  \item $V_K(0)=0$，$V_K(R>0)=+\infty$ 或显式终端罚函数。否则算法会通过不交易获得虚假零成本。
  \item 枚举每期 $0,1,2,3$，筛选总和为 6；按同一目标重算，最优 $(3,2,1)$、成本 $19.2$，与递归 DP 相同。
  \item $\kappa$ 或库存风险越高，多头状态下报价中心下降越多，ask 更积极、bid 更保守；但成交强度响应若未建模，不能断言期末库存必然下降。
  \item 若整条报价序列使用事后库存，则未来成交影响过去报价。正确循环必须是读取当前状态、报价、观察成交、更新状态，再进入下一步。
  \item 修复包至少含决策时到达价、方向、计划与实际量、逐笔成交、费用、未完成量、停止原因、基点与现金两种口径、对照基准和时间戳。
  \item 压力矩阵交叉扫描完成率分布、临时/永久冲击、期限和损失阈值；所有控制器复用相同随机样本，并报告完成率、IS、尾部和触停概率。
\end{enumerate}

事先确定 解析参照：实际成交 5、剩余 4、partial-fill IS 为 $3.2$；DP 与完整枚举均给出 $(3,2,1)$ 和目标 $19.2$；bid 成交后下一轮报价为 $99.4/100.4$。

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\microstructure-simulation-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 若结果为 $Y_A,Y_B$，则 $\operatorname{Var}(Y_A-Y_B)=\operatorname{Var}Y_A+\operatorname{Var}Y_B-2\operatorname{Cov}(Y_A,Y_B)$。共享事件流通常提高协方差，从而减少差值噪声。
  \item 在本章符号约定下，可写 $\pi_i=h+q_i\Delta m_i$；全路径为 $\sum_i\pi_i$。若采用库存现金计算记录，则必须证明与该简式在边界条件下等价。
  \item 独立重放多条完整路径，对每条计算三项统计，再对路径统计做 t 区间、百分位 bootstrap 或成对差区间；应同时报告路径数和随机种子。
  \item 事先确定分段常数 $s(t)$，模拟非齐次过程；计算 $\int_{t_{i-1}}^{t_i}\widehat\lambda(u)\,du$，检查其是否近似单位指数且无剩余季节模式。
  \item trade id 非递增、编号重复应在解析层拒绝；被动委托导致最高买价不低于最低卖价时，订单簿不变量必须失败且不保留半更新状态。
  \item 单路径没有采样不确定性。至少补共同随机数、路径分布、库存尾部、成交率、费用、冲击、未完成状态、参数选择窗口和压力场景。
  \item 公开成交没有自己的订单编号、前方数量、隐藏委托、撤单或接收延迟；同一成交序列可对应许多不同队列状态，所以概率不可识别。
  \item 收益领先不是理解前提。
\end{enumerate}

事先确定的独立基准包含 500 个事件；每个事件共享方向与成交均匀数。

静态和控制分别成交 21/23 次；PnL 为 $-0.280297/-0.027929$，期末库存 $-1/1$，最大绝对库存 $3/2$。

SEC 摘要加权 cancel-to-trade 比率为 $55.3908$，隐含执行概率约 $0.044470$。

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\ml-alpha-research-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item MSE 服务数值预测，排序损失服务横截面次序，收益加权目标接近决策效用；三者仍需成本后组合检验。
  \item 零成交意味着订单没有改变旧仓位。只有成交的订单增量才进入新仓位。
  \item 第一期目标/实际为 $[1,0,-1]$；第二期目标 $[-1,1,0]$、订单 $[-2,1,1]$、成交后 $[0,1,-1]$。毛收益 $0.03+0.02=0.05$，换手 4、成本 0.004、净收益 0.046。
  \item 若 $L_R=-n^{-1}\sum s_ir_i$ 且没有范数或仓位约束，把正确方向分数乘任意正数会继续降低损失。
  \item 借券失败将对应卖单成交比例设为 0，并保留旧仓位；换手只累计实际成交增量。
  \item 结合本章模型，解释预测、成交和成本如何共同决定结果，并说明所用假设。
  \item 好预测不等于可交易。应从分数重建目标、实际成交、成本和容量；净收益为负时不能以预测指标替代结论。
  \item 结合本章模型，解释预测、成交和成本如何共同决定结果，并说明所用假设。
\end{enumerate}

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\multifactor-estimation-solutions.tex

```tex
\section*{解析或独立计算参照 核对}
透明 ridge 系数为 $0.465574$，透明 lasso 系数为 $0.470000$。
零范数特征列必须在迭代前被拒绝。


```

## D:\Latex\math-for-quant\tex\lower\chapters\multifactor-estimation-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 条件数与实现交叉检查支持数值稳定；抽样和误差协议支持统计识别；经济机制、可交易定义与外部证据支持解释。
  \item 高相关特征可互相替代，轻微样本扰动会改变被保留者。应报告重采样选择频率而非只报一次非零系数。
  \item 梯度为 $X^{\mathsf T}(X\beta-y)+\lambda\beta$，令其为零得到 $(X^{\mathsf T}X+\lambda I)\beta=X^{\mathsf T}y$。
  \item 一维子问题的解是对 $x_j^{\mathsf T}r_{-j}$ 软阈值后除以 $x_j^{\mathsf T}x_j$。
  \item 在相同截距和损失缩放下，透明与库系数应落在声明容差内；否则先检查目标契约。
  \item 零列令坐标更新分母为零，应直接拒绝；近共线列应由奇异值和条件数报告。
  \item 该研究材料把最终期用于选择惩罚参数，因此所谓测试表现有选择偏差；需重新划分未触碰测试期。
  \item WDI 数据不是资产收益，样本极小且无交易机制或因果识别；只可演示外部数据责任链。
\end{enumerate}

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\multifactor-research-solutions.tex

```tex
\section*{解析或独立计算参照 核对}
两期等权分组毛收益为 $0.035$ 与 $0.030$，期均值为 $0.0325$。
扣除 $0.002$ 换手成本和 $0.0005$ 容量冲击后，净收益为 $0.0300$。


```

## D:\Latex\math-for-quant\tex\lower\chapters\multifactor-research-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 信号是时点输入，排序是变换，目标权重是决策，成交权重是实现状态，净收益是后来市场结果扣除摩擦；每步都需单独记录。
  \item 等权改变组内集中度，市值权重引入规模暴露，重叠持有把多个 vintage 合成并引入收益自相关。
  \item 两期等权分组毛收益为 $0.035$ 与 $0.030$，期均值为 $0.0325$。平均双边换手 2 对应成本 $0.002$；再减容量冲击 $0.0005$，净收益 $0.0300$。
  \item $w_t=|A_t|^{-1}\sum_{s\in A_t}w_t^{(s)}$；相邻期共享 vintage，所以普通独立标准误不再适用。
  \item 敏感性表必须固定资产池、时间切分和摩擦，只改变分组、权重或持有期中的一项，并完整报告所有结果。
  \item 每期断言 gross 等于仓位点乘下一期收益；外生收益字段应删除。
  \item 20 项子集不是原检验族。恢复 500 项尝试后重做校正；若计算记录缺失，只能声明无法审计。
  \item 涨跌停、停牌、T+1、融券和成交完成率都影响可交易权重；删除不可交易观察会形成选择偏差。
  \item 结合本章模型，解释预测、成交和成本如何共同决定结果，并说明所用假设。
\end{enumerate}

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\portfolio-risk-optimization-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item $P$ 的每行定义一个绝对或相对观点，$q$ 是其期望收益，$\Omega$ 是观点误差协方差，$\tau\Sigma$ 是先验均值不确定性。减小 $\Omega$ 只表示提高置信度。
  \item 合并 $(\mu-\pi)^\top(\tau\Sigma)^{-1}(\mu-\pi)$ 与 $(P\mu-q)^\top\Omega^{-1}(P\mu-q)$，令一阶导数为零，即得正文精度公式。
  \item 对每个情景加入 $u_s\ge-r_s^\top w-z$ 和 $u_s\ge0$，目标为 $z+[(1-\alpha)N]^{-1}\sum_su_s$；连同满仓和 bounds 即为 LP。
  \item 事先确定八情景最优权重为 $(0.2,0.8)$，最坏两个情景的平均损失为 $0.013$。枚举必须应用同一 0.8 上限，否则会得到不可比答案。
  \item 上限 0.6 时可行权重落在 $[0.4,0.6]$；应重新求解而不是把原解裁剪，并报告上限乘子或枚举边界是否活跃。
  \item 裁剪会改变权重和、风险、换手与成本，且裁剪后的点一般不满足原问题的最优性。应把边界直接写进可行域。
  \item 均值不确定性惩罚降低暴露于低置信度收益的权重；ridge 稳定协方差谱；成本惩罚交易距离。数值形状相似不等于经济含义相同。
\end{enumerate}

事先确定 解析参照：BL 后验均值为 $(0.05375,0.03)$；CVaR LP 与枚举均得到 $(0.2,0.8)$ 和 $0.013$；稳健成本感知再平衡得到 $(0.8,0.2)$、双边换手 $0.8$、现金成本 80。
% MFQ source replacements removed=1 unit=PortfolioOptimization

\input{tex/common/source-mapped-exercises}
\MFQMappedPortfolioOptimizationSolutions

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\portfolio-risk-tail-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item VaR/ES 描述分布尾部；历史压力重演已知联合冲击；假设压力评估指定条件；反向压力寻找触发资本或流动性阈值的冲击尺度。
  \item 若尾部质量不是整数，积分分位数把边界观察按剩余概率质量加权，再除以总尾部质量；这避免随意全纳入或全排除。
  \item 代入 $sx$ 得 $L(sx)=-s d^\top x-s^2\sum_i g_ix_i^2/2$。令其等于阈值 $C$，求最小非负根并验证第一处穿越。
  \item 500 点在 95\% 下有 25 个尾部观察，状态 warn；99\% 下只有 5 个，应按默认阈值拒绝。总样本量没有改变这一区别。
  \item 课程二阶例的线性/非线性单位损失为 $4.4/4.016$。全重估应同时更新现货、曲线、波动率曲面和剩余期限，并报告模型失败。
  \item 四点样本在 99\% 下有效尾部数为 0.04，任何 ES 数字几乎只是最大值重命名。报告应拒绝或明确标成教学演示。
  \item $s^*$ 是沿已声明方向的条件阈值。只有另建冲击概率模型，才可能讨论达到该阈值的概率；方向选择本身也有模型风险。
\end{enumerate}

事先确定 解析参照：95\% VaR 为 $7.498998$，ES 为 $7.759519$，有效尾部数 25、状态 warn；ES 区间为
\[
 [7.627635,\,7.854910].
\]
反向压力最小尺度为 $3.125$，损失重估为 10。
% MFQ source replacements removed=1 unit=PortfolioTail

\input{tex/common/source-mapped-exercises}
\MFQMappedPortfolioTailSolutions

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\stat-arb-estimation-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item 预测为 $p(s_t\mid y_{1:t-1})$，滤波为 $p(s_t\mid y_{1:t})$，平滑为 $p(s_t\mid y_{1:T})$。只有前两者可按相应时点在线获得。
  \item 交换两套状态参数与转移矩阵行列不改变观测似然；经济命名需要额外排序约束，且应报告该约束。
  \item $K=P^-H/(H^2P^-+R)$；$m^+=m^-+K(y-Hm^-)$；$P^+=(1-KH)P^-$。
  \item $\pi_{t|t-1}=\pi_{t-1|t-1}P$；再计算 $\tilde\pi_{t,j}=f_j(y_t)\pi_{t|t-1,j}$ 并除以总和。
  \item 较大 $Q/R$ 提高增益、缩短响应时间，但也放大观测冲击；敏感性报告应同时给误差与平滑度。
  \item 至少保存每次初值、收敛状态、对数似然、转移矩阵和对齐后的概率路径。近似同似然但不同参数说明弱识别。
  \item 修复是只保存当时滤波状态并从下一滚动窗重新估计。
\end{enumerate}

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\stat-arb-research-solutions.tex

```tex
\begin{enumerate}[leftmargin=*]
  \item purge 删除标签结果跨入验证区的训练样本；embargo 在已用于选择的验证区后留空，避免相邻依赖或持仓跨入最终交易期。
  \item 目标仓位是意图；真实收益只能使用实际成交仓位。部分成交、停牌、借券失败与延迟都会令二者不同。
  \item 实际仓位 $[1,0,0,0]$：第二期反向订单为 $-2$，成交一半后平仓。毛收益 $1(0.02)=0.02$；换手 $1+1=2$；成本 $2(0.002)=0.004$；净收益 $0.016$。
  \item 持有期 2 时第二期不能因新负预测立即翻空，目标仓位保留为 1；模型需明确持有年龄是在决策前还是后递增，并以测试事先确定。
  \item 为每笔订单保存目标量、旧仓位、订单增量、可借数量、成交比例、原因码与时间；失败订单不改变旧仓位，不能强制归零或从样本中静默删除。
  \item 拟合截止晚于训练结束必须失败；成交比例为 0 时，仓位必须等于旧仓位。
  \item 结合本章模型，解释预测、成交和成本如何共同决定结果，并说明所用假设。
\end{enumerate}

\section*{逐题推导与核对}
```

## D:\Latex\math-for-quant\tex\lower\chapters\portfolio-risk-tail-questions.tex

```tex
\MFQTransition{错误研究材料审计清单}

对第 6 题不要只写“样本太少”。把结论拆成可复核的证据：

\begin{center}
\begin{tabular}{@{}p{0.18\textwidth}p{0.36\textwidth}p{0.38\textwidth}@{}}
  \toprule
  审计项 & 必须报告的证据 & 最小恢复动作 \\
  \midrule
  尾部有效性 & 置信水平、样本总数、有效尾部观察数与分位点分辨率
    & 样本不足时返回 \texttt{warn} 或 \texttt{reject}，而不是只打印一个 ES 数字。 \\
  估计不确定性 & 分位数约定、重采样方案、区间端点与随机种子
    & 增加 bootstrap 区间，并说明依赖结构是否允许独立重采样。 \\
  模型边界 & 估值时点、数据来源、非线性重估范围、成本与不可交易约束
    & 用历史压力和反向压力补充统计尾部，并把未覆盖风险写入限制声明。 \\
  \bottomrule
\end{tabular}
\end{center}

开放 Capstone 的提交物应包含一条可复现命令、独立手算或库实现对照、失败注入、完整限制声明，以及从原始输入到最终报告的文件清单。审计者必须能够区分“计算成功”“统计证据足够”和“研究结论可行动”这三个不同状态。
% MFQ source replacements removed=1 unit=PortfolioTail

\input{tex/common/source-mapped-exercises}
\MFQMappedPortfolioTailQuestions


```
