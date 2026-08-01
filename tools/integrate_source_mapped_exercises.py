from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
COMMON = r"\input{tex/common/source-mapped-exercises}"


UNITS = {
    "UpperChFive": ("tex/upper/chapters/ch05-questions.tex", "tex/upper/chapters/ch05-hints.tex", "tex/upper/solutions/ch05.tex"),
    "UpperChSix": ("tex/upper/chapters/ch06-questions.tex", "tex/upper/chapters/ch06-hints.tex", "tex/upper/solutions/ch06.tex"),
    "UpperChSeven": ("tex/upper/chapters/ch07-questions.tex", "tex/upper/chapters/ch07-hints.tex", "tex/upper/solutions/ch07.tex"),
    "UpperChEight": ("tex/upper/chapters/ch08-questions.tex", "tex/upper/chapters/ch08-hints.tex", "tex/upper/solutions/ch08.tex"),
    "UpperChNine": ("tex/upper/chapters/ch09-questions.tex", "tex/upper/chapters/ch09-hints.tex", "tex/upper/solutions/ch09.tex"),
    "UpperChEleven": ("tex/upper/chapters/ch11-questions.tex", "tex/upper/chapters/ch11-hints.tex", "tex/upper/solutions/ch11.tex"),
    "UpperThirteen": ("tex/upper/chapters/ch13-questions.tex", "tex/upper/chapters/ch13-hints.tex", "tex/upper/solutions/ch13.tex"),
    "UpperFourteen": ("tex/upper/chapters/ch14-questions.tex", "tex/upper/chapters/ch14-hints.tex", "tex/upper/solutions/ch14.tex"),
    "DerivativesStochastic": ("tex/lower/chapters/derivatives-stochastic-questions.tex", "tex/lower/chapters/derivatives-stochastic-hints.tex", "tex/lower/chapters/derivatives-stochastic-solutions.tex"),
    "DerivativesNumerics": ("tex/lower/chapters/derivatives-numerics-questions.tex", "tex/lower/chapters/derivatives-numerics-hints.tex", "tex/lower/chapters/derivatives-numerics-solutions.tex"),
    "PortfolioOptimization": ("tex/lower/chapters/portfolio-risk-optimization-questions.tex", "tex/lower/chapters/portfolio-risk-optimization-hints.tex", "tex/lower/chapters/portfolio-risk-optimization-solutions.tex"),
    "PortfolioTail": ("tex/lower/chapters/portfolio-risk-tail-questions.tex", "tex/lower/chapters/portfolio-risk-tail-hints.tex", "tex/lower/chapters/portfolio-risk-tail-solutions.tex"),
}

REPLACEMENT_SELECTORS = {
    "UpperChFive": {
        "questions": ("一个损失在训练中下降", "样本协方差使均值--方差闭式解"),
        "solutions": ("先写损失是", "第一层用有限差分"),
    },
    "UpperChSix": {
        "questions": ("对 $A_\\varepsilon$ 比较", "因子协方差矩阵出现", "回归系数在样本窗口", "研究员建议“统一加 ridge"),
        "solutions": ("对坐标扰动", "先核对对称化", "预测稳定但参数翻倍", "先说明 ridge 目标"),
    },
    "UpperChSeven": {
        "questions": ("离散变量 $D$", "样本均值在滚动窗口", "团队用相关矩阵管理风险"),
        "solutions": ("对 $D$ 直接有限求和", "先报告尾指数估计方法", "直接检查联合下尾事件"),
    },
    "UpperChEight": {
        "questions": ("只在“成功成交”样本中", "团队声称模型估计"),
        "solutions": ("成交由报价激进度", "定义 $\\mathcal F_t$ 只含"),
    },
    "UpperChNine": {
        "questions": ("样本均值随样本增长看似稳定",),
        "solutions": ("先滚动检查位置",),
    },
    "UpperChEleven": {
        "questions": ("连续 Brownian 风险模型",),
        "solutions": ("明确 Brownian 连续路径",),
    },
    "UpperThirteen": {
        "questions": ("如何为带收益、风险、换手",),
        "solutions": ("证据包应含数据与估计窗口",),
    },
    "UpperFourteen": {
        "questions": ("同一因子归约在不同线程数", "求解器成功、残差极小"),
        "solutions": ("若合同只要求业务量级一致", "将求解器成功解释为"),
    },
    "DerivativesStochastic": {
        "questions": ("Novikov 失败为什么不等于",),
        "solutions": ("Novikov 是保证随机指数",),
    },
    "DerivativesNumerics": {
        "questions": ("财政部 par yield 快照",),
        "solutions": ("财政部 par yield 仍需",),
    },
    "PortfolioOptimization": {
        "questions": ("加入部分成交与整数头寸",),
        "solutions": ("先生成目标订单",),
    },
    "PortfolioTail": {
        "questions": ("开放 Capstone",),
        "solutions": ("Capstone 应包含哈希",),
    },
}


def append_once(relative: str, marker: str, command: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + COMMON + "\n" + command + "\n", encoding="utf-8")


def remove_replaced_items(relative: str, stem: str, selectors: tuple[str, ...]) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = len(selectors)
    audit = f"% MFQ source replacements removed={count} unit={stem}"
    already_removed = audit in text
    split_at = text.find(COMMON)
    prefix = text if split_at < 0 else text[:split_at]
    suffix = "" if split_at < 0 else text[split_at:]
    if not already_removed:
        spans = list(re.finditer(r"(?ms)^\s*\\item\b.*?(?=^\s*\\item\b|^\s*\\end\{enumerate\})", prefix))
        selected = []
        for selector in selectors:
            matches = [match for match in spans if selector in match.group(0)]
            if len(matches) != 1:
                raise ValueError(f"{relative}: selector {selector!r} matched {len(matches)} items")
            selected.append(matches[0])
        if len({match.start() for match in selected}) != count:
            raise ValueError(f"{relative}: replacement selectors are not unique")
        for match in sorted(selected, key=lambda item: item.start(), reverse=True):
            prefix = prefix[:match.start()] + prefix[match.end():]
    prefix = re.sub(
        r"(?ms)^\s*\\(?:subsection|section)\*\{[^}]+\}\s*\\begin\{enumerate\}(?:\[[^]]*\])?\s*\\end\{enumerate\}\s*",
        "\n",
        prefix,
    )
    if already_removed:
        path.write_text(prefix.rstrip() + "\n\n" + suffix.lstrip(), encoding="utf-8")
    else:
        path.write_text(prefix.rstrip() + "\n" + audit + "\n\n" + suffix.lstrip(), encoding="utf-8")


def main() -> int:
    for stem, (questions, hints, solutions) in UNITS.items():
        selectors = REPLACEMENT_SELECTORS[stem]
        remove_replaced_items(questions, stem, selectors["questions"])
        remove_replaced_items(solutions, stem, selectors["solutions"])
        append_once(questions, f"MFQMapped{stem}Questions", rf"\MFQMapped{stem}Questions")
        append_once(hints, f"MFQMapped{stem}Hints", rf"\MFQMapped{stem}Hints")
        append_once(solutions, f"MFQMapped{stem}Solutions", rf"\MFQMapped{stem}Solutions")
    print(f"integrated_units={len(UNITS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
