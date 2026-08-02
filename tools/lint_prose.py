"""Report prose-style warnings for the reader-facing TeX sources.

This is deliberately advisory.  The book contains technical appendices and
research-audit material where project vocabulary is appropriate; the linter
helps find places where that vocabulary has leaked into ordinary exposition.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


ENGINEERING_TERMS = (
    "门禁",
    "验收",
    "oracle",
    "fixture",
    "schema",
    "稳定拒绝",
    "逐字比较",
    "package root",
    "stdout",
    "stderr",
    "故障注入",
    "账本",
    "研究包",
)
ROUTE_PREFIXES = {
    "multifactor": "多因子与计量",
    "stat-arb": "时间序列与统计套利",
    "ml-alpha": "机器学习 Alpha",
    "derivatives": "衍生品定价与对冲",
    "portfolio-risk": "组合与风险",
    "microstructure": "高频、微观结构与执行",
}


def _reader_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for base in (root / "tex" / "upper" / "chapters", root / "tex" / "lower" / "chapters"):
        for path in sorted(base.glob("*.tex")):
            if any(token in path.stem for token in ("-hints", "-questions", "-solutions")):
                continue
            files.append(path)
    return files


def _strip_technical_blocks(text: str) -> str:
    text = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", text, flags=re.S)
    text = re.sub(r"\\begin\{implementationnote\}.*?\\end\{implementationnote\}", "", text, flags=re.S)
    return text


def lint(root: Path) -> list[str]:
    warnings: list[str] = []
    route_leads: Counter[str] = Counter()
    route_files: defaultdict[str, list[Path]] = defaultdict(list)

    for path in _reader_files(root):
        raw = path.read_text(encoding="utf-8")
        prose = _strip_technical_blocks(raw)
        rel = path.relative_to(root).as_posix()

        leads = len(re.findall(r"\\MFQLead\{", raw))
        if leads > 6:
            warnings.append(f"{rel}: MFQLead 使用 {leads} 次（建议不超过 6 次）")

        for prefix, label in ROUTE_PREFIXES.items():
            if path.stem.startswith(prefix):
                route_leads[label] += leads
                route_files[label].append(path)

        counts = Counter()
        for term in ENGINEERING_TERMS:
            counts[term] = len(re.findall(re.escape(term), prose, flags=re.I))
        dense = [f"{term}={count}" for term, count in counts.items() if count >= 4]
        if dense:
            warnings.append(f"{rel}: 正文工程词密度偏高（{', '.join(dense)}）")

        outside_notes = re.sub(
            r"\\begin\{implementationnote\}.*?\\end\{implementationnote\}",
            "",
            raw,
            flags=re.S,
        )
        path_count = len(re.findall(r"\\path\{", outside_notes))
        if path_count:
            warnings.append(f"{rel}: implementationnote 之外出现 {path_count} 个路径标记")

        titles = re.findall(r"\\(?:section|subsection)\*?\{([^{}]+)\}", raw)
        duplicates = [title for title, count in Counter(titles).items() if count > 1]
        if duplicates:
            warnings.append(f"{rel}: 重复标题 {duplicates}")

    for label, count in route_leads.items():
        if count > 6:
            names = ", ".join(p.name for p in route_files[label])
            warnings.append(f"路线 {label}: MFQLead 合计 {count} 次（{names}）")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    warnings = lint(args.root.resolve())
    if not warnings:
        print("prose-lint: no warnings")
        return 0
    print("prose-lint: advisory warnings")
    for warning in warnings:
        print(f"- {warning}")
    print(f"total_warnings={len(warnings)} (non-blocking)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
