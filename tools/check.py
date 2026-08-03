"""Run the two lightweight maintenance checks used by the book project."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING_TERMS = (
    "门禁", "验收", "oracle", "fixture", "schema", "稳定拒绝", "逐字比较",
    "package root", "stdout", "stderr", "故障注入", "账本", "研究包",
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
    return re.sub(r"\\begin\{implementationnote\}.*?\\end\{implementationnote\}", "", text, flags=re.S)


def lint_prose(root: Path) -> list[str]:
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
        counts = {
            term: len(re.findall(re.escape(term), prose, flags=re.I))
            for term in ENGINEERING_TERMS
        }
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
            names = ", ".join(path.name for path in route_files[label])
            warnings.append(f"路线 {label}: MFQLead 合计 {count} 次（{names}）")
    return warnings


def run_prose(root: Path) -> int:
    warnings = lint_prose(root.resolve())
    if not warnings:
        print("prose-lint: no warnings")
        return 0
    print("prose-lint: advisory warnings")
    for warning in warnings:
        print(f"- {warning}")
    print(f"total_warnings={len(warnings)} (non-blocking)")
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run_template(source: Path | None, vendored_only: bool) -> int:
    if source is None and not vendored_only:
        print("provide --source or --vendored-only", file=sys.stderr)
        return 2
    provenance = json.loads(
        (ROOT / "docs" / "template-provenance.json").read_text(encoding="utf-8")
    )
    files = provenance["files"]
    if source is not None:
        source = source.resolve()
        declared = {str(record["source"]) for record in files}
        observed = {
            path.relative_to(source).as_posix()
            for path in source.rglob("*")
            if path.is_file()
        }
        if declared != observed:
            missing = sorted(declared - observed)
            extra = sorted(observed - declared)
            print(f"external template inventory changed: missing={missing} extra={extra}", file=sys.stderr)
            return 1
        for record in files:
            name = record["source"]
            expected = record["sha256"]
            path = source / name
            if not path.is_file() or _sha256(path) != expected:
                print(f"external template changed: {name}", file=sys.stderr)
                return 1

    vendored = [record for record in files if record.get("vendored")]
    for record in vendored:
        path = ROOT / record["vendored"]
        expected = record.get("vendored_sha256", record["sha256"])
        if not path.is_file() or _sha256(path) != expected:
            print(f"vendored asset does not match its source: {record['vendored']}", file=sys.stderr)
            return 1
    print(f"template={provenance['template']}")
    print(f"external-baseline={'passed files=' + str(len(files)) if source is not None else 'not-available ci-vendored-check=used'}")
    print(f"vendored-assets=passed files={len(vendored)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="check", required=True)
    prose = subparsers.add_parser("prose", help="report advisory reader-facing prose warnings")
    prose.add_argument("--root", type=Path, default=ROOT)
    template = subparsers.add_parser("template", help="verify ElegantBook provenance hashes")
    template.add_argument("--source", type=Path)
    template.add_argument("--vendored-only", action="store_true")
    args = parser.parse_args()
    if args.check == "prose":
        return run_prose(args.root)
    return run_template(args.source, args.vendored_only)


if __name__ == "__main__":
    raise SystemExit(main())
