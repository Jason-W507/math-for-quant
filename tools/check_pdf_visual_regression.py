from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "curriculum" / "visual-regression.json"
DEFAULT_BASELINE = ROOT / "curriculum" / "visual-regression-baseline.json"


def difference_hash(page: fitz.Page, hash_size: int) -> str:
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(1.0, 1.0),
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    samples = memoryview(pixmap.samples)
    values: list[int] = []
    for y in range(hash_size):
        top = y * pixmap.height // hash_size
        bottom = max(top + 1, (y + 1) * pixmap.height // hash_size)
        for x in range(hash_size):
            left = x * pixmap.width // hash_size
            right = max(left + 1, (x + 1) * pixmap.width // hash_size)
            total = 0
            count = 0
            for source_y in range(top, bottom):
                start = source_y * pixmap.width + left
                block = samples[start : start + (right - left)]
                total += sum(block)
                count += len(block)
            values.append(int(total / count < 254.0))
    packed = 0
    for value in values:
        packed = (packed << 1) | value
    return f"{packed:0{hash_size * hash_size // 4}x}"


def resolve_page(document: fitz.Document, selector: dict[str, object]) -> int:
    if "page" in selector:
        index = int(selector["page"]) - 1
        if not 0 <= index < document.page_count:
            raise ValueError(f"page selector is outside document: {selector['page']}")
        return index
    if "bookmark" in selector:
        marker = str(selector["bookmark"])
        matches = [page - 1 for _, title, page in document.get_toc() if title == marker]
        if len(matches) != 1:
            raise ValueError(f"bookmark selector must match exactly one page: {marker!r}")
        return matches[0]
    raise ValueError("visual page selector requires either 'page' or 'bookmark'")


def validate_pdf_structure(
    document: fitz.Document, publication: dict[str, object]
) -> None:
    expected_metadata = publication.get("metadata", {})
    for field in ("title", "author"):
        expected = str(expected_metadata.get(field, ""))
        if expected and document.metadata.get(field) != expected:
            raise ValueError(
                f"{publication['id']}: PDF metadata {field} differs from {expected!r}"
            )
    if not document.get_toc():
        raise ValueError(f"{publication['id']}: PDF has no bookmarks")
    fonts = {
        font
        for page in document
        for font in page.get_fonts(full=True)
    }
    if not fonts or any(font[0] <= 0 for font in fonts):
        raise ValueError(f"{publication['id']}: PDF contains an unembedded font")
    for page_number, page in enumerate(document):
        for link in page.get_links():
            destination = int(link.get("page", -1))
            if destination >= document.page_count:
                raise ValueError(
                    f"{publication['id']}: link on page {page_number + 1} "
                    "targets a page outside the PDF"
                )
            if link.get("kind") == fitz.LINK_URI and not link.get("uri"):
                raise ValueError(
                    f"{publication['id']}: empty external link on page {page_number + 1}"
                )


def observed_hashes(config: dict[str, object], root: Path = ROOT) -> dict[str, str]:
    output: dict[str, str] = {}
    size = int(config["hash_size"])
    for publication in config["publications"]:
        pdf = root / str(publication["pdf"])
        if not pdf.is_file():
            raise ValueError(f"visual regression PDF is missing: {pdf}")
        with fitz.open(pdf) as document:
            if publication.get("metadata"):
                validate_pdf_structure(document, publication)
            for selector in publication["pages"]:
                page_index = resolve_page(document, selector)
                key = f"{publication['id']}:{selector['id']}"
                output[key] = difference_hash(document[page_index], size)
    return output


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record or check key-page PDF visual hashes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--record", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    observed = observed_hashes(config)
    if args.record:
        args.baseline.write_text(
            json.dumps(
                {"schema_version": 1, "hashes": observed},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"visual-baseline=recorded pages={len(observed)}")
        return 0
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["hashes"]
    if set(observed) != set(baseline):
        raise SystemExit("visual regression page inventory differs from baseline")
    limit = int(config["maximum_hamming_distance"])
    failures = {
        key: hamming(value, baseline[key])
        for key, value in observed.items()
        if hamming(value, baseline[key]) > limit
    }
    if failures:
        raise SystemExit(f"visual regression exceeded distance {limit}: {failures}")
    print(f"visual-regression=passed pages={len(observed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
