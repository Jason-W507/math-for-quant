"""Render only publication pages that contain registered figure IDs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "figures" / "figure-manifest.json"
OUTPUT = ROOT / "tmp" / "figures" / "book-pages"


def page_image(page: fitz.Page, width: int = 430) -> Image.Image:
    scale = width / page.rect.width
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def make_sheet(images: list[tuple[str, Image.Image]], target: Path) -> None:
    columns = 4
    tile_width, tile_height = 470, 660
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(images):
        left = (index % columns) * tile_width + 20
        top = (index // columns) * tile_height + 35
        image.thumbnail((430, 600))
        sheet.paste(image, (left, top))
        draw.text((left, top - 22), label, fill="#2D3748", font=font)
    sheet.save(target)


def main() -> int:
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for volume in ("upper", "lower"):
        ids = {record["id"] for record in records if record["volume"] == volume}
        pdf = ROOT / "output" / "pdf" / f"math-for-quant-{volume}.pdf"
        document = fitz.open(pdf)
        selected: list[tuple[str, Image.Image]] = []
        found: set[str] = set()
        for page_index, page in enumerate(document):
            text = page.get_text()
            page_ids = sorted(figure_id for figure_id in ids if figure_id in text)
            if page_ids:
                found.update(page_ids)
                selected.append((f"p{page_index + 1}: {', '.join(page_ids)}", page_image(page)))
        document.close()
        missing = sorted(ids - found)
        if missing:
            raise ValueError(f"{volume} figure IDs not found in PDF text: {missing}")
        for sheet_index in range(0, len(selected), 16):
            batch = selected[sheet_index:sheet_index + 16]
            target = OUTPUT / f"{volume}-{sheet_index // 16 + 1}.png"
            make_sheet(batch, target)
            print(target.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
