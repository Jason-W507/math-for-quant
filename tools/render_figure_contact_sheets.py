"""Render cached figure PDFs into compact QA contact sheets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "figures" / "figure-manifest.json"
OUTPUT = ROOT / "tmp" / "figures"


def render_pdf(path: Path, width: int = 640) -> Image.Image:
    document = fitz.open(path)
    page = document[0]
    scale = width / page.rect.width
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    document.close()
    return image


def main() -> int:
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    records = [record for record in records if record["cached_asset"]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for volume in ("upper", "lower"):
        selected = [record for record in records if record["volume"] == volume]
        columns = 3
        tile_width, tile_height = 680, 430
        rows = math.ceil(len(selected) / columns)
        sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(selected):
            image = render_pdf(ROOT / record["cached_asset"])
            image.thumbnail((640, 360))
            left = (index % columns) * tile_width + 20
            top = (index // columns) * tile_height + 35
            sheet.paste(image, (left, top))
            draw.text((left, 10 + (index // columns) * tile_height), record["id"], fill="#2D3748", font=font)
        target = OUTPUT / f"{volume}-figures-contact-sheet.png"
        sheet.save(target)
        print(target.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
