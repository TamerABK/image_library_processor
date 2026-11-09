"""Build local review sheets from a completed Phase 8G.3 audit replay."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def create(audit: Path, photos: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (audit / "experimental_variants.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(dict)
    for row in rows:
        if row["preference_id"] == "face_1_eyes_1" and row["variant"] in ("baseline", "plus_c"):
            groups[row["group_id"]].setdefault(row["image_id"], {})[row["variant"]] = row
    paths = {p.name: p for p in photos.rglob("*") if p.is_file()}
    for group_id, images in groups.items():
        ordered = sorted(images)
        sheet = Image.new("RGB", (480 * len(ordered), 430), "white")
        draw = ImageDraw.Draw(sheet)
        for index, name in enumerate(ordered):
            x = index * 480
            path = paths.get(name)
            if path:
                try:
                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source)
                        image.thumbnail((460, 285))
                        sheet.paste(image.convert("RGB"), (x + (480-image.width)//2, 5))
                except OSError:
                    draw.text((x+10, 30), "Image unavailable", fill="black")
            baseline = images[name].get("baseline", {})
            composition = images[name].get("plus_c", {})
            labels = (name,
                      f"V1: {baseline.get('v1_winner', '')}  V2: {baseline.get('baseline_v2_winner', '')}",
                      f"F: {baseline.get('face_score', '')[:8]}  T: {baseline.get('technical_score', '')[:8]}",
                      f"M: unavailable  C: {composition.get('composition_score', '')[:8]}",
                      f"Baseline: {baseline.get('score', '')[:8]}  C variant: {composition.get('score', '')[:8]}",
                      f"C status: {composition.get('unavailable_reasons', '') or 'experimental'}")
            for line, label in enumerate(labels):
                draw.text((x+8, 295+line*20), label[:72], fill="black")
        sheet.save(output / f"{group_id}.jpg", quality=85)
    return len(groups)


if __name__ == "__main__":
    print(create(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
