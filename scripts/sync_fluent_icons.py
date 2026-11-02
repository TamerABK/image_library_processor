"""Fetch the small, pinned subset of Microsoft's Fluent System Icons used by Room36.

Run from the repository root. The source is the official upstream repository;
theme-color copies keep every original path intact for reliable Qt rendering.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

COMMIT = "a347af95478f989038cccbf085978865264c048a"  # release 1.1.344
ROOT = Path(__file__).resolve().parents[1] / "assets" / "icons" / "fluent"
# Qt Quick's layer colorization is unreliable with the offscreen renderer used
# by our Windows UI tests. Keep original geometry and pre-tint only theme colors.
PALETTE = {"white": "#FFFFFF", "blue": "#00008E", "midblue": "#4B44E0"}
WARNING_TINTS = {"red": "#D92D20", "yellow": "#B77900"}
# Fluent asset directory -> stem and sizes actually used by the interface.
ICONS = {
    "Arrow Left": ("arrow_left", (16, 20)),
    "Arrow Sort": ("arrow_sort", (20,)),
    "Arrow Download": ("arrow_download", (20, 48)),
    "Arrow Upload": ("arrow_upload", (20,)),
    "Arrow Undo": ("arrow_undo", (20,)),
    "Calendar": ("calendar", (20,)),
    "Chevron Down": ("chevron_down", (16, 20)),
    "Chevron Right": ("chevron_right", (16, 20)),
    "Clock": ("clock", (20,)),
    "Delete": ("delete", (20,)),
    "Dismiss": ("dismiss", (20,)),
    "Edit": ("edit", (20,)),
    "Folder Open": ("folder_open", (20, 24)),
    "Grid": ("grid", (20,)),
    "Home": ("home", (20, 24)),
    "Image": ("image", (20, 24)),
    "Image Multiple": ("image_multiple", (20,)),
    "More Horizontal": ("more_horizontal", (20,)),
    "People": ("people", (20,)),
    "Person": ("person", (20, 24)),
    "Premium": ("premium", (16, 20)),
    "Search": ("search", (20,)),
    "Settings": ("settings", (20, 24)),
    "Star": ("star", (20,)),
    "View Desktop": ("view_desktop", (20,)),
    "Warning": ("warning", (20,)),
}


def fetch(item: tuple[str, str, int, str]) -> str:
    group, stem, size, style = item
    filename = f"{stem}_{size}_{style}.svg"
    path = ROOT / filename
    upstream = f"assets/{group}/SVG/ic_fluent_{filename}"
    url = "https://raw.githubusercontent.com/microsoft/fluentui-system-icons/" + COMMIT + "/" + quote(upstream, safe="/")
    with urlopen(url, timeout=30) as response:
        content = response.read()
    if not content.startswith(b"<svg ") or b"#212121" not in content:
        raise ValueError(f"Unexpected SVG content: {url}")
    path.write_bytes(content)
    for color_name, hex_color in PALETTE.items():
        (ROOT / f"{stem}_{size}_{style}_{color_name}.svg").write_bytes(
            content.replace(b"#212121", hex_color.encode("ascii")))
    if stem == "warning":
        for color_name, hex_color in WARNING_TINTS.items():
            (ROOT / f"{stem}_{size}_{style}_{color_name}.svg").write_bytes(
                content.replace(b"#212121", hex_color.encode("ascii")))
    return filename


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [(group, stem, size, style) for group, (stem, sizes) in ICONS.items()
            for size in sizes for style in ("regular", "filled")]
    with ThreadPoolExecutor(max_workers=8) as workers:
        names = list(workers.map(fetch, jobs))
    print(f"Fetched {len(names)} official Fluent SVGs at {COMMIT}; generated {len(names) * len(PALETTE)} theme tints")


if __name__ == "__main__":
    main()
