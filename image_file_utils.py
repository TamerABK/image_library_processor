from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from image_loader import default_image_loader


def normalize_extensions(
    extensions: Iterable[str],
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()

    for extension in extensions:
        value = extension.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    return tuple(normalized)


def resolve_selected_extensions(
    supported_extensions: Iterable[str],
    selected_extensions: Iterable[str] | None = None,
) -> tuple[str, ...]:
    supported = normalize_extensions(supported_extensions)
    if selected_extensions is None:
        return supported

    selected = normalize_extensions(selected_extensions)
    filtered = tuple(
        extension
        for extension in selected
        if extension in supported
    )

    return filtered or supported


def find_supported_files(
    folder: str | Path,
    supported_extensions: Iterable[str],
    selected_extensions: Iterable[str] | None = None,
    orientation_filter: str | None = None,
) -> list[Path]:
    root = Path(folder)
    allowed = set(
        resolve_selected_extensions(
            supported_extensions,
            selected_extensions,
        )
    )

    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed
        and matches_orientation(path, orientation_filter)
    ]


def matches_orientation(
    path: Path,
    orientation_filter: str | None,
) -> bool:
    if orientation_filter not in {"landscape", "portrait"}:
        return True

    metadata = default_image_loader.read_metadata(path)
    if metadata is None:
        return False
    width, height = metadata.width, metadata.height

    if orientation_filter == "landscape":
        return width > height
    return height > width


def discover_supported_extensions(
    folder: str | Path,
    supported_extensions: Iterable[str],
) -> list[str]:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []

    supported = set(normalize_extensions(supported_extensions))
    found = {
        path.suffix.lower()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    }
    return sorted(found)
