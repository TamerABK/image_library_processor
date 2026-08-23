from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))

    return Path(__file__).resolve().parent


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def qml_root() -> Path:
    return resource_path("qml")


def qml_path(*parts: str) -> Path:
    return qml_root().joinpath(*parts)


def assets_root() -> Path:
    return resource_path("assets")


def asset_path(*parts: str) -> Path:
    return assets_root().joinpath(*parts)


def model_path(filename: str) -> Path:
    return resource_path("onnx_models", filename)


def app_data_root() -> Path:
    override = os.environ.get("IMAGE_DEDUPLICATOR_APP_DATA_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parent

    root.mkdir(parents=True, exist_ok=True)
    return root


def app_data_path(filename: str) -> Path:
    return app_data_root() / filename
