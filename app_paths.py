from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))

    return Path(__file__).resolve().parent


def model_path(filename: str) -> Path:
    return resource_root() / "onnx_models" / filename


def app_data_root() -> Path:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parent

    root.mkdir(parents=True, exist_ok=True)
    return root


def app_data_path(filename: str) -> Path:
    return app_data_root() / filename
