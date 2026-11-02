"""Copy license/notice files for Room36's pinned runtime wheels.

Run with the interpreter used for the Windows release build after installing
requirements for the selected execution-provider profile.
"""
from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1] / "third_party_licenses"
RUNTIME = (
    "flatbuffers", "ImageHash", "joblib", "ml-dtypes", "narwhals", "numpy",
    "onnx", "onnxruntime", "onnxruntime-directml", "onnxruntime-gpu",
    "opencv-python", "packaging", "Pillow", "protobuf", "PySide6",
    "PySide6-Addons", "PySide6-Essentials", "shiboken6", "PyWavelets",
    "scikit-learn", "scipy", "Send2Trash", "threadpoolctl",
    "typing_extensions", "rawpy",
)


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    count = 0
    for name in RUNTIME:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        source = Path(dist._path)  # dist-info directory of this exact installed wheel
        license_files = sorted(path for path in source.rglob("*") if path.is_file()
                               and (path.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE"))
                                    or path.name == "ThirdPartyNotices.txt"))
        for path in license_files:
            target = ROOT / source.name / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
    print(f"Copied {count} license/notice files from installed runtime wheels")


if __name__ == "__main__":
    main()
