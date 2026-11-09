"""Download pinned OpenCV Zoo FER checkpoints with LFS SHA-256 checks."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

UPSTREAM_COMMIT = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/" + UPSTREAM_COMMIT + "/models/facial_expression_recognition/"
MODELS = {
    "fp32": ("facial_expression_recognition_mobilefacenet_2022july.onnx", 4791892,
             "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9"),
    "int8": ("facial_expression_recognition_mobilefacenet_2022july_int8.onnx", 1364007,
             "f0d7093aff10e2638c734c5f18a6a7eabd2b9239b20bdb9b8090865a6f69a1ed"),
    "int8bq": ("facial_expression_recognition_mobilefacenet_2022july_int8bq.onnx", 1376702,
               "0c3eaf9d0a7d442c0aa3beb3234243e1cdff9ad8871fb3cec346e90874caf57d"),
}


def verify(path: Path, size: int, digest: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest() == digest


def acquire(directory: Path, variants=("fp32", "int8", "int8bq")):
    directory.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        name, size, digest = MODELS[variant]
        target = directory / name
        if verify(target, size, digest):
            continue
        temporary = directory / (name + ".download")
        try:
            with urlopen(BASE + name, timeout=120) as source, temporary.open("wb") as sink:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(block)
            if not verify(temporary, size, digest):
                raise ValueError(f"Invalid download: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return {variant: directory / MODELS[variant][0] for variant in variants}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("audit_results/expression_benchmark/models"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    args = parser.parse_args()
    for variant, path in acquire(args.output, args.models).items():
        print(variant, path)
