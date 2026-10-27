"""Read-only JPEG I/O/decode experiment; production ImageLoader is unchanged.

Run with a real image directory and an output directory outside it. The first
pass has unknown OS-cache state; subsequent passes are labeled repeat-read.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from image_loader import ImageLoader
from .benchmark import _select_paths
from .profiling_resources import ResourceSampler


def _measure(path: Path, flag: int, *, check: bool) -> dict:
    steps = {}
    start = time.perf_counter_ns()
    with path.open("rb") as handle:
        compressed = handle.read()
    steps["read_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    image = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), flag)
    steps["imdecode_ns"] = time.perf_counter_ns() - start
    if image is None:
        raise ValueError("OpenCV could not decode image")
    start = time.perf_counter_ns()
    copied = image.copy()
    steps["array_copy_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    largest = max(image.shape[:2])
    if largest > 2048:
        scale = 2048 / largest
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        steps["resize_ns"] = time.perf_counter_ns() - start
    else:
        resized = image
        steps["resize_ns"] = 0
    start = time.perf_counter_ns()
    cv2.cvtColor(resized, cv2.COLOR_BGR2RGB if resized.ndim == 3 else cv2.COLOR_GRAY2BGR)
    steps["color_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    height, width = image.shape[:2]
    # Representative allocation/crop only; no face detection or actual face box.
    crop = image[height // 4:3 * height // 4, width // 4:3 * width // 4].copy()
    steps["synthetic_face_crop_ns"] = time.perf_counter_ns() - start
    steps["decoded_bytes"] = image.nbytes
    steps["compressed_bytes"] = len(compressed)
    steps["width"] = width
    steps["height"] = height
    if check:
        start = time.perf_counter_ns()
        original = cv2.imread(str(path), flag)
        steps["imread_ns"] = time.perf_counter_ns() - start
        steps["equivalent"] = bool(original is not None and original.shape == image.shape
                                   and np.array_equal(original, image))
        del original
    del copied, crop
    return steps


def _summarize(records: list[dict], wall: float, resources: dict, workers: int) -> dict:
    keys = ("read_ns", "imdecode_ns", "array_copy_ns", "resize_ns", "color_ns",
            "synthetic_face_crop_ns", "imread_ns")
    timings = {}
    for key in keys:
        values = [r[key] / 1e6 for r in records if key in r]
        if values:
            timings[key.removesuffix("_ns") + "_ms"] = {
                "sum": sum(values), "median": statistics.median(values),
                "p95": sorted(values)[int(.95 * (len(values) - 1))], "mean": statistics.mean(values)}
    return {"workers": workers, "images": len(records), "wall_seconds": wall,
            "images_per_second": len(records) / wall if wall else None,
            "timings": timings,
            "compressed_bytes": sum(r["compressed_bytes"] for r in records),
            "decoded_bytes": sum(r["decoded_bytes"] for r in records),
            "dimensions": {"min_pixels": min((r["width"] * r["height"] for r in records), default=None),
                           "median_pixels": statistics.median((r["width"] * r["height"] for r in records)) if records else None,
                           "max_pixels": max((r["width"] * r["height"] for r in records), default=None)},
            "equivalence_checked": sum("equivalent" in r for r in records),
            "equivalence_failures": sum(r.get("equivalent") is False for r in records),
            "resources": resources}


def run_once(paths: list[Path], *, workers: int, check: bool, interval: float) -> dict:
    loader = ImageLoader()
    flags = []
    for path in paths:
        metadata = loader.read_metadata(path)
        flags.append(loader._raster_read_flag(path.suffix.lower(), metadata,
            max_dimension=loader.max_decode_dimension, grayscale=False)
            if metadata else cv2.IMREAD_COLOR)
    sampler = ResourceSampler(dataset=str(paths[0].parent), interval=interval)
    sampler.start()
    started = time.perf_counter_ns()
    try:
        if workers == 1:
            records = [_measure(path, flag, check=check) for path, flag in zip(paths, flags)]
        else:
            records = []
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="room36-decode-diag") as pool:
                for offset in range(0, len(paths), workers * 2):
                    futures = [pool.submit(_measure, path, flag, check=check)
                               for path, flag in zip(paths[offset:offset + workers * 2],
                                                     flags[offset:offset + workers * 2])]
                    records.extend(f.result() for f in futures)
    finally:
        wall = (time.perf_counter_ns() - started) / 1e9
        sampler.stop()
    return _summarize(records, wall, sampler.report(), workers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--sample-interval", type=float, default=.25)
    args = parser.parse_args(argv)
    dataset = args.dataset.resolve(strict=True)
    output = args.out.resolve()
    if output == dataset or dataset in output.parents:
        parser.error("output must be outside the dataset")
    if args.size < 1 or args.repeat < 1 or not args.workers or min(args.workers) < 1:
        parser.error("size, repeat and workers must be positive")
    paths = [p for p in _select_paths(dataset, args.size) if p.suffix.lower() in {".jpg", ".jpeg"}]
    if not paths:
        parser.error("no JPEG photos selected")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for repeat in range(args.repeat):
        for workers in args.workers:
            result = run_once(paths, workers=workers, check=False, interval=args.sample_interval)
            result["cache_state"] = "initial-unknown" if repeat == 0 else "repeat-read-likely-warm"
            result["repeat"] = repeat + 1
            results.append(result)
            print(f"pass {repeat+1} workers {workers}: {result['wall_seconds']:.3f}s")
    # Equivalence is a separate final pass, so it cannot warm the first
    # measured read or double one run's JPEG work.
    verify_loader = ImageLoader()
    verification = []
    for path in paths:
        metadata = verify_loader.read_metadata(path)
        flag = verify_loader._raster_read_flag(path.suffix.lower(), metadata,
            max_dimension=verify_loader.max_decode_dimension, grayscale=False) if metadata else cv2.IMREAD_COLOR
        verification.append(_measure(path, flag, check=True))
    equivalence_failures = sum(row["equivalent"] is False for row in verification)
    report = {"dataset_photo_count": len(paths), "source_paths_omitted": True,
              "equivalence_checked": len(verification),
              "equivalence_failures": equivalence_failures,
              "note": "PhysicalDisk counters are device-wide; process read bytes are logical and may hit OS cache. "
                      "Synthetic crop is an allocation diagnostic, not a detected face crop.",
              "runs": results}
    (output / "decode-diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Room36 JPEG read/decode diagnostic", "",
             report["note"], "",
             f"Separate post-timing pixel-equivalence check: {len(verification)} JPEGs, "
             f"{equivalence_failures} failures.", "",
             "| Pass | Workers | Cache state | Wall s | Images/s | Read ms | Decode ms | Copy ms |",
             "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in results:
        t = row["timings"]
        lines.append(f"| {row['repeat']} | {row['workers']} | {row['cache_state']} | "
            f"{row['wall_seconds']:.3f} | {row['images_per_second']:.2f} | "
            f"{t['read_ms']['sum']:.1f} | {t['imdecode_ms']['sum']:.1f} | "
            f"{t['array_copy_ms']['sum']:.1f} |")
    (output / "decode-diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if equivalence_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
