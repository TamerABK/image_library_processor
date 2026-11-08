"""Bounded, experimental image-level regional sharpness for confirmed duplicates.

No face inputs, inference, or blur-classification changes. Versioned separately
from the shadow ranker; callers may retain raw-log-v1 when evidence is weak.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from image_loader import default_image_loader

VERSION = "regional-ratio-v3-mean-gradient"
MAX_DIMENSION = 1024
MAX_GROUP_IMAGES = 32
GRID = 6


@dataclass(frozen=True)
class RegionalResult:
    score: float | None
    confidence: float
    valid_patches: int
    status: str


def _signature(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(json.dumps((str(path.resolve()), stat.st_size,
                                      stat.st_mtime_ns, VERSION, MAX_DIMENSION)).encode()).hexdigest()


class RegionalCache:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS regional_v2 "
                                "(key TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def get(self, key):
        row = self.connection.execute("SELECT payload FROM regional_v2 WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, result):
        self.connection.execute("INSERT OR REPLACE INTO regional_v2 VALUES (?,?)",
                                (key, json.dumps(result)))
        self.connection.commit()

    def close(self):
        self.connection.close()


def patch_values(gray: np.ndarray) -> list[float]:
    """Fine/coarse edge ratio with low-contrast and noise rejection."""
    if gray is None or gray.ndim != 2 or min(gray.shape) < 96:
        return []
    values = []
    h, w = gray.shape
    for y in range(GRID):
        for x in range(GRID):
            patch = gray[y*h//GRID:(y+1)*h//GRID, x*w//GRID:(x+1)*w//GRID]
            if patch.std() < 8:
                continue
            smoothed = cv2.GaussianBlur(patch, (0, 0), .6)
            coarse = cv2.GaussianBlur(patch, (0, 0), 1.6)
            def edge(image):
                gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
                gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
                return cv2.magnitude(gx, gy)
            fine, broad = edge(smoothed), edge(coarse)
            mask = broad >= max(8., float(np.percentile(broad, 80)))
            if mask.sum() < 32:
                continue
            ratio = float(np.mean(fine) / (np.mean(broad) + 1.))
            residual = np.abs(patch.astype(np.float32) - smoothed.astype(np.float32))
            flat = broad < np.percentile(broad, 25)
            noise = float(np.median(residual[flat])) if flat.any() else 0.
            values.append(max(0., min(1., (ratio - 1.) / (ratio + 1.) - .001 * noise)))
    return values


def summarize(values: list[float]) -> RegionalResult:
    if len(values) < 8:
        return RegionalResult(None, 0., len(values), "insufficient_regions")
    ordered = sorted(values)
    trimmed = ordered[len(ordered)//10:len(ordered)-len(ordered)//10]
    return RegionalResult(float(np.median(trimmed)), min(1., len(values)/24.),
                          len(values), "measured")


def register(reference: np.ndarray, candidate: np.ndarray):
    """Translation only; reject weak or implausibly large shifts."""
    if abs(reference.shape[1] / reference.shape[0] - candidate.shape[1] / candidate.shape[0]) > .02:
        return None, 0., "aspect_mismatch"
    size = (256, 256)
    a = cv2.resize(reference, size).astype(np.float32)
    b = cv2.resize(candidate, size).astype(np.float32)
    shift, response = cv2.phaseCorrelate(a, b)
    if response < .3 or max(abs(shift[0]), abs(shift[1])) > 30:
        return None, float(response), "registration_rejected"
    dx = shift[0] * candidate.shape[1] / 256
    dy = shift[1] * candidate.shape[0] / 256
    if reference.shape != candidate.shape:
        return None, float(response), "dimension_mismatch"
    # Integer translation avoids softening only the candidate by interpolation.
    aligned = np.roll(candidate, (-round(dy), -round(dx)), axis=(0, 1))
    h, w = aligned.shape
    return aligned[h//8:-h//8, w//8:-w//8], float(response), "registered"


def evaluate_group(paths, cache: RegionalCache, *, loader=default_image_loader,
                   cancel=None, registered=True):
    """Process one confirmed group; at most 32 bounded grayscale buffers."""
    unique = tuple(dict.fromkeys(Path(p) for p in paths))
    reused = len(paths) - len(unique)
    if len(unique) > MAX_GROUP_IMAGES:
        return {}, dict(status="group_limit", additional_decodes=0, cache_hits=0)
    if cancel is not None and cancel():
        return {}, dict(status="cancelled", additional_decodes=0, cache_hits=0)
    started = time.perf_counter()
    try:
        group_key = "group:" + hashlib.sha256(json.dumps(
            [_signature(path) for path in unique]).encode()).hexdigest()
    except OSError:
        return {}, dict(status="source_missing", additional_decodes=0, cache_hits=0)
    if registered:
        stored_group = cache.get(group_key)
        if stored_group is not None:
            return stored_group, dict(status="cached", additional_decodes=0,
                                      cache_hits=len(unique), decode_reuse=reused,
                                      runtime_seconds=time.perf_counter()-started)
    images, baseline, hits = {}, {}, 0
    for path in unique:
        if cancel is not None and cancel():
            return {}, dict(status="cancelled", additional_decodes=len(images), cache_hits=hits)
        try:
            key = _signature(path)
            stored = cache.get(key)
            if stored is not None:
                baseline[str(path)] = RegionalResult(**stored)
                hits += 1
            # Registered comparison needs pixels; baseline-only cache hits do not.
            if registered or stored is None:
                image = loader.load_grayscale(path, max_dimension=MAX_DIMENSION)
                if image is None:
                    baseline[str(path)] = RegionalResult(None, 0., 0, "decode_failed")
                    continue
                images[str(path)] = image
                if stored is None:
                    baseline[str(path)] = summarize(patch_values(image))
                    cache.put(key, baseline[str(path)].__dict__)
        except (OSError, ValueError, cv2.error):
            baseline[str(path)] = RegionalResult(None, 0., 0, "decode_failed")
    registered_results = {}
    statuses = {}
    if registered and images:
        reference_path = next(iter(images))
        reference = images[reference_path]
        h, w = reference.shape
        registered_results[reference_path] = summarize(patch_values(
            reference[h//8:-h//8, w//8:-w//8]))
        statuses[reference_path] = "reference"
        for path, image in images.items():
            if path == reference_path:
                continue
            aligned, response, status = register(reference, image)
            statuses[path] = status
            registered_results[path] = (summarize(patch_values(aligned)) if aligned is not None
                                        else RegionalResult(None, 0., 0, status))
    count = len(images)
    images.clear()
    results = {path: dict(baseline=baseline.get(path, RegionalResult(None, 0., 0, "missing")).__dict__,
                       registered=registered_results.get(path, RegionalResult(None, 0., 0, "not_run")).__dict__,
                       registration_status=statuses.get(path, "not_run")) for path in map(str, unique)}
    if registered and all(item["baseline"]["score"] is not None for item in results.values()):
        cache.put(group_key, results)
    return results, {
            "status": "complete", "additional_decodes": count, "cache_hits": hits,
            "decode_reuse": reused, "runtime_seconds": time.perf_counter()-started}
