"""Replay a completed face-quality audit from CSV and cached raw blur measurements.

Usage: python -B -m app.tools.recalibrate_v2_audit AUDIT_DIR OUTPUT_DIR
No images or models are loaded.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from automatic_scan.keeper_ranker_v2 import Preferences, VERSION, rank_group
from automatic_scan.keeper_ranker_v2_adapter import METRICS, extract_faces, photo_input, technical_feature


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def replay(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tracemalloc.start()
    photos = {r["image_path"]: r for r in read(source / "photo_metrics.csv")}
    faces = defaultdict(list)
    for row in read(source / "face_metrics.csv"):
        faces[row["image_path"]].append(row)
    groups = defaultdict(list)
    for row in read(source / "ranking_decisions.csv"):
        groups[row["group_id"]].append(row)
    original = {(r["group_id"], r["preference_id"]): r
                for r in read(source / "ranking_v1_v2_comparison.csv")}
    connection = sqlite3.connect(f"file:{(source / 'audit_cache.sqlite3').as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        blur = {r["path"]: dict(r) for r in connection.execute(
            "SELECT path, laplacian, sobel, local_contrast, final_score, status FROM blur_scan_cache")}
    finally:
        connection.close()
    face_rows, technical_rows, result_rows = [], [], []
    for path, photo in photos.items():
        raw = blur.get(path)
        calibrated = technical_feature(raw)
        technical_rows.append(dict(image_id=path, **{key: raw.get(key) if raw else ""
            for key in ("laplacian", "sobel", "local_contrast", "final_score", "status")},
            calibrated_score=calibrated.value, calibration_confidence=calibrated.confidence))
        _, diagnostics = extract_faces(faces[path], float(photo["width"] or 0),
                                       float(photo["height"] or 0), rows=True)
        for row, diagnostic in zip(faces[path], diagnostics):
            unavailable = [name for name, prefix in METRICS.items()
                           if not row.get(prefix + "_ranking_score") or
                           not row.get(prefix + "_confidence") or
                           float(row[prefix + "_confidence"]) <= 0]
            face_rows.append(dict(image_id=path, **diagnostic,
                                  unavailable_metrics="|".join(unavailable)))
    both = defaultdict(int)
    for group_id, members in groups.items():
        paths = [r["candidate_image_id"] for r in sorted(members, key=lambda r: int(r["final_position"]))]
        legacy_saturated = all(blur.get(path) and blur[path]["laplacian"] >= 400
                               and blur[path]["sobel"] >= 60 for path in paths)
        for face_on in (True, False):
            for eye_on in (True, False):
                pref_id = f"face_{int(face_on)}_eyes_{int(eye_on)}"
                candidates = tuple(photo_input(path, faces[path], blur.get(path),
                    float(photos[path]["width"] or 0), float(photos[path]["height"] or 0), rows=True)
                    for path in paths)
                ranking = rank_group(candidates, Preferences(face_on, eye_on))
                if ranking[0].face_score is not None and ranking[0].technical_score is not None:
                    both[pref_id] += 1
                old = original.get((group_id, pref_id), {})
                for position, score in enumerate(ranking, 1):
                    result_rows.append(dict(group_id=group_id, preference_id=pref_id,
                        v1_winner=paths[0], original_v2_winner=old.get("v2_winner", ""),
                        corrected_v2_winner=ranking[0].image_id,
                        original_v2_margin=old.get("v2_margin", ""),
                        corrected_v2_margin=ranking[0].margin_to_next,
                        legacy_edges_saturated=legacy_saturated,
                        position=position, **asdict(score)))
    write(output / "face_diagnostics.csv", face_rows)
    write(output / "technical_diagnostics.csv", technical_rows)
    write(output / "ranking_comparison.csv", result_rows)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    summary = dict(version=VERSION, source=str(source), groups=len(groups),
                   both_components=dict(both), runtime_seconds=time.perf_counter() - started,
                   peak_python_allocated_bytes=peak, blur_cache_rows=len(blur),
                   images_decoded=0, models_run=0)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
