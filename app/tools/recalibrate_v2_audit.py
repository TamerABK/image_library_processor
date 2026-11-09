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
from dataclasses import asdict, replace
from pathlib import Path

from automatic_scan.keeper_ranker_v2 import Preferences, VERSION, rank_group
from automatic_scan.keeper_ranker_v2 import Feature
from automatic_scan.keeper_ranker_v2_adapter import METRICS, extract_faces, photo_input, technical_feature
from automatic_scan.regional_sharpness_v2 import RegionalCache, evaluate_group, VERSION as REGIONAL_VERSION
from automatic_scan.experimental_quality_v2 import (composition_observation,
    moment_observation, experimental_score, VERSION as EXPERIMENT_VERSION)


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


def replay(source: Path, output: Path, *, regional=False):
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
    phase2a_file = source.parent / "phase8g2a_replay" / "ranking_comparison.csv"
    phase2a = {(r["group_id"], r["preference_id"]): r
               for r in read(phase2a_file) if r.get("position") == "1"} if phase2a_file.exists() else {}
    connection = sqlite3.connect(f"file:{(source / 'audit_cache.sqlite3').as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        blur = {r["path"]: dict(r) for r in connection.execute(
            "SELECT path, laplacian, sobel, local_contrast, final_score, status FROM blur_scan_cache")}
    finally:
        connection.close()
    face_rows, technical_rows, result_rows, regional_rows, variant_rows = [], [], [], [], []
    regional_cache = RegionalCache(output / "regional_cache.sqlite3") if regional else None
    regional_stats = dict(additional_decodes=0, cache_hits=0, decode_reuse=0, runtime_seconds=0.)
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
        regional_data = {}
        if regional_cache is not None:
            regional_data, stats = evaluate_group(paths, regional_cache)
            for key in regional_stats:
                regional_stats[key] += stats.get(key, 0)
            for path, data in regional_data.items():
                regional_rows.append(dict(group_id=group_id, image_id=Path(path).name,
                    **{f"regional_{k}": v for k, v in data["baseline"].items()},
                    **{f"registered_{k}": v for k, v in data["registered"].items()},
                    registration_status=data["registration_status"]))
        legacy_saturated = all(blur.get(path) and blur[path]["laplacian"] >= 400
                               and blur[path]["sobel"] >= 60 for path in paths)
        for face_on in (True, False):
            for eye_on in (True, False):
                pref_id = f"face_{int(face_on)}_eyes_{int(eye_on)}"
                candidates = tuple(photo_input(path, faces[path], blur.get(path),
                    float(photos[path]["width"] or 0), float(photos[path]["height"] or 0),
                    rows=True, include_faces=face_on)
                    for path in paths)
                for method in (("raw_log", "regional", "registered") if regional else ("raw_log",)):
                    selected = candidates
                    effective_method = method
                    if method != "raw_log":
                        proposal = tuple(replace(candidate, technical=Feature(
                            regional_data.get(candidate.image_id, {}).get(
                                "baseline" if method == "regional" else "registered", {}).get("score"),
                            regional_data.get(candidate.image_id, {}).get(
                                "baseline" if method == "regional" else "registered", {}).get("confidence", 0)))
                            for candidate in candidates)
                        if all(candidate.technical.usable() for candidate in proposal):
                            selected = proposal
                        else:
                            effective_method = method + "_fallback_raw_log"
                    ranking = rank_group(selected, Preferences(face_on, eye_on))
                    if method == "raw_log":
                        counts = [len(faces[path]) for path in paths]
                        comparable = len(set(counts)) == 1 and counts[0] > 0
                        observations = {}
                        for path in paths:
                            photo = photos[path]
                            moment = moment_observation()
                            composition = (composition_observation(faces[path],
                                float(photo["width"] or 0), float(photo["height"] or 0))
                                if face_on and comparable else composition_observation([], 0, 0))
                            observations[path] = (moment, composition)
                        for variant, use_m, use_c in (("baseline", False, False),
                                                       ("plus_m", True, False),
                                                       ("plus_c", False, True),
                                                       ("plus_m_c", True, True)):
                            # Face-off stays technical-only for every variant.
                            use_m = use_m and face_on
                            use_c = use_c and face_on
                            group_c = use_c and all(observations[p][1].available for p in paths)
                            scores = []
                            for score in ranking:
                                m, c = observations[score.image_id]
                                value, reasons = experimental_score(score, m, c,
                                    use_moment=use_m, use_composition=use_c and group_c)
                                if use_c and not group_c:
                                    value, reasons = None, ("group_face_count_or_geometry_incomparable",)
                                scores.append((score.image_id, value, reasons, m, c))
                            ordered = sorted(scores, key=lambda item: (item[1] is not None,
                                item[1] if item[1] is not None else 0, item[0]), reverse=True)
                            for position, (path, value, reasons, m, c) in enumerate(ordered, 1):
                                variant_rows.append(dict(group_id=group_id, preference_id=pref_id,
                                    variant=variant, image_id=Path(path).name, position=position,
                                    score=value, winner=Path(ordered[0][0]).name if ordered[0][1] is not None else "",
                                    moment_score=m.score, moment_confidence=m.confidence,
                                    moment_coverage=m.coverage, moment_reasons="|".join(m.reasons),
                                    composition_score=c.score, composition_confidence=c.confidence,
                                    composition_coverage=c.coverage,
                                    composition_reasons="|".join(c.reasons),
                                    unavailable_reasons="|".join(reasons),
                                    experimental_version=EXPERIMENT_VERSION))
                    if method == "raw_log" and ranking[0].face_score is not None and ranking[0].technical_score is not None:
                        both[pref_id] += 1
                    old = original.get((group_id, pref_id), {})
                    earlier = phase2a.get((group_id, pref_id), {})
                    for position, score in enumerate(ranking, 1):
                        result_rows.append(dict(group_id=group_id, preference_id=pref_id,
                            technical_method=effective_method, v1_winner=Path(paths[0]).name,
                            original_v2_winner=Path(old.get("v2_winner", "")).name,
                            phase8g2a_v2_winner=Path(earlier.get("corrected_v2_winner", "")).name,
                            corrected_v2_winner=Path(ranking[0].image_id).name,
                            original_v2_margin=old.get("v2_margin", ""),
                            corrected_v2_margin=ranking[0].margin_to_next,
                            legacy_edges_saturated=legacy_saturated,
                            position=position, **{**asdict(score), "image_id": Path(score.image_id).name}))
    if regional_cache is not None:
        regional_cache.close()
    for row in face_rows + technical_rows:
        row["image_id"] = Path(row["image_id"]).name
    write(output / "face_diagnostics.csv", face_rows)
    write(output / "technical_diagnostics.csv", technical_rows)
    write(output / "ranking_comparison.csv", result_rows)
    write(output / "regional_diagnostics.csv", regional_rows)
    write(output / "experimental_variants.csv", variant_rows)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    summary = dict(version=VERSION, source=str(source), groups=len(groups),
                   both_components=dict(both), runtime_seconds=time.perf_counter() - started,
                   peak_python_allocated_bytes=peak, blur_cache_rows=len(blur),
                   images_decoded=regional_stats["additional_decodes"], models_run=0,
                   regional_version=REGIONAL_VERSION if regional else None,
                   regional_stats=regional_stats)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(Path(sys.argv[1]), Path(sys.argv[2]),
                            regional="--regional" in sys.argv), indent=2))
