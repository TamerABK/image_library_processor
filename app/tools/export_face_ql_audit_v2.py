"""Flatten the existing Phase 8G.2 audit into one candidate-level V1/V2 CSV."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from automatic_scan.keeper_ranker_v2 import FACE_WEIGHTS, PRIORS, Face, Feature, _face_value


FACE_PREFIXES = {
    "sharpness": "image_quality_sharpness", "eyes": "eyes",
    "pose": "pose_metric", "exposure": "image_quality_exposure",
    "contrast": "image_quality_contrast", "detail": "image_quality_detail_availability",
    "detector": "detector_metric", "visible": "visible_face_metric",
    "reliability": "measurement_reliability",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: str | None) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the source audit directory")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    faces = _rows(source / "face_metrics.csv")
    photos = {row["image_path"]: row for row in _rows(source / "photo_metrics.csv")}
    v1 = {(row["group_id"], row["candidate_image_id"]): row
          for row in _rows(source / "ranking_decisions.csv")}
    v2 = _rows(source / "ranking_v2_features.csv")
    comparisons = {(row["group_id"], row["preference_id"]): row
                   for row in _rows(source / "ranking_v1_v2_comparison.csv")}
    blur: dict[str, dict[str, object]] = {}
    cache_db = source / "audit_cache.sqlite3"
    checkpoint_source = manifest.get("checkpoint_source")
    if checkpoint_source:
        original_cache = Path(checkpoint_source).resolve().parent / "audit_cache.sqlite3"
        if original_cache.is_file():
            cache_db = original_cache
    if cache_db.is_file():
        with sqlite3.connect(cache_db.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            blur = {row["path"]: dict(row) for row in connection.execute(
                "SELECT path, laplacian, sobel, local_contrast, lap_norm, sobel_norm, "
                "contrast_norm, final_score, status FROM blur_scan_cache")}
    faces_by_photo: dict[str, list[dict[str, str]]] = {}
    for row in faces:
        faces_by_photo.setdefault(row["image_path"], []).append(row)
    for rows in faces_by_photo.values():
        rows.sort(key=lambda row: int(row["face_index"]))
    max_faces = max((len(faces_by_photo.get(row["image_id"], [])) for row in v2), default=0)
    face_columns = list(faces[0]) if faces else []
    v1_columns = list(next(iter(v1.values()))) if v1 else []
    v2_columns = list(v2[0]) if v2 else []
    photo_columns = list(next(iter(photos.values()))) if photos else []
    comparison_columns = list(next(iter(comparisons.values()))) if comparisons else []
    base = ["group_id", "preference_id", "candidate_image_id", "detected_face_count",
            "v2_primary_face_count", "v2_face_threshold", "v2_primary_rule",
            "v2_score_unit", "v1_rank_order", "v2_rank_order"]
    base += [f"v1_{name}" for name in v1_columns if name not in {"group_id", "candidate_image_id"}]
    base += [f"v2_{name}" for name in v2_columns if name not in {"group_id", "preference_id", "image_id"}]
    base += [f"comparison_{name}" for name in comparison_columns if name not in {"group_id", "preference_id"}]
    base += [f"photo_{name}" for name in photo_columns if name != "image_path"]
    blur_columns = ("laplacian", "sobel", "local_contrast", "lap_norm",
                    "sobel_norm", "contrast_norm", "final_score", "status")
    base += [f"technical_{name}" for name in blur_columns]
    base += [f"weight_{name}" for name in FACE_WEIGHTS]
    base += [f"neutral_prior_{name}" for name in PRIORS]
    for index in range(max_faces):
        base += [f"face_{index:02d}_{name}" for name in face_columns]
        base += [f"face_{index:02d}_v2_geometry_primary",
                 f"face_{index:02d}_v2_prominence_weight",
                 f"face_{index:02d}_v2_effective_score"]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "face_quality_v1_v2_all_values.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=base)
        writer.writeheader()
        for result in v2:
            group, preference, path = result["group_id"], result["preference_id"], result["image_id"]
            old = v1[group, path]
            comparison = comparisons[group, preference]
            photo = photos[path]
            candidate_faces = faces_by_photo.get(path, [])
            largest = max((_number(face.get("geometry_relative_area")) or 0
                           for face in candidate_faces), default=0)
            used = tuple(ast.literal_eval(result["used_face_features"]))
            row = dict(group_id=group, preference_id=preference,
                       candidate_image_id=path, detected_face_count=len(candidate_faces),
                       v2_primary_face_count=0, v2_face_threshold=.55,
                       v2_primary_rule="relative_area >= max(0.008, 0.68 * largest)",
                       v2_score_unit="0-100", v1_rank_order=old["final_position"],
                       v2_rank_order=result["v2_position"])
            row.update({f"v1_{name}": value for name, value in old.items()
                        if name not in {"group_id", "candidate_image_id"}})
            row.update({f"v2_{name}": value for name, value in result.items()
                        if name not in {"group_id", "preference_id", "image_id"}})
            row.update({f"comparison_{name}": value for name, value in comparison.items()
                        if name not in {"group_id", "preference_id"}})
            row.update({f"photo_{name}": value for name, value in photo.items() if name != "image_path"})
            row.update({f"technical_{name}": blur.get(path, {}).get(name, "") for name in blur_columns})
            row.update({f"weight_{name}": value for name, value in FACE_WEIGHTS.items()})
            row.update({f"neutral_prior_{name}": value for name, value in PRIORS.items()})
            primary_count = 0
            for index, face in enumerate(candidate_faces):
                row.update({f"face_{index:02d}_{name}": value for name, value in face.items()})
                area = _number(face.get("geometry_relative_area")) or 0
                primary = area >= max(.008, .68 * largest) and face.get("analysis_missing") in ("0", "", None)
                primary_count += int(primary)
                row[f"face_{index:02d}_v2_geometry_primary"] = int(primary)
                if primary:
                    row[f"face_{index:02d}_v2_prominence_weight"] = max(.25, min(2., area / max(largest, .0001)))
                if primary and used:
                    features = {name: Feature(_number(face.get(prefix + "_ranking_score")),
                                              _number(face.get(prefix + "_confidence")) or 0)
                                for name, prefix in FACE_PREFIXES.items()}
                    if all(features[name].usable() for name in used):
                        row[f"face_{index:02d}_v2_effective_score"] = _face_value(
                            Face(features, prominence=1), used)
            row["v2_primary_face_count"] = primary_count
            writer.writerow(row)
    audit_files = ("face_metrics.csv", "photo_metrics.csv", "ranking_decisions.csv",
                   "ranking_v2_features.csv", "ranking_v1_v2_comparison.csv", "manifest.json")
    metadata = {"source_audit": str(source), "dataset_signature": manifest.get("dataset_signature"),
                "source_git_commit": manifest.get("git_commit"),
                "export_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "rows": len(v2), "columns": len(base), "max_faces_per_candidate": max_faces,
                "source_sha256": {name: _digest(source / name) for name in audit_files},
                "blur_cache_sha256": _digest(cache_db) if cache_db.is_file() else None,
                "csv_sha256": _digest(destination)}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.source, args.output))


if __name__ == "__main__":
    main()
