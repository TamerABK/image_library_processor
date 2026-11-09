"""Audit-only NSC subject measurements; no production ranking changes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil

from automatic_scan.subject_quality_experiment import (VERSION, register, match_faces,
    face_region, region_metrics, detection_space_size)
from face_processing.cache import ImageFaceAnalysisCache
from image_loader import ImageLoader


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, sorted({key for row in records for key in row}))
        writer.writeheader()
        writer.writerows(records)


def run(audit, output, max_dimension=1024):
    output.mkdir(parents=True, exist_ok=True)
    groups = [row for row in read(audit / "ranking_v1_v2_comparison.csv")
              if row["preference_id"] == "face_1_eyes_1"]
    photos = {row["image_path"]: row for row in read(audit / "photo_metrics.csv")}
    scores = {row["image_id"]: row for row in read(audit / "ranking_v2_features.csv")
              if row["preference_id"] == "face_1_eyes_1"}
    face_cache = ImageFaceAnalysisCache(audit / "audit_cache.sqlite3")
    loader = ImageLoader(max_decode_dimension=max_dimension)
    per_photo, per_face, matches, scenes, rankings, runtime, ablations = [], [], [], [], [], [], []
    images = {}
    face_rows = {}
    controlled_rows = []
    peak = psutil.Process().memory_info().rss
    start = time.perf_counter()
    for group in groups:
        paths = group["candidate_ids"].split("|")
        group_started = time.perf_counter()
        for name in paths:
            if name in images:
                continue
            path = Path(name)
            try:
                stat = path.stat()
                cached = face_cache.get(path, stat.st_size, stat.st_mtime_ns)
                photo = photos[name]
                original = (int(photo["width"]), int(photo["height"]))
                source_size = detection_space_size(*original)
                image = loader.load_for_scan(path, max_dimension=max_dimension)
                if image is None or cached is None:
                    raise ValueError("image_or_face_cache_unavailable")
                images[name] = image
                face_rows[name] = []
                for index, face in enumerate(cached):
                    roi, bounds = face_region(image, face.bbox, source_size)
                    metric = region_metrics(roi)
                    if not controlled_rows and min(roi.shape[:2]) >= 64:
                        rng = np.random.default_rng(3604)
                        motion_kernel = np.zeros((9, 9), np.float32)
                        motion_kernel[4, :] = 1/9
                        altered = {
                            "baseline": roi,
                            "gaussian_blur": cv2.GaussianBlur(roi, (9, 9), 2),
                            "motion_blur": cv2.filter2D(roi, -1, motion_kernel),
                            "chroma_noise": np.clip(roi.astype(np.int16) + rng.normal(0, 12, roi.shape), 0, 255).astype(np.uint8),
                            "overexposure": np.clip(roi.astype(np.int16)+60, 0, 255).astype(np.uint8),
                            "underexposure": np.clip(roi.astype(np.int16)-60, 0, 255).astype(np.uint8),
                            "occlusion": roi.copy(),
                        }
                        altered["occlusion"][:roi.shape[0]//3, :] = 0
                        for kind, sample in altered.items():
                            result = region_metrics(sample)
                            controlled_rows.append(dict(source_image=path.name, face_index=index,
                                perturbation=kind, laplacian=result.get("laplacian"),
                                tenengrad=result.get("tenengrad"),
                                anisotropy=result.get("anisotropy"),
                                bright_clip_bgr=json.dumps(result.get("bright_clip_bgr")),
                                median_luminance=result.get("median_luminance")))
                    row = dict(image_id=path.name, face_index=index, source_path=name,
                               x=face.bbox[0], y=face.bbox[1], w=face.bbox[2], h=face.bbox[3],
                               decoded_box=list(bounds), **metric)
                    per_face.append(row)
                    face_rows[name].append(row)
                per_photo.append(dict(image_id=path.name, source_path=name,
                                      face_count=len(cached), decoded_width=image.shape[1],
                                      decoded_height=image.shape[0], v1_winner=int(name==group["v1_winner"]),
                                      v2_score=scores.get(name, {}).get("score", ""),
                                      status="measured"))
            except (OSError, ValueError, KeyError, cv2.error) as exc:
                per_photo.append(dict(image_id=path.name, source_path=name,
                                      face_count=0, status="unavailable:" + type(exc).__name__,
                                      reason=str(exc)[:120]))
            peak = max(peak, psutil.Process().memory_info().rss)
        reference = paths[0]
        group_links = 0
        if reference in images:
            for name in paths[1:]:
                if name not in images:
                    continue
                registration = register(images[reference], images[name])
                links, state = match_faces(face_rows[reference], face_rows[name], registration,
                                           images[reference].shape, images[name].shape)
                group_links += len(links)
                scenes.append(dict(group_id=group["group_id"], image_a=Path(reference).name,
                                   image_b=Path(name).name, registration=registration.status,
                                   inliers=registration.inliers, inlier_ratio=registration.ratio,
                                   reprojection_error=registration.error,
                                   correspondence_status=state))
                for link in links:
                    left, right = face_rows[reference][link["a"]], face_rows[name][link["b"]]
                    matches.append(dict(group_id=group["group_id"], image_a=Path(reference).name,
                        image_b=Path(name).name, face_a=link["a"], face_b=link["b"],
                        box_a=json.dumps(left["decoded_box"]), box_b=json.dumps(right["decoded_box"]),
                        laplacian_a=left.get("laplacian"), laplacian_b=right.get("laplacian"),
                        laplacian_difference=(right["laplacian"]-left["laplacian"]
                                              if left.get("laplacian") is not None and right.get("laplacian") is not None else ""),
                        **{key: value for key, value in link.items() if key not in ("a", "b")}))
        rankings.append(dict(group_id=group["group_id"], v1_winner=Path(group["v1_winner"]).name,
                             v2_winner=Path(group["v2_winner"]).name, tentative_face_links=group_links,
                             experimental_winner="", status="inconclusive_no_verified_identity_or_human_labels",
                             baseline_margin=group["v2_margin"]))
        for variant, component in (("A", "production_V1"), ("B", "default_shadow_V2"),
                                   ("C", "face_local_color"), ("D", "detail_blur"),
                                   ("E", "matched_faces"), ("F", "group_completeness"),
                                   ("G", "matched_moment"), ("H", "stable_scene"),
                                   ("I", "combined")):
            baseline = group["v1_winner"] if variant == "A" else group["v2_winner"] if variant == "B" else ""
            ablations.append(dict(group_id=group["group_id"], variant=variant,
                component=component, winner=Path(baseline).name if baseline else "",
                status="observed_baseline" if baseline else "diagnostic_only_insufficient_validation",
                tentative_links=group_links if variant in ("E", "F", "G", "I") else ""))
        runtime.append(dict(group_id=group["group_id"], seconds=time.perf_counter()-group_started,
                            decoded_candidates=sum(name in images for name in paths)))
        for name in paths:
            images.pop(name, None)
            face_rows.pop(name, None)
    for filename, rows in (("photo_metrics.csv", per_photo), ("face_metrics.csv", per_face),
                           ("matched_faces.csv", matches), ("scene_regions.csv", scenes),
                           ("ranking_comparison.csv", rankings), ("group_runtime.csv", runtime),
                           ("controlled_perturbations.csv", controlled_rows),
                           ("ablation_results.csv", ablations)):
        write(output / filename, rows)
    signature = hashlib.sha256(json.dumps([(p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
        for g in groups for p in g["candidate_ids"].split("|")]).encode()).hexdigest()
    summary = dict(version=VERSION, source_signature=signature, groups=len(groups),
        candidates=len(per_photo), tentative_links=len(matches),
        verified_identity_links=0, additional_decodes=sum(r["decoded_candidates"] for r in runtime),
        elapsed_seconds=time.perf_counter()-start, peak_process_rss=peak,
        experimental_winner_changes=0, winner_changes_evaluated=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write(output / "human_labels_template.csv", [
        dict(group_id=g["group_id"], technical_preference="", moment_preference="",
             composition_preference="", overall_keeper="", confidence="") for g in groups])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("audit_results/nsc_fresh"))
    parser.add_argument("--output", type=Path, default=Path("audit_results/phase8g4_subject_quality"))
    parser.add_argument("--max-dimension", type=int, default=1024)
    args = parser.parse_args()
    if not 256 <= args.max_dimension <= 2048:
        parser.error("max-dimension must be 256..2048")
    print(json.dumps(run(args.audit, args.output, args.max_dimension), indent=2))


if __name__ == "__main__":
    main()
