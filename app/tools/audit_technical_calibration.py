"""Isolated Phase 8G.5 technical calibration on confirmed NSC duplicates."""
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

from automatic_scan.technical_quality_v2_calibration import VERSION, measure_technical
from automatic_scan.color_integrity_experiment import measure_color
from automatic_scan.keeper_ranker_v2_adapter import technical_feature, extract_faces
from automatic_scan.subject_quality_experiment import detection_space_size, face_region
from blur_detector.blur_detector import BlurDetector
from face_processing.cache import ImageFaceAnalysisCache
from image_loader import ImageLoader


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, sorted({k for row in records for k in row}))
        writer.writeheader()
        writer.writerows(records)


def cache_key(path, maximum, face_cache_revision=""):
    stat = path.stat()
    return hashlib.sha256(json.dumps((str(path.resolve()), stat.st_size, stat.st_mtime_ns,
        maximum, VERSION, "ImageLoader-BGR8-v1", "face-roi-inset-v1",
        face_cache_revision)).encode()).hexdigest()


def controlled(image, detector):
    rng = np.random.default_rng(3605)
    variants = {"baseline": image}
    for sigma in (1, 2, 3):
        variants[f"gaussian_{sigma}"] = cv2.GaussianBlur(image, (0, 0), sigma)
    for strength in (5, 12, 25):
        variants[f"luma_noise_{strength}"] = np.clip(
            image.astype(np.int16)+rng.normal(0, strength, image.shape[:2])[:, :, None], 0, 255).astype(np.uint8)
        variants[f"chroma_noise_{strength}"] = np.clip(
            image.astype(np.int16)+rng.normal(0, strength, image.shape), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    for strength in (5, 12):
        variants[f"blur_plus_noise_{strength}"] = np.clip(
            blurred.astype(np.int16)+rng.normal(0, strength, image.shape), 0, 255).astype(np.uint8)
    for gain in (.5, 1, 2):
        variants[f"unsharp_{gain}"] = cv2.addWeighted(image, 1+gain, cv2.GaussianBlur(image, (0, 0), 1.2), -gain, 0)
    for quality in (90, 60, 25):
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            variants[f"jpeg_{quality}"] = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    records = []
    for name, altered in variants.items():
        result = measure_technical(altered)
        legacy = detector.detect_image(altered)
        old = technical_feature(legacy)
        records.append(dict(variant=name, baseline_raw_log=old.value,
                            baseline_laplacian=legacy.laplacian,
                            baseline_sobel=legacy.sobel,
                            persistence=result.edge_persistence,
                            detail=result.detail_score,
                            noise_aware_detail=result.noise_aware_detail,
                            luminance_noise_floor=result.noise_floor,
                            chroma_noise=result.chroma_noise,
                            chroma_coverage=result.chroma_coverage,
                            candidate_score=result.score, status=result.status))
    return records


def run(audit, output, maximum=1024):
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"
    cache_dir.mkdir(exist_ok=True)
    groups = [r for r in read(audit/"ranking_v1_v2_comparison.csv")
              if r["preference_id"] == "face_1_eyes_1"]
    photos = {r["image_path"]: r for r in read(audit/"photo_metrics.csv")}
    face_rows = {}
    for row in read(audit/"face_metrics.csv"):
        face_rows.setdefault(row["image_path"], []).append(row)
    face_cache = ImageFaceAnalysisCache(audit/"audit_cache.sqlite3")
    # Face measurements and prominence can change without the source JPEG changing.
    face_revision = ":".join(str(path.stat().st_mtime_ns) for path in
        (audit/"audit_cache.sqlite3", audit/"face_metrics.csv"))
    loader = ImageLoader(max_decode_dimension=maximum)
    detector = BlurDetector(cache_path=output/"experimental_blur.sqlite3")
    per_photo, per_face, comparisons, timings = [], [], [], []
    count_hits = count_decodes = 0
    start = time.perf_counter()
    peak = psutil.Process().memory_info().rss
    control_rows = []
    for group in groups:
        members = group["candidate_ids"].split("|")
        group_metrics = []
        for name in members:
            path = Path(name)
            t0 = time.perf_counter()
            stages = {"decode_seconds": 0., "whole_seconds": 0.,
                      "color_seconds": 0., "faces_seconds": 0.}
            try:
                key = cache_key(path, maximum, face_revision)
                cache_file = cache_dir/(key+".json")
                was_hit = cache_file.is_file()
                if was_hit:
                    payload = json.loads(cache_file.read_text(encoding="utf-8"))
                    count_hits += 1
                else:
                    stat = path.stat()
                    faces = face_cache.get(path, stat.st_size, stat.st_mtime_ns)
                    tick = time.perf_counter()
                    image = loader.load_for_scan(path, max_dimension=maximum)
                    stages["decode_seconds"] = time.perf_counter()-tick
                    count_decodes += 1
                    if image is None:
                        raise ValueError("decode_failed")
                    tick = time.perf_counter()
                    whole = measure_technical(image).to_dict()
                    stages["whole_seconds"] = time.perf_counter()-tick
                    tick = time.perf_counter()
                    color = measure_color(image)
                    stages["color_seconds"] = time.perf_counter()-tick
                    face_payload = []
                    tick = time.perf_counter()
                    if faces is not None:
                        source = detection_space_size(int(photos[name]["width"]), int(photos[name]["height"]))
                        _, diagnostics = extract_faces(face_rows.get(name, []), *source, rows=True)
                        for index, face in enumerate(faces):
                            roi, box = face_region(image, face.bbox, source)
                            if min(roi.shape[:2]) >= 20:
                                inset = max(1, int(min(roi.shape[:2])*.08))
                                roi = roi[inset:-inset, inset:-inset]
                            result = measure_technical(roi).to_dict()
                            face_payload.append(dict(face_index=index, box=box, prominence=(
                                diagnostics[index]["prominence"] if index < len(diagnostics) else 0),
                                primary=(diagnostics[index]["primary"] if index < len(diagnostics) else False),
                                face_clip_bgr=[float(v) for v in np.mean(roi >= 254, axis=(0, 1))]
                                    if roi.size else None,
                                **result))
                    stages["faces_seconds"] = time.perf_counter()-tick
                    if not control_rows and face_payload:
                        valid = next((f for f in face_payload if f["status"] == "measured"), None)
                        if valid:
                            roi, _ = face_region(image, faces[valid["face_index"]].bbox, source)
                            control_rows = controlled(roi, detector)
                            for row in control_rows:
                                row["source_image"] = path.name
                    payload = dict(whole=whole, color=color, faces=face_payload,
                                   decoded_size=list(image.shape[:2]))
                    cache_file.write_text(json.dumps(payload), encoding="utf-8")
                    del image
                baseline = technical_feature({field: photos[name].get("technical_raw_"+field)
                    for field in ("laplacian", "sobel", "local_contrast")})
                whole = payload["whole"]
                row = dict(group_id=group["group_id"], image_id=path.name,
                    source_path=name, decoded_width=payload["decoded_size"][1],
                    decoded_height=payload["decoded_size"][0],
                    baseline_score=baseline.value, baseline_confidence=baseline.confidence,
                    candidate_score=whole["score"], candidate_status=whole["status"],
                    edge_persistence=whole["edge_persistence"], detail=whole["detail_score"],
                    noise_aware_detail=whole["noise_aware_detail"],
                    noise_floor=whole["noise_floor"], chroma_noise=whole["chroma_noise"],
                    chroma_coverage=whole["chroma_coverage"], face_regions=len(payload["faces"]),
                    color_clipping=payload["color"].get("clipping_max"),
                    candidate_confidence=whole["confidence"])
                per_photo.append(row)
                group_metrics.append(row)
                for item in payload["faces"]:
                    per_face.append(dict(group_id=group["group_id"], image_id=path.name,
                        **{k: json.dumps(v) if isinstance(v, (list, tuple)) else v for k, v in item.items()}))
                timings.append(dict(image_id=path.name, elapsed_seconds=time.perf_counter()-t0,
                                    cache_hit=int(was_hit), **stages))
            except (OSError, ValueError, KeyError, cv2.error) as exc:
                per_photo.append(dict(group_id=group["group_id"], image_id=path.name,
                                      candidate_status="unavailable:"+type(exc).__name__))
            peak = max(peak, psutil.Process().memory_info().rss)
        if len(group_metrics) >= 2:
            raw = [r["baseline_score"] for r in group_metrics if r["baseline_score"] is not None]
            comparisons.append(dict(group_id=group["group_id"], baseline_margin=(
                max(raw)-min(raw) if len(raw) == len(group_metrics) else ""),
                candidate_margin="", candidate_winner="", status="score_withheld_unvalidated",
                v1_winner=Path(group["v1_winner"]).name, v2_winner=Path(group["v2_winner"]).name))
    for filename, records in (("photo_metrics.csv", per_photo), ("face_metrics.csv", per_face),
                              ("group_comparisons.csv", comparisons), ("controlled.csv", control_rows),
                              ("timings.csv", timings)):
        write(output/filename, records)
    summary = dict(version=VERSION, groups=len(groups), candidates=len(per_photo),
        additional_decodes=count_decodes, cache_hits=count_hits,
        elapsed_seconds=time.perf_counter()-start, peak_process_rss=peak,
        candidate_score_active=False, experimental_winner_changes_evaluated=False)
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write(output/"human_labels_template.csv", [dict(group_id=g["group_id"], technical_preference="",
        focus_reason="", noise_reason="", color_reason="", cannot_determine="") for g in groups])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("audit_results/nsc_fresh"))
    parser.add_argument("--output", type=Path, default=Path("audit_results/phase8g5_technical_calibration"))
    parser.add_argument("--max-dimension", type=int, default=1024)
    args = parser.parse_args()
    if not 256 <= args.max_dimension <= 2048:
        parser.error("max-dimension must be 256..2048")
    print(json.dumps(run(args.audit, args.output, args.max_dimension), indent=2))


if __name__ == "__main__":
    main()
