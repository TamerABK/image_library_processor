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

from automatic_scan.technical_quality_v2_calibration import VERSION, measure_technical, chroma_noise_multiscale
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
        maximum, VERSION, "audit-schema-v2", "ImageLoader-BGR8-v1", "face-roi-adaptive-inset-v2",
        face_cache_revision)).encode()).hexdigest()


def controlled(image, detector):
    rng = np.random.default_rng(3605)
    variants = {"baseline": image}
    lab = cv2.cvtColor(image.astype(np.float32)/255., cv2.COLOR_BGR2Lab)
    base_lab = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2Lab)
    def lab_variant(luma=0., chroma=0., correlated=False):
        changed = lab.copy()
        if luma:
            changed[:, :, 0] += rng.normal(0, luma, image.shape[:2])
        if chroma:
            noise = rng.normal(0, chroma, (*image.shape[:2], 2)).astype(np.float32)
            if correlated:
                noise = cv2.GaussianBlur(noise, (0, 0), 1.5)
            changed[:, :, 1:3] += noise
        changed[:, :, 0] = np.clip(changed[:, :, 0], 0, 100)
        bgr = np.clip(cv2.cvtColor(changed, cv2.COLOR_Lab2BGR)*255, 0, 255).astype(np.uint8)
        observed = cv2.cvtColor(bgr.astype(np.float32)/255., cv2.COLOR_BGR2Lab)
        delta = observed-base_lab
        return bgr, float(np.std(delta[:, :, 0])), float(np.std(delta[:, :, 1:3]))
    contamination = {"baseline": (0., 0.)}
    for sigma in (1, 2, 3):
        variants[f"gaussian_{sigma}"] = cv2.GaussianBlur(image, (0, 0), sigma)
    for strength in (5, 12, 25):
        for prefix, kwargs in (("luma_noise", dict(luma=strength/4)),
                               ("chroma_noise", dict(chroma=strength/4)),
                               ("combined_noise", dict(luma=strength/4, chroma=strength/4)),
                               ("correlated_chroma", dict(chroma=strength/4, correlated=True))):
            key = f"{prefix}_{strength}"
            variants[key], *contamination[key] = lab_variant(**kwargs)
    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    for strength in (5, 12):
        variants[f"blur_plus_noise_{strength}"] = np.clip(
            blurred.astype(np.int16)+rng.normal(0, strength, image.shape), 0, 255).astype(np.uint8)
    for strength in (5, 12):
        variants[f"blur_plus_luma_{strength}"], *contamination[f"blur_plus_luma_{strength}"] = lab_variant(luma=strength/4)
        variants[f"blur_plus_luma_{strength}"] = cv2.GaussianBlur(variants[f"blur_plus_luma_{strength}"], (0, 0), 2)
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
                            chroma_multiscale=chroma_noise_multiscale(altered),
                            chroma_coverage=result.chroma_coverage,
                            observed_luma_std=contamination.get(name, (None, None))[0],
                            observed_chroma_std=contamination.get(name, (None, None))[1],
                            artifact_measurement=result.artifact_measurement,
                            artifact_reliability=result.artifact_reliability,
                            detail_reliability=result.detail_reliability,
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
    loader = ImageLoader(max_decode_dimension=None if maximum == 0 else maximum)
    detector = BlurDetector(cache_path=output/"experimental_blur.sqlite3")
    per_photo, per_face, comparisons, timings = [], [], [], []
    count_hits = count_decodes = 0
    start = time.perf_counter()
    peak = psutil.Process().memory_info().rss
    control_rows = []
    control_groups = set()
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
                    if maximum == 0 and int(photos[name]["width"])*int(photos[name]["height"]) > 32_000_000:
                        raise ValueError("native_memory_budget_exceeded")
                    image = loader.load_for_scan(path, max_dimension=None if maximum == 0 else maximum)
                    stages["decode_seconds"] = time.perf_counter()-tick
                    count_decodes += 1
                    if image is None:
                        raise ValueError("decode_failed")
                    tick = time.perf_counter()
                    whole_image = (cv2.resize(image, (round(image.shape[1]*1024/max(image.shape[:2])),
                                               round(image.shape[0]*1024/max(image.shape[:2]))),
                                               interpolation=cv2.INTER_AREA)
                                   if maximum == 0 and max(image.shape[:2]) > 1024 else image)
                    whole = measure_technical(whole_image).to_dict()
                    stages["whole_seconds"] = time.perf_counter()-tick
                    tick = time.perf_counter()
                    color = measure_color(whole_image)
                    stages["color_seconds"] = time.perf_counter()-tick
                    face_payload = []
                    tick = time.perf_counter()
                    if faces is not None:
                        source = detection_space_size(int(photos[name]["width"]), int(photos[name]["height"]))
                        _, diagnostics = extract_faces(face_rows.get(name, []), *source, rows=True)
                        for index, face in enumerate(faces):
                            roi, box = face_region(image, face.bbox, source)
                            original_size = list(roi.shape[:2])
                            if min(roi.shape[:2]) >= 48:
                                inset = max(1, min(3, int(min(roi.shape[:2])*.03)))
                                roi = roi[inset:-inset, inset:-inset]
                            result = measure_technical(roi).to_dict()
                            if min(original_size) < 32:
                                result["status"] = "insufficient_source_resolution" if maximum == 0 else "insufficient_analysis_resolution"
                            face_payload.append(dict(face_index=index, box=box, prominence=(
                                diagnostics[index]["prominence"] if index < len(diagnostics) else 0),
                                primary=(diagnostics[index]["primary"] if index < len(diagnostics) else False),
                                face_clip_bgr=[float(v) for v in np.mean(roi >= 254, axis=(0, 1))]
                                    if roi.size else None,
                                face_shadow_clip_bgr=[float(v) for v in np.mean(roi <= 1, axis=(0, 1))]
                                    if roi.size else None,
                                roi_height=roi.shape[0], roi_width=roi.shape[1],
                                **result))
                    stages["faces_seconds"] = time.perf_counter()-tick
                    if group["group_id"] not in control_groups and face_payload:
                        valid = next((f for f in face_payload if f["status"] == "measured"), None)
                        if valid:
                            roi, _ = face_region(image, faces[valid["face_index"]].bbox, source)
                            group_controls = controlled(roi, detector)
                            for row in group_controls:
                                row["source_image"] = path.name
                                row["group_id"] = group["group_id"]
                            control_rows.extend(group_controls)
                            control_groups.add(group["group_id"])
                    payload = dict(whole=whole, color=color, faces=face_payload,
                                   detection_status="compatible_detection" if faces is not None else "missing_compatible_detection",
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
                    detection_status=payload["detection_status"],
                    color_clipping=payload["color"].get("clipping_max"),
                    detail_coverage=whole.get("detail_coverage"), detail_reliability=whole.get("detail_reliability"),
                    noise_coverage=whole.get("noise_coverage"), noise_reliability=whole.get("noise_reliability"),
                    artifact_measurement=whole.get("artifact_measurement"),
                    artifact_reliability=whole.get("artifact_reliability"),
                    chroma_reliability=whole.get("chroma_reliability"),
                    candidate_confidence=whole["confidence"])
                per_photo.append(row)
                group_metrics.append(row)
                for item in payload["faces"]:
                    per_face.append(dict(group_id=group["group_id"], image_id=path.name,
                        **{k: json.dumps(v) if isinstance(v, (list, tuple)) else v for k, v in item.items()}))
                timings.append(dict(group_id=group["group_id"], image_id=path.name, elapsed_seconds=time.perf_counter()-t0,
                                    cache_hit=int(was_hit), **stages))
            except (OSError, ValueError, KeyError, cv2.error) as exc:
                per_photo.append(dict(group_id=group["group_id"], image_id=path.name,
                                      candidate_status="processing_failure:"+type(exc).__name__))
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
    if args.max_dimension != 0 and not 256 <= args.max_dimension <= 2048:
        parser.error("max-dimension must be 0 (native) or 256..2048")
    print(json.dumps(run(args.audit, args.output, args.max_dimension), indent=2))


if __name__ == "__main__":
    main()
