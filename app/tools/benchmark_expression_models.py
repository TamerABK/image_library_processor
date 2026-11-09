"""Local, read-only OpenCV Zoo FER comparison on cached Room36 faces."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.tools.acquire_expression_models import MODELS, UPSTREAM_COMMIT, verify
from image_loader import ImageLoader

LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
TEMPLATE = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                     [41.5493,92.3655],[70.7299,92.2041]], dtype=np.float64)


def align_official(image, landmarks):
    """OpenCV Zoo nonreflective least-squares similarity and default warpAffine."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5,2) or not np.isfinite(points).all():
        return None
    x, y = TEMPLATE[:,0:1], TEMPLATE[:,1:2]
    design = np.vstack((np.hstack((x,y,np.ones((5,1)),np.zeros((5,1)))),
                        np.hstack((y,-x,np.zeros((5,1)),np.ones((5,1))))))
    if np.linalg.matrix_rank(design) < 4:
        return None
    solved = np.linalg.lstsq(design, np.r_[points[:,0],points[:,1]], rcond=-1)[0]
    scale_cos, scale_sin, tx, ty = solved
    inverse = np.array([[scale_cos,-scale_sin,0],
                        [scale_sin,scale_cos,0],[tx,ty,1]])
    transform = np.linalg.inv(inverse)[:,0:2].T
    return cv2.warpAffine(image, transform, (112,112))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def category(row):
    area = float(row.get("geometry_relative_area") or 0)
    pose = abs(float(row.get("head_pose_yaw_degrees") or 0))
    light = float(row.get("image_quality_median_luminance") or 128)
    if area < .008:
        return "small"
    if pose >= 30:
        return "angled"
    if light < 55:
        return "low_light"
    if not row.get("eyes_ranking_score"):
        return "eyes_uncertain"
    return "other"


def select_faces(rows, sample_size, seed):
    pools = defaultdict(list)
    for row in rows:
        if row.get("analysis_missing") == "1":
            continue
        pools[category(row)].append(row)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    chosen, photo_count = [], defaultdict(int)
    order = sorted(pools)
    while len(chosen) < sample_size and any(pools.values()):
        progress = False
        for group in order:
            while pools[group] and photo_count[pools[group][-1]["image_path"]] >= 2:
                pools[group].pop()
            if pools[group]:
                row = pools[group].pop()
                chosen.append(row)
                photo_count[row["image_path"]] += 1
                progress = True
                if len(chosen) >= sample_size:
                    break
        if not progress:
            break
    return chosen


def preprocess(crop):
    """OpenCV Zoo: aligned BGR 112² -> RGB -> [0,1] -> (x-.5)/.5 -> NCHW."""
    if crop is None or crop.shape != (112, 112, 3):
        raise ValueError("Expected aligned BGR crop of shape 112x112x3")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return np.ascontiguousarray(((rgb - .5) / .5).transpose(2, 0, 1)[None])


def probabilities(output):
    vector = np.asarray(output, dtype=np.float64).reshape(-1)
    if vector.size != 7 or not np.isfinite(vector).all():
        raise ValueError("Invalid seven-class model output")
    if np.min(vector) >= 0 and abs(float(vector.sum()) - 1.) < 1e-3:
        return vector / vector.sum()
    shifted = vector - vector.max()
    value = np.exp(shifted)
    return value / value.sum()


def load_crops(selected, cache_db, root):
    connection = sqlite3.connect(f"file:{Path(cache_db).resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    loader = ImageLoader(max_decode_dimension=None)
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["image_path"]].append(row)
    samples, failed = [], []
    for path, rows in grouped.items():
        image = loader.load_for_scan(Path(path), max_dimension=None)
        if image is None:
            failed.extend((r, "decode_failed") for r in rows)
            continue
        for row in rows:
            item = connection.execute("SELECT landmarks,landmarks_rows,landmarks_cols FROM image_face_analysis_faces "
                "WHERE entry_path=? AND face_index=?", (path, int(row["face_index"]))).fetchone()
            if item is None or item["landmarks"] is None:
                failed.append((row, "landmarks_missing"))
                continue
            points = np.frombuffer(item["landmarks"], dtype=np.float32).reshape(
                item["landmarks_rows"], item["landmarks_cols"])
            crop = align_official(image, points)
            if crop is None:
                failed.append((row, "alignment_failed"))
                continue
            samples.append((row, preprocess(crop), crop))
        del image
    connection.close()
    return samples, failed


def review_sheets(samples, predictions, output, count):
    if count <= 0:
        return 0
    lookup = {(r["variant"], r["image_id"], r["face_index"]): r
              for r in predictions if r["provider"] == "cpu"}
    ranked = []
    for row, _, crop in samples:
        key = (Path(row["image_path"]).name, row["face_index"])
        fp = lookup.get(("fp32", *key))
        q = lookup.get(("int8", *key))
        if fp is None:
            continue
        disagreement = int(q is not None and fp["top1"] != q["top1"])
        difficulty = int(category(row) != "other")
        ranked.append((disagreement, difficulty, key, row, crop))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = ranked[:count]
    target = output / "review_sheets"
    target.mkdir(exist_ok=True)
    for sheet_index in range(0, len(selected), 8):
        canvas = Image.new("RGB", (800, 4*175), "white")
        draw = ImageDraw.Draw(canvas)
        for position, (disagreement, _, key, row, crop) in enumerate(selected[sheet_index:sheet_index+8]):
            x, y = (position % 2)*400, (position // 2)*175
            canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (x, y))
            draw.text((x+120, y+2), f"{key[0]} face {key[1]}  {'DISAGREE' if disagreement else ''}", fill="black")
            draw.text((x+120, y+20), f"bbox {row.get('bbox_width')}x{row.get('bbox_height')} yaw {row.get('head_pose_yaw_degrees')}", fill="black")
            for line, variant in enumerate(("fp32", "int8", "int8bq")):
                item = lookup.get((variant, *key))
                label = variant + ": " + ((item["top3"][:37]) if item else "unavailable")
                draw.text((x+120, y+42+line*19), label, fill="black")
        canvas.save(target / f"review_{sheet_index//8+1:02d}.jpg", quality=85)
    return len(selected)


def run(args):
    import onnxruntime as ort
    import psutil
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    selected = select_faces(read_csv(args.face_csv), args.sample_size, args.seed)
    load_started = time.perf_counter()
    samples, failed = load_crops(selected, args.cache_source, args.dataset)
    load_seconds = time.perf_counter() - load_started
    if not samples:
        raise ValueError("No usable cached face landmarks/crops")
    process = psutil.Process()
    available = ort.get_available_providers()
    models, predictions, timings, memory, comparison = [], [], [], [], []
    vectors = {}
    for variant in args.models:
        name, size, digest = MODELS[variant]
        path = args.model_dir / name
        if not verify(path, size, digest):
            raise ValueError(f"Missing or SHA-256-invalid model: {path}")
        for requested in args.providers:
            provider = "CPUExecutionProvider" if requested == "cpu" else "DmlExecutionProvider"
            record = dict(variant=variant, provider=requested, filename=name, bytes=size,
                          sha256=digest, source_commit=UPSTREAM_COMMIT, status="")
            try:
                import onnx
                graph = onnx.load(str(path), load_external_data=False)
                record["opset"] = "|".join(f"{item.domain or 'ai.onnx'}:{item.version}"
                    for item in graph.opset_import)
                record["operators"] = "|".join(sorted({node.op_type for node in graph.graph.node}))
            except ImportError:
                record["opset"] = "onnx_package_unavailable"
            if provider not in available:
                record["status"] = "provider_unavailable"
                models.append(record)
                continue
            try:
                options = ort.SessionOptions()
                options.log_severity_level = 3
                if requested == "directml":
                    options.enable_profiling = True
                    options.profile_file_prefix = str(output / f"profile_{variant}_{requested}")
                rss_before = process.memory_info().rss
                started = time.perf_counter()
                session = ort.InferenceSession(str(path), sess_options=options,
                    providers=[provider] if requested == "cpu" else [provider, "CPUExecutionProvider"])
                record["init_seconds"] = time.perf_counter() - started
                record["actual_providers"] = "|".join(session.get_providers())
                record["input_metadata"] = json.dumps([(i.name, i.shape, i.type) for i in session.get_inputs()])
                record["output_metadata"] = json.dumps([(o.name, o.shape, o.type) for o in session.get_outputs()])
                record["rss_session_delta"] = process.memory_info().rss - rss_before
                inputs = session.get_inputs()
                if len(inputs) != 1 or inputs[0].shape != [1, 3, 112, 112]:
                    record["status"] = "unsupported_input_signature"
                    models.append(record)
                    del session
                    continue
                feed_name = inputs[0].name
                sample = samples[0][1]
                started = time.perf_counter()
                session.run(None, {feed_name: sample})
                record["first_inference_ms"] = (time.perf_counter()-started)*1000
                warmup_started = time.perf_counter()
                for _ in range(5):
                    session.run(None, {feed_name: sample})
                record["warmup_ms"] = (time.perf_counter()-warmup_started)*1000
                record["status"] = "measured"
                rss_peak = process.memory_info().rss
                for index, (row, tensor, _) in enumerate(samples * args.repetitions):
                    starts = time.perf_counter()
                    raw = session.run(None, {feed_name: tensor})[0]
                    elapsed = (time.perf_counter()-starts)*1000
                    probabilities_started = time.perf_counter()
                    prob = probabilities(raw)
                    post_ms = (time.perf_counter()-probabilities_started)*1000
                    key = (row["image_path"], row["face_index"])
                    vectors[(variant, requested, key)] = prob
                    top = np.argsort(-prob)[:3]
                    predictions.append(dict(variant=variant, provider=requested,
                        image_id=Path(key[0]).name, face_index=key[1], category=category(row),
                        top1=LABELS[top[0]], confidence=float(prob[top[0]]),
                        top3="|".join(f"{LABELS[i]}:{prob[i]:.4f}" for i in top),
                        entropy=float(-sum(p*math.log(p) for p in prob if p > 0)),
                        probabilities=json.dumps([float(x) for x in prob])))
                    timings.append(dict(variant=variant, provider=requested, batch_size=1,
                        repetition=index//len(samples), index=index%len(samples),
                        inference_ms=elapsed, postprocess_ms=post_ms))
                    rss_peak = max(rss_peak, process.memory_info().rss)
                record["median_ms"] = statistics.median(t["inference_ms"] for t in timings
                    if t["variant"] == variant and t["provider"] == requested)
                record["p95_ms"] = float(np.percentile([t["inference_ms"] for t in timings
                    if t["variant"] == variant and t["provider"] == requested], 95))
                record["p90_ms"] = float(np.percentile([t["inference_ms"] for t in timings
                    if t["variant"] == variant and t["provider"] == requested], 90))
                record["throughput_faces_s"] = 1000 / statistics.mean(t["inference_ms"] for t in timings
                    if t["variant"] == variant and t["provider"] == requested)
                memory.append(dict(variant=variant, provider=requested, rss_before=rss_before,
                                   rss_session=process.memory_info().rss, rss_peak=rss_peak,
                                   gpu_memory_bytes="unavailable"))
                if requested == "directml":
                    profile_path = Path(session.end_profiling())
                    events = json.loads(profile_path.read_text(encoding="utf-8"))
                    counts = defaultdict(int)
                    for event in events:
                        provider_name = event.get("args", {}).get("provider")
                        if provider_name:
                            counts[provider_name] += 1
                    record["profiled_node_providers"] = json.dumps(dict(counts))
                del session
            except Exception as exc:
                record["status"] = type(exc).__name__
                record["error"] = str(exc)[:300]
            models.append(record)
    for (variant, provider, key), prob in vectors.items():
        if variant == "fp32":
            continue
        reference = vectors.get(("fp32", provider, key))
        if reference is None:
            continue
        comparison.append(dict(variant=variant, provider=provider,
            image_id=Path(key[0]).name, face_index=key[1],
            top1_agree=int(np.argmax(prob) == np.argmax(reference)),
            top2_agree=int(set(np.argsort(-prob)[:2]) == set(np.argsort(-reference)[:2])),
            probability_mae=float(np.mean(np.abs(prob-reference)))))
    predictions = list({(r["variant"], r["provider"], r["image_id"], r["face_index"]): r
                        for r in predictions}.values())
    write_csv(output / "model_comparison.csv", models,
              list(dict.fromkeys(k for row in models for k in row)))
    write_csv(output / "expression_predictions.csv", predictions,
              ["variant","provider","image_id","face_index","category","top1","confidence","top3","entropy","probabilities"])
    write_csv(output / "prediction_comparison.csv", comparison,
              ["variant","provider","image_id","face_index","top1_agree","top2_agree","probability_mae"])
    write_csv(output / "latency_results.csv", timings,
              ["variant","provider","batch_size","repetition","index","inference_ms","postprocess_ms"])
    write_csv(output / "memory_results.csv", memory,
              ["variant","provider","rss_before","rss_session","rss_peak","gpu_memory_bytes"])
    write_csv(output / "review_labels_template.csv",
              [dict(image_id=Path(r["image_path"]).name, face_index=r["face_index"],
                    visible_expression="", ambiguous="", unusable_crop="", uncertain="") for r in selected],
              ["image_id","face_index","visible_expression","ambiguous","unusable_crop","uncertain"])
    reviewed = review_sheets(samples, predictions, output, args.review_count)
    summary = dict(seed=args.seed, selected=len(selected), usable=len(samples), failed=len(failed),
        review_faces=reviewed,
        category_counts={c: sum(category(r)==c for r in selected) for c in
                         sorted({category(r) for r in selected})},
        preprocessing_and_decode_seconds=load_seconds, available_providers=available,
        requested_batch_sizes=args.batch_sizes, supported_batch_sizes=[1],
        repetitions=args.repetitions,
        model_commit=UPSTREAM_COMMIT, models=models, expression_accuracy="unavailable_without_labels")
    (output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("audit_results/expression_benchmark/models"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--providers", nargs="+", choices=("cpu", "directml"), default=["cpu", "directml"])
    parser.add_argument("--dataset", type=Path, default=Path("room36_test_set/NSC_Meeting"))
    parser.add_argument("--face-csv", type=Path, default=Path("audit_results/phase8g2_full_20261103/face_metrics.csv"))
    parser.add_argument("--cache-source", type=Path, default=Path("audit_results/phase8g2_full_20261103/audit_cache.sqlite3"))
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=3603)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1,4,8,16,32])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--review-count", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("audit_results/expression_benchmark"))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
