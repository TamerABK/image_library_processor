from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_paths import model_path
from face_analyzer.classical_metrics import legacy_tonal_metrics
from face_analyzer.config import FaceAnalyzerConfig
from face_analyzer.default_face_analyzer import DefaultFaceAnalyzer
from face_analyzer.geometry import aligned_or_fallback_crop, to_gray
from face_analyzer.ranking import (
    apply_group_relative_ranking,
    describe_distribution,
    kendall_tau,
    pairwise_accuracy,
    spearman_correlation,
)
from face_detector.scrfd_face_detector import SCRFDFaceDetector


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _print_stats(name: str, values: list[float]) -> None:
    stats = describe_distribution(values)
    print(name)
    print(f"  count: {int(stats['count'])}")
    print(f"  minimum: {stats['minimum']:.6f}")
    print(f"  10th percentile: {stats['p10']:.6f}")
    print(f"  25th percentile: {stats['p25']:.6f}")
    print(f"  median: {stats['median']:.6f}")
    print(f"  75th percentile: {stats['p75']:.6f}")
    print(f"  90th percentile: {stats['p90']:.6f}")
    print(f"  95th percentile: {stats['p95']:.6f}")
    print(f"  99th percentile: {stats['p99']:.6f}")
    print(f"  maximum: {stats['maximum']:.6f}")
    print(f"  percentage exactly 0: {stats['pct_eq_0']:.2f}%")
    print(f"  percentage exactly 1: {stats['pct_eq_1']:.2f}%")
    print(f"  unique-value count: {int(stats['unique_value_count'])}")


def _read_group_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = Path(row["image_path"]).resolve()
            mapping[os.fspath(image_path)] = str(row["group_id"])
    return mapping


def _print_correlations(rows: list[dict[str, object]], metric_names: list[str]) -> None:
    if len(rows) < 2:
        print("Not enough rows for correlation analysis.")
        return
    print("Metric correlations")
    matrix = np.asarray(
        [
            [float(row[name]) for name in metric_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    correlations = np.corrcoef(matrix, rowvar=False)
    for row_index, left_name in enumerate(metric_names):
        for col_index in range(row_index + 1, len(metric_names)):
            right_name = metric_names[col_index]
            correlation = float(correlations[row_index, col_index])
            suffix = "  HIGH" if abs(correlation) > 0.85 else ""
            print(f"  {left_name} vs {right_name}: {correlation:.3f}{suffix}")


def _read_pairwise_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_from_analysis(
    *,
    image_path: Path,
    face_index: int,
    group_id: str | None,
    legacy_exposure_score: float,
    legacy_contrast_score: float,
    analysis,
) -> dict[str, object]:
    return {
        "image_path": os.fspath(image_path),
        "face_index": face_index,
        "group_id": "" if group_id is None else group_id,
        "old_exposure_score": legacy_exposure_score,
        "old_contrast_score": legacy_contrast_score,
        "raw_exposure_score": analysis.image_quality.raw_exposure_score,
        "display_exposure_score": analysis.image_quality.display_exposure_score,
        "exposure_ranking_score": analysis.image_quality.exposure_ranking_score,
        "exposure_confidence": analysis.image_quality.exposure.confidence,
        "contrast_quality_score": analysis.image_quality.contrast_quality_score,
        "contrast_ranking_score": analysis.image_quality.contrast_ranking_score,
        "contrast_confidence": analysis.image_quality.contrast.confidence,
        "focus_sharpness_score": analysis.image_quality.focus_sharpness_score,
        "focus_ranking_score": analysis.image_quality.focus_ranking_score,
        "detail_availability_score": analysis.image_quality.detail_availability_score,
        "detail_availability_ranking_score": analysis.image_quality.detail_availability_ranking_score,
        "sharpness_ranking_score": analysis.image_quality.sharpness_ranking_score,
        "sharpness_confidence": analysis.image_quality.sharpness.confidence,
        "combined_eye_open_score": "" if analysis.combined_eye_open_score is None else analysis.combined_eye_open_score,
        "eye_ranking_score": "" if analysis.eye_ranking_score is None else analysis.eye_ranking_score,
        "eye_confidence": analysis.eye_confidence,
        "pose_quality_score": "" if analysis.pose_score is None else analysis.pose_score,
        "pose_ranking_score": "" if analysis.pose_ranking_score is None else analysis.pose_ranking_score,
        "pose_confidence": analysis.pose_confidence,
        "measurement_reliability_score": analysis.measurement_reliability_score,
        "global_selection_score": analysis.global_selection_score,
        "group_relative_score": analysis.group_relative_score,
        "final_group_score": analysis.final_group_score,
        "selection_score": analysis.selection_score,
        "detector_confidence": analysis.detector_confidence,
        "detector_ranking_score": analysis.detector_metric.ranking_score,
        "visible_face_ratio": analysis.visible_face_ratio,
        "visible_face_ranking_score": analysis.visible_face_metric.ranking_score,
        "median_luminance": analysis.image_quality.median_luminance,
        "p05_luminance": analysis.image_quality.p05_luminance,
        "p10_luminance": analysis.image_quality.p10_luminance,
        "p25_luminance": analysis.image_quality.p25_luminance,
        "p75_luminance": analysis.image_quality.p75_luminance,
        "p90_luminance": analysis.image_quality.p90_luminance,
        "p95_luminance": analysis.image_quality.p95_luminance,
        "dark_clip_ratio": analysis.image_quality.dark_clip_ratio,
        "bright_clip_ratio": analysis.image_quality.bright_clip_ratio,
        "usable_tonal_range": analysis.image_quality.usable_tonal_range,
        "clipping_score": analysis.image_quality.clipping_score,
        "luminance_score": analysis.image_quality.luminance_score,
        "tonal_information_score": analysis.image_quality.tonal_information_score,
        "shadow_detail_score": analysis.image_quality.shadow_detail_score,
        "highlight_detail_score": analysis.image_quality.highlight_detail_score,
        "tonal_balance_score": analysis.image_quality.tonal_balance_score,
        "broad_tonal_range": analysis.image_quality.broad_tonal_range,
        "interquartile_range": analysis.image_quality.interquartile_range,
        "broad_contrast_score": analysis.image_quality.broad_contrast_score,
        "interquartile_contrast_score": analysis.image_quality.interquartile_contrast_score,
        "local_contrast_raw": analysis.image_quality.local_contrast_raw,
        "local_contrast_score": analysis.image_quality.local_contrast_score,
        "laplacian_variance": analysis.image_quality.laplacian_variance,
        "tenengrad_energy": analysis.image_quality.tenengrad_energy,
        "high_frequency_energy_ratio": analysis.image_quality.high_frequency_energy_ratio,
        "detail_availability_measure": analysis.image_quality.detail_availability_measure,
        "pose_yaw_score": "" if analysis.pose.yaw_score is None else analysis.pose.yaw_score,
        "pose_pitch_score": "" if analysis.pose.pitch_score is None else analysis.pose.pitch_score,
        "pose_roll_score": "" if analysis.pose.roll_score is None else analysis.pose.roll_score,
        "warnings": "|".join(analysis.warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze face-quality ranking fields, export explicit CSV columns, "
            "and print metric distributions and correlations."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument(
        "--group-map-csv",
        help=(
            "Optional CSV with columns image_path,group_id. When present, duplicate-group "
            "relative scores are blended into final_group_score."
        ),
    )
    parser.add_argument(
        "--pairwise-csv",
        help=(
            "Optional CSV with columns image_a,image_b,preferred_image and optional "
            "group_id for ranking-agreement evaluation."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    csv_out = Path(args.csv_out).resolve()
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    group_map = _read_group_map(
        None if args.group_map_csv is None else Path(args.group_map_csv).resolve()
    )
    pairwise_rows = _read_pairwise_rows(
        None if args.pairwise_csv is None else Path(args.pairwise_csv).resolve()
    )

    config = FaceAnalyzerConfig()
    detector = SCRFDFaceDetector(model_path("scrfd_10g_bnkps.onnx"))
    analyzer = DefaultFaceAnalyzer(
        eye_state_model_path=model_path("open_closed_eye.onnx"),
        head_pose_model_path=model_path("sixdrepnet.onnx"),
        config=config,
    )

    collected: list[dict[str, object]] = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)

    for image_path in _iter_image_paths(input_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        faces = detector.detect(image, image_path)
        if not faces:
            continue

        for face_index, face in enumerate(faces):
            analysis = analyzer.analyze(image, face)

            aligned_face, _alignment_confidence = aligned_or_fallback_crop(
                image,
                face.bbox,
                face.landmarks,
                output_size=config.aligned_face_size,
            )
            gray = to_gray(
                cv2.resize(
                    aligned_face,
                    (config.aligned_face_size, config.aligned_face_size),
                    interpolation=(
                        cv2.INTER_AREA
                        if max(aligned_face.shape[:2]) > config.aligned_face_size
                        else cv2.INTER_CUBIC
                    ),
                )
            ).astype(np.float32)
            legacy = legacy_tonal_metrics(gray.reshape(-1))
            group_id = group_map.get(os.fspath(image_path.resolve()))
            collected.append(
                {
                    "image_path": image_path.resolve(),
                    "face_index": face_index,
                    "analysis": analysis,
                    "legacy_exposure_score": legacy.exposure_score,
                    "legacy_contrast_score": legacy.contrast_score,
                    "group_id": group_id,
                }
            )
            if group_id is not None:
                grouped_indices[group_id].append(len(collected) - 1)

    for group_id, indices in grouped_indices.items():
        updated_analyses = apply_group_relative_ranking(
            [collected[index]["analysis"] for index in indices],
            weights=config.ranking_weights,
            priors=config.metric_priors,
        )
        for local_index, updated in zip(indices, updated_analyses):
            collected[local_index]["analysis"] = updated

    rows = [
        _row_from_analysis(
            image_path=item["image_path"],
            face_index=int(item["face_index"]),
            group_id=item["group_id"],
            legacy_exposure_score=float(item["legacy_exposure_score"]),
            legacy_contrast_score=float(item["legacy_contrast_score"]),
            analysis=item["analysis"],
        )
        for item in collected
    ]

    if rows:
        with csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"CSV written to: {csv_out}")
    _print_stats("old exposure score", [float(row["old_exposure_score"]) for row in rows])
    _print_stats("display exposure score", [float(row["display_exposure_score"]) for row in rows])
    _print_stats("exposure ranking score", [float(row["exposure_ranking_score"]) for row in rows])
    _print_stats("old contrast score", [float(row["old_contrast_score"]) for row in rows])
    _print_stats("contrast quality score", [float(row["contrast_quality_score"]) for row in rows])
    _print_stats("contrast ranking score", [float(row["contrast_ranking_score"]) for row in rows])
    _print_stats("sharpness ranking score", [float(row["sharpness_ranking_score"]) for row in rows])
    _print_stats("final group score", [float(row["final_group_score"]) for row in rows])
    _print_correlations(
        rows,
        [
            "sharpness_ranking_score",
            "eye_ranking_score",
            "pose_ranking_score",
            "exposure_ranking_score",
            "contrast_ranking_score",
            "final_group_score",
        ],
    )

    sorted_rows = sorted(rows, key=lambda row: (float(row["final_group_score"]), str(row["image_path"])))
    if sorted_rows:
        print("Bottom-ranked faces")
        for row in sorted_rows[:5]:
            print(
                f"  {row['image_path']} face={row['face_index']} final={float(row['final_group_score']):.4f}"
            )
        print("Top-ranked faces")
        for row in reversed(sorted_rows[-5:]):
            print(
                f"  {row['image_path']} face={row['face_index']} final={float(row['final_group_score']):.4f}"
            )

    if grouped_indices:
        print("Top-ranked photo per duplicate group")
        for group_id, indices in sorted(grouped_indices.items()):
            best_row = max(
                (rows[index] for index in indices),
                key=lambda row: float(row["final_group_score"]),
            )
            print(
                f"  group={group_id} best={best_row['image_path']} score={float(best_row['final_group_score']):.4f}"
            )

    if rows:
        print("Largest ranking disagreements between metrics")
        disagreements = sorted(
            (
                (
                    max(
                        float(row["sharpness_ranking_score"]),
                        float(row["exposure_ranking_score"]),
                        float(row["contrast_ranking_score"]),
                        0.0 if row["eye_ranking_score"] == "" else float(row["eye_ranking_score"]),
                        0.0 if row["pose_ranking_score"] == "" else float(row["pose_ranking_score"]),
                    )
                    - min(
                        float(row["sharpness_ranking_score"]),
                        float(row["exposure_ranking_score"]),
                        float(row["contrast_ranking_score"]),
                        0.0 if row["eye_ranking_score"] == "" else float(row["eye_ranking_score"]),
                        0.0 if row["pose_ranking_score"] == "" else float(row["pose_ranking_score"]),
                    ),
                    row,
                )
                for row in rows
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for disagreement, row in disagreements[:5]:
            print(
                f"  {row['image_path']} face={row['face_index']} spread={disagreement:.4f}"
            )

    if pairwise_rows:
        best_score_by_image: dict[str, float] = {}
        for row in rows:
            path = str(Path(str(row["image_path"])).resolve())
            best_score_by_image[path] = max(
                best_score_by_image.get(path, 0.0),
                float(row["final_group_score"]),
            )

        scores_a: list[float] = []
        scores_b: list[float] = []
        preferred_is_a: list[bool] = []
        human_rank_values: list[float] = []
        model_rank_values: list[float] = []
        human_wins_by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for pairwise_row in pairwise_rows:
            image_a = str(Path(pairwise_row["image_a"]).resolve())
            image_b = str(Path(pairwise_row["image_b"]).resolve())
            preferred_image = str(Path(pairwise_row["preferred_image"]).resolve())
            if image_a not in best_score_by_image or image_b not in best_score_by_image:
                continue
            score_a = best_score_by_image[image_a]
            score_b = best_score_by_image[image_b]
            scores_a.append(score_a)
            scores_b.append(score_b)
            preferred_is_a.append(preferred_image == image_a)
            group_id = pairwise_row.get("group_id", "")
            if group_id:
                human_wins_by_group[group_id][preferred_image] += 1

        print("Pairwise ranking agreement")
        print(f"  pairwise accuracy: {pairwise_accuracy(scores_a, scores_b, preferred_is_a):.3f}")

        if human_wins_by_group:
            top1_total = 0
            top1_correct = 0
            for group_id, win_map in sorted(human_wins_by_group.items()):
                labeled_images = [image for image in win_map if image in best_score_by_image]
                if len(labeled_images) < 2:
                    continue
                human_best = max(labeled_images, key=lambda image: win_map[image])
                model_best = max(labeled_images, key=lambda image: best_score_by_image[image])
                top1_total += 1
                top1_correct += int(human_best == model_best)
                for image in labeled_images:
                    human_rank_values.append(float(win_map[image]))
                    model_rank_values.append(best_score_by_image[image])
            if human_rank_values and model_rank_values:
                print(f"  kendall tau: {kendall_tau(model_rank_values, human_rank_values):.3f}")
                print(f"  spearman correlation: {spearman_correlation(model_rank_values, human_rank_values):.3f}")
            if top1_total:
                print(f"  top-1 duplicate-group accuracy: {top1_correct / top1_total:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
