from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from face_analyzer.ranking import CalibrationCurve, RankingCalibrationProfile


DEFAULT_METRICS = {
    "exposure": "display_exposure_score",
    "contrast": "contrast_quality_score",
    "focus_sharpness": "focus_sharpness_score",
    "detail_availability": "detail_availability_score",
    "eyes": "combined_eye_open_score",
    "pose": "pose_quality_score",
    "detector_confidence": "detector_confidence",
    "visible_face_ratio": "visible_face_ratio",
}


def _suggest_curve(values: list[float]) -> CalibrationCurve:
    array = np.asarray(sorted(values), dtype=np.float64)
    percentiles = np.asarray([1, 10, 25, 40, 55, 70, 82, 90, 96, 99], dtype=np.float64)
    source_values = np.percentile(array, percentiles).astype(float)
    for index in range(1, source_values.size):
        if source_values[index] <= source_values[index - 1]:
            source_values[index] = min(1.0, source_values[index - 1] + 1e-4)
    target_values = np.asarray([0.01, 0.06, 0.16, 0.28, 0.42, 0.58, 0.74, 0.86, 0.95, 0.99], dtype=np.float64)
    return CalibrationCurve(
        source_values=tuple(float(value) for value in source_values),
        target_values=tuple(float(value) for value in target_values),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read a labeled face-ranking CSV, summarize percentile knots, and "
            "write a versioned calibration profile for review."
        )
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_json = Path(args.output_json).resolve()

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("Input CSV is empty.")

    curves: dict[str, CalibrationCurve] = {}
    for metric_name, column_name in DEFAULT_METRICS.items():
        values = [
            float(row[column_name])
            for row in rows
            if row.get(column_name) not in (None, "")
        ]
        if len(values) < 2:
            continue
        curves[metric_name] = _suggest_curve(values)
        print(metric_name)
        print(f"  samples: {len(values)}")
        print(f"  source_values: {list(curves[metric_name].source_values)}")
        print(f"  target_values: {list(curves[metric_name].target_values)}")

    base_profile = RankingCalibrationProfile()
    profile = RankingCalibrationProfile(
        version=base_profile.version + 1,
        exposure=curves.get("exposure", base_profile.exposure),
        contrast=curves.get("contrast", base_profile.contrast),
        focus_sharpness=curves.get("focus_sharpness", base_profile.focus_sharpness),
        detail_availability=curves.get("detail_availability", base_profile.detail_availability),
        eyes=curves.get("eyes", base_profile.eyes),
        pose=curves.get("pose", base_profile.pose),
        detector_confidence=curves.get("detector_confidence", base_profile.detector_confidence),
        visible_face_ratio=curves.get("visible_face_ratio", base_profile.visible_face_ratio),
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(profile.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Calibration profile written to: {output_json}")
    print("Review the suggested knots before deploying them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
