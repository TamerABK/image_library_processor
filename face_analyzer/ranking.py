from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .math_utils import EPSILON, clamp01


def _tuple_of_floats(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class CalibrationCurve:
    source_values: tuple[float, ...]
    target_values: tuple[float, ...]

    def __post_init__(self) -> None:
        source_values = _tuple_of_floats(self.source_values)
        target_values = _tuple_of_floats(self.target_values)
        if len(source_values) != len(target_values):
            raise ValueError("Calibration curves require matching source and target lengths.")
        if len(source_values) < 2:
            raise ValueError("Calibration curves require at least two knots.")
        for previous, current in zip(source_values, source_values[1:]):
            if current <= previous:
                raise ValueError("Calibration curve source knots must be strictly increasing.")
        for previous, current in zip(target_values, target_values[1:]):
            if current < previous:
                raise ValueError("Calibration curve target knots must be monotonic.")
        object.__setattr__(self, "source_values", source_values)
        object.__setattr__(self, "target_values", target_values)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "source_values": [float(value) for value in self.source_values],
            "target_values": [float(value) for value in self.target_values],
        }


@dataclass(frozen=True)
class MetricPriorConfig:
    exposure: float = 0.55
    contrast: float = 0.50
    sharpness: float = 0.45
    detail_availability: float = 0.45
    eyes: float = 0.55
    pose: float = 0.50
    detector_confidence: float = 0.50
    visible_face: float = 0.50
    measurement_reliability: float = 0.50


@dataclass(frozen=True)
class RankingWeights:
    sharpness: float = 0.30
    eyes: float = 0.23
    pose: float = 0.15
    exposure: float = 0.10
    contrast: float = 0.07
    detector_confidence: float = 0.04
    measurement_reliability: float = 0.04
    visible_face: float = 0.02
    detail_availability: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpness": float(self.sharpness),
            "eyes": float(self.eyes),
            "pose": float(self.pose),
            "exposure": float(self.exposure),
            "contrast": float(self.contrast),
            "detector_confidence": float(self.detector_confidence),
            "measurement_reliability": float(self.measurement_reliability),
            "visible_face": float(self.visible_face),
            "detail_availability": float(self.detail_availability),
        }


def _default_exposure_curve() -> CalibrationCurve:
    return CalibrationCurve(
        source_values=(0.20, 0.40, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85, 0.88, 0.92),
        target_values=(0.01, 0.08, 0.20, 0.35, 0.50, 0.64, 0.75, 0.84, 0.92, 0.98),
    )


def _default_contrast_curve() -> CalibrationCurve:
    return CalibrationCurve(
        source_values=(0.00, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.82, 0.90, 0.97),
        target_values=(0.00, 0.03, 0.10, 0.22, 0.38, 0.56, 0.72, 0.84, 0.93, 0.99),
    )


def _default_focus_curve() -> CalibrationCurve:
    return CalibrationCurve(
        source_values=(0.00, 0.08, 0.16, 0.28, 0.42, 0.58, 0.72, 0.84, 0.93, 1.00),
        target_values=(0.00, 0.02, 0.07, 0.16, 0.30, 0.50, 0.69, 0.84, 0.95, 1.00),
    )


def _default_detail_curve() -> CalibrationCurve:
    return CalibrationCurve(
        source_values=(0.00, 0.10, 0.20, 0.32, 0.46, 0.60, 0.74, 0.86, 0.94, 1.00),
        target_values=(0.00, 0.03, 0.08, 0.17, 0.30, 0.48, 0.68, 0.85, 0.95, 1.00),
    )


def _default_eye_curve() -> CalibrationCurve:
    return CalibrationCurve(
        source_values=(0.00, 0.20, 0.50, 0.75, 0.90, 0.97, 0.995, 1.00),
        target_values=(0.00, 0.03, 0.12, 0.35, 0.62, 0.80, 0.94, 1.00),
    )


def _default_pose_curve() -> CalibrationCurve:
    return CalibrationCurve(
        source_values=(0.00, 0.20, 0.40, 0.58, 0.72, 0.84, 0.92, 0.97, 1.00),
        target_values=(0.00, 0.04, 0.12, 0.26, 0.45, 0.66, 0.82, 0.93, 1.00),
    )


def _default_detector_curve() -> CalibrationCurve:
    return CalibrationCurve(
        source_values=(0.10, 0.25, 0.40, 0.55, 0.70, 0.82, 0.90, 0.95, 0.99, 1.00),
        target_values=(0.01, 0.05, 0.12, 0.24, 0.42, 0.60, 0.75, 0.86, 0.96, 1.00),
    )


def _default_visible_face_curve() -> CalibrationCurve:
    return CalibrationCurve(
        source_values=(0.20, 0.40, 0.55, 0.68, 0.78, 0.86, 0.92, 0.96, 0.99, 1.00),
        target_values=(0.00, 0.03, 0.10, 0.22, 0.40, 0.58, 0.75, 0.88, 0.97, 1.00),
    )


@dataclass(frozen=True)
class RankingCalibrationProfile:
    version: int = 1
    exposure: CalibrationCurve = field(default_factory=_default_exposure_curve)
    contrast: CalibrationCurve = field(default_factory=_default_contrast_curve)
    focus_sharpness: CalibrationCurve = field(default_factory=_default_focus_curve)
    detail_availability: CalibrationCurve = field(default_factory=_default_detail_curve)
    eyes: CalibrationCurve = field(default_factory=_default_eye_curve)
    pose: CalibrationCurve = field(default_factory=_default_pose_curve)
    detector_confidence: CalibrationCurve = field(default_factory=_default_detector_curve)
    visible_face_ratio: CalibrationCurve = field(default_factory=_default_visible_face_curve)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "exposure": self.exposure.to_dict(),
            "contrast": self.contrast.to_dict(),
            "focus_sharpness": self.focus_sharpness.to_dict(),
            "detail_availability": self.detail_availability.to_dict(),
            "eyes": self.eyes.to_dict(),
            "pose": self.pose.to_dict(),
            "detector_confidence": self.detector_confidence.to_dict(),
            "visible_face_ratio": self.visible_face_ratio.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RankingCalibrationProfile":
        return cls(
            version=int(data["version"]),
            exposure=CalibrationCurve(**data["exposure"]),
            contrast=CalibrationCurve(**data["contrast"]),
            focus_sharpness=CalibrationCurve(**data["focus_sharpness"]),
            detail_availability=CalibrationCurve(**data["detail_availability"]),
            eyes=CalibrationCurve(**data["eyes"]),
            pose=CalibrationCurve(**data["pose"]),
            detector_confidence=CalibrationCurve(**data["detector_confidence"]),
            visible_face_ratio=CalibrationCurve(**data["visible_face_ratio"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RankingCalibrationProfile":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def write(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def apply_calibration_curve(
    value: float | None,
    curve: CalibrationCurve,
) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    clipped_value = clamp01(float(value))
    return clamp01(
        float(
            np.interp(
                clipped_value,
                np.asarray(curve.source_values, dtype=np.float64),
                np.asarray(curve.target_values, dtype=np.float64),
            )
        )
    )


def confidence_blended_score(
    score: float | None,
    confidence: float,
    *,
    prior: float,
) -> float:
    clipped_confidence = clamp01(confidence)
    clipped_prior = clamp01(prior)
    if score is None or not math.isfinite(score):
        return clipped_prior
    return clamp01(
        clipped_confidence * clamp01(float(score))
        + (1.0 - clipped_confidence) * clipped_prior
    )


def percentile_rank(
    value: float,
    sorted_reference_values: np.ndarray,
) -> float:
    reference = np.asarray(sorted_reference_values, dtype=np.float64).reshape(-1)
    if reference.size == 0:
        return 0.5
    left = int(np.searchsorted(reference, value, side="left"))
    right = int(np.searchsorted(reference, value, side="right"))
    return clamp01((left + right) / (2.0 * reference.size))


def weighted_sum(entries: list[tuple[float | None, float]]) -> float:
    total = 0.0
    total_weight = 0.0
    for value, weight in entries:
        if value is None or not math.isfinite(value):
            continue
        clipped_weight = max(0.0, float(weight))
        if clipped_weight <= 0.0:
            continue
        total += clamp01(float(value)) * clipped_weight
        total_weight += clipped_weight
    if total_weight <= EPSILON:
        return 0.0
    return clamp01(total / total_weight)


def describe_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "minimum": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "maximum": 0.0,
            "pct_eq_0": 0.0,
            "pct_eq_1": 0.0,
            "unique_value_count": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    rounded = np.round(array, 8)
    return {
        "count": float(array.size),
        "minimum": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
        "pct_eq_0": float(np.mean(array == 0.0) * 100.0),
        "pct_eq_1": float(np.mean(array == 1.0) * 100.0),
        "unique_value_count": float(np.unique(rounded).size),
    }


def _as_numpy(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def pairwise_accuracy(
    scores_a: list[float],
    scores_b: list[float],
    preferred_is_a: list[bool],
) -> float:
    if not scores_a:
        return 0.0
    correct = 0
    for score_a, score_b, preferred_a in zip(scores_a, scores_b, preferred_is_a):
        predicted_a = score_a >= score_b
        correct += int(predicted_a == bool(preferred_a))
    return correct / max(len(preferred_is_a), 1)


def spearman_correlation(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0
    from scipy.stats import spearmanr

    return float(spearmanr(_as_numpy(values_a), _as_numpy(values_b)).statistic)


def kendall_tau(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0
    from scipy.stats import kendalltau

    return float(kendalltau(_as_numpy(values_a), _as_numpy(values_b)).statistic)


def apply_group_relative_ranking(
    analyses: list["FaceAnalysisResult"],
    *,
    weights: RankingWeights,
    priors: MetricPriorConfig,
    global_weight: float = 0.75,
    group_weight: float = 0.25,
) -> list["FaceAnalysisResult"]:
    if not analyses:
        return analyses

    from .models import FaceAnalysisResult

    metrics = [
        (
            "sharpness",
            [analysis.image_quality.sharpness.confidence_blended_ranking(priors.sharpness) for analysis in analyses],
            [weights.sharpness for _analysis in analyses],
        ),
        (
            "eyes",
            [analysis.eyes.confidence_blended_ranking(priors.eyes) for analysis in analyses],
            [analysis.eye_weight for analysis in analyses],
        ),
        (
            "pose",
            [analysis.pose.metric.confidence_blended_ranking(priors.pose) for analysis in analyses],
            [weights.pose for _analysis in analyses],
        ),
        (
            "exposure",
            [analysis.image_quality.exposure.confidence_blended_ranking(priors.exposure) for analysis in analyses],
            [weights.exposure for _analysis in analyses],
        ),
        (
            "contrast",
            [analysis.image_quality.contrast.confidence_blended_ranking(priors.contrast) for analysis in analyses],
            [weights.contrast for _analysis in analyses],
        ),
        (
            "detector",
            [analysis.detector_metric.confidence_blended_ranking(priors.detector_confidence) for analysis in analyses],
            [weights.detector_confidence for _analysis in analyses],
        ),
        (
            "measurement_reliability",
            [analysis.measurement_reliability.confidence_blended_ranking(priors.measurement_reliability) for analysis in analyses],
            [weights.measurement_reliability for _analysis in analyses],
        ),
        (
            "visible_face",
            [analysis.visible_face_metric.confidence_blended_ranking(priors.visible_face) for analysis in analyses],
            [weights.visible_face for _analysis in analyses],
        ),
        (
            "detail_availability",
            [analysis.image_quality.detail_availability.confidence_blended_ranking(priors.detail_availability) for analysis in analyses],
            [weights.detail_availability for _analysis in analyses],
        ),
    ]

    sorted_references = {
        name: np.sort(np.asarray(values, dtype=np.float64))
        for name, values, _metric_weights in metrics
    }

    updated: list[FaceAnalysisResult] = []
    for index, analysis in enumerate(analyses):
        group_metric_score = weighted_sum(
            [
                (
                    percentile_rank(values[index], sorted_references[name]),
                    metric_weights[index],
                )
                for name, values, metric_weights in metrics
            ]
        )
        final_group_score = clamp01(
            global_weight * analysis.global_selection_score
            + group_weight * group_metric_score
        )
        updated.append(
            replace(
                analysis,
                group_relative_score=group_metric_score,
                final_group_score=final_group_score,
                selection_score=final_group_score,
            )
        )
    return updated
