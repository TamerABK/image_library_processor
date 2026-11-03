"""Adapt existing scan results to shadow ranking inputs; never run models."""
from __future__ import annotations

from .keeper_ranker_v2 import Face, Feature, Photo


def photo_input(path: str, faces: list, blur: object | None) -> Photo:
    measured = [face.analysis for face in faces if face.analysis is not None]
    largest = max((analysis.geometry.relative_area for analysis in measured), default=0.0)
    important = []
    for analysis in measured:
        area = analysis.geometry.relative_area
        if area < max(.008, .68 * largest):
            continue
        metrics = {
            "sharpness": analysis.image_quality.sharpness,
            "eyes": analysis.eyes,
            "pose": analysis.pose.metric,
            "exposure": analysis.image_quality.exposure,
            "contrast": analysis.image_quality.contrast,
            "detail": analysis.image_quality.detail_availability,
            "detector": analysis.detector_metric,
            "visible": analysis.visible_face_metric,
            "reliability": analysis.measurement_reliability,
        }
        features = {name: Feature(metric.ranking_score, metric.confidence)
                    for name, metric in metrics.items()}
        important.append(Face(features, prominence=max(.25, min(2., area / max(largest, .0001)))))
    technical = Feature(None) if blur is None else Feature(getattr(blur, "final_score", None))
    return Photo(path, tuple(important), technical)
