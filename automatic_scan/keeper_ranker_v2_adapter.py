"""Shared model-free V2 extraction for production shadow and audit."""
from __future__ import annotations

from math import exp, hypot, isfinite, log1p, sqrt
from .keeper_ranker_v2 import Face, Feature, Photo

METRICS = {"sharpness": "image_quality_sharpness", "eyes": "eyes", "pose": "pose_metric",
           "exposure": "image_quality_exposure", "contrast": "image_quality_contrast",
           "detail": "image_quality_detail_availability", "detector": "detector_metric",
           "visible": "visible_face_metric", "reliability": "measurement_reliability"}
TECHNICAL_VERSION = "raw-log-v1"


def number(value):
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (TypeError, ValueError):
        return None


def technical_feature(blur) -> Feature:
    if blur is None:
        return Feature(None, 0)
    get = blur.get if isinstance(blur, dict) else lambda key: getattr(blur, key, None)
    lap, sob, contrast = (number(get(key)) for key in ("laplacian", "sobel", "local_contrast"))
    if any(v is None or v < 0 for v in (lap, sob, contrast)):
        return Feature(None, 0)
    # Fixed, provisional anchors from blur-classification scales; no group fitting.
    l, s, c = log1p(lap) / log1p(400.), log1p(sob) / log1p(60.), log1p(contrast) / log1p(50.)
    agreement = exp(-abs(l - s))
    edge = sqrt(l * s) * agreement
    # Smooth tonal support and metric agreement limit noise/texture advantage.
    support = max(.01, c + .25)
    supported = edge / (1. + edge / support)
    value = .55 * supported / (1. + supported) + .45 * c / (1. + c)
    confidence = min(1., agreement * min(1., (lap + sob + contrast) / 15.))
    return Feature(value, confidence)


def extract_faces(faces, width=None, height=None, *, rows=False):
    """Eligibility uses detection and geometry; prominence uses no eye evidence."""
    candidates, diagnostics = [], []
    for index, face in enumerate(faces):
        source = face if rows else face.analysis
        bbox = tuple(number(face.get(k)) or 0 for k in
                     ("bbox_x", "bbox_y", "bbox_width", "bbox_height")) if rows else face.bbox
        confidence = number(face.get("detector_confidence_observed") if rows else face.confidence)
        area = number(face.get("geometry_relative_area")) if rows else (
            number(source.geometry.relative_area) if source is not None else None)
        metrics = (None if source is None or (rows and number(face.get("analysis_missing"))) else
                   {name: Feature(number(source.get(prefix + "_ranking_score")),
                                  number(source.get(prefix + "_confidence")) or 0)
                    for name, prefix in METRICS.items()} if rows else
                   {"sharpness": source.image_quality.sharpness, "eyes": source.eyes,
                    "pose": source.pose.metric, "exposure": source.image_quality.exposure,
                    "contrast": source.image_quality.contrast,
                    "detail": source.image_quality.detail_availability,
                    "detector": source.detector_metric, "visible": source.visible_face_metric,
                    "reliability": source.measurement_reliability})
        if metrics is not None and not rows:
            metrics = {name: Feature(metric.ranking_score, metric.confidence)
                       for name, metric in metrics.items()}
        reason = ("analysis_missing" if metrics is None else
                  "invalid_geometry" if bbox[2] < 8 or bbox[3] < 8 or not area or area <= 0 else
                  "low_detection_confidence" if confidence is None or confidence < .3 else
                  "no_eye_independent_evidence" if not any(
                      metrics[k].usable() for k in METRICS if k not in ("eyes", "reliability")) else "")
        diagnostics.append(dict(face_index=index, relative_area=area,
                                detector_confidence=confidence, eligible=not bool(reason),
                                primary=False, prominence=0., reason=reason))
        if not reason:
            candidates.append((index, metrics, bbox, area, confidence))
    if not candidates:
        return (), diagnostics
    areas = sorted(c[3] for c in candidates)
    median = areas[len(areas) // 2]
    result = []
    for index, metrics, bbox, area, confidence in candidates:
        relative = area / median
        primary = relative >= .16
        centrality = (max(0., 1. - hypot((bbox[0] + bbox[2]/2)/width - .5,
                                            (bbox[1] + bbox[3]/2)/height - .5) / .71)
                      if width and height and width > 0 and height > 0 else .5)
        prominence = max(.25, min(2., (.7 + .3 * centrality) *
                                    (.6 + .4 * min(2., sqrt(relative))) *
                                    (.8 + .2 * confidence)))
        diagnostics[index].update(primary=primary, prominence=prominence,
                                  reason="" if primary else "background_scale")
        result.append(Face(metrics, prominence, primary))
    return tuple(result), diagnostics


def photo_input(path, faces, blur, width=None, height=None, *, rows=False):
    extracted, _ = extract_faces(faces, width, height, rows=rows)
    return Photo(path, extracted, technical_feature(blur))
