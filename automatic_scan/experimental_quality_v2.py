"""Auditable, disabled-by-default moment and framing experiments.

These observations never enter production or default shadow ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

VERSION = "moment-composition-experiment-v1"
WEIGHTS = {"face": .40, "moment": .25, "composition": .20, "technical": .15}


@dataclass(frozen=True)
class Observation:
    score: float | None
    confidence: float
    coverage: float
    available: bool
    reasons: tuple[str, ...]
    explanation: str


def moment_observation(*, expression=None, temporal=None) -> Observation:
    """No expression model or validated temporal signal is installed."""
    return Observation(None, 0., 0., False,
                       ("expression_model_unverified", "temporal_quality_unvalidated"),
                       "No validated moment-quality observation")


def composition_observation(rows, width, height) -> Observation:
    """Measure only definite edge clipping, with strict count comparability upstream."""
    if not width or not height or width <= 0 or height <= 0 or not rows:
        return Observation(None, 0., 0., False, ("geometry_unavailable",),
                           "Frame or face geometry unavailable")
    valid = []
    for row in rows:
        try:
            if int(row.get("analysis_missing", 0)):
                continue
            x, y, w, h = (float(row[k]) for k in
                          ("bbox_x", "bbox_y", "bbox_width", "bbox_height"))
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        # Margin under one percent of frame size is a possible clipping warning.
        clipped = x <= .01 * width or y <= .01 * height or \
                  x + w >= .99 * width or y + h >= .99 * height
        valid.append(clipped)
    if not valid:
        return Observation(None, 0., 0., False, ("face_geometry_unavailable",),
                           "No comparable face boxes")
    fraction = sum(valid) / len(valid)
    return Observation(1. - fraction, 1., len(valid) / len(rows), True, (),
                       f"{sum(valid)} of {len(valid)} detected faces near frame edges")


def experimental_score(baseline, moment: Observation, composition: Observation,
                       *, use_moment=False, use_composition=False):
    """Group caller must pass common availability; never score missing as zero."""
    if baseline.score is None:
        return None, ("baseline_unavailable",)
    if use_moment and not moment.available:
        return None, ("moment_unavailable",)
    if use_composition and not composition.available:
        return None, ("composition_unavailable",)
    if not use_moment and not use_composition:
        return baseline.score, ()
    active = {"technical": baseline.technical_score}
    if baseline.face_score is not None:
        active["face"] = baseline.face_score
    if use_moment:
        active["moment"] = moment.score
    if use_composition:
        active["composition"] = composition.score
    if any(value is None for value in active.values()):
        return None, ("component_unavailable",)
    total = sum(WEIGHTS[key] for key in active)
    return 100 * sum(WEIGHTS[key] * value for key, value in active.items()) / total - baseline.penalty, ()
