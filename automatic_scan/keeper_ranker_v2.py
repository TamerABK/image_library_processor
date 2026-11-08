"""Pure, shadow-only keeper scoring from already measured features."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite


VERSION = "keeper-v2.2-evidence-v1-raw-log-v1"
# Operational reporting thresholds in score points, provisional without labels.
NEAR_TIE_POINTS = .5
MEASUREMENT_RESOLUTION_POINTS = .15
FACE_WEIGHTS = {"sharpness": .30, "eyes": .23, "pose": .15,
                "exposure": .10, "contrast": .07, "detail": .05,
                "detector": .04, "visible": .02, "reliability": .04}
PRIORS = {"sharpness": .45, "eyes": .55, "pose": .50, "exposure": .55,
          "contrast": .50, "detail": .45, "detector": .50,
          "visible": .50, "reliability": .50}
VISUAL = frozenset(("sharpness", "pose", "exposure", "contrast", "detail", "visible"))
# Provisional coverage gates. Supporting detector/reliability cannot establish sufficiency.
MIN_VISUAL_COVERAGE = .5
MIN_EYE_COVERAGE = .1
MIN_METRIC_CONFIDENCE = .25


@dataclass(frozen=True)
class Feature:
    value: float | None
    confidence: float = 1.0

    def usable(self) -> bool:
        return (self.value is not None and isfinite(self.value)
                and 0 <= self.value <= 1 and isfinite(self.confidence)
                and self.confidence > 0)


@dataclass(frozen=True)
class Face:
    features: dict[str, Feature]
    prominence: float
    primary: bool = True


@dataclass(frozen=True)
class Penalty:
    kind: str
    category: str  # eye, face, or technical
    severity: float
    confidence: float


@dataclass(frozen=True)
class Photo:
    image_id: str
    faces: tuple[Face, ...]
    technical: Feature
    penalties: tuple[Penalty, ...] = ()


@dataclass(frozen=True)
class Preferences:
    prioritize_primary_face_quality: bool = True
    prefer_open_eyes: bool = True
    acceptable_face_score: float = .55
    penalty_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    image_id: str
    score: float | None
    base_score: float | None
    penalty: float
    face_score: float | None
    technical_score: float | None
    weighted_primary_mean: float | None
    primary_p20: float | None
    acceptable_primary_fraction: float | None
    face_weight: float
    technical_weight: float
    used_face_features: tuple[str, ...]
    uncertain: bool
    explanation: str
    tie_class: str = "distinct"
    margin_to_next: float | None = None
    quality_preference: bool = True
    technical_discriminative: bool = True
    face_evidence_incomplete: bool = False
    face_evidence_sufficient: bool = False
    eye_coverage: float = 0.0
    face_coverage: float = 0.0
    uncertainty_reasons: tuple[str, ...] = ()
    enabled_components: tuple[str, ...] = ()
    unavailable_components: tuple[str, ...] = ()


def _coverage(faces: tuple[Face, ...], key: str) -> float:
    total = sum(max(.25, min(2., f.prominence)) for f in faces)
    return (sum(max(.25, min(2., f.prominence)) * min(1., f.features[key].confidence)
                for f in faces if key in f.features and f.features[key].usable()
                and f.features[key].confidence >= MIN_METRIC_CONFIDENCE) / total if total else 0.)


def _face_value(face: Face, weights: dict[str, float] | tuple[str, ...],
                priors: dict[str, float] | None = None) -> float:
    if not isinstance(weights, dict):
        weights = {key: FACE_WEIGHTS[key] for key in weights}
    priors = PRIORS if priors is None else priors
    weight = sum(weights.values())
    total = 0.
    for key, share in weights.items():
        feature = face.features.get(key, Feature(None))
        confidence = min(1., feature.confidence) if feature.usable() and feature.confidence >= MIN_METRIC_CONFIDENCE else 0.
        total += share * (confidence * feature.value + (1. - confidence) * priors[key]) if confidence else share * priors[key]
    return total / weight


def _aggregate(faces: tuple[Face, ...], weights: dict[str, float], priors: dict[str, float], threshold: float):
    values = sorted(((_face_value(face, weights, priors), face) for face in faces), key=lambda item: item[0])
    weights = [min(2.0, max(.25, face.prominence)) for _, face in values]
    mean = sum(value * weight for (value, _), weight in zip(values, weights)) / sum(weights)
    position = .2 * (len(values) - 1)
    low, high = int(position), min(len(values) - 1, int(position) + 1)
    p20 = values[low][0] * (high - position) + values[high][0] * (position - low) if high != low else values[low][0]
    acceptable = sum(value >= threshold for value, _ in values) / len(values)
    return .55 * mean + .25 * p20 + .20 * acceptable, mean, p20, acceptable


def rank_group(photos: tuple[Photo, ...], preferences: Preferences = Preferences()) -> list[Score]:
    """Use the same available component set for every candidate in one group."""
    if not photos:
        return []
    face_enabled = preferences.prioritize_primary_face_quality
    eye_enabled = face_enabled and preferences.prefer_open_eyes
    faces_by_photo = ([tuple(face for face in photo.faces if face.primary) for photo in photos]
                      if face_enabled else [() for _ in photos])
    coverage = ({key: [ _coverage(faces, key) for faces in faces_by_photo]
                 for key in FACE_WEIGHTS if key != "reliability"}
                if face_enabled and all(faces_by_photo) else {})
    visual = tuple(sorted(key for key in VISUAL if coverage and
                          min(coverage[key]) >= MIN_VISUAL_COVERAGE))
    # One weak visual proxy cannot activate the entire 40/55 face component.
    face_available = bool(len(visual) >= 2 and
                          ("sharpness" in visual or "pose" in visual))
    eye_coverage = min(coverage["eyes"]) if coverage else 0.
    eye_available = bool(face_available and eye_enabled and eye_coverage >= MIN_EYE_COVERAGE)
    common = tuple(sorted((*visual, "detector" if coverage and
                           min(coverage["detector"]) >= MIN_VISUAL_COVERAGE else "") +
                          (("eyes",) if eye_available else ()))) if face_available else ()
    common = tuple(key for key in common if key)
    weights = {key: FACE_WEIGHTS[key] * (eye_coverage if key == "eyes" else 1.)
               for key in common}
    priors = dict(PRIORS)
    if eye_available:
        observed = [(face.features["eyes"].value, min(1., face.features["eyes"].confidence))
                    for faces in faces_by_photo for face in faces if
                    face.features.get("eyes", Feature(None)).usable() and
                    face.features["eyes"].confidence >= MIN_METRIC_CONFIDENCE]
        priors["eyes"] = sum(v * c for v, c in observed) / sum(c for _, c in observed)
    technical_available = all(photo.technical.usable() for photo in photos)
    face_weight = 40 / 55 if face_available and technical_available else float(face_available)
    technical_weight = 1 - face_weight if technical_available else 0.0
    result = []
    for photo, faces in zip(photos, faces_by_photo):
        aggregate = _aggregate(faces, weights, priors, preferences.acceptable_face_score) if face_available else None
        technical = photo.technical.value if technical_available else None
        base = (100 * ((face_weight * aggregate[0] if aggregate else 0.0)
                        + (technical_weight * technical if technical is not None else 0.0))
                 if face_available or technical_available else None)
        penalty = sum(preferences.penalty_weights.get(item.kind, 0.0) * item.severity
                      for item in photo.penalties if item.confidence >= .8
                      and item.category in ({"technical"} if not preferences.prioritize_primary_face_quality
                                            else ({"technical", "face", "eye"} if preferences.prefer_open_eyes
                                                  else {"technical", "face"})))
        final = max(0.0, base - penalty) if base is not None else None
        reasons = []
        if face_enabled and not face_available:
            reasons.append("insufficient_visual_face_evidence")
        if eye_enabled and face_available and not eye_available:
            reasons.append("eye_coverage_insufficient")
        if face_available and any(_coverage(faces, key) < .999 for key in common):
            reasons.append("face_measurements_partial")
        if not technical_available:
            reasons.append("technical_unavailable")
        elif photo.technical.confidence < .5:
            reasons.append("technical_low_confidence")
        incomplete = bool(face_enabled and any(r.startswith(("insufficient_visual", "eye_", "face_")) for r in reasons))
        result.append(Score(photo.image_id, final, base, penalty,
                            aggregate[0] if aggregate else None, technical,
                            aggregate[1] if aggregate else None,
                            aggregate[2] if aggregate else None,
                            aggregate[3] if aggregate else None,
                            face_weight, technical_weight, common,
                            bool(reasons),
                            "face and technical" if face_available and technical_available else
                            "technical only" if technical_available else
                            "face only" if face_available else "quality unavailable",
                            face_evidence_incomplete=incomplete,
                            face_evidence_sufficient=face_available,
                            eye_coverage=_coverage(faces, "eyes") if eye_enabled else 0.,
                            face_coverage=min((_coverage(faces, key) for key in visual), default=0.) if face_enabled else 0.,
                            uncertainty_reasons=tuple(reasons),
                            enabled_components=(("face",) if face_enabled else ()) + ("technical",),
                            unavailable_components=(("face",) if face_enabled and not face_available else ()) +
                                                   (("eyes",) if eye_enabled and not eye_available else ()) +
                                                   (("technical",) if not technical_available else ())))
    result.sort(key=lambda item: (item.score is not None,
                                 item.score if item.score is not None else 0,
                                 item.image_id), reverse=True)
    technical_values = [item.technical_score for item in result if item.technical_score is not None]
    discriminative = (len(technical_values) > 1 and
                      max(technical_values) - min(technical_values) >= .002)
    annotated = []
    for index, item in enumerate(result):
        following = result[index + 1] if index + 1 < len(result) else None
        margin = (item.score - following.score if following and item.score is not None
                  and following.score is not None else None)
        tie = ("unavailable" if item.score is None else "last" if margin is None else
               "exact" if margin == 0 else "below_resolution" if margin < MEASUREMENT_RESOLUTION_POINTS else
               "near" if margin < NEAR_TIE_POINTS else "distinct")
        tie_reasons = item.uncertainty_reasons + (("near_tie",) if tie in ("exact", "below_resolution", "near") else ())
        annotated.append(replace(item, tie_class=tie, margin_to_next=margin,
                                 quality_preference=bool(margin is not None and
                                                         margin >= MEASUREMENT_RESOLUTION_POINTS),
                                 technical_discriminative=discriminative,
                                 uncertain=bool(tie_reasons), uncertainty_reasons=tie_reasons))
    return annotated
