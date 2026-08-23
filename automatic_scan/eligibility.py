from __future__ import annotations

import math
from dataclasses import dataclass

from face_processing.models import DetectedFace

from .config import AutomaticScanConfig
from .models import ImageFaceQualityAssessment, PeoplePictureAssessment


@dataclass(frozen=True, slots=True)
class _FaceProminence:
    index: int
    prominence: float
    center_distance: float
    frame_span_fraction: float


def assess_people_picture(
    image_width: int | None,
    image_height: int | None,
    faces: list[DetectedFace],
    config: AutomaticScanConfig,
) -> PeoplePictureAssessment:
    if not faces:
        return PeoplePictureAssessment(
            is_people_picture=False,
            confidence=0.0,
            primary_face_indices=(),
            supporting_reasons=("no_faces_detected",),
            detected_face_count=0,
            reliable_face_count=0,
        )

    thresholds = config.people_picture
    prominences: list[_FaceProminence] = []
    supporting_reasons: list[str] = []

    for index, face in enumerate(faces):
        analysis = face.analysis
        if analysis is None:
            continue
        reliability = analysis.measurement_reliability_score
        if reliability < thresholds.minimum_face_reliability:
            continue

        bbox_x, bbox_y, bbox_w, bbox_h = face.bbox
        if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
            center_distance = 0.5
            frame_span_fraction = 0.0
        else:
            center_x = (bbox_x + (bbox_w / 2.0)) / image_width
            center_y = (bbox_y + (bbox_h / 2.0)) / image_height
            dx = center_x - 0.5
            dy = center_y - 0.5
            center_distance = min(math.sqrt(dx * dx + dy * dy) / 0.71, 1.0)
            frame_span_fraction = max(
                bbox_w / max(image_width, 1),
                bbox_h / max(image_height, 1),
            )

        centrality = 1.0 - center_distance
        prominence = (
            0.30 * analysis.relative_face_area
            + 0.18 * frame_span_fraction
            + 0.18 * analysis.visible_face_ratio
            + 0.16 * analysis.global_selection_score
            + 0.12 * reliability
            + 0.06 * centrality
        )
        prominences.append(
            _FaceProminence(
                index=index,
                prominence=prominence,
                center_distance=center_distance,
                frame_span_fraction=frame_span_fraction,
            )
        )

    if not prominences:
        return PeoplePictureAssessment(
            is_people_picture=False,
            confidence=0.0,
            primary_face_indices=(),
            supporting_reasons=("faces_not_reliable_enough",),
            detected_face_count=len(faces),
            reliable_face_count=0,
        )

    ordered = sorted(prominences, key=lambda item: item.prominence, reverse=True)
    strongest = ordered[0]
    primary_indices = tuple(
        item.index
        for item in ordered
        if item.prominence >= strongest.prominence * thresholds.primary_face_prominence_fraction
    )

    primary_faces = [
        faces[index]
        for index in primary_indices
        if faces[index].analysis is not None
    ]
    largest_area = max(
        (face.analysis.relative_face_area for face in primary_faces if face.analysis is not None),
        default=0.0,
    )
    total_area = sum(
        face.analysis.relative_face_area
        for face in primary_faces
        if face.analysis is not None
    )
    largest_span = max(
        (
            item.frame_span_fraction
            for item in ordered
            if item.index in primary_indices
        ),
        default=0.0,
    )
    average_centrality = sum(
        1.0 - item.center_distance
        for item in ordered
        if item.index in primary_indices
    ) / max(len(primary_indices), 1)
    best_primary_selection_score = max(
        (
            face.analysis.global_selection_score
            for face in primary_faces
            if face.analysis is not None
        ),
        default=None,
    )
    effective_total_area = total_area if len(primary_indices) >= 2 else 0.0
    base_confidence = max(
        0.0,
        min(
            1.0,
            0.35 * min(largest_area / max(thresholds.primary_face_area_threshold, 1e-6), 1.0)
            + 0.20 * min(largest_span / max(thresholds.primary_face_span_threshold, 1e-6), 1.0)
            + 0.20 * min(
                effective_total_area / max(thresholds.total_primary_face_area_threshold, 1e-6),
                1.0,
            )
            + 0.15 * average_centrality
            + 0.10 * min(len(primary_indices) / 2.0, 1.0),
        ),
    )
    confidence_multiplier = 1.0
    if largest_area >= thresholds.primary_face_area_threshold:
        supporting_reasons.append("largest_face_prominent")
    if largest_span >= thresholds.primary_face_span_threshold:
        supporting_reasons.append("face_scale_prominent")
    total_face_area_prominent = (
        len(primary_indices) >= 2
        and total_area >= thresholds.total_primary_face_area_threshold
    )
    if total_face_area_prominent:
        supporting_reasons.append("total_face_area_prominent")
    if average_centrality >= 0.55:
        supporting_reasons.append("faces_near_frame_center")
    if len(primary_indices) >= 2:
        supporting_reasons.append("multiple_primary_faces")

    if (
        len(primary_indices) >= 2
        and largest_span >= thresholds.multi_face_confidence_min_span
        and average_centrality >= 0.40
        and (best_primary_selection_score or 0.0) >= thresholds.multi_face_confidence_min_selection_score
    ):
        confidence_multiplier = min(
            1.0
            + thresholds.multi_face_confidence_multiplier_per_additional_face
            * max(len(primary_indices) - 1, 0),
            thresholds.multi_face_confidence_multiplier_cap,
        )
        if confidence_multiplier > 1.0:
            supporting_reasons.append("additional_face_count_multiplier")

    confidence = min(base_confidence * confidence_multiplier, 1.0)

    is_people_picture = bool(
        primary_indices
        and (
            largest_area >= thresholds.primary_face_area_threshold
            or largest_span >= thresholds.primary_face_span_threshold
            or total_face_area_prominent
            or confidence >= thresholds.people_picture_confidence_threshold
        )
    )
    return PeoplePictureAssessment(
        is_people_picture=is_people_picture,
        confidence=confidence,
        primary_face_indices=primary_indices,
        supporting_reasons=tuple(supporting_reasons) or ("faces_too_small_or_incidental",),
        detected_face_count=len(faces),
        reliable_face_count=len(prominences),
        largest_primary_face_area=largest_area,
        total_primary_face_area=total_area,
        largest_primary_face_span=largest_span,
        average_primary_centrality=average_centrality,
        best_primary_selection_score=best_primary_selection_score,
    )


def assess_face_quality(
    faces: list[DetectedFace],
    people_picture: PeoplePictureAssessment,
    config: AutomaticScanConfig,
) -> ImageFaceQualityAssessment:
    thresholds = config.face_quality_thresholds()
    if not people_picture.is_people_picture or not people_picture.primary_face_indices:
        if people_picture.is_people_picture and not people_picture.primary_face_indices:
            terminal_failure = thresholds.missing_primary_faces_terminal
            reasons = ("no_usable_primary_faces_detected",)
            warnings = ("no_primary_faces_detected_for_people_picture",)
            return ImageFaceQualityAssessment(
                is_people_picture=True,
                primary_face_count=0,
                acceptable_primary_face_count=0,
                weighted_primary_average=None,
                worst_primary_score=None,
                best_primary_score=None,
                closed_eye_primary_count=0,
                unusable_primary_count=0,
                terminal_failure=terminal_failure,
                warnings=warnings,
                reasons=reasons,
            )
        return ImageFaceQualityAssessment(
            is_people_picture=False,
            primary_face_count=0,
            acceptable_primary_face_count=0,
            weighted_primary_average=None,
            worst_primary_score=None,
            best_primary_score=None,
            closed_eye_primary_count=0,
            unusable_primary_count=0,
            terminal_failure=False,
            warnings=(),
            reasons=("not_a_people_picture",),
        )

    primary_faces = [
        faces[index]
        for index in people_picture.primary_face_indices
        if index < len(faces) and faces[index].analysis is not None
    ]
    if not primary_faces:
        return ImageFaceQualityAssessment(
            is_people_picture=True,
            primary_face_count=0,
            acceptable_primary_face_count=0,
            weighted_primary_average=None,
            worst_primary_score=None,
            best_primary_score=None,
            closed_eye_primary_count=0,
            unusable_primary_count=0,
            terminal_failure=False,
            warnings=("primary_faces_missing_analysis",),
            reasons=("primary_faces_missing_analysis",),
        )

    weights: list[float] = []
    scores: list[float] = []
    warnings: list[str] = []
    reasons: list[str] = []
    acceptable_count = 0
    unusable_count = 0
    closed_eye_count = 0
    dark_closed_eye_count = 0
    dark_primary_count = 0
    low_pick_score_count = 0

    for face in primary_faces:
        assert face.analysis is not None
        analysis = face.analysis
        score = analysis.global_selection_score
        pick_score = analysis.selection_score
        reliability = analysis.measurement_reliability_score
        weight = max(analysis.relative_face_area, 0.01) * max(reliability, 0.20)
        weights.append(weight)
        scores.append(score)
        if score >= thresholds.worst_primary_threshold:
            acceptable_count += 1
        if score < thresholds.unusable_primary_threshold:
            unusable_count += 1
        if pick_score < thresholds.minimum_primary_pick_score_threshold:
            low_pick_score_count += 1
            warnings.append("low_primary_pick_score")
        if reliability < thresholds.low_reliability_warning_threshold:
            warnings.append("low_primary_face_reliability")
        eye_open_score = analysis.combined_eye_open_score
        eye_compromised = analysis.eye_state.has_confident_closed_eye or (
            eye_open_score is not None
            and analysis.eye_confidence >= thresholds.eye_closed_confidence_threshold
            and eye_open_score < thresholds.minimum_primary_eye_open_score
        )
        if eye_compromised:
            closed_eye_count += 1
            warnings.append("closed_eye_primary_face")
        if analysis.exposure_score < thresholds.minimum_primary_exposure_score:
            dark_primary_count += 1
            warnings.append("low_primary_face_exposure")
        if analysis.contrast_score < thresholds.minimum_primary_contrast_score:
            warnings.append("low_primary_face_contrast")
        if analysis.pose_score is not None and analysis.pose_score < 0.36:
            warnings.append("extreme_pose_primary_face")
        if (
            eye_compromised
            and analysis.exposure_score < thresholds.minimum_primary_exposure_score
        ):
            dark_closed_eye_count += 1

    weighted_average = sum(score * weight for score, weight in zip(scores, weights)) / max(
        sum(weights),
        1e-6,
    )
    worst_score = min(scores)
    best_score = max(scores)
    terminal_failure = False

    if config.faces.hard_exclude_dark_primary_faces and dark_primary_count >= len(primary_faces):
        terminal_failure = True
        reasons.append("dark_primary_faces")
    elif unusable_count >= len(primary_faces):
        terminal_failure = True
        reasons.append("all_primary_faces_unusable")
    elif low_pick_score_count >= len(primary_faces):
        terminal_failure = True
        reasons.append("low_primary_pick_scores")
    elif (
        weighted_average < thresholds.weighted_average_threshold
        and worst_score < thresholds.worst_primary_threshold
    ):
        terminal_failure = True
        reasons.append("poor_primary_face_quality")

    small_group = len(primary_faces) <= 2
    if (
        thresholds.closed_eyes_terminal_for_small_group
        and small_group
        and closed_eye_count >= len(primary_faces)
    ):
        terminal_failure = True
        reasons.append("all_primary_faces_closed_eyes")
    elif (
        thresholds.dark_closed_eye_terminal_for_small_group
        and small_group
        and dark_closed_eye_count >= len(primary_faces)
    ):
        terminal_failure = True
        reasons.append("primary_faces_dark_and_eye_compromised")

    if acceptable_count and unacceptable_but_not_terminal(acceptable_count, len(primary_faces)):
        warnings.append("mixed_primary_face_quality")
    if worst_score < thresholds.worst_primary_threshold and not terminal_failure:
        warnings.append("weak_primary_face_present")
    if weighted_average < thresholds.weighted_average_threshold and not terminal_failure:
        warnings.append("primary_face_quality_penalty")

    return ImageFaceQualityAssessment(
        is_people_picture=True,
        primary_face_count=len(primary_faces),
        acceptable_primary_face_count=acceptable_count,
        weighted_primary_average=weighted_average,
        worst_primary_score=worst_score,
        best_primary_score=best_score,
        closed_eye_primary_count=closed_eye_count,
        unusable_primary_count=unusable_count,
        terminal_failure=terminal_failure,
        warnings=tuple(dict.fromkeys(warnings)),
        reasons=tuple(dict.fromkeys(reasons)) or ("acceptable_primary_faces_present",),
    )


def unacceptable_but_not_terminal(acceptable_count: int, total_count: int) -> bool:
    return acceptable_count > 0 and acceptable_count < total_count
