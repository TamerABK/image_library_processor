from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ContrastConfig, ExposureConfig, SharpnessConfig
from .geometry import to_gray
from .math_utils import clamp01, descending_smoothstep, sigmoid, smoothstep
from .models import FaceImageQuality, MetricResult
from .ranking import RankingCalibrationProfile, apply_calibration_curve


@dataclass(frozen=True)
class LegacyTonalMetrics:
    exposure_score: float
    contrast_score: float


def central_face_mask(
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(width // 2, int(round(height * 0.52))),
        axes=(max(1, int(width * 0.34)), max(1, int(height * 0.43))),
        angle=0.0,
        startAngle=0.0,
        endAngle=360.0,
        color=1,
        thickness=-1,
    )
    return mask.astype(bool)


def shape_quality_score(
    score: float,
    *,
    gamma: float = 1.25,
) -> float:
    score = clamp01(score)
    gamma = max(float(gamma), 1e-6)
    return float(score ** gamma)


def exposure_penalty(
    score: float,
    *,
    acceptable_threshold: float = 0.78,
    maximum_penalty: float = 0.12,
    exponent: float = 1.5,
) -> float:
    score = clamp01(score)
    acceptable_threshold = max(float(acceptable_threshold), 1e-6)
    maximum_penalty = max(0.0, float(maximum_penalty))
    exponent = max(1e-6, float(exponent))

    if score >= acceptable_threshold:
        return 0.0

    deficiency = (acceptable_threshold - score) / acceptable_threshold
    return float(
        min(maximum_penalty, maximum_penalty * (deficiency ** exponent))
    )


def legacy_tonal_metrics(
    face_luminance: np.ndarray,
) -> LegacyTonalMetrics:
    pixels = np.asarray(face_luminance, dtype=np.float32).reshape(-1)
    if pixels.size == 0:
        return LegacyTonalMetrics(exposure_score=0.0, contrast_score=0.0)

    p05, p10, p50, p90, p95 = np.percentile(
        pixels,
        [5, 10, 50, 90, 95],
    ).astype(float)
    dark_clip_ratio = float(np.mean(pixels <= 5.0))
    bright_clip_ratio = float(np.mean(pixels >= 250.0))

    underexposure_penalty = 1.0 - smoothstep(25.0, 58.0, p50)
    overexposure_penalty = smoothstep(198.0, 232.0, p50)
    luminance_score = clamp01(
        1.0 - max(underexposure_penalty, overexposure_penalty)
    )

    clipping_penalty = clamp01(
        0.65 * smoothstep(0.015, 0.16, dark_clip_ratio)
        + 0.65 * smoothstep(0.010, 0.12, bright_clip_ratio)
    )
    exposure_score = clamp01(
        0.82 * luminance_score + 0.18 * (1.0 - clipping_penalty)
    )

    robust_range = max(0.0, p95 - p05)
    central_range = max(0.0, p90 - p10)
    standard_deviation = float(np.std(pixels))
    contrast_score = clamp01(
        0.40 * smoothstep(28.0, 78.0, robust_range)
        + 0.35 * smoothstep(22.0, 62.0, central_range)
        + 0.25 * smoothstep(16.0, 46.0, standard_deviation)
    )

    return LegacyTonalMetrics(
        exposure_score=exposure_score,
        contrast_score=contrast_score,
    )


def _normalize_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    clipped = tuple(max(0.0, float(weight)) for weight in weights)
    total = sum(clipped)
    if total <= 0.0:
        uniform = 1.0 / max(len(clipped), 1)
        return tuple(uniform for _ in clipped)
    return tuple(weight / total for weight in clipped)


def exposure_diagnostics(
    face_luminance: np.ndarray,
    *,
    config: ExposureConfig,
) -> dict[str, float]:
    pixels = np.asarray(face_luminance, dtype=np.float32).reshape(-1)
    if pixels.size == 0:
        return {
            "median_luminance": 0.0,
            "p05_luminance": 0.0,
            "p10_luminance": 0.0,
            "p50_luminance": 0.0,
            "p90_luminance": 0.0,
            "p95_luminance": 0.0,
            "dark_clip_ratio": 0.0,
            "bright_clip_ratio": 0.0,
            "usable_tonal_range": 0.0,
            "clipping_score": 0.0,
            "luminance_score": 0.0,
            "tonal_information_score": 0.0,
            "raw_exposure_score": 0.0,
            "display_exposure_score": 0.0,
            "exposure_score": 0.0,
            "shadow_detail_score": 0.0,
            "highlight_detail_score": 0.0,
            "tonal_balance_score": 0.0,
            "exposure_ranking_basis": 0.0,
        }

    p05, p10, p50, p90, p95 = np.percentile(
        pixels,
        [5, 10, 50, 90, 95],
    ).astype(float)
    dark_clip_ratio = float(np.mean(pixels <= config.dark_clip_value))
    bright_clip_ratio = float(np.mean(pixels >= config.bright_clip_value))

    shadow_usability = smoothstep(
        config.median_shadow_low,
        config.median_shadow_good,
        p50,
    )
    highlight_usability = descending_smoothstep(
        config.median_highlight_good,
        config.median_highlight_high,
        p50,
    )
    median_usability = shadow_usability * highlight_usability
    shadow_detail_measure = p05 / (
        p05 + max(float(config.shadow_detail_scale), 1e-6)
    )
    highlight_detail_measure = (255.0 - p95) / (
        (255.0 - p95) + max(float(config.highlight_detail_scale), 1e-6)
    )
    (
        median_usability_weight,
        shadow_detail_weight,
        highlight_detail_weight,
    ) = _normalize_weights(
        (
            config.median_usability_weight,
            config.shadow_detail_weight,
            config.highlight_detail_weight,
        )
    )
    luminance_score = clamp01(
        median_usability_weight * median_usability
        + shadow_detail_weight * shadow_detail_measure
        + highlight_detail_weight * highlight_detail_measure
    )

    dark_clip_penalty = smoothstep(
        config.dark_clip_start,
        config.dark_clip_end,
        dark_clip_ratio,
    )
    bright_clip_penalty = smoothstep(
        config.bright_clip_start,
        config.bright_clip_end,
        bright_clip_ratio,
    )
    clipping_score = clamp01(
        1.0 - 0.55 * dark_clip_penalty - 0.65 * bright_clip_penalty
    )

    usable_tonal_range = max(0.0, p95 - p05)
    tonal_information_score = smoothstep(
        config.tonal_range_low,
        config.tonal_range_good,
        usable_tonal_range,
    ) * (
        usable_tonal_range
        / (usable_tonal_range + max(float(config.tonal_range_good), 1e-6))
    )

    clipping_weight, luminance_weight, tonal_information_weight = (
        _normalize_weights(
            (
                config.clipping_weight,
                config.luminance_weight,
                config.tonal_information_weight,
            )
        )
    )
    raw_exposure_score = clamp01(
        clipping_weight * clipping_score
        + luminance_weight * luminance_score
        + tonal_information_weight * tonal_information_score
    )
    display_exposure_score = shape_quality_score(
        raw_exposure_score,
        gamma=config.score_gamma,
    )

    shadow_detail_score = 1.0 - smoothstep(
        config.dark_clip_start,
        config.dark_clip_end,
        dark_clip_ratio,
    )
    highlight_detail_score = 1.0 - smoothstep(
        config.bright_clip_start,
        config.bright_clip_end,
        bright_clip_ratio,
    )
    lower_span = max(0.0, p50 - p10)
    upper_span = max(0.0, p90 - p50)
    combined_span = lower_span + upper_span
    imbalance = (
        0.0
        if combined_span <= 1e-6
        else abs(lower_span - upper_span) / combined_span
    )
    tonal_balance_score = clamp01(
        1.0 - 0.35 * smoothstep(0.35, 0.90, imbalance)
    )
    exposure_ranking_basis = clamp01(
        0.78 * raw_exposure_score
        + 0.12 * ((shadow_detail_score + highlight_detail_score) * 0.5)
        + 0.10 * tonal_balance_score
    )

    return {
        "median_luminance": p50,
        "p05_luminance": p05,
        "p10_luminance": p10,
        "p50_luminance": p50,
        "p90_luminance": p90,
        "p95_luminance": p95,
        "dark_clip_ratio": dark_clip_ratio,
        "bright_clip_ratio": bright_clip_ratio,
        "usable_tonal_range": usable_tonal_range,
        "clipping_score": clipping_score,
        "luminance_score": clamp01(luminance_score),
        "tonal_information_score": tonal_information_score,
        "raw_exposure_score": raw_exposure_score,
        "display_exposure_score": display_exposure_score,
        "exposure_score": display_exposure_score,
        "shadow_detail_score": shadow_detail_score,
        "highlight_detail_score": highlight_detail_score,
        "tonal_balance_score": tonal_balance_score,
        "exposure_ranking_basis": exposure_ranking_basis,
    }


def _tile_local_contrast(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    config: ContrastConfig,
) -> float:
    height, width = gray.shape
    tile_height = max(1, height // max(config.tile_rows, 1))
    tile_width = max(1, width // max(config.tile_cols, 1))

    tile_standard_deviations: list[float] = []
    for row_index in range(config.tile_rows):
        start_y = row_index * tile_height
        end_y = height if row_index == config.tile_rows - 1 else (row_index + 1) * tile_height
        for col_index in range(config.tile_cols):
            start_x = col_index * tile_width
            end_x = width if col_index == config.tile_cols - 1 else (col_index + 1) * tile_width
            tile_mask = mask[start_y:end_y, start_x:end_x]
            if tile_mask.size == 0:
                continue
            if float(np.mean(tile_mask)) < config.tile_mask_coverage_threshold:
                continue
            tile_pixels = gray[start_y:end_y, start_x:end_x][tile_mask]
            if tile_pixels.size < 4:
                continue
            tile_standard_deviations.append(float(np.std(tile_pixels)))

    if not tile_standard_deviations:
        return 0.0
    return float(np.median(tile_standard_deviations))


def contrast_diagnostics(
    face_luminance: np.ndarray,
    *,
    gray_image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    config: ContrastConfig,
) -> dict[str, float]:
    pixels = np.asarray(face_luminance, dtype=np.float32).reshape(-1)
    if pixels.size == 0:
        return {
            "p10_luminance": 0.0,
            "p25_luminance": 0.0,
            "p50_luminance": 0.0,
            "p75_luminance": 0.0,
            "p90_luminance": 0.0,
            "broad_tonal_range": 0.0,
            "interquartile_range": 0.0,
            "broad_contrast_score": 0.0,
            "interquartile_contrast_score": 0.0,
            "local_contrast_raw": 0.0,
            "local_contrast_score": 0.0,
            "contrast_quality_score": 0.0,
            "contrast_score": 0.0,
        }

    p10, p25, p50, p75, p90 = np.percentile(
        pixels,
        [10, 25, 50, 75, 90],
    ).astype(float)

    broad_tonal_range = max(0.0, p90 - p10)
    interquartile_range = max(0.0, p75 - p25)
    broad_contrast_score = broad_tonal_range / (
        broad_tonal_range + max(float(config.broad_range_scale), 1e-6)
    )
    interquartile_contrast_score = interquartile_range / (
        interquartile_range + max(float(config.interquartile_range_scale), 1e-6)
    )

    local_contrast_raw = 0.0
    if gray_image is not None and mask is not None:
        local_contrast_raw = _tile_local_contrast(
            gray_image,
            mask,
            config=config,
        )
    else:
        local_contrast_raw = float(np.std(pixels))
    local_contrast_score = local_contrast_raw / (
        local_contrast_raw + max(float(config.local_contrast_scale), 1e-6)
    )

    broad_weight, interquartile_weight, local_weight = _normalize_weights(
        (
            config.broad_weight,
            config.interquartile_weight,
            config.local_weight,
        )
    )
    contrast_quality_score = clamp01(
        broad_weight * broad_contrast_score
        + interquartile_weight * interquartile_contrast_score
        + local_weight * local_contrast_score
    )

    return {
        "p10_luminance": p10,
        "p25_luminance": p25,
        "p50_luminance": p50,
        "p75_luminance": p75,
        "p90_luminance": p90,
        "broad_tonal_range": broad_tonal_range,
        "interquartile_range": interquartile_range,
        "broad_contrast_score": clamp01(broad_contrast_score),
        "interquartile_contrast_score": clamp01(interquartile_contrast_score),
        "local_contrast_raw": local_contrast_raw,
        "local_contrast_score": clamp01(local_contrast_score),
        "contrast_quality_score": contrast_quality_score,
        "contrast_score": contrast_quality_score,
    }


class ClassicalFaceQualityAssessor:
    """Fixed-scale, interpretable measurements from an aligned face crop.

    Absolute quality stays stable and interpretable. Ranking calibration stays
    separate so the same face score does not change when an unrelated folder is added.
    """

    def __init__(
        self,
        *,
        normalized_size: int = 160,
        exposure_config: ExposureConfig | None = None,
        contrast_config: ContrastConfig | None = None,
        sharpness_config: SharpnessConfig | None = None,
        calibration_profile: RankingCalibrationProfile | None = None,
    ) -> None:
        self._normalized_size = max(96, int(normalized_size))
        self._exposure_config = exposure_config or ExposureConfig()
        self._contrast_config = contrast_config or ContrastConfig()
        self._sharpness_config = sharpness_config or SharpnessConfig()
        self._calibration_profile = calibration_profile or RankingCalibrationProfile()

    def assess(
        self,
        aligned_face: np.ndarray,
        *,
        original_face_minimum_dimension: int,
        alignment_confidence: float,
    ) -> FaceImageQuality:
        if aligned_face.size == 0:
            unavailable = MetricResult(
                raw_value=None,
                quality_score=None,
                confidence=0.0,
                ranking_score=None,
            )
            return FaceImageQuality(
                focus_sharpness=unavailable,
                detail_availability=unavailable,
                sharpness=unavailable,
                exposure=unavailable,
                contrast=unavailable,
                laplacian_variance=0.0,
                tenengrad_energy=0.0,
                high_frequency_energy_ratio=0.0,
                detail_availability_measure=0.0,
                median_luminance=0.0,
                p05_luminance=0.0,
                p95_luminance=0.0,
                dark_clip_ratio=0.0,
                bright_clip_ratio=0.0,
                usable_tonal_range=0.0,
                clipping_score=0.0,
                luminance_score=0.0,
                tonal_information_score=0.0,
                raw_exposure_score=0.0,
                display_exposure_score=0.0,
                shadow_detail_score=0.0,
                highlight_detail_score=0.0,
                tonal_balance_score=0.0,
                p10_luminance=0.0,
                p25_luminance=0.0,
                p75_luminance=0.0,
                p90_luminance=0.0,
                broad_tonal_range=0.0,
                interquartile_range=0.0,
                broad_contrast_score=0.0,
                interquartile_contrast_score=0.0,
                local_contrast_raw=0.0,
                local_contrast_score=0.0,
                contrast_quality_score=0.0,
            )

        resized = cv2.resize(
            aligned_face,
            (self._normalized_size, self._normalized_size),
            interpolation=(
                cv2.INTER_AREA
                if max(aligned_face.shape[:2]) > self._normalized_size
                else cv2.INTER_CUBIC
            ),
        )
        gray = to_gray(resized).astype(np.float32)
        mask = central_face_mask(*gray.shape)
        pixels = gray[mask]

        detail_availability_measure = float(original_face_minimum_dimension)
        detail_availability_score = smoothstep(
            self._sharpness_config.detail_low_dimension,
            self._sharpness_config.detail_good_dimension,
            detail_availability_measure,
        )
        detail_availability_ranking = apply_calibration_curve(
            detail_availability_score,
            self._calibration_profile.detail_availability,
        )
        detail_availability_confidence = clamp01(alignment_confidence)

        # Mild smoothing limits false sharpness from noise and JPEG ringing.
        filtered = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.55)
        laplacian = cv2.Laplacian(filtered, cv2.CV_32F, ksize=3)
        laplacian_variance = float(np.var(laplacian[mask]))

        sobel_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
        gradient_energy = sobel_x * sobel_x + sobel_y * sobel_y
        tenengrad_energy = float(np.mean(gradient_energy[mask]))

        high_pass = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.1)
        high_frequency_energy_ratio = float(
            np.mean(np.abs(high_pass[mask])) / (np.std(gray[mask]) + 1e-6)
        )

        laplacian_score = sigmoid(
            (
                math.log1p(laplacian_variance)
                - math.log1p(self._sharpness_config.focus_log_laplacian_center)
            )
            / self._sharpness_config.focus_log_laplacian_width
        )
        tenengrad_score = sigmoid(
            (
                math.log1p(tenengrad_energy)
                - math.log1p(self._sharpness_config.focus_log_tenengrad_center)
            )
            / self._sharpness_config.focus_log_tenengrad_width
        )
        high_frequency_score = sigmoid(
            (
                high_frequency_energy_ratio
                - self._sharpness_config.high_frequency_ratio_center
            )
            / max(self._sharpness_config.high_frequency_ratio_width, 1e-6)
        )
        focus_sharpness_score = clamp01(
            self._sharpness_config.laplacian_weight * laplacian_score
            + self._sharpness_config.tenengrad_weight * tenengrad_score
            + self._sharpness_config.high_frequency_weight * high_frequency_score
        )
        focus_ranking_score = apply_calibration_curve(
            focus_sharpness_score,
            self._calibration_profile.focus_sharpness,
        )

        exposure = exposure_diagnostics(
            pixels,
            config=self._exposure_config,
        )
        contrast = contrast_diagnostics(
            pixels,
            gray_image=gray,
            mask=mask,
            config=self._contrast_config,
        )

        signal_confidence = 1.0 - 0.60 * clamp01(
            exposure["dark_clip_ratio"] + exposure["bright_clip_ratio"]
        )
        focus_confidence = clamp01(signal_confidence * alignment_confidence)
        tonal_confidence = clamp01(signal_confidence * alignment_confidence)

        sharpness_ranking_score = clamp01(
            0.75 * (focus_ranking_score or 0.0)
            + 0.25 * (detail_availability_ranking or 0.0)
        )
        sharpness_quality_score = clamp01(
            0.75 * focus_sharpness_score + 0.25 * detail_availability_score
        )

        exposure_ranking_score = apply_calibration_curve(
            exposure["exposure_ranking_basis"],
            self._calibration_profile.exposure,
        )
        contrast_ranking_score = apply_calibration_curve(
            contrast["contrast_quality_score"],
            self._calibration_profile.contrast,
        )

        return FaceImageQuality(
            focus_sharpness=MetricResult(
                raw_value=focus_sharpness_score,
                quality_score=focus_sharpness_score,
                confidence=focus_confidence,
                ranking_score=focus_ranking_score,
            ),
            detail_availability=MetricResult(
                raw_value=detail_availability_measure,
                quality_score=detail_availability_score,
                confidence=detail_availability_confidence,
                ranking_score=detail_availability_ranking,
            ),
            sharpness=MetricResult(
                raw_value=focus_sharpness_score,
                quality_score=sharpness_quality_score,
                confidence=focus_confidence,
                ranking_score=sharpness_ranking_score,
            ),
            exposure=MetricResult(
                raw_value=exposure["raw_exposure_score"],
                quality_score=exposure["display_exposure_score"],
                confidence=tonal_confidence,
                ranking_score=exposure_ranking_score,
            ),
            contrast=MetricResult(
                raw_value=contrast["contrast_quality_score"],
                quality_score=contrast["contrast_quality_score"],
                confidence=tonal_confidence,
                ranking_score=contrast_ranking_score,
            ),
            laplacian_variance=laplacian_variance,
            tenengrad_energy=tenengrad_energy,
            high_frequency_energy_ratio=high_frequency_energy_ratio,
            detail_availability_measure=detail_availability_measure,
            median_luminance=exposure["median_luminance"],
            p05_luminance=exposure["p05_luminance"],
            p95_luminance=exposure["p95_luminance"],
            dark_clip_ratio=exposure["dark_clip_ratio"],
            bright_clip_ratio=exposure["bright_clip_ratio"],
            usable_tonal_range=exposure["usable_tonal_range"],
            clipping_score=exposure["clipping_score"],
            luminance_score=exposure["luminance_score"],
            tonal_information_score=exposure["tonal_information_score"],
            raw_exposure_score=exposure["raw_exposure_score"],
            display_exposure_score=exposure["display_exposure_score"],
            shadow_detail_score=exposure["shadow_detail_score"],
            highlight_detail_score=exposure["highlight_detail_score"],
            tonal_balance_score=exposure["tonal_balance_score"],
            p10_luminance=contrast["p10_luminance"],
            p25_luminance=contrast["p25_luminance"],
            p75_luminance=contrast["p75_luminance"],
            p90_luminance=contrast["p90_luminance"],
            broad_tonal_range=contrast["broad_tonal_range"],
            interquartile_range=contrast["interquartile_range"],
            broad_contrast_score=contrast["broad_contrast_score"],
            interquartile_contrast_score=contrast["interquartile_contrast_score"],
            local_contrast_raw=contrast["local_contrast_raw"],
            local_contrast_score=contrast["local_contrast_score"],
            contrast_quality_score=contrast["contrast_quality_score"],
        )
