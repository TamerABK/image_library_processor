from __future__ import annotations

from dataclasses import dataclass, field

from .ranking import MetricPriorConfig, RankingCalibrationProfile, RankingWeights


SelectionWeights = RankingWeights


@dataclass(frozen=True)
class EmbeddingWeights:
    """Weights used to judge recognition usefulness.

    Eye state is intentionally absent: a blink can still produce a useful
    identity embedding.
    """

    sharpness: float = 0.48
    pose: float = 0.27
    visible_face: float = 0.15
    detector_confidence: float = 0.10


@dataclass(frozen=True)
class ExposureConfig:
    dark_clip_value: float = 5.0
    bright_clip_value: float = 250.0

    dark_clip_start: float = 0.015
    dark_clip_end: float = 0.20
    bright_clip_start: float = 0.010
    bright_clip_end: float = 0.14

    median_shadow_low: float = 22.0
    median_shadow_good: float = 68.0
    median_highlight_good: float = 202.0
    median_highlight_high: float = 238.0
    shadow_detail_scale: float = 68.0
    highlight_detail_scale: float = 53.0
    median_usability_weight: float = 0.50
    shadow_detail_weight: float = 0.25
    highlight_detail_weight: float = 0.25

    tonal_range_low: float = 18.0
    tonal_range_good: float = 58.0

    clipping_weight: float = 0.50
    luminance_weight: float = 0.35
    tonal_information_weight: float = 0.15
    score_gamma: float = 1.25


@dataclass(frozen=True)
class ExposurePenaltyConfig:
    acceptable_threshold: float = 0.78
    maximum_penalty: float = 0.12
    exponent: float = 1.5


@dataclass(frozen=True)
class ContrastConfig:
    broad_range_scale: float = 48.0
    interquartile_range_scale: float = 24.0
    local_contrast_scale: float = 16.0
    broad_weight: float = 0.40
    interquartile_weight: float = 0.30
    local_weight: float = 0.30
    tile_rows: int = 4
    tile_cols: int = 4
    tile_mask_coverage_threshold: float = 0.45


@dataclass(frozen=True)
class SharpnessConfig:
    focus_log_laplacian_center: float = 145.0
    focus_log_laplacian_width: float = 0.72
    focus_log_tenengrad_center: float = 820.0
    focus_log_tenengrad_width: float = 0.78
    high_frequency_ratio_center: float = 0.30
    high_frequency_ratio_width: float = 0.18
    laplacian_weight: float = 0.46
    tenengrad_weight: float = 0.34
    high_frequency_weight: float = 0.20
    detail_low_dimension: float = 36.0
    detail_good_dimension: float = 104.0


@dataclass(frozen=True)
class FaceAnalyzerConfig:
    aligned_face_size: int = 160
    minimum_reliable_face_size: int = 48

    # Open Model Zoo open-closed-eye-0001 thresholds.
    eye_open_threshold: float = 0.70
    eye_closed_threshold: float = 0.88
    eye_minimum_decision_confidence: float = 0.60
    eye_low_signal_confidence_threshold: float = 0.18

    # The source crop should stay smaller than the 32 x 32 model input.
    eye_source_min_width: int = 12
    eye_source_min_height: int = 8
    eye_model_input_size: int = 32
    eye_crop_width_ratio: float = 0.58
    eye_crop_height_ratio: float = 0.38
    eye_neutral_score: float = 0.60
    eye_probability_rounding_decimals: int = 4
    eye_minimum_weight_at_high_yaw: float = 0.12
    eye_full_weight_yaw_degrees: float = 25.0
    eye_minimal_weight_yaw_degrees: float = 55.0

    # Do not force semantic decisions for highly oblique faces.
    maximum_eye_yaw_degrees: float = 46.0
    maximum_eye_pitch_degrees: float = 38.0

    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    sharpness: SharpnessConfig = field(default_factory=SharpnessConfig)
    ranking_weights: RankingWeights = field(default_factory=RankingWeights)
    metric_priors: MetricPriorConfig = field(default_factory=MetricPriorConfig)
    calibration_profile: RankingCalibrationProfile = field(
        default_factory=RankingCalibrationProfile
    )
    embedding_weights: EmbeddingWeights = field(default_factory=EmbeddingWeights)

    @property
    def selection_weights(self) -> RankingWeights:
        return self.ranking_weights
