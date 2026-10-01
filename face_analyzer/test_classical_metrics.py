from __future__ import annotations

import unittest

import numpy as np

from .classical_metrics import (
    ClassicalFaceQualityAssessor,
    contrast_diagnostics,
    exposure_diagnostics,
    exposure_penalty,
    shape_quality_score,
)
from .config import ContrastConfig, ExposureConfig, ExposurePenaltyConfig
from .math_utils import clamp01, descending_smoothstep, smoothstep
from .ranking import CalibrationCurve, RankingCalibrationProfile


def _gradient(
    start: float,
    end: float,
    *,
    height: int = 48,
    width: int = 48,
) -> np.ndarray:
    row = np.linspace(start, end, width, dtype=np.float32)
    return np.tile(row, (height, 1))


class SmoothstepTests(unittest.TestCase):
    def test_descending_smoothstep_handles_descending_edges(self) -> None:
        self.assertGreater(
            descending_smoothstep(202.0, 238.0, 210.0),
            descending_smoothstep(202.0, 238.0, 236.0),
        )
        self.assertAlmostEqual(
            descending_smoothstep(202.0, 238.0, 202.0),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            descending_smoothstep(202.0, 238.0, 238.0),
            0.0,
            places=6,
        )

    def test_smoothstep_ascending_still_behaves_normally(self) -> None:
        self.assertLess(smoothstep(22.0, 68.0, 28.0), smoothstep(22.0, 68.0, 60.0))


class ExposureMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExposureConfig()

    def test_empty_pixels_return_zero_scores(self) -> None:
        diagnostics = exposure_diagnostics(np.asarray([], dtype=np.float32), config=self.config)
        self.assertEqual(diagnostics["raw_exposure_score"], 0.0)
        self.assertEqual(diagnostics["exposure_score"], 0.0)

    def test_mid_brightness_unclipped_face_is_high(self) -> None:
        diagnostics = exposure_diagnostics(_gradient(60.0, 170.0), config=self.config)
        self.assertGreater(diagnostics["raw_exposure_score"], 0.75)

    def test_very_dark_face_scores_lower(self) -> None:
        dark = exposure_diagnostics(_gradient(2.0, 24.0), config=self.config)
        normal = exposure_diagnostics(_gradient(60.0, 170.0), config=self.config)
        self.assertLess(dark["raw_exposure_score"], normal["raw_exposure_score"])

    def test_very_bright_face_scores_lower(self) -> None:
        bright = exposure_diagnostics(_gradient(226.0, 254.0), config=self.config)
        normal = exposure_diagnostics(_gradient(60.0, 170.0), config=self.config)
        self.assertLess(bright["raw_exposure_score"], normal["raw_exposure_score"])

    def test_dark_clipping_reduces_score(self) -> None:
        clipped = np.concatenate(
            [
                np.zeros((48, 24), dtype=np.float32),
                np.full((48, 24), 72.0, dtype=np.float32),
            ],
            axis=1,
        )
        diagnostics = exposure_diagnostics(clipped, config=self.config)
        self.assertLess(diagnostics["clipping_score"], 0.7)
        self.assertLess(diagnostics["raw_exposure_score"], 0.7)

    def test_highlight_clipping_reduces_score(self) -> None:
        clipped = np.concatenate(
            [
                np.full((48, 24), 180.0, dtype=np.float32),
                np.full((48, 24), 255.0, dtype=np.float32),
            ],
            axis=1,
        )
        diagnostics = exposure_diagnostics(clipped, config=self.config)
        self.assertLess(diagnostics["clipping_score"], 0.7)
        self.assertLess(diagnostics["raw_exposure_score"], 0.75)

    def test_low_key_but_detailed_face_remains_acceptable(self) -> None:
        diagnostics = exposure_diagnostics(_gradient(28.0, 92.0), config=self.config)
        self.assertGreater(diagnostics["raw_exposure_score"], 0.6)

    def test_high_key_but_detailed_face_remains_acceptable(self) -> None:
        diagnostics = exposure_diagnostics(_gradient(150.0, 220.0), config=self.config)
        self.assertGreater(diagnostics["raw_exposure_score"], 0.6)

    def test_raw_and_shaped_scores_remain_bounded(self) -> None:
        diagnostics = exposure_diagnostics(_gradient(40.0, 210.0), config=self.config)
        self.assertGreaterEqual(diagnostics["raw_exposure_score"], 0.0)
        self.assertLessEqual(diagnostics["raw_exposure_score"], 1.0)
        self.assertGreaterEqual(diagnostics["exposure_score"], 0.0)
        self.assertLessEqual(diagnostics["exposure_score"], 1.0)

    def test_gamma_one_leaves_score_unchanged(self) -> None:
        self.assertAlmostEqual(shape_quality_score(0.73, gamma=1.0), 0.73, places=6)

    def test_higher_gamma_mildly_lowers_sub_one_scores(self) -> None:
        self.assertLess(shape_quality_score(0.73, gamma=1.25), 0.73)

    def test_acceptable_inputs_do_not_all_collapse_to_exactly_one(self) -> None:
        mid = exposure_diagnostics(_gradient(60.0, 170.0), config=self.config)
        low_key = exposure_diagnostics(_gradient(30.0, 92.0), config=self.config)
        high_key = exposure_diagnostics(_gradient(148.0, 220.0), config=self.config)
        scores = [
            mid["raw_exposure_score"],
            low_key["raw_exposure_score"],
            high_key["raw_exposure_score"],
        ]
        self.assertLess(sum(score == 1.0 for score in scores), len(scores))
        self.assertGreater(len(set(scores)), 1)


class ContrastMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ContrastConfig()

    def test_flat_image_receives_low_contrast_score(self) -> None:
        diagnostics = contrast_diagnostics(np.full((48, 48), 96.0, dtype=np.float32), config=self.config)
        self.assertLess(diagnostics["contrast_score"], 0.15)

    def test_moderate_tonal_range_receives_middle_score(self) -> None:
        diagnostics = contrast_diagnostics(_gradient(70.0, 150.0), config=self.config)
        self.assertGreater(diagnostics["contrast_score"], 0.35)
        self.assertLess(diagnostics["contrast_score"], 0.8)

    def test_wide_tonal_range_receives_high_score(self) -> None:
        diagnostics = contrast_diagnostics(_gradient(10.0, 240.0), config=self.config)
        self.assertGreater(diagnostics["contrast_score"], 0.7)

    def test_contrast_scores_remain_bounded(self) -> None:
        diagnostics = contrast_diagnostics(_gradient(0.0, 255.0), config=self.config)
        self.assertGreaterEqual(diagnostics["contrast_score"], 0.0)
        self.assertLessEqual(diagnostics["contrast_score"], 1.0)

    def test_high_contrast_approaches_but_does_not_equal_one(self) -> None:
        diagnostics = contrast_diagnostics(_gradient(0.0, 255.0), config=self.config)
        self.assertLess(diagnostics["contrast_score"], 1.0)

    def test_broad_and_local_components_are_stored(self) -> None:
        diagnostics = contrast_diagnostics(_gradient(20.0, 200.0), config=self.config)
        self.assertGreater(diagnostics["broad_tonal_range"], 0.0)
        self.assertGreater(diagnostics["interquartile_range"], 0.0)
        self.assertGreater(diagnostics["broad_contrast_score"], 0.0)
        self.assertGreater(diagnostics["local_contrast_score"], 0.0)


class RankingCalibrationTests(unittest.TestCase):
    def test_calibration_curve_requires_monotonic_targets(self) -> None:
        with self.assertRaises(ValueError):
            CalibrationCurve(
                source_values=(0.1, 0.2, 0.3),
                target_values=(0.1, 0.4, 0.2),
            )

    def test_useful_exposure_ranking_separates_good_faces(self) -> None:
        assessor = ClassicalFaceQualityAssessor(
            calibration_profile=RankingCalibrationProfile(),
        )
        scores = [
            assessor.assess(
                np.dstack([_gradient(40.0, 110.0)] * 3).astype(np.uint8),
                original_face_minimum_dimension=96,
                alignment_confidence=1.0,
            ).exposure_ranking_score,
            assessor.assess(
                np.dstack([_gradient(55.0, 150.0)] * 3).astype(np.uint8),
                original_face_minimum_dimension=96,
                alignment_confidence=1.0,
            ).exposure_ranking_score,
            assessor.assess(
                np.dstack([_gradient(70.0, 175.0)] * 3).astype(np.uint8),
                original_face_minimum_dimension=96,
                alignment_confidence=1.0,
            ).exposure_ranking_score,
        ]
        self.assertGreater(max(scores) - min(scores), 0.02)

    def test_useful_contrast_ranking_separates_good_faces(self) -> None:
        assessor = ClassicalFaceQualityAssessor(
            calibration_profile=RankingCalibrationProfile(),
        )
        scores = [
            assessor.assess(
                np.dstack([_gradient(60.0, 130.0)] * 3).astype(np.uint8),
                original_face_minimum_dimension=96,
                alignment_confidence=1.0,
            ).contrast_ranking_score,
            assessor.assess(
                np.dstack([_gradient(50.0, 165.0)] * 3).astype(np.uint8),
                original_face_minimum_dimension=96,
                alignment_confidence=1.0,
            ).contrast_ranking_score,
            assessor.assess(
                np.dstack([_gradient(35.0, 200.0)] * 3).astype(np.uint8),
                original_face_minimum_dimension=96,
                alignment_confidence=1.0,
            ).contrast_ranking_score,
        ]
        self.assertGreater(max(scores) - min(scores), 0.02)

    def test_small_numeric_noise_does_not_cause_large_rank_jump(self) -> None:
        assessor = ClassicalFaceQualityAssessor(
            calibration_profile=RankingCalibrationProfile(),
        )
        base = assessor.assess(
            np.dstack([_gradient(60.0, 170.0)] * 3).astype(np.uint8),
            original_face_minimum_dimension=96,
            alignment_confidence=1.0,
        )
        noisy = assessor.assess(
            np.dstack([_gradient(61.0, 171.0)] * 3).astype(np.uint8),
            original_face_minimum_dimension=96,
            alignment_confidence=1.0,
        )
        self.assertLess(
            abs(base.exposure_ranking_score - noisy.exposure_ranking_score),
            0.05,
        )
        self.assertLess(
            abs(base.contrast_ranking_score - noisy.contrast_ranking_score),
            0.05,
        )

    def test_small_face_reduces_detail_availability_but_not_focus_score(self) -> None:
        assessor = ClassicalFaceQualityAssessor(
            calibration_profile=RankingCalibrationProfile(),
        )
        image = np.dstack([_gradient(40.0, 180.0, height=96, width=96)] * 3).astype(
            np.uint8
        )
        small = assessor.assess(
            image,
            original_face_minimum_dimension=40,
            alignment_confidence=1.0,
        )
        large = assessor.assess(
            image,
            original_face_minimum_dimension=120,
            alignment_confidence=1.0,
        )
        self.assertLess(
            small.detail_availability_score,
            large.detail_availability_score,
        )
        self.assertAlmostEqual(
            small.focus_sharpness_score,
            large.focus_sharpness_score,
            places=6,
        )


class ExposurePenaltyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExposurePenaltyConfig()

    def test_above_threshold_has_zero_penalty(self) -> None:
        self.assertEqual(
            exposure_penalty(
                0.90,
                acceptable_threshold=self.config.acceptable_threshold,
                maximum_penalty=self.config.maximum_penalty,
                exponent=self.config.exponent,
            ),
            0.0,
        )

    def test_small_deficiency_has_small_penalty(self) -> None:
        small = exposure_penalty(
            0.75,
            acceptable_threshold=self.config.acceptable_threshold,
            maximum_penalty=self.config.maximum_penalty,
            exponent=self.config.exponent,
        )
        poor = exposure_penalty(
            0.25,
            acceptable_threshold=self.config.acceptable_threshold,
            maximum_penalty=self.config.maximum_penalty,
            exponent=self.config.exponent,
        )
        self.assertGreater(small, 0.0)
        self.assertLess(small, poor)

    def test_penalty_never_exceeds_maximum(self) -> None:
        penalty = exposure_penalty(
            0.0,
            acceptable_threshold=self.config.acceptable_threshold,
            maximum_penalty=self.config.maximum_penalty,
            exponent=self.config.exponent,
        )
        self.assertLessEqual(penalty, self.config.maximum_penalty)

    def test_final_selection_score_stays_bounded(self) -> None:
        penalty = exposure_penalty(
            0.15,
            acceptable_threshold=self.config.acceptable_threshold,
            maximum_penalty=self.config.maximum_penalty,
            exponent=self.config.exponent,
        )
        final_score = clamp01(0.08 - penalty)
        self.assertGreaterEqual(final_score, 0.0)
        self.assertLessEqual(final_score, 1.0)


if __name__ == "__main__":
    unittest.main()
