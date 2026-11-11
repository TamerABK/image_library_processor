import unittest
import numpy as np

from automatic_scan.technical_quality_v2_calibration import measure_technical
from automatic_scan.technical_pairwise_calibration import calibrate, split_events, validate_label
from app.tools.review_technical_pairs import sample_pairs


class EvidenceContractTests(unittest.TestCase):
    def test_flat_image_keeps_clipping_independent_of_detail(self):
        image = np.full((128, 128, 3), 255, np.uint8)
        result = measure_technical(image)
        self.assertEqual(result.status, "insufficient_structured_detail")
        self.assertEqual(result.clipping_bgr, (1., 1., 1.))
        self.assertEqual(result.clipping_coverage, 1.)
        self.assertEqual(result.noise_coverage, 0.)
        self.assertEqual(result.luminance_noise_coverage, 0.)
        self.assertEqual(result.chroma_noise_coverage, result.chroma_coverage)
        self.assertIsNone(result.predictive_confidence)
        self.assertIsNone(result.score)

    def test_low_texture_color_keeps_chroma_without_luminance_noise(self):
        image = np.full((128, 128, 3), (80, 120, 160), np.uint8)
        result = measure_technical(image)
        self.assertEqual(result.status, "insufficient_structured_detail")
        self.assertIsNotNone(result.chroma_noise)
        self.assertGreater(result.chroma_noise_coverage, 0)
        self.assertEqual(result.luminance_noise_coverage, 0)

    def test_unsupported_does_not_claim_confidence(self):
        result = measure_technical(None)
        self.assertIsNone(result.score)
        self.assertIsNone(result.predictive_confidence)
        self.assertEqual(result.clipping_coverage, 0)


class ReviewCalibrationTests(unittest.TestCase):
    def test_seeded_sampler_and_score_blind_schema(self):
        groups = [dict(group_id="g1", candidate_ids="C:/event/CAM/a.jpg|C:/event/CAM/b.jpg|C:/event/CAM/c.jpg")]
        first = sample_pairs(groups, 3605)
        self.assertEqual(first, sample_pairs(groups, 3605))
        self.assertEqual(len(first), 3)
        self.assertEqual(len({p["pair_id"] for p in first}), 3)
        self.assertTrue(all(p["technical_preference"] == "" for p in first))
        self.assertTrue(all("score" not in p for p in first))

    def test_event_split_never_leaks(self):
        split = split_events(["shoot1", "shoot1", "shoot2", "shoot3"], 10)
        self.assertEqual(set(split), {"shoot1", "shoot2", "shoot3"})
        self.assertEqual(split, split_events(["shoot1", "shoot2", "shoot3"], 10))

    def test_labels_and_insufficient_data_withhold_score(self):
        pair = dict(pair_id="p", event_id="e", group_id="g", photo_a="a", photo_b="b",
                    technical_preference="equal")
        self.assertEqual(validate_label(pair), pair)
        self.assertEqual(calibrate([pair], {}).status, "withheld")
        self.assertIsNone(calibrate([pair], {}).score)
        pair["technical_preference"] = "cannot_determine"
        self.assertEqual(calibrate([pair], {}).pairwise_status, "abstain")
        pair["technical_preference"] = "unknown"
        with self.assertRaises(ValueError):
            validate_label(pair)


if __name__ == "__main__":
    unittest.main()
