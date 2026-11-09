import unittest

import cv2
import numpy as np

from automatic_scan.color_integrity_experiment import measure_color, compare_white_balance


class ColorIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.base = np.full((256, 256, 3), 80, np.uint8)
        cv2.circle(self.base, (128, 128), 55, (130, 130, 130), -1)

    def test_rgb_channel_clipping_direction(self):
        a = self.base.copy()
        b = self.base.copy()
        a[:, :20, 2] = 255
        b[:, :40, 2] = 255
        first, second = measure_color(a), measure_color(b)
        self.assertLess(first["high_r"], second["high_r"])
        self.assertEqual(first["high_b"], 0)
        self.assertEqual(second["clipping_confidence"], 0)  # saturation is not a proved defect

    def test_chroma_noise_rises_then_becomes_unavailable(self):
        estimates = []
        for severity in (0, 4, 8, 16):
            rng = np.random.default_rng(3603)
            noisy = np.clip(self.base.astype(np.int16) + rng.normal(0, severity, self.base.shape), 0, 255).astype(np.uint8)
            estimates.append(measure_color(noisy))
        self.assertLess(estimates[0]["chroma_noise"], estimates[1]["chroma_noise"])
        self.assertLess(estimates[1]["chroma_noise"], estimates[2]["chroma_noise"])
        self.assertIsNone(estimates[3]["chroma_noise"])

    def test_ambiguous_saturated_scene_is_not_penalized(self):
        saturated = np.full_like(self.base, (0, 0, 255))
        result = measure_color(saturated)
        self.assertEqual(result["high_r"], 1)
        self.assertEqual(result["clipping_confidence"], 0)

    def test_white_balance_difference_has_no_direction(self):
        a = measure_color(self.base)
        warm = self.base.copy()
        warm[:, :, 2] += 10
        b = measure_color(warm)
        delta, status = compare_white_balance(a, b)
        self.assertEqual(status, "non_directional_difference")
        self.assertGreater(delta, 0)

    def test_changed_lighting_rejects_white_balance_comparison(self):
        a = measure_color(self.base)
        b = measure_color(np.clip(self.base.astype(np.int16) + 50, 0, 255).astype(np.uint8))
        self.assertEqual(compare_white_balance(a, b)[1], "lighting_or_exposure_changed")

    def test_colored_edge_is_not_called_aberration(self):
        image = self.base.copy()
        image[:, 128:, :] = (20, 20, 230)
        result = measure_color(image)
        self.assertEqual(result["fringing_confidence"], 0)

    def test_unavailable_pixels(self):
        self.assertEqual(measure_color(None)["status"], "unsupported_pixels")
        self.assertEqual(measure_color(np.zeros((10, 10, 3), np.uint8))["status"], "insufficient_resolution")


if __name__ == "__main__":
    unittest.main()
