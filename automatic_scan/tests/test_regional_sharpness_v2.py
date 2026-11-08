import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from automatic_scan.regional_sharpness_v2 import (RegionalCache, evaluate_group,
    patch_values, register, summarize, VERSION)


class RegionalSharpnessTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.image = np.full((384, 384), 105, np.uint8)
        for _ in range(45):
            x, y = rng.integers(20, 350, 2)
            cv2.line(self.image, (int(x), int(y)), (int(x)+20, int(y)+10), 205, 2)

    def test_defocus_and_registration(self):
        blurred = cv2.GaussianBlur(self.image, (0, 0), 2)
        sharp_score = summarize(patch_values(self.image))
        blur_score = summarize(patch_values(blurred))
        self.assertIsNotNone(sharp_score.score)
        self.assertGreater(sharp_score.score, blur_score.score)
        shifted = np.roll(self.image, 5, axis=1)
        aligned, response, status = register(self.image, shifted)
        self.assertEqual(status, "registered")
        self.assertGreater(response, .3)
        self.assertEqual(aligned.shape, (288, 288))
        self.assertEqual(register(self.image, np.zeros_like(self.image))[2],
                         "registration_rejected")

    def test_controlled_perturbations_remain_bounded(self):
        rng = np.random.default_rng(4)
        noisy = np.clip(self.image.astype(np.int16) + rng.normal(0, 12, self.image.shape), 0, 255).astype(np.uint8)
        exposed = np.clip(self.image.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
        cropped = cv2.resize(self.image[10:-10, 10:-10], (384, 384))
        for image in (noisy, exposed, cropped, np.roll(self.image, 3, axis=0)):
            result = summarize(patch_values(image))
            self.assertIsNotNone(result.score)
            self.assertGreaterEqual(result.score, 0)
            self.assertLessEqual(result.score, 1)

    def test_cache_and_selective_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{i}.png" for i in range(2)]
            for path in paths:
                path.write_bytes(b"fixture")
            class Loader:
                calls = 0
                def load_grayscale(self, path, *, max_dimension):
                    self.calls += 1
                    self.max_dimension = max_dimension
                    return self_image
            self_image = self.image
            loader = Loader()
            cache = RegionalCache(root / "regional.sqlite3")
            try:
                result, stats = evaluate_group(paths + paths[:1], cache, loader=loader, registered=False)
                self.assertEqual(stats["additional_decodes"], 2)
                self.assertEqual(loader.calls, 2)
                self.assertEqual(len(result), 2)
                _, stats = evaluate_group(paths, cache, loader=loader, registered=False)
                self.assertEqual(stats["cache_hits"], 2)
                self.assertEqual(stats["additional_decodes"], 0)
                _, stats = evaluate_group(paths, cache, loader=loader, registered=True)
                self.assertEqual(stats["additional_decodes"], 2)
                _, stats = evaluate_group(paths, cache, loader=loader, registered=True)
                self.assertEqual(stats["additional_decodes"], 0)
                self.assertEqual(stats["cache_hits"], 2)
                paths[0].write_bytes(b"changed fixture")
                _, stats = evaluate_group(paths, cache, loader=loader, registered=False)
                self.assertEqual(stats["cache_hits"], 1)
                self.assertEqual(stats["additional_decodes"], 1)
                _, stats = evaluate_group(paths, cache, loader=loader, cancel=lambda: True)
                self.assertEqual(stats["status"], "cancelled")
            finally:
                cache.close()
        self.assertIn("regional", VERSION)


if __name__ == "__main__":
    unittest.main()
