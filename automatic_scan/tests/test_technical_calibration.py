import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from automatic_scan.technical_quality_v2_calibration import measure_technical, chroma_noise_float_lab
from automatic_scan.subject_quality_experiment import detection_space_size, face_region
from app.tools.audit_technical_calibration import cache_key
from app.tools.report_technical_calibration import build, independent_face_observations


class TechnicalCalibrationTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:256, :256]
        pattern = np.clip(120 + 35*np.sin(xx/5) + 25*np.cos(yy/7), 0, 255).astype(np.uint8)
        self.image = np.repeat(pattern[:, :, None], 3, axis=2)
        cv2.circle(self.image, (128, 128), 65, (150, 155, 160), -1)
        cv2.line(self.image, (30, 30), (220, 220), (210, 210, 210), 3)

    def test_blur_and_noise_do_not_improve_diagnostic_detail(self):
        base = measure_technical(self.image)
        blurred = measure_technical(cv2.GaussianBlur(self.image, (0, 0), 2))
        rng = np.random.default_rng(3605)
        noisy = np.clip(self.image.astype(np.int16) + rng.normal(0, 15, self.image.shape), 0, 255).astype(np.uint8)
        noisy_result = measure_technical(noisy)
        blur_noisy = np.clip(cv2.GaussianBlur(self.image, (0, 0), 2).astype(np.int16) +
                             rng.normal(0, 15, self.image.shape), 0, 255).astype(np.uint8)
        self.assertLess(blurred.noise_aware_detail, base.noise_aware_detail)
        self.assertLess(noisy_result.noise_aware_detail, base.noise_aware_detail)
        self.assertLess(measure_technical(blur_noisy).noise_aware_detail, base.noise_aware_detail)
        self.assertIsNone(base.score)  # Oversharpening remains unresolved.

    def test_float_lab_noise_stays_measurable_at_high_noise(self):
        estimates = []
        for severity in (0, 8, 20, 40):
            rng = np.random.default_rng(4)
            noisy = np.clip(self.image.astype(np.int16)+rng.normal(0, severity, self.image.shape),
                            0, 255).astype(np.uint8)
            estimate, coverage, status = chroma_noise_float_lab(noisy)
            self.assertEqual(status, "measured")
            self.assertGreaterEqual(coverage, .08)
            estimates.append(estimate)
        self.assertLess(estimates[0], estimates[1])
        self.assertLess(estimates[1], estimates[2])
        self.assertLess(estimates[2], estimates[3])

    def test_saturation_is_not_a_quality_score(self):
        red = np.zeros_like(self.image)
        red[:, :, 2] = 255
        result = measure_technical(red)
        self.assertIsNone(result.score)

    def test_processing_artifacts_cannot_activate_a_score(self):
        blur = cv2.GaussianBlur(self.image, (0, 0), 1.2)
        oversharp = cv2.addWeighted(self.image, 3, blur, -2, 0)
        ok, encoded = cv2.imencode(".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, 25])
        self.assertTrue(ok)
        jpeg = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.assertIsNone(measure_technical(oversharp).score)
        self.assertIsNone(measure_technical(jpeg).score)

    def test_scene_without_faces_is_measurable(self):
        result = measure_technical(self.image)
        self.assertEqual(result.status, "measured")
        self.assertIsNotNone(result.noise_aware_detail)

    def test_face_coordinates_and_versioned_cache_key(self):
        dimensions = detection_space_size(6240, 4160)
        _, box = face_region(np.zeros((520, 780, 3), np.uint8),
                             (714, 409, 181, 223), dimensions)
        self.assertEqual(box[0], 271)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"photo.jpg"
            path.write_bytes(b"first")
            first = cache_key(path, 1024)
            self.assertNotEqual(first, cache_key(path, 2048))
            self.assertNotEqual(first, cache_key(path, 1024, "updated-face-cache"))
            path.write_bytes(b"changed image bytes")
            self.assertNotEqual(first, cache_key(path, 1024))

    def test_face_roi_scales_from_detection_space_without_upsampling(self):
        source = detection_space_size(6000, 4000)
        box = (800, 500, 180, 210)
        dimensions = []
        for width, height in ((900, 600), (1800, 1200), (6000, 4000)):
            image = np.zeros((height, width, 3), np.uint8)
            roi, mapped = face_region(image, box, source)
            self.assertEqual(roi.shape[1], mapped[2]-mapped[0])
            self.assertEqual(roi.shape[0], mapped[3]-mapped[1])
            dimensions.append(roi.shape[1])
        self.assertLess(dimensions[0], dimensions[1])
        self.assertLess(dimensions[1], dimensions[2])

    def test_unavailable_input(self):
        self.assertEqual(measure_technical(None).status, "unsupported_image")
        self.assertEqual(measure_technical(np.zeros((8, 8, 3), np.uint8)).status, "unsupported_image")

    def test_chroma_and_luminance_perturbations_are_separate(self):
        for base in (self.image, cv2.GaussianBlur(self.image, (0, 0), 2)):
            lab = cv2.cvtColor(base.astype(np.float32)/255, cv2.COLOR_BGR2Lab)
            rng = np.random.default_rng(82)
            luma, chroma = lab.copy(), lab.copy()
            luma[:, :, 0] = np.clip(luma[:, :, 0] + rng.normal(0, 6, lab.shape[:2]), 0, 100)
            chroma[:, :, 1:3] += rng.normal(0, 6, (*lab.shape[:2], 2))
            convert = lambda a: np.clip(cv2.cvtColor(a, cv2.COLOR_Lab2BGR)*255, 0, 255).astype(np.uint8)
            original_estimate = chroma_noise_float_lab(base)[0]
            luma_estimate = chroma_noise_float_lab(convert(luma))[0]
            chroma_estimate = chroma_noise_float_lab(convert(chroma))[0]
            self.assertGreater(chroma_estimate-original_estimate, 3)
            self.assertLess(abs(luma_estimate-original_estimate), 2)

    def test_confidence_fields_are_independent(self):
        measured = measure_technical(self.image)
        tiny = measure_technical(self.image[:40, :40])
        self.assertNotEqual(measured.detail_reliability, "unavailable")
        self.assertEqual(tiny.chroma_reliability, "unavailable")
        self.assertIsNotNone(tiny.noise_aware_detail)
        self.assertNotEqual(tiny.detail_reliability, "unavailable")
        self.assertGreater(tiny.detail_coverage, 0)
        self.assertEqual(tiny.chroma_coverage, 0)

    def test_artifact_observations_change_under_processing(self):
        blur = cv2.GaussianBlur(self.image, (0, 0), 1.2)
        oversharp = cv2.addWeighted(self.image, 3, blur, -2, 0)
        original = measure_technical(self.image)
        strong = measure_technical(oversharp)
        self.assertLess(strong.edge_persistence, original.edge_persistence)
        self.assertIsNone(strong.score)
        ok, encoded = cv2.imencode(".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, 20])
        self.assertTrue(ok)
        jpeg = measure_technical(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
        self.assertIsNotNone(jpeg.artifact_measurement)
        self.assertGreater(jpeg.artifact_measurement, original.artifact_measurement)
        self.assertIsNone(jpeg.score)

    def test_face_index_does_not_establish_correspondence(self):
        rows = [dict(image_id="a.jpg", face_index="0", status="measured"),
                dict(image_id="b.jpg", face_index="0", status="measured")]
        a = independent_face_observations(rows, "a.jpg")
        b = independent_face_observations(rows, "b.jpg")
        self.assertNotEqual(a[0]["region_label"], b[0]["region_label"])
        self.assertNotIn("match", a[0])

    def test_report_requires_real_audit_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Real audit directory"):
                build(Path(directory)/"missing", Path(directory)/"report.pdf")


if __name__ == "__main__":
    unittest.main()
