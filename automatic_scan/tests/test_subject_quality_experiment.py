import tempfile
import csv
import json
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from automatic_scan.subject_quality_experiment import (
    Registration, detection_space_size, face_region, map_point, match_faces, region_metrics, register)
from app.tools.report_subject_quality import magnified_region, build


class SubjectQualityTests(unittest.TestCase):
    def test_detection_coordinates_are_not_original_coordinates(self):
        source = detection_space_size(6240, 4160)
        self.assertAlmostEqual(source[0], 2048)
        image = np.zeros((520, 780, 3), np.uint8)
        roi, box = face_region(image, (714, 409, 181, 223), source)
        self.assertEqual(box[0], 271)
        self.assertGreater(roi.shape[0], 80)

    def test_geometric_match_and_ambiguity(self):
        a = [dict(decoded_box=[10, 10, 50, 50]), dict(decoded_box=[100, 10, 140, 50])]
        b = [dict(decoded_box=[101, 10, 141, 50]), dict(decoded_box=[11, 10, 51, 50])]
        reg = Registration(np.eye(3), 100, .9, .2, "registered")
        links, status = match_faces(a, b, reg, None, None)
        self.assertEqual({(x["a"], x["b"]) for x in links}, {(0, 1), (1, 0)})
        self.assertEqual(status, "tentative_without_embeddings")
        self.assertLess(max(x["confidence"] for x in links), 1)
        missing, status = match_faces(a, [], reg, None, None)
        self.assertEqual(missing, [])
        self.assertEqual(status, "registration_or_faces_unavailable")

    def test_registration_translation(self):
        rng = np.random.default_rng(8)
        image = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        shifted = np.roll(image, 3, axis=1)
        result = register(image, shifted)
        self.assertEqual(result.status, "registered")
        point = map_point(result.matrix, (100, 100))
        self.assertAlmostEqual(point[0], 103, delta=2)

    def test_blur_and_noise_are_not_interchangeable(self):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        clean = region_metrics(image)["laplacian"]
        blurred = region_metrics(cv2.GaussianBlur(image, (9, 9), 2))["laplacian"]
        noisy = region_metrics(np.clip(image.astype(np.int16) +
            rng.normal(0, 20, image.shape), 0, 255).astype(np.uint8))["laplacian"]
        self.assertLess(blurred, clean)
        self.assertGreater(noisy, clean)

    def test_magnified_region_uses_analyzed_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"photo.png"
            original = np.zeros((200, 400, 3), np.uint8)
            original[40:80, 100:140] = (255, 0, 0)
            Image.fromarray(original).save(path)
            crop = magnified_region(path, [50, 20, 70, 40], (200, 100), (100, 100))
            self.assertGreater(np.asarray(crop)[:, :, 0].mean(), 75)

    def test_pdf_generation_from_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root/"a.png", root/"b.png"
            Image.new("RGB", (200, 100), "#839ab1").save(first)
            Image.new("RGB", (200, 100), "#849ab1").save(second)
            def table(name, records):
                with (root/name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
            (root/"summary.json").write_text(json.dumps(dict(groups=1, candidates=2,
                tentative_links=1, elapsed_seconds=.2, peak_process_rss=1000000)))
            table("scene_regions.csv", [dict(group_id="g", image_a="a.png", image_b="b.png",
                inliers=50, inlier_ratio=.9, reprojection_error=.5, registration="registered",
                correspondence_status="tentative_without_embeddings")])
            table("matched_faces.csv", [dict(group_id="g", image_a="a.png", image_b="b.png",
                box_a="[50,20,70,40]", box_b="[50,20,70,40]",
                laplacian_a=100, laplacian_b=110, laplacian_difference=10, confidence=.7)])
            table("photo_metrics.csv", [dict(image_id="a.png", source_path=str(first),
                decoded_width=200, decoded_height=100, v1_winner=1, v2_score=50),
                dict(image_id="b.png", source_path=str(second), decoded_width=200,
                     decoded_height=100, v1_winner=0, v2_score=50.1)])
            table("ranking_comparison.csv", [dict(group_id="g", v1_winner="a.png",
                v2_winner="b.png", tentative_face_links=1)])
            table("controlled_perturbations.csv", [dict(perturbation="baseline", laplacian=100)])
            output = root/"report.pdf"
            result = build(root, output)
            self.assertEqual(result["outcome"], "inconclusive")
            self.assertGreater(output.stat().st_size, 1000)
            self.assertTrue((root/"report_page_3.png").is_file())


if __name__ == "__main__":
    unittest.main()
