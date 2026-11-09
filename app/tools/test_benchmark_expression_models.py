import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.tools.acquire_expression_models import verify
from app.tools.acquire_expression_models import MODELS
from app.tools.benchmark_expression_models import (LABELS, TEMPLATE, align_official, category,
    preprocess, probabilities, select_faces)


class ExpressionBenchmarkTests(unittest.TestCase):
    def test_local_model_signatures_if_acquired(self):
        directory = Path("audit_results/expression_benchmark/models")
        if not all((directory / item[0]).exists() for item in MODELS.values()):
            self.skipTest("Optional OpenCV models are not present in CI")
        import onnxruntime as ort
        for variant, (name, size, digest) in MODELS.items():
            path = directory / name
            self.assertTrue(verify(path, size, digest))
            options = ort.SessionOptions()
            options.log_severity_level = 3
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            if variant == "int8bq":
                self.assertEqual([item.name for item in session.get_inputs()],
                                 ["data", "fc.weight"])
            else:
                self.assertEqual(session.get_inputs()[0].shape, [1,3,112,112])
                output = session.run(None, {"data": np.zeros((1,3,112,112),np.float32)})[0]
                self.assertEqual(output.shape, (1,7))
                self.assertTrue(np.isfinite(output).all())
    def test_hash_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(b"test")
            self.assertTrue(verify(path, 4, hashlib.sha256(b"test").hexdigest()))
            self.assertFalse(verify(path, 5, hashlib.sha256(b"test").hexdigest()))
            self.assertFalse(verify(path, 4, "0"*64))

    def test_official_rgb_normalization_and_shape(self):
        image = np.zeros((112,112,3), np.uint8)
        image[0,0] = [0,127,255]  # BGR
        tensor = preprocess(image)
        self.assertEqual(tensor.shape, (1,3,112,112))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(float(tensor[0,0,0,0]), 1.)
        self.assertAlmostEqual(float(tensor[0,2,0,0]), -1.)
        with self.assertRaises(ValueError):
            preprocess(image[:10])

    def test_probabilities_and_class_order(self):
        self.assertEqual(LABELS, ("angry","disgust","fearful","happy",
                                  "neutral","sad","surprised"))
        logits = probabilities([[-10,-10,-10,10,-10,-10,-10]])
        self.assertEqual(int(np.argmax(logits)), 3)
        self.assertAlmostEqual(float(logits.sum()), 1.)
        already = probabilities([[0,0,0,1,0,0,0]])
        self.assertEqual(int(np.argmax(already)), 3)
        with self.assertRaises(ValueError):
            probabilities([[1,2]])

    def test_official_alignment_identity(self):
        image = np.arange(112*112*3, dtype=np.uint8).reshape(112,112,3)
        aligned = align_official(image, TEMPLATE)
        self.assertTrue(np.array_equal(aligned, image))
        self.assertIsNone(align_official(image, np.zeros((4,2))))

    def test_stratified_selection_is_seeded_and_photo_capped(self):
        rows = [dict(image_path=f"photo{i//4}.jpg", face_index=str(i),
                     geometry_relative_area=".001" if i % 2 else ".02",
                     analysis_missing="0") for i in range(80)]
        a = select_faces(rows, 30, 3603)
        b = select_faces(rows, 30, 3603)
        self.assertEqual(a,b)
        self.assertEqual(len(a),30)
        self.assertLessEqual(max(sum(r["image_path"] == path for r in a)
                                 for path in {r["image_path"] for r in a}),2)
        self.assertEqual(category(rows[1]),"small")


if __name__ == "__main__":
    unittest.main()
