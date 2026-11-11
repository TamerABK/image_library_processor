import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.tools.audit_full_nsc_quality import flag_parts, run, source_key, summarize
from app.tools.report_full_nsc_quality import phase_stress


class FullPopulationAuditTests(unittest.TestCase):
    def test_zero_denominator_and_missing_values(self):
        self.assertEqual(summarize([None, ""])["count"], 0)
        self.assertIsNone(summarize([None])["median"])
        self.assertEqual(summarize([1, 2, 3])["median"], 2)

    def test_flags_are_independent_and_thresholds_unchanged(self):
        self.assertEqual(flag_parts(dict(edge_persistence=.64, artifact_measurement=1.0))["persistence_flag"], True)
        self.assertEqual(flag_parts(dict(edge_persistence=.9, artifact_measurement=1.51))["blocking_flag"], True)
        self.assertFalse(flag_parts(dict(edge_persistence=None, artifact_measurement=None))["flag_evaluable"])
        self.assertFalse(flag_parts(dict(edge_persistence=.65, artifact_measurement=1.5))["any_flag"])

    def test_source_revision_invalidates_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"test.jpg";path.write_bytes(b"one")
            first=source_key(path,path.stat(),"face-v1",32_000_000)
            self.assertNotEqual(first,source_key(path,path.stat(),"face-v2",32_000_000))
            path.write_bytes(b"two-changed")
            self.assertNotEqual(first,source_key(path,path.stat(),"face-v1",32_000_000))

    def test_phase_stress_uses_real_local_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=np.zeros((128,128,3),np.uint8)
            image[:,:,0]=np.arange(128,dtype=np.uint8)[None,:]
            Image.fromarray(image).save(root/"sample.jpg")
            rows=phase_stress(root,[dict(relative_path="sample.jpg",any_flag="False")],root,sample=2)
            self.assertEqual(len(rows),8)
            self.assertEqual({r["offset"] for r in rows},set(range(8)))

    def test_full_population_resume_and_unsupported_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/"photos";audit=base/"audit";out=base/"output"
            root.mkdir();audit.mkdir()
            for name in ("a.jpg","b.jpg","notes.txt"):
                (root/name).write_bytes(name.encode())
            (audit/"photo_metrics.csv").write_text("image_path\n")
            (audit/"face_metrics.csv").write_text("image_path\n")
            (audit/"ranking_v1_v2_comparison.csv").write_text("preference_id,candidate_ids\n")
            (audit/"audit_cache.sqlite3").write_bytes(b"stub")
            calls=[]
            def fake_measure(path, source_root, *_args):
                calls.append(path.name)
                return dict(photo=dict(relative_path=path.name,face_count=0,flag_evaluable=True,
                    persistence_flag=False,blocking_flag=False,any_flag=False,both_flag=False,
                    detail_score=.2,noise_aware_detail=.1),faces=[],timing=dict(
                    decode_s=.01,whole_s=.01,color_s=.01,face_s=0.,total_s=.03))
            with patch("app.tools.audit_full_nsc_quality.ImageFaceAnalysisCache"), patch(
                    "app.tools.audit_full_nsc_quality.measure_one",side_effect=fake_measure):
                cold=run(root,audit,out)
                warm=run(root,audit,out,resume=True)
            self.assertEqual(cold["discovered"],3)
            self.assertEqual((cold["measured"],cold["unsupported"],cold["additional_decodes"]),(2,1,2))
            self.assertEqual((warm["measured"],warm["cache_hits"],warm["additional_decodes"]),(2,2,0))
            self.assertEqual(calls,["a.jpg","b.jpg"])


if __name__=="__main__":unittest.main()
