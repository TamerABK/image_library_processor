import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from automatic_scan.models import DuplicateRankingBreakdown
from duplicate_detector.models import PhotoInfo
from app.tools import audit_face_quality as audit


def _breakdown(**changes):
    values = dict(acceptable_primary_face_count=1, worst_primary_score=0.5,
                  weighted_primary_average=0.5, eye_open_score=1.0,
                  pose_visibility_score=0.0, image_sharpness_score=0.0,
                  exposure_contrast_score=0.0, overall_quality_score=0.0,
                  group_visual_centrality=0.5, capture_timestamp=1.0,
                  path_key="a", explanation=())
    values.update(changes)
    return DuplicateRankingBreakdown(**values)


class FakeServices:
    requests = 0

    def __init__(self, config, **kwargs):
        self.config = config

    def get_face_analysis_cached(self, record):
        self.__class__.requests += 1
        return []

    def get_duplicate_photo_cached(self, record):
        return None

    def index_duplicate_photo(self, record):
        return PhotoInfo(record.path, record.width, record.height, record.file_size,
                         0, 0)

    def generate_duplicate_candidates(self, photos):
        return []

    def verify_duplicate_groups(self, photos, pairs):
        return [photos]


class FaceAuditToolTests(unittest.TestCase):
    def test_first_lexicographic_difference_is_reported(self):
        winner = _breakdown(worst_primary_score=0.500001, weighted_primary_average=0.1)
        loser = _breakdown(worst_primary_score=0.5, weighted_primary_average=0.9)
        self.assertEqual(("worst_primary_score", 0.000001),
                         (lambda x: (x[0], round(x[1], 6)))(audit._first_difference(winner, loser)))

    def test_isolated_csv_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "photos"
            source.mkdir()
            for name in ("a.jpg", "b.jpg"):
                Image.new("RGB", (64, 48), "blue").save(source / name)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()}
            output = base / "audit"
            args = argparse.Namespace(input=str(source), output=str(output), mode="full",
                cache_source=None, checkpoint_source=None, groups_csv=None, labels_csv=None, provider="cpu",
                max_images=None, contact_sheets=True)
            FakeServices.requests = 0
            with patch.object(audit, "AutomaticScanServices", FakeServices), \
                 patch.object(audit, "_model_fingerprints", return_value={}), \
                 patch.object(audit, "_git_sha", return_value="test-sha"):
                audit.run(args)
                self.assertEqual(2, FakeServices.requests)
                imported = argparse.Namespace(**vars(args))
                imported.output = str(base / "imported")
                imported.checkpoint_source = str(output / "image_checkpoint.jsonl")
                audit.run(imported)
                self.assertEqual(2, FakeServices.requests)
                audit.run(args)
                self.assertEqual(2, FakeServices.requests)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()})
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("complete", manifest["status"])
            self.assertEqual(1, manifest["comparison_group_count"])
            for name in ("face_metrics", "photo_metrics", "ranking_decisions",
                         "pairwise_comparisons", "audit_summary"):
                with (output / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
                    self.assertTrue(list(csv.DictReader(stream)) or name == "face_metrics")
            self.assertEqual(1, len(list((output / "contact_sheets").glob("*.jpg"))))

    def test_output_cannot_be_inside_source_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "photos"
            source.mkdir()
            Image.new("RGB", (20, 20)).save(source / "a.jpg")
            args = argparse.Namespace(input=str(source), output=str(source / "audit"),
                mode="full", cache_source=None, checkpoint_source=None, groups_csv=None, labels_csv=None,
                provider="cpu", max_images=None, contact_sheets=False)
            with self.assertRaisesRegex(ValueError, "outside"):
                audit.run(args)

    def test_top1_summary_counts_a_single_human_winner(self):
        ranking = [dict(group_id="g", candidate_image_id="a", final_position=1, scenario="single_person"),
                   dict(group_id="g", candidate_image_id="b", final_position=2, scenario="single_person")]
        pairs = [dict(group_id="g", image_a="a", image_b="b", human_preference="a",
                      agreement=1)]
        photos = [dict(image_path=name, primary_face_count=1, worst_primary_score=.7,
                       face_count=1, analyzed_face_count=1, error="", terminal_failure=0)
                  for name in ("a", "b")]
        summary = audit._summaries(photos, ranking, pairs)[0]
        self.assertEqual(1, summary["top1_labeled_groups"])
        self.assertEqual(1.0, summary["top1_agreement"])
        self.assertEqual(1.0, summary["pairwise_accuracy"])

    def test_pairwise_labels_reject_noncandidate_preference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "labels.csv"
            labels.write_text("image_a,image_b,preferred_image\na.jpg,b.jpg,c.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid preferred_image"):
                audit._labels(labels, root)
            labels.write_text("image_a,image_b,preferred_image\na.jpg,b.jpg,a.jpg\n", encoding="utf-8")
            parsed = audit._labels(labels, root)
            self.assertEqual(str((root / "a.jpg").resolve()), next(iter(parsed.values())))


if __name__ == "__main__":
    unittest.main()
