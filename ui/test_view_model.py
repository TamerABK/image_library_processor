from __future__ import annotations

import unittest

from automatic_scan.models import PeoplePictureAssessment, PipelineImageState
from ui.view_model import PhotoCleanerViewModel


class PhotoCleanerViewModelBadgeTests(unittest.TestCase):
    def test_non_people_picture_gets_badge(self) -> None:
        image = PipelineImageState(
            path="/tmp/example.jpg",
            file_size=1,
            mtime_ns=1,
            extension=".jpg",
            orientation="landscape",
            capture_timestamp=None,
            width=1600,
            height=1000,
            people_picture_assessment=PeoplePictureAssessment(
                is_people_picture=False,
                confidence=0.12,
                primary_face_indices=(),
                supporting_reasons=("no_faces_detected",),
            ),
        )

        self.assertEqual(
            PhotoCleanerViewModel._automatic_badge_text(image),
            "No people pic",
        )

    def test_people_picture_has_no_badge(self) -> None:
        image = PipelineImageState(
            path="/tmp/example.jpg",
            file_size=1,
            mtime_ns=1,
            extension=".jpg",
            orientation="landscape",
            capture_timestamp=None,
            width=1600,
            height=1000,
            people_picture_assessment=PeoplePictureAssessment(
                is_people_picture=True,
                confidence=0.84,
                primary_face_indices=(0,),
                supporting_reasons=("largest_face_prominent",),
            ),
        )

        self.assertEqual(PhotoCleanerViewModel._automatic_badge_text(image), "")


if __name__ == "__main__":
    unittest.main()
