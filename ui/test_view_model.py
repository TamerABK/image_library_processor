from __future__ import annotations

import unittest
from unittest.mock import patch

from automatic_scan.config import ExecutionProvider
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


class PhotoCleanerViewModelProviderTests(unittest.TestCase):
    def test_directml_choice_appears_only_when_runtime_exposes_it(self) -> None:
        with patch("ui.view_model.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
            view_model = PhotoCleanerViewModel()
        self.assertIn("DirectML", view_model.state.available_execution_providers)
        view_model.set_execution_provider("DirectML")
        self.assertIs(
            view_model._current_automatic_config().performance.execution_provider,
            ExecutionProvider.DIRECTML,
        )

        with patch("ui.view_model.get_available_providers", return_value=["CPUExecutionProvider"]):
            cpu_view_model = PhotoCleanerViewModel()
        self.assertNotIn("DirectML", cpu_view_model.state.available_execution_providers)


if __name__ == "__main__":
    unittest.main()
