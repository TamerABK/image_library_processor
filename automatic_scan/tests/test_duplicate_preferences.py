import unittest
from dataclasses import replace

from automatic_scan.duplicate_selection import _ranking_sort_key
from automatic_scan.models import DuplicateRankingBreakdown


class DuplicatePreferenceTests(unittest.TestCase):
    def test_preferences_only_change_their_existing_ranking_criteria(self):
        base = DuplicateRankingBreakdown(
            acceptable_primary_face_count=1, worst_primary_score=0.6,
            weighted_primary_average=0.6, eye_open_score=1.0,
            pose_visibility_score=0.0, image_sharpness_score=0.0,
            exposure_contrast_score=0.0, overall_quality_score=0.0,
            group_visual_centrality=0.5, capture_timestamp=None,
            path_key="same", explanation=(),
        )
        better_face = replace(base, worst_primary_score=0.7,
                              eye_open_score=0.0, group_visual_centrality=0.2)
        better_eye = replace(base, worst_primary_score=0.6,
                             eye_open_score=1.0, group_visual_centrality=0.8)
        self.assertGreater(_ranking_sort_key(better_face), _ranking_sort_key(better_eye))
        self.assertLess(_ranking_sort_key(better_face, prioritize_primary_face_quality=False),
                        _ranking_sort_key(better_eye, prioritize_primary_face_quality=False))
        self.assertGreater(_ranking_sort_key(better_eye),
                           _ranking_sort_key(replace(better_eye, eye_open_score=0.0)))
        self.assertEqual(_ranking_sort_key(better_eye, prefer_open_eyes=False),
                         _ranking_sort_key(replace(better_eye, eye_open_score=0.0),
                                           prefer_open_eyes=False))


if __name__ == "__main__":
    unittest.main()
