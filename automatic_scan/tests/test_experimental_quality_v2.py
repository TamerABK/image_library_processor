import unittest
from types import SimpleNamespace

from automatic_scan.experimental_quality_v2 import (Observation,
    composition_observation, experimental_score, moment_observation)


class ExperimentalQualityTests(unittest.TestCase):
    def test_moment_is_explicitly_unavailable(self):
        self.assertFalse(moment_observation().available)
        self.assertIsNone(moment_observation().score)

    def test_clipping_is_geometry_only(self):
        center = dict(bbox_x=40, bbox_y=40, bbox_width=20, bbox_height=20,
                      analysis_missing=0)
        edge = dict(center, bbox_x=0)
        self.assertEqual(composition_observation([center], 100, 100).score, 1)
        self.assertEqual(composition_observation([edge], 100, 100).score, 0)
        self.assertFalse(composition_observation([], 100, 100).available)

    def test_variants_do_not_impute_missing_moment(self):
        baseline = SimpleNamespace(score=70, face_score=.7, technical_score=.7, penalty=0)
        unavailable = moment_observation()
        framed = Observation(1, 1, 1, True, (), "no clipping")
        self.assertEqual(experimental_score(baseline, unavailable, framed)[0], 70)
        self.assertIsNone(experimental_score(baseline, unavailable, framed,
                                             use_moment=True)[0])
        self.assertGreater(experimental_score(baseline, unavailable, framed,
                                              use_composition=True)[0], 70)


if __name__ == "__main__":
    unittest.main()
