from dataclasses import replace
import unittest

from automatic_scan.keeper_ranker_v2 import FACE_WEIGHTS, Face, Feature, Penalty, Photo, Preferences, rank_group


def face(eye=.8, sharp=.7, pose=.8):
    values = {key: Feature(.7) for key in FACE_WEIGHTS}
    values.update(eyes=Feature(eye), sharpness=Feature(sharp), pose=Feature(pose))
    return Face(values, prominence=1)


class KeeperRankerV2Tests(unittest.TestCase):
    def test_eye_off_excludes_eye_and_eye_dependent_reliability(self):
        a = Photo("a", (face(eye=.1),), Feature(.7))
        bface = face(eye=.9)
        bface = replace(bface, features={**bface.features, "reliability": Feature(.2)})
        b = Photo("b", (bface,), Feature(.7))
        scores = rank_group((a, b), Preferences(prefer_open_eyes=False))
        self.assertEqual(scores[0].score, scores[1].score)
        self.assertEqual(scores[0].face_score, scores[1].face_score)
        self.assertNotIn("eyes", scores[0].used_face_features)
        self.assertNotIn("reliability", scores[0].used_face_features)
        self.assertNotEqual(*[s.score for s in rank_group((a, b))])

    def test_face_off_is_technical_only_and_ties_ignore_faces(self):
        a = Photo("a", (face(sharp=.1),), Feature(.6))
        b = Photo("b", (face(sharp=.9), face(pose=.1)), Feature(.6))
        result = rank_group((a, b), Preferences(prioritize_primary_face_quality=False))
        self.assertEqual([s.score for s in result], [60, 60])
        self.assertTrue(all(s.face_score is None and s.face_weight == 0 for s in result))
        self.assertEqual(result[0].image_id, "b")
        self.assertEqual([s.score for s in rank_group((a, b), Preferences(False, False))], [60, 60])

    def test_group_common_missingness_and_no_quality_fallback(self):
        a = Photo("a", (face(),), Feature(.9))
        b = Photo("b", (), Feature(.8))
        scores = rank_group((a, b))
        self.assertTrue(all(s.face_score is None and s.technical_weight == 1 for s in scores))
        missing = rank_group((replace(a, technical=Feature(None)), replace(b, technical=Feature(None))),
                             Preferences(prioritize_primary_face_quality=False))
        self.assertTrue(all(s.score is None for s in missing))

    def test_penalty_gates_follow_preferences_and_confidence(self):
        penalties = (Penalty("blink", "eye", 1, .95),
                     Penalty("face_blur", "face", 1, .95),
                     Penalty("technical_defect", "technical", 1, .95),
                     Penalty("uncertain", "technical", 1, .2))
        photo = Photo("a", (face(),), Feature(.8), penalties)
        weights = {"blink": 4, "face_blur": 5, "technical_defect": 6, "uncertain": 50}
        self.assertEqual(rank_group((photo,), Preferences(penalty_weights=weights))[0].penalty, 15)
        self.assertEqual(rank_group((photo,), Preferences(prefer_open_eyes=False,
                       penalty_weights=weights))[0].penalty, 11)
        self.assertEqual(rank_group((photo,), Preferences(prioritize_primary_face_quality=False,
                       penalty_weights=weights))[0].penalty, 6)


if __name__ == "__main__":
    unittest.main()
