from dataclasses import replace
import unittest

from automatic_scan.keeper_ranker_v2 import FACE_WEIGHTS, Face, Feature, Penalty, Photo, Preferences, rank_group
from automatic_scan.keeper_ranker_v2_adapter import extract_faces, photo_input, technical_feature, TECHNICAL_VERSION
from automatic_scan.keeper_ranker_v2 import VERSION
from types import SimpleNamespace


def face(eye=.8, sharp=.7, pose=.8):
    values = {key: Feature(.7) for key in FACE_WEIGHTS}
    values.update(eyes=Feature(eye), sharpness=Feature(sharp), pose=Feature(pose))
    return Face(values, prominence=1)


class KeeperRankerV2Tests(unittest.TestCase):
    def test_small_group_faces_and_background(self):
        def row(i, area, x):
            values = {"bbox_x": x, "bbox_y": 50, "bbox_width": 30,
                      "bbox_height": 30, "geometry_relative_area": area,
                      "detector_confidence_observed": .9, "analysis_missing": 0}
            for key in FACE_WEIGHTS:
                prefix = {"sharpness": "image_quality_sharpness", "eyes": "eyes",
                          "pose": "pose_metric", "exposure": "image_quality_exposure",
                          "contrast": "image_quality_contrast", "detail": "image_quality_detail_availability",
                          "detector": "detector_metric", "visible": "visible_face_metric",
                          "reliability": "measurement_reliability"}[key]
                values[prefix + "_ranking_score"] = .7
                values[prefix + "_confidence"] = 1
            return values
        rows = [row(i, .001, i * 50) for i in range(8)]
        faces, diag = extract_faces(rows, 3000, 2000, rows=True)
        self.assertEqual(sum(f.primary for f in faces), 8)
        self.assertTrue(all(d["eligible"] for d in diag))
        rows += [row(8, .00001, 1000)]
        faces, diag = extract_faces(rows, 3000, 2000, rows=True)
        self.assertFalse(faces[-1].primary)
        self.assertEqual(diag[-1]["reason"], "background_scale")
        # Live and flattened cache paths use the same feature extraction.
        live = []
        for row_data in rows:
            quality = SimpleNamespace(**{name: SimpleNamespace(ranking_score=.7, confidence=1)
                for name in ("sharpness", "exposure", "contrast", "detail_availability")})
            metric = SimpleNamespace(ranking_score=.7, confidence=1)
            analysis = SimpleNamespace(geometry=SimpleNamespace(relative_area=row_data["geometry_relative_area"]),
                image_quality=quality, eyes=metric, pose=SimpleNamespace(metric=metric),
                detector_metric=metric, visible_face_metric=metric, measurement_reliability=metric)
            live.append(SimpleNamespace(bbox=(row_data["bbox_x"], row_data["bbox_y"],30,30),
                                        confidence=.9, analysis=analysis))
        self.assertEqual(photo_input("x", rows, None, 3000, 2000, rows=True).faces,
                         photo_input("x", live, None, 3000, 2000).faces)

    def test_raw_technical_keeps_saturated_differences(self):
        low = technical_feature(dict(laplacian=500, sobel=70, local_contrast=55))
        high = technical_feature(dict(laplacian=800, sobel=90, local_contrast=55))
        noisy = technical_feature(dict(laplacian=5000, sobel=500, local_contrast=5))
        self.assertNotEqual(low.value, high.value)
        self.assertLess(noisy.value, high.value)
        self.assertIsNone(technical_feature({}).value)
        self.assertIn(TECHNICAL_VERSION, VERSION)

    def test_ties_are_not_quality_preferences(self):
        equal = rank_group((Photo("a", (), Feature(.7)), Photo("b", (), Feature(.7))),
                           Preferences(False, False))
        self.assertEqual(equal[0].image_id, "b")
        self.assertEqual(equal[0].tie_class, "exact")
        self.assertFalse(equal[0].quality_preference)
        near = rank_group((Photo("a", (), Feature(.7005)), Photo("b", (), Feature(.7))),
                          Preferences(False, False))
        self.assertEqual(near[0].tie_class, "below_resolution")
        self.assertFalse(near[0].quality_preference)

    def test_eye_off_is_independent_of_all_eye_derived_values(self):
        a = face()
        b = replace(a, features={**a.features, "eyes": Feature(.01),
                                 "reliability": Feature(.01)})
        p = Preferences(True, False)
        result = rank_group((Photo("a", (a,), Feature(.6)),
                             Photo("b", (b,), Feature(.6))), p)
        self.assertEqual(result[0].score, result[1].score)
        self.assertEqual(result[0].face_score, result[1].face_score)

    def test_partial_eyes_use_evidence_without_missingness_reward(self):
        complete = Photo("complete", (face(eye=.1), face(eye=.8)), Feature(.6))
        partial = Photo("partial", (face(eye=.1), replace(face(eye=.8),
            features={**face().features, "eyes": Feature(None, 0)})), Feature(.6))
        ranked = rank_group((complete, partial))
        self.assertIn("eyes", ranked[0].used_face_features)
        self.assertTrue(all(score.eye_coverage > 0 for score in ranked))
        self.assertLess(next(s for s in ranked if s.image_id == "partial").eye_coverage, 1)
        self.assertTrue(any(s.face_evidence_incomplete for s in ranked))
        without_eyes = rank_group((complete, partial), Preferences(True, False))
        self.assertEqual(without_eyes[0].score, without_eyes[1].score)

    def test_detector_alone_cannot_activate_face_weight(self):
        detector = Face({"detector": Feature(.9)}, 1)
        photographs = (Photo("a", (detector,), Feature(.6)),
                       Photo("b", (detector,), Feature(.7)))
        ranked = rank_group(photographs)
        self.assertTrue(all(s.face_weight == 0 and not s.face_evidence_sufficient for s in ranked))
        self.assertTrue(all("insufficient_visual_face_evidence" in s.uncertainty_reasons for s in ranked))
        one_visual = Face({"detector": Feature(.9), "sharpness": Feature(.7)}, 1)
        self.assertEqual(rank_group((Photo("a", (one_visual,), Feature(.6)),
                                     Photo("b", (one_visual,), Feature(.7))))[0].face_weight, 0)

    def test_technical_only_uncertainty_depends_on_technical_evidence(self):
        photos = (Photo("a", (face(),), Feature(.9)), Photo("b", (), Feature(.7)))
        on = rank_group(photos, Preferences(False, True))
        off = rank_group(photos, Preferences(False, False))
        self.assertEqual([(s.score, s.uncertain, s.tie_class) for s in on],
                         [(s.score, s.uncertain, s.tie_class) for s in off])
        self.assertFalse(any(s.uncertain for s in on))

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
