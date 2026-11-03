# Phase 8G.2: shadow keeper ranking V2

Starting branch commit: `eaa75cff895ecdea682fa0feafae5b563682ba70`.
V1 remains the sole production keeper selector. V2 scores are stored separately
under `ranking_v2` and exported by the face audit tool. They never change
reviews, embeddings, clustering, or automatic keeper decisions.

## Formula and available evidence

The proposed long-term weights are 40% important-face quality, 25% moment,
20% composition, and 15% technical quality. Moment and composition have no
validated measurements; V2 marks them unavailable. With usable face and
technical evidence, active weights are 40/55 and 15/55. With face quality
disabled, technical quality receives 100%. With no comparable component, the
score is null and candidates use deterministic image-ID ordering.

The face component is 55% bounded-prominence weighted primary mean, 25% lower
20th percentile, and 20% fraction above a provisional 0.55 face threshold.
Prominence is capped at 2.0 and floored at 0.25. Primary subjects are selected
by geometry (relative area at least max(0.008, 0.68 times the largest face));
this keeps eye measurements out of primary selection. The face score uses
existing calibrated metric rankings and their confidence availability. The
same available metric set is applied to every candidate in a duplicate group.
No candidate gets extra weight from a privately missing measurement.

With `prefer_open_eyes=false`, V2 excludes both the eyes metric and measurement
reliability, since reliability may contain eye confidence. It does not use
`global_selection_score`, eye-dependent V1 acceptance, or eye tie-breakers.
With `prioritize_primary_face_quality=false`, V2 ignores all face records and
uses only whole-image blur quality. This rule takes precedence over the eye
preference. Missing technical quality stays missing. The score is on 0–100.

The penalty interface is reserved: V2 currently applies **zero** penalties
because posed/candid intent and defect confidence have not been validated.
The unpenalized score is therefore the final score. No photo is deleted or
rejected by V2.

## Compatibility and exports

Face-analysis stage compatibility includes `ImageFaceAnalysisCache` schema
version 2, the analyzer configuration, and content-based model hashes cached
by path and file metadata. V1 ranking compatibility includes the face-analysis
key so pre-fix aggregates are not reused. Embedding compatibility remains a
separate stage; ranking-only changes do not modify ArcFace embeddings.

Run the audit with `--mode full --include-blur` to calculate the existing
whole-image blur measurement in the isolated audit cache. The audit tool writes
`ranking_v2_features.csv`,
`ranking_v1_v2_comparison.csv`, and `ranking_v2_summary.json`, for all four
preference combinations. It uses only compatible cached blur results as
technical evidence. A prior audit checkpoint that did not contain whole-image
blur evidence cannot support a technical-only quality comparison; V2 reports
that uncertainty instead of using face-level sharpness. Winner disagreement
is descriptive, not evidence that V2 is more accurate. Human-labeled event
comparisons are required to calibrate weights and acceptance thresholds.

Future work may add validated expression/moment and composition measurements
and, separately, held-out evaluation of personal preferences. Neither is part
of this implementation.

## Offline event audit

The isolated 2026-11-03 run at `audit_results/phase8g2_release_20261103`
analyzed 1,732 photos, 8,453 faces and eight duplicate groups, with zero photo
failures. It evaluated all four preference combinations. With both face and
eye preferences enabled, V2 selected a different top candidate in four of
eight groups. Six groups lacked comparable face evidence, and all eight
groups had a top-two V2 margin below one point. These are **not** accuracy
results and many apparent reversals are uncertain.

For IMG_3951/3952 (the V2 tie-break selected IMG_3953) and IMG_3955/3956,
face evidence was not comparable across all group members and the whole-image
blur score saturated at 1.0; the order is a deterministic fallback, not a
quality judgment. For IMG_4184/4183, both components were available: V1
preferred IMG_4184, while V2 scored IMG_4183 about 0.45 points higher
(86.44 versus 86.00). This small reversal requires human review. With face
quality disabled, eye preference has no effect, as required. No human labels
were supplied, so no improvement claim is possible.
