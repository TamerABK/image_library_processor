# Phase 8G.2b — V2 scoring reliability and regional experiment

## Scope

Fetched `origin/qt-qml-export-phase8d` before editing; local and remote HEAD
both matched `e04fa1561f1254283a8c48127e61a68d5947edaa`. V1 ranking,
duplicate grouping, face inference, blur classification, and deletion decisions
are unchanged. Production V2 remains a shadow result. The selected technical
method remains `raw-log-v1`; regional methods are audit experiments.

## Evidence policy (`keeper-v2.2-evidence-v1-raw-log-v1`)

Metrics are enabled by preference, evaluated per face, then admitted at the
group level by prominence-weighted measurement coverage. A usable measurement
must have confidence at least 0.25. Each admitted visual metric must cover at
least 50% of the selected face prominence in **every** candidate. At least two
visual metrics must qualify, including sharpness or pose, before the 40/55
face component is active. Visual metrics are sharpness, pose, exposure,
contrast, detail and visibility. Detector confidence is supporting metadata;
it cannot activate face scoring by itself or with only one visual metric.
These coverage gates are provisional engineering choices, not trained
photographic preferences. The 55/25/20 face aggregation and top-level 40/55
and 15/55 component weights are unchanged.

Eyes are handled separately. When enabled, each candidate needs at least 10%
prominence-weighted reliable eye coverage. Missing eyes receive the pooled
confidence-weighted eye value from the duplicate group; measured eyes blend
with that value according to confidence. The eye feature's effective weight is
multiplied by the **minimum** candidate eye coverage, so sparse evidence
cannot dominate a group. This avoids assuming missing eyes are open or closed
and avoids increasing the eye contribution when a photo has fewer assessed
faces. Confident closed-eye evidence remains measured. Insufficient group eye
coverage disables only the eye metric and is reported. The group pool is a
neutral comparison reference, not a claim about the missing person's eyes.

With eyes off, neither eye values nor eye-dependent reliability enter scoring,
eligibility, prominence or tie handling. With face quality off, face extraction
is skipped in the production shadow adapter and the ranker reads only technical
features. Face-off/eyes-on and face-off/eyes-off have identical results.

## Uncertainty

`Score.uncertainty_reasons` distinguishes insufficient visual face evidence,
insufficient eye coverage, partial face measurements, missing technical data,
low-confidence technical data, and near ties. `enabled_components` and
`unavailable_components` distinguish disabled versus unavailable components.
Face-off mode does not add a face-related uncertainty reason. All eight NSC
technical-only top comparisons remain uncertain **because their cached
raw-log margins are near ties**, not because face scoring was intentionally
disabled. A separate regression test verifies a clear technical-only margin
is not uncertain. Existing 0.15 and 0.5 score-point thresholds remain
provisional; no repeatability study justifies a change yet.

## Regional sharpness experiment (`regional-ratio-v3-mean-gradient`)

The experiment loads grayscale images at at most 1,024 pixels on the long
side, only for confirmed duplicate-group candidates, with at most 32 unique
images in memory per group. It divides each image into 36 regions, rejects
low-contrast or edge-poor regions, and uses a fine-to-coarse gradient ratio
with a small flat-region noise adjustment. Region scores use a trimmed median.
The registered variant uses phase correlation to estimate translation, rejects
weak or implausible registrations, shifts by integer pixels without blurring
only the candidate, and evaluates an identical central crop. The technical
experiment never receives face boxes, identities, eyes or poses. The audit
falls back to raw-log for the *whole group* if any candidate lacks regional
evidence. This method is deliberately experimental: texture, noise, subject
motion and local contrast can still confound it. CPBD was considered but not
added; its edge-width approach would add implementation and processing cost
without labels to validate the benefit. [OpenCV documentation](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html)
describes phase correlation's translation estimate and response; the [original CPBD paper](https://pubmed.ncbi.nlm.nih.gov/21447451/)
describes a perceptual blur metric. These sources motivate experiments, not
claims that our provisional ratio is perceptually accurate.

The regional cache is separate from face, blur and duplicate caches. Keys
include resolved path, size, modification time, algorithm version and maximum
decode dimension; registered group keys include all candidate signatures.
Cancellation is checked before processing and before each decode. A failed
decode or registration cannot change V1. The audit reuses each decoded image
within its group and releases buffers after evaluation; it does not keep a
library-wide decoded collection. There was no prefetched decode available
in the replay. A deferred production stage would avoid delaying V1, but is
not enabled without evidence that this method is worth its cost.

## NSC Meeting comparison

The previous audit supplied 1,732 photo records, 8 duplicate groups and 17
unique group candidates. New CSV/JSON exports in `audit_results/phase8g2b_final2`
include V1, original V2, Phase 8G.2a, new cached V2, regional B and registered
C rankings across all four preference combinations. Exported image IDs are
filenames rather than absolute local paths. Seven of eight groups have
sufficient face evidence; five of eight use actual eye evidence when enabled,
up from one with the previous all-face intersection. One group still falls
back to technical-only because visual face evidence is insufficient. The
corrected small-face group eligibility from Phase 8G.2a is unchanged.

For technical-only ranking, cached A had eight near top comparisons; regional
B had seven and registered C had six. B and C each changed four of eight
technical-only winners relative to A. With face scoring enabled, B and C did
not change the eight top winners in the replay, although margins shifted.
The first registered implementation used interpolation and favored its
reference image; that bias was found and removed before this benchmark.
No human preference labels are available, and most regional margins remain
near ties. Therefore the experiment does **not** justify replacing raw-log.
The audit exports provide per-candidate scores, margins, tie classes and
registration status for the named IMG_2928/2929, 3438/3439, 3951–3953,
3955/3956, 4183/4184 and 8190/8191 comparisons. Winner changes alone do not
establish identifiable photographic improvement.

Cold replay took 3.79 seconds overall, with 2.05 seconds in regional evaluation,
17 additional decodes, zero models, and 88.4 MB peak tracked Python allocations.
Warm replay took 1.74 seconds, with 17 regional cache hits and zero additional
decodes. The cached raw-log replay was about 1.39 seconds. These are local
replay measurements, not a full-scan latency estimate. Each group's images
are bounded and released before the next group.

## Validation and limits

Focused tests cover partial eyes, preference isolation, detector-only fallback,
technical-only uncertainty, synthetic defocus, translations, noise/exposure/
crop boundedness, cache invalidation, duplicate references and cancellation.
Controlled transformations check implementation behavior but do not prove
real-world keeper accuracy. The experiment still lacks human labels, robust
registration under framing changes, a noise sensitivity study, and a measured
confidence interval for the near-tie thresholds. No composition, moment,
expression or new ONNX model was added.
The local Windows CI command (`python -B -m unittest discover -p "test_*.py"`)
passed: 547 tests, 10 skipped, zero failures in 213.284 seconds. A final
targeted run passed 49 tests. The full log is
`audit_results/phase8g2b_windows_suite.txt`.
