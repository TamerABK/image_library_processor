"""Opt-in pairwise technical calibration; no production ranking imports this module."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

VERSION = "technical-pairwise-calibration-v1"
LABELS = {"A", "B", "equal", "both_unacceptable", "cannot_determine"}
FEATURES = ("noise_aware_detail", "chroma_noise", "artifact_measurement")


def split_events(events, seed=3605):
    """Keep every group from one shoot on one side of the split."""
    unique = sorted(set(events))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    holdout = set(unique[:max(1, round(len(unique)*.25))]) if len(unique) > 1 else set()
    return {event: ("validation" if event in holdout else "train") for event in unique}


def validate_label(row):
    required = ("pair_id", "event_id", "group_id", "photo_a", "photo_b", "technical_preference")
    if any(not row.get(key) for key in required):
        raise ValueError("Incomplete pair identity or technical preference")
    if row["technical_preference"] not in LABELS:
        raise ValueError("Unknown technical preference")
    if row["photo_a"] == row["photo_b"]:
        raise ValueError("A pair needs two different photos")
    return row


@dataclass(frozen=True)
class CalibrationResult:
    version: str
    status: str
    score: float | None
    pairwise_status: str
    reasons: tuple[str, ...]
    parameters: dict


def calibrate(pairs, metric_rows, *, seed=3605, minimum_groups=50):
    """Fit only with independent event-level validation and complete features.

    Output is a research artifact, never a production Feature. Equal and
    unavailable judgments are counted but excluded from binary fitting.
    """
    labeled = [validate_label(p) for p in pairs if p.get("technical_preference")]
    distinct_groups = {(p["event_id"], p["group_id"]) for p in labeled}
    events = {p["event_id"] for p in labeled}
    fingerprint = hashlib.sha256(json.dumps(sorted((p["pair_id"], p["technical_preference"])
                                                for p in labeled)).encode()).hexdigest()
    base = dict(seed=seed, fingerprint=fingerprint, labeled_pairs=len(labeled),
                independent_groups=len(distinct_groups), events=len(events))
    if len(distinct_groups) < minimum_groups or len(events) < 3:
        return CalibrationResult(VERSION, "withheld", None, "abstain",
                                 ("insufficient_independent_labeled_groups_or_events",), base)
    split = split_events(events, seed)
    samples = []
    for pair in labeled:
        if pair["technical_preference"] not in ("A", "B"):
            continue
        a, b = metric_rows.get(pair["photo_a"]), metric_rows.get(pair["photo_b"])
        if not a or not b or any(a.get(k) is None or b.get(k) is None for k in FEATURES):
            continue
        difference = np.array([float(a[k])-float(b[k]) for k in FEATURES])
        samples.append((difference, int(pair["technical_preference"] == "A"), split[pair["event_id"]]))
    train = [(x, y) for x, y, side in samples if side == "train"]
    valid = [(x, y) for x, y, side in samples if side == "validation"]
    if len(train) < 30 or len(valid) < 10 or len({y for _, y in train}) < 2:
        return CalibrationResult(VERSION, "withheld", None, "abstain",
                                 ("insufficient_complete_train_validation_pairs",), base)
    x = np.stack([v for v, _ in train]); y = np.array([v for _, v in train])
    scale = np.maximum(np.subtract(*np.percentile(x, [75, 25], axis=0)), 1e-6)
    x = x/scale
    def objective(coef):
        z = x@coef
        return float(np.mean(np.logaddexp(0, z)-y*z)+.1*np.sum(coef*coef))
    fit = minimize(objective, np.zeros(len(FEATURES)), method="L-BFGS-B")
    if not fit.success:
        return CalibrationResult(VERSION, "withheld", None, "abstain", ("fit_failed",), base)
    vx = np.stack([v for v, _ in valid])/scale
    vy = np.array([target for _, target in valid])
    accuracy = float(np.mean((vx@fit.x > 0) == vy))
    majority = max(float(np.mean(vy)), float(1-np.mean(vy)))
    parameters = dict(base, features=FEATURES, scale=scale.tolist(), coefficients=fit.x.tolist(),
                      validation_accuracy=accuracy, validation_majority_baseline=majority,
                      validation_pairs=len(valid), split=split)
    if accuracy <= majority:
        return CalibrationResult(VERSION, "withheld", None, "abstain",
                                 ("no_heldout_advantage_over_majority_baseline",), parameters)
    # This model is pairwise only. An absolute 0..1 image Feature needs a
    # separate anchored calibration, so score stays unavailable even here.
    return CalibrationResult(VERSION, "pairwise_research_only", None, "abstain",
                             ("absolute_feature_not_calibrated",), parameters)
