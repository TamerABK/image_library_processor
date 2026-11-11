"""Experimental noise-aware image technical measurements, never a default V2 input."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import cv2
import numpy as np

VERSION = "technical-calibration-v6-evidence-contract"


@dataclass(frozen=True)
class TechnicalQualityResult:
    version: str
    status: str
    score: float | None
    detail_score: float | None
    noise_aware_detail: float | None
    edge_persistence: float | None
    noise_floor: float | None
    chroma_noise: float | None
    chroma_coverage: float
    clipping_bgr: tuple[float, float, float] | None
    confidence: float
    explanation: str
    detail_coverage: float = 0.
    detail_reliability: str = "unavailable"
    noise_coverage: float = 0.
    noise_reliability: str = "unavailable"
    artifact_measurement: float | None = None
    artifact_coverage: float = 0.
    artifact_reliability: str = "unavailable"
    clipping_coverage: float = 0.
    clipping_reliability: str = "unavailable"
    chroma_reliability: str = "unavailable"
    unavailable_reasons: tuple[str, ...] = ()
    luminance_noise_coverage: float = 0.
    chroma_noise_coverage: float = 0.
    empirical_repeatability: float | None = None
    defect_direction_reliability: str = "unvalidated"
    predictive_confidence: float | None = None

    def to_dict(self):
        return asdict(self)


def _gradient(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def chroma_noise_float_lab(image):
    """Robust median residual in OpenCV float Lab a/b; not a DeltaE value."""
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or min(image.shape[:2]) < 64:
        return None, 0., "unsupported_pixels"
    lab = cv2.cvtColor(image.astype(np.float32)/255., cv2.COLOR_BGR2Lab)
    luma = lab[:, :, 0]
    chroma = lab[:, :, 1:3]
    # Eligibility uses blurred luminance, so increasing sensor noise does not
    # automatically remove every low-texture pixel.
    smooth_luma = cv2.GaussianBlur(luma, (0, 0), 2.5)
    texture = _gradient(smooth_luma)
    eligible = (texture <= np.percentile(texture, 35)) & (luma > 3) & (luma < 97)
    coverage = float(np.mean(eligible))
    if coverage < .08:
        return None, coverage, "insufficient_low_texture_regions"
    median_bgr = cv2.medianBlur(image, 3).astype(np.float32)/255.
    median_chroma = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2Lab)[:, :, 1:3]
    residual = np.linalg.norm(chroma-median_chroma, axis=2)[eligible]
    # Robust location plus dispersion. Report raw units without claiming
    # that colorful microtexture has been perfectly removed.
    center = np.median(residual)
    return float(center + 1.4826*np.median(np.abs(residual-center))), coverage, "measured"


def chroma_noise_multiscale(image):
    """Alternative high-pass Lab chroma diagnostic, with the same eligibility mask."""
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or min(image.shape[:2]) < 64:
        return None
    lab = cv2.cvtColor(image.astype(np.float32)/255., cv2.COLOR_BGR2Lab)
    smooth = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 2.5)
    eligible = _gradient(smooth) <= np.percentile(_gradient(smooth), 35)
    coarse = cv2.GaussianBlur(lab[:, :, 1:3], (0, 0), 1.5)
    residual = np.linalg.norm(lab[:, :, 1:3]-coarse, axis=2)[eligible]
    return float(1.4826*np.median(np.abs(residual-np.median(residual)))) if residual.size else None


def _blocking_ratio(gray):
    """8-pixel boundary excess; evidence of block structure, not proof of JPEG damage."""
    if min(gray.shape) < 32:
        return None
    scores = []
    for axis in (0, 1):
        signal = gray if axis == 1 else gray.T
        boundary = np.abs(signal[:, 8::8]-signal[:, 7:-1:8])
        ordinary = np.abs(signal[:, 4::8]-signal[:, 3:-1:8])
        if boundary.size and ordinary.size:
            scores.append(float(np.mean(boundary))/(float(np.mean(ordinary))+.002))
    return max(scores) if scores else None


def measure_technical(image):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
        return TechnicalQualityResult(VERSION, "unsupported_image", None, None, None, None, None, None, 0.,
                                      None, 0., "Unsupported or too-small decoded image",
                                      unavailable_reasons=("unsupported_or_too_small",))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    clipping = tuple(float(v) for v in np.mean(image >= 254, axis=(0, 1)))
    chroma, coverage, _ = chroma_noise_float_lab(image)
    fine = _gradient(cv2.GaussianBlur(gray, (0, 0), .65))
    coarse = _gradient(cv2.GaussianBlur(gray, (0, 0), 2.0))
    mask = coarse >= max(.015, float(np.percentile(coarse, 80)))
    if int(mask.sum()) < 64:
        return TechnicalQualityResult(VERSION, "insufficient_structured_detail", None, None, None, None, None,
                                      chroma, coverage, clipping, 0., "Insufficient structured edges",
                                      chroma_reliability="measured_unvalidated" if chroma is not None else "unavailable",
                                      clipping_coverage=1., clipping_reliability="decoded_saturation_only",
                                      chroma_noise_coverage=coverage,
                                      unavailable_reasons=("insufficient_structured_edges",))
    # Persistent coarse edges are less sensitive to uncorrelated sensor noise.
    persistence = float(np.median(coarse[mask] / (fine[mask]+.002)))
    persistence = min(1., max(0., persistence))
    detail = float(np.median(fine[mask]))
    flat = coarse < np.percentile(coarse, 30)
    residual = np.abs(gray-cv2.GaussianBlur(gray, (0, 0), .7))
    noise = float(np.median(residual[flat])) if flat.any() else None
    adjusted = max(0., detail*persistence-noise) if noise is not None else None
    blocking = _blocking_ratio(gray)
    artifact_status = ("suspected_processing_artifact" if persistence < .65 or
                       (blocking is not None and blocking > 1.5) else "direction_unvalidated")
    # This score is deliberately uncalibrated. It cannot enter ranking until
    # controlled and labeled evidence support a mapping and noise gate.
    return TechnicalQualityResult(VERSION, "measured", None, detail, adjusted, persistence, noise,
                                  chroma, coverage, clipping, 0.,
                                  "Structured detail, noise and artifact observations; quality direction unvalidated",
                                  detail_coverage=float(mask.mean()),
                                  detail_reliability="artifact_suspected" if artifact_status == "suspected_processing_artifact" else "measured_unvalidated",
                                  noise_coverage=float(flat.mean()), noise_reliability="measured_unvalidated",
                                  artifact_measurement=blocking, artifact_coverage=1.,
                                  artifact_reliability=artifact_status, clipping_coverage=1.,
                                  clipping_reliability="decoded_saturation_only",
                                  chroma_reliability="measured_unvalidated" if chroma is not None else "unavailable",
                                  luminance_noise_coverage=float(flat.mean()), chroma_noise_coverage=coverage,
                                  unavailable_reasons=("chroma_unavailable",) if chroma is None else ())
