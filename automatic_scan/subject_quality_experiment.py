"""Audit-only subject and scene measurements for confirmed duplicate groups."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

VERSION = "subject-quality-experiment-v1"
DETECTION_MAX_DIMENSION = 2048


def detection_space_size(width, height):
    """Face boxes are cached in ImageLoader's 2048-pixel detection space."""
    scale = min(1.0, DETECTION_MAX_DIMENSION/max(width, height))
    return width*scale, height*scale


@dataclass(frozen=True)
class Registration:
    matrix: np.ndarray | None
    inliers: int
    ratio: float
    error: float | None
    status: str


def register(a, b):
    """Map coordinates in a to b; abstain on weak or inconsistent geometry."""
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1200)
    ka, da = orb.detectAndCompute(gray_a, None)
    kb, db = orb.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return Registration(None, 0, 0, None, "insufficient_keypoints")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [x for x, y in pairs if x.distance < .72*y.distance]
    if len(good) < 12:
        return Registration(None, 0, 0, None, "insufficient_matches")
    src = np.float32([ka[x.queryIdx].pt for x in good])
    dst = np.float32([kb[x.trainIdx].pt for x in good])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or mask is None:
        return Registration(None, 0, 0, None, "homography_failed")
    included = mask.ravel().astype(bool)
    inliers = int(included.sum())
    ratio = inliers / len(good)
    projected = cv2.perspectiveTransform(src[included][None], matrix)[0]
    error = float(np.median(np.linalg.norm(projected - dst[included], axis=1))) if inliers else None
    if inliers < 12 or ratio < .45 or error is None or error > 3:
        return Registration(None, inliers, ratio, error, "registration_rejected")
    return Registration(matrix, inliers, ratio, error, "registered")


def map_point(matrix, point):
    if matrix is None:
        return None
    result = cv2.perspectiveTransform(np.float32([[point]]), matrix)[0, 0]
    return tuple(float(x) for x in result) if np.isfinite(result).all() else None


def match_faces(faces_a, faces_b, registration, size_a, size_b):
    """Provisional one-to-one links: registered centers + visual crop agreement.

    Geometry alone never certifies a person identity; all links stay tentative
    without independent embeddings.
    """
    if registration.matrix is None or not faces_a or not faces_b:
        return [], "registration_or_faces_unavailable"
    costs = np.full((len(faces_a), len(faces_b)), 1e6, np.float64)
    for i, fa in enumerate(faces_a):
        ax0, ay0, ax1, ay1 = fa["decoded_box"]
        center = ((ax0+ax1)/2, (ay0+ay1)/2)
        mapped = map_point(registration.matrix, center)
        if mapped is None:
            continue
        for j, fb in enumerate(faces_b):
            bx0, by0, bx1, by1 = fb["decoded_box"]
            target = ((bx0+bx1)/2, (by0+by1)/2)
            distance = float(np.hypot(mapped[0]-target[0], mapped[1]-target[1]))
            scale = max(20., ((ax1-ax0)+(ay1-ay0)+(bx1-bx0)+(by1-by0))/4)
            costs[i, j] = distance/scale
    rr, cc = linear_sum_assignment(costs)
    links = []
    for i, j in zip(rr, cc):
        row = costs[i]
        alternate = min((v for k, v in enumerate(row) if k != j), default=1e6)
        margin = alternate - costs[i, j]
        if costs[i, j] < .35 and margin > .15:
            links.append(dict(a=i, b=j, normalized_distance=float(costs[i, j]),
                              ambiguity_margin=float(margin), status="tentative_geometry",
                              confidence=float(min(.7, registration.ratio * (1-costs[i, j])))))
    return links, "tentative_without_embeddings" if links else "ambiguous_or_unmatched"


def face_region(image, box, source_size):
    sx, sy = image.shape[1]/source_size[0], image.shape[0]/source_size[1]
    x, y, w, h = box
    x0, y0 = max(0, int(x*sx)), max(0, int(y*sy))
    x1, y1 = min(image.shape[1], int((x+w)*sx)), min(image.shape[0], int((y+h)*sy))
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def region_metrics(image):
    if image is None or min(image.shape[:2]) < 16:
        return dict(status="insufficient_region")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray.astype(np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1)
    energy_x, energy_y = float(np.mean(gx*gx)), float(np.mean(gy*gy))
    lap = float(cv2.Laplacian(values, cv2.CV_32F).var())
    clipped = np.mean((image <= 1) | (image >= 254), axis=(0, 1))
    # A diagnostic proxy; periodic scene content can mimic JPEG blocks.
    vertical = float(np.mean(np.abs(values[:, 8::8] - values[:, 7:-1:8]))) if image.shape[1] > 16 else 0
    horizontal = float(np.mean(np.abs(values[8::8, :] - values[7:-1:8, :]))) if image.shape[0] > 16 else 0
    return dict(status="measured", laplacian=lap, tenengrad=energy_x+energy_y,
                anisotropy=abs(energy_x-energy_y)/(energy_x+energy_y+1),
                possible_motion_smear="uncertain", jpeg_boundary_proxy=(vertical+horizontal)/2,
                bright_clip_bgr=[float(x) for x in np.mean(image >= 254, axis=(0, 1))],
                dark_clip_bgr=[float(x) for x in np.mean(image <= 1, axis=(0, 1))],
                median_luminance=float(np.median(gray)),
                region_pixels=int(gray.size))
