"""Pure projections of the Phase 6 People aggregate for Qt presentation."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import os


def _face_identity(face: dict) -> tuple[str, tuple[int, int, int, int]]:
    path = os.path.normcase(os.path.normpath(str(face["path"])))
    bbox = tuple(int(value) for value in face["bbox"])
    if len(bbox) != 4:
        raise ValueError("Invalid cached face bounds")
    return path, bbox


def stable_cluster_id(cluster: dict) -> str:
    """Content identity; Phase 6's integer cluster label is intentionally ignored."""
    faces = sorted(_face_identity(face) for face in cluster["faces"])
    if not faces:
        raise ValueError("Empty People cluster")
    payload = json.dumps(faces, ensure_ascii=False, separators=(",", ":"))
    return "faces-v1:" + sha256(payload.encode("utf-8")).hexdigest()


def project_people_rows(payload: dict, names: dict[int, str], ignored: set[str]):
    known_by_id = defaultdict(list)
    for face in payload.get("known", ()):
        person_id = int(face["person_id"])
        if person_id in names:
            known_by_id[person_id].append(face)
    known = []
    for person_id, faces in known_by_id.items():
        # The aggregate has no quality score. Use the largest face, then a
        # stable path/box order; this never changes the stored detector bounds.
        representative = min(faces, key=lambda face: (
            -max(0, int(face["bbox"][2])) * max(0, int(face["bbox"][3])),
            os.path.normcase(str(face["path"])), tuple(face["bbox"])))
        known.append(dict(key=str(person_id), personId=person_id, name=names[person_id],
                          faceCount=len(faces), photoCount=len({str(face["path"]) for face in faces}),
                          representativePhotoId=str(representative["path"]),
                          representativeBBox=list(representative["bbox"]),
                          paths=sorted({str(face["path"]) for face in faces})))
    known.sort(key=lambda row: (row["name"].casefold(), row["personId"]))

    unknown = []
    for cluster in payload.get("unknown", ()):
        key = stable_cluster_id(cluster)
        faces = cluster["faces"]
        paths = sorted({str(face["path"]) for face in faces})
        unknown.append(dict(key=key, clusterId=key, faceCount=len(faces),
                            name="",
                            photoCount=len(paths), representativePhotoId=str(cluster["representative"]),
                            representativeBBox=list(cluster["representative_bbox"]),
                            ignored=key in ignored, paths=paths, faces=faces))
    unknown.sort(key=lambda row: (-row["faceCount"], row["key"]))
    return known, unknown
