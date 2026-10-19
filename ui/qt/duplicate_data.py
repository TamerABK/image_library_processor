"""Small presentation projection of the central duplicate and ranking payloads."""
from __future__ import annotations

import hashlib

from automatic_scan.duplicate_selection import _group_id


def group_revision(paths: list[str], revisions: dict[str, tuple[int, int]]) -> str | None:
    digest = hashlib.sha256()
    for path in sorted(paths):
        revision = revisions.get(path)
        if revision is None:
            return None
        digest.update(path.encode("utf-8"))
        digest.update(f"\0{revision[0]}:{revision[1]}\0".encode("ascii"))
    return digest.hexdigest()


def project_duplicate_rows(duplicates, ranking, revisions, persisted, *, ranking_failed=False,
                           blur_reviews=None):
    blur_reviews = blur_reviews or {}
    ranked = {row["group_id"]: row for row in ranking or ()}
    rows = []
    for members in duplicates or ():
        paths = sorted(set(members))
        if len(paths) < 2:
            continue
        group_id = _group_id(paths)
        recommendation = ranked.get(group_id)
        recommended = list(recommendation["keeper_paths"]) if recommendation else []
        signature = group_revision(paths, revisions)
        saved = persisted.get(group_id, {})
        valid = bool(signature and saved.get("signature") == signature and
                     set(saved.get("keepers", ())).issubset(paths) and
                     set(saved.get("intents", {})).issubset(paths) and
                     (not recommended or len(saved.get("keepers", ())) == len(recommended)))
        keepers = list(saved["keepers"]) if valid else recommended
        intents = dict(saved.get("intents", {})) if valid else {}
        persisted_reviewed = bool(saved.get("reviewed")) if valid else False
        conflicting_paths = [path for path, intent in intents.items()
                             if (intent == "REMOVE" and blur_reviews.get(path) in ("KEEP", "REVIEW"))
                             or (intent == "KEEP" and blur_reviews.get(path) == "REMOVE")]
        reviewed = persisted_reviewed and not conflicting_paths
        representative = keepers[0] if keepers else paths[0]
        explanations = (recommendation or {}).get("ranking", {}).get(
            recommended[0] if recommended else "", {}).get("explanation", ())
        rows.append(dict(groupId=group_id, paths=paths, photoCount=len(paths),
                         recommendedKeepers=recommended, currentKeepers=keepers,
                         recommendedKeeperCount=len(recommended),
                         representativePhotoId=representative, recommendationState=(
                             "ready" if recommendation else "unavailable" if ranking_failed else "pending"),
                         explanation=list(explanations), intents=intents,
                         removeCount=sum(value == "REMOVE" for value in intents.values()),
                         persistedReviewed=persisted_reviewed, reviewed=reviewed,
                         hasConflict=bool(conflicting_paths), conflictingPaths=conflicting_paths,
                         reviewState=("Decision conflict" if conflicting_paths else
                                      "Reviewed" if reviewed else "Needs review"),
                         manualOverride=bool(valid and keepers != recommended) if recommended else bool(valid),
                         signature=signature or ""))
    return rows
