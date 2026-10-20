"""Plain, project-scoped projection of explicit removal decisions."""
from __future__ import annotations

from pathlib import Path


def project_trash_rows(available, decisions, duplicate_rows, favorites, revisions, *, review_ready=True):
    available = set(available)
    favorites = set(favorites)
    members = {}
    for group in duplicate_rows:
        if not group.get("signature"):
            continue
        for path, intent in group.get("intents", {}).items():
            if path in available and intent in ("KEEP", "REMOVE"):
                members.setdefault(path, []).append((group["groupId"], group["signature"], intent))

    rows = []
    for path in sorted(available):
        decision = decisions.get(path)
        blur = decision.review_state if decision else None
        groups = members.get(path, [])
        sources = []
        if blur == "REMOVE":
            sources.append("Blur")
        if any(intent == "REMOVE" for _, _, intent in groups):
            sources.append("Near duplicates")
        if not sources:
            continue
        revision = revisions.get(path)
        if revision is None and decision:
            revision = (decision.file_size, decision.mtime_ns)
        changed = bool(decision and revision and (decision.file_size, decision.mtime_ns) != revision)
        conflict = ("Near duplicates says Remove; Blur says " + blur.title()
                    if "Near duplicates" in sources and blur in ("KEEP", "REVIEW") else
                    "Blur says Remove; Near duplicates says Keep"
                    if "Blur" in sources and any(intent == "KEEP" for _, _, intent in groups) else "")
        favorite = path in favorites
        reasons = []
        if changed:
            reasons.append("Photo changed since review. Reopen the project.")
        if conflict:
            reasons.append("Decision conflict: " + conflict + ".")
        if favorite:
            reasons.append("Favorite: remove it from Favorites or cancel removal.")
        if revision is None:
            reasons.append("File revision is unavailable. Reopen the project.")
        if not review_ready:
            reasons.append("Review choices are still loading or unavailable.")
        reason = " ".join(reasons)
        rows.append(dict(photoId=path, filename=Path(path).name,
                         removalSources=sources, duplicateGroups={gid: signature for gid, signature, intent in groups
                                                                      if intent == "REMOVE"},
                         blurRemove=blur == "REMOVE", favorite=favorite, conflictReason=reason,
                         executable=not bool(reason), fileSize=revision[0] if revision else 0,
                         revision=revision, badgeText=("Decision conflict" if conflict else
                             "Favorite" if favorite else "Changed" if changed else " · ".join(sources))))
    return rows
