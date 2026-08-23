from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app_paths import app_data_path

from .models import ManualOverrideKind, PipelineImageState, PipelineImageStatus


STATE_VERSION = 1


@dataclass(slots=True)
class DeletionTransactionRecord:
    created_at_utc: str
    paths: list[str]
    failed_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PersistedImageDecision:
    path: str
    file_size: int
    mtime_ns: int
    selected_for_deletion: bool
    manually_restored: bool
    manually_kept: bool
    manual_override: str | None
    duplicate_group_id: str | None = None
    is_duplicate_keeper: bool | None = None


@dataclass(slots=True)
class FolderAutomaticState:
    folder: str
    current_tab: str = "Scenes"
    ignored_unknown_cluster_ids: set[str] = field(default_factory=set)
    image_decisions: dict[str, PersistedImageDecision] = field(default_factory=dict)
    deletion_transactions: list[DeletionTransactionRecord] = field(default_factory=list)


class AutomaticStateStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root = Path(root_dir) if root_dir is not None else app_data_path("automatic_scan_state")
        self._root.mkdir(parents=True, exist_ok=True)

    def load(self, folder: str | Path) -> FolderAutomaticState:
        path = self._state_path(folder)
        if not path.exists():
            return FolderAutomaticState(folder=str(Path(folder).resolve()))
        payload = json.loads(path.read_text(encoding="utf-8"))
        decisions = {
            item["path"]: PersistedImageDecision(
                path=item["path"],
                file_size=int(item["file_size"]),
                mtime_ns=int(item["mtime_ns"]),
                selected_for_deletion=bool(item["selected_for_deletion"]),
                manually_restored=bool(item["manually_restored"]),
                manually_kept=bool(item["manually_kept"]),
                manual_override=item.get("manual_override"),
                duplicate_group_id=item.get("duplicate_group_id"),
                is_duplicate_keeper=item.get("is_duplicate_keeper"),
            )
            for item in payload.get("image_decisions", [])
        }
        transactions = [
            DeletionTransactionRecord(
                created_at_utc=str(item["created_at_utc"]),
                paths=list(item.get("paths", [])),
                failed_paths=list(item.get("failed_paths", [])),
            )
            for item in payload.get("deletion_transactions", [])
        ]
        return FolderAutomaticState(
            folder=str(payload.get("folder", Path(folder).resolve())),
            current_tab=str(payload.get("current_tab", "Scenes")),
            ignored_unknown_cluster_ids=set(payload.get("ignored_unknown_cluster_ids", [])),
            image_decisions=decisions,
            deletion_transactions=transactions,
        )

    def save(self, state: FolderAutomaticState) -> None:
        path = self._state_path(state.folder)
        payload = {
            "version": STATE_VERSION,
            "folder": state.folder,
            "current_tab": state.current_tab,
            "ignored_unknown_cluster_ids": sorted(state.ignored_unknown_cluster_ids),
            "image_decisions": [
                {
                    "path": decision.path,
                    "file_size": decision.file_size,
                    "mtime_ns": decision.mtime_ns,
                    "selected_for_deletion": decision.selected_for_deletion,
                    "manually_restored": decision.manually_restored,
                    "manually_kept": decision.manually_kept,
                    "manual_override": decision.manual_override,
                    "duplicate_group_id": decision.duplicate_group_id,
                    "is_duplicate_keeper": decision.is_duplicate_keeper,
                }
                for decision in sorted(state.image_decisions.values(), key=lambda item: item.path)
            ],
            "deletion_transactions": [
                {
                    "created_at_utc": transaction.created_at_utc,
                    "paths": transaction.paths,
                    "failed_paths": transaction.failed_paths,
                }
                for transaction in state.deletion_transactions
            ],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def apply(self, state: FolderAutomaticState, images: list[PipelineImageState]) -> None:
        valid_paths = {image.path for image in images}
        stale_paths = [path for path in state.image_decisions if path not in valid_paths]
        for stale_path in stale_paths:
            state.image_decisions.pop(stale_path, None)

        for image in images:
            decision = state.image_decisions.get(image.path)
            if decision is None:
                continue
            if decision.file_size != image.file_size or decision.mtime_ns != image.mtime_ns:
                state.image_decisions.pop(image.path, None)
                continue
            image.selected_for_deletion = bool(decision.selected_for_deletion)
            image.manually_restored = bool(decision.manually_restored)
            image.manually_kept = bool(decision.manually_kept)
            if decision.manual_override:
                image.manual_override = ManualOverrideKind(decision.manual_override)
            if image.manually_restored and not image.is_duplicate_keeper:
                image.status = PipelineImageStatus.MANUALLY_RESTORED
            elif image.manually_kept and not image.is_duplicate_keeper:
                image.status = PipelineImageStatus.MANUALLY_KEPT

    def capture(self, state: FolderAutomaticState, images: list[PipelineImageState]) -> None:
        state.image_decisions = {}
        for image in images:
            state.image_decisions[image.path] = PersistedImageDecision(
                path=image.path,
                file_size=image.file_size,
                mtime_ns=image.mtime_ns,
                selected_for_deletion=image.selected_for_deletion,
                manually_restored=image.manually_restored,
                manually_kept=image.manually_kept,
                manual_override=None if image.manual_override is None else image.manual_override.value,
                duplicate_group_id=image.duplicate_group_id,
                is_duplicate_keeper=image.is_duplicate_keeper,
            )

    def record_deletion_transaction(
        self,
        state: FolderAutomaticState,
        *,
        paths: list[str],
        failed_paths: list[str],
    ) -> None:
        state.deletion_transactions.append(
            DeletionTransactionRecord(
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                paths=list(paths),
                failed_paths=list(failed_paths),
            )
        )

    def _state_path(self, folder: str | Path) -> Path:
        resolved_folder = str(Path(folder).resolve())
        digest = hashlib.sha256(resolved_folder.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.json"
