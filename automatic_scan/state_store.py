from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from app_paths import app_data_path
from app_state_database import APP_DATABASE_FILENAME, AppStateDatabase

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
    review_state: str | None = None


@dataclass(slots=True)
class FolderAutomaticState:
    folder: str
    current_tab: str = "Scenes"
    ignored_unknown_cluster_ids: set[str] = field(default_factory=set)
    image_decisions: dict[str, PersistedImageDecision] = field(default_factory=dict)
    deletion_transactions: list[DeletionTransactionRecord] = field(default_factory=list)


class AutomaticStateStore:
    def __init__(self, root_dir: str | Path | None = None, *, database_path: str | Path | None = None) -> None:
        self._root = Path(root_dir) if root_dir is not None else app_data_path("automatic_scan_state")
        path = database_path or (self._root / APP_DATABASE_FILENAME if root_dir is not None
                                 else app_data_path(APP_DATABASE_FILENAME))
        self._database = AppStateDatabase(path)
        self._migrate_legacy_states()

    def load(self, folder: str | Path) -> FolderAutomaticState:
        key = str(Path(folder).resolve())
        with self._database.connect() as db:
            row = db.execute("SELECT folder, current_tab FROM folder_state WHERE folder=?", (key,)).fetchone()
            if row is not None:
                decisions = {
                    item["path"]: PersistedImageDecision(
                        path=item["path"], file_size=item["file_size"], mtime_ns=item["mtime_ns"],
                        selected_for_deletion=bool(item["selected_for_deletion"]),
                        manually_restored=bool(item["manually_restored"]), manually_kept=bool(item["manually_kept"]),
                        manual_override=item["manual_override"], duplicate_group_id=item["duplicate_group_id"],
                        is_duplicate_keeper=None if item["is_duplicate_keeper"] is None else bool(item["is_duplicate_keeper"]),
                        review_state=item["review_state"])
                    for item in db.execute("SELECT * FROM image_decisions WHERE folder=?", (key,))
                }
                ignored = {item[0] for item in db.execute(
                    "SELECT cluster_id FROM ignored_unknown_clusters WHERE folder=?", (key,))}
                transactions = []
                for item in db.execute("SELECT id, created_at_utc FROM deletion_transactions WHERE folder=? ORDER BY id", (key,)):
                    paths = {"paths": [], "failed_paths": []}
                    for part in db.execute("SELECT kind, path FROM deletion_transaction_paths "
                                           "WHERE transaction_id=? ORDER BY kind, position", (item["id"],)):
                        paths[part["kind"]].append(part["path"])
                    transactions.append(DeletionTransactionRecord(item["created_at_utc"], paths["paths"], paths["failed_paths"]))
                return FolderAutomaticState(key, row["current_tab"], ignored, decisions, transactions)
        path = self._state_path(key)
        if path.is_file():
            try:
                state = self._load_legacy(path, key)
            except (OSError, ValueError, KeyError, TypeError, OverflowError):
                return FolderAutomaticState(folder=key)
            self.save(state)
            return state
        return FolderAutomaticState(folder=key)

    def ignored_clusters(self, folder: str | Path) -> set[str]:
        key = str(Path(folder).resolve())
        with self._database.connect() as db:
            return {row[0] for row in db.execute(
                "SELECT cluster_id FROM ignored_unknown_clusters WHERE folder=?", (key,))}

    def set_cluster_ignored(self, folder: str | Path, cluster_id: str, ignored: bool) -> None:
        """Persist one cluster without rewriting unrelated photo review state."""
        key = str(Path(folder).resolve())
        with self._database.connect() as db:
            if ignored:
                db.execute("INSERT OR IGNORE INTO folder_state(folder) VALUES (?)", (key,))
                db.execute("INSERT OR IGNORE INTO ignored_unknown_clusters VALUES (?,?)",
                           (key, cluster_id))
            else:
                db.execute("DELETE FROM ignored_unknown_clusters WHERE folder=? AND cluster_id=?",
                           (key, cluster_id))

    @staticmethod
    def _load_legacy(path: Path, fallback_folder: str | None = None) -> FolderAutomaticState:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Legacy folder state is not an object")
        folder = payload.get("folder", fallback_folder)
        if not isinstance(folder, str) or not folder:
            raise ValueError("Legacy folder state is missing its folder")
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
                review_state=item.get("review_state"),
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
            folder=folder,
            current_tab=str(payload.get("current_tab", "Scenes")),
            ignored_unknown_cluster_ids=set(payload.get("ignored_unknown_cluster_ids", [])),
            image_decisions=decisions,
            deletion_transactions=transactions,
        )

    def save(self, state: FolderAutomaticState) -> None:
        folder = str(Path(state.folder).resolve())
        with self._database.connect() as db:
            db.execute("INSERT INTO folder_state VALUES (?,?) ON CONFLICT(folder) "
                       "DO UPDATE SET current_tab=excluded.current_tab", (folder, state.current_tab))
            db.execute("DELETE FROM ignored_unknown_clusters WHERE folder=?", (folder,))
            db.executemany("INSERT INTO ignored_unknown_clusters VALUES (?,?)",
                           ((folder, value) for value in sorted(state.ignored_unknown_cluster_ids)))
            db.execute("DELETE FROM image_decisions WHERE folder=?", (folder,))
            db.executemany("INSERT INTO image_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                (folder, item.path, item.file_size, item.mtime_ns, int(item.selected_for_deletion),
                 int(item.manually_restored), int(item.manually_kept), item.manual_override,
                 item.duplicate_group_id, None if item.is_duplicate_keeper is None else int(item.is_duplicate_keeper),
                 item.review_state)
                for item in state.image_decisions.values()))
            db.execute("DELETE FROM deletion_transactions WHERE folder=?", (folder,))
            for transaction in state.deletion_transactions:
                cursor = db.execute("INSERT INTO deletion_transactions(folder, created_at_utc) VALUES (?,?)",
                                    (folder, transaction.created_at_utc))
                db.executemany("INSERT INTO deletion_transaction_paths VALUES (?,?,?,?)", (
                    (cursor.lastrowid, kind, position, path)
                    for kind, values in (("paths", transaction.paths), ("failed_paths", transaction.failed_paths))
                    for position, path in enumerate(values)))

    def _migrate_legacy_states(self) -> None:
        with self._database.connect() as db:
            if db.execute("SELECT 1 FROM app_migrations WHERE migration_key='folder_state_json_v1'").fetchone():
                return
        failed = False
        if self._root.is_dir():
            for path in self._root.glob("*.json"):
                try:
                    state = self._load_legacy(path)
                    with self._database.connect() as db:
                        exists = db.execute("SELECT 1 FROM folder_state WHERE folder=?",
                                            (str(Path(state.folder).resolve()),)).fetchone()
                    if not exists:
                        self.save(state)
                except (OSError, ValueError, KeyError, TypeError, OverflowError, sqlite3.DatabaseError):
                    failed = True  # Keep the JSON for a later repair/import attempt.
        if not failed:
            with self._database.connect() as db:
                db.execute("INSERT OR IGNORE INTO app_migrations VALUES ('folder_state_json_v1')")

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
        previous = state.image_decisions
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
                review_state=previous[image.path].review_state if image.path in previous else None,
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
