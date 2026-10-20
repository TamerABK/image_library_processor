"""Durable Room 36 application state, separate from disposable analysis caches."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3

from app_paths import app_data_path


APP_DATABASE_FILENAME = "room36_app.sqlite3"


class AppStateDatabase:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else app_data_path(APP_DATABASE_FILENAME)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS app_migrations (
                    migration_key TEXT PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    folder TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_opened REAL NOT NULL DEFAULT 0,
                    photo_count INTEGER NOT NULL DEFAULT -1 CHECK (photo_count >= -1)
                );
                CREATE TABLE IF NOT EXISTS folder_state (
                    folder TEXT PRIMARY KEY,
                    current_tab TEXT NOT NULL DEFAULT 'Scenes'
                );
                CREATE TABLE IF NOT EXISTS ignored_unknown_clusters (
                    folder TEXT NOT NULL REFERENCES folder_state(folder) ON DELETE CASCADE,
                    cluster_id TEXT NOT NULL,
                    PRIMARY KEY (folder, cluster_id)
                );
                CREATE TABLE IF NOT EXISTS favorite_photos (
                    folder TEXT NOT NULL REFERENCES folder_state(folder) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    PRIMARY KEY (folder, path)
                );
                CREATE TABLE IF NOT EXISTS image_decisions (
                    folder TEXT NOT NULL REFERENCES folder_state(folder) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    selected_for_deletion INTEGER NOT NULL CHECK (selected_for_deletion IN (0, 1)),
                    manually_restored INTEGER NOT NULL CHECK (manually_restored IN (0, 1)),
                    manually_kept INTEGER NOT NULL CHECK (manually_kept IN (0, 1)),
                    manual_override TEXT,
                    duplicate_group_id TEXT,
                    is_duplicate_keeper INTEGER CHECK (is_duplicate_keeper IN (0, 1)),
                    review_state TEXT,
                    PRIMARY KEY (folder, path)
                );
                CREATE TABLE IF NOT EXISTS duplicate_group_reviews (
                    folder TEXT NOT NULL REFERENCES folder_state(folder) ON DELETE CASCADE,
                    group_id TEXT NOT NULL,
                    revision_signature TEXT NOT NULL,
                    keeper_paths TEXT NOT NULL,
                    member_intents TEXT NOT NULL,
                    reviewed INTEGER NOT NULL CHECK (reviewed IN (0, 1)),
                    PRIMARY KEY (folder, group_id)
                );
                CREATE TABLE IF NOT EXISTS deletion_transactions (
                    id INTEGER PRIMARY KEY,
                    folder TEXT NOT NULL REFERENCES folder_state(folder) ON DELETE CASCADE,
                    created_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deletion_transaction_paths (
                    transaction_id INTEGER NOT NULL REFERENCES deletion_transactions(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK (kind IN ('paths', 'failed_paths')),
                    position INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    PRIMARY KEY (transaction_id, kind, position)
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=30000")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
