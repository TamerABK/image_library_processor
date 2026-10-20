"""Project-scoped favorite metadata, independent of review decisions."""
from __future__ import annotations

from pathlib import Path

from app_state_database import AppStateDatabase
from .photo_model import photo_identity


class FavoritesStore:
    def __init__(self, database_path: str | Path | None = None):
        self._database = AppStateDatabase(database_path)

    def favorite_paths(self, folder: str | Path) -> set[str]:
        folder = photo_identity(folder)
        with self._database.connect() as db:
            return {row[0] for row in db.execute(
                "SELECT path FROM favorite_photos WHERE folder = ?", (folder,))}

    def is_favorite(self, folder: str | Path, path: str | Path) -> bool:
        folder, path = photo_identity(folder), photo_identity(path)
        with self._database.connect() as db:
            return db.execute("SELECT 1 FROM favorite_photos WHERE folder = ? AND path = ?",
                              (folder, path)).fetchone() is not None

    def set_favorite(self, folder: str | Path, path: str | Path, favorite: bool) -> None:
        folder, path = photo_identity(folder), photo_identity(path)
        with self._database.connect() as db:
            if favorite:
                db.execute("INSERT OR IGNORE INTO folder_state(folder) VALUES (?)", (folder,))
                db.execute("INSERT OR IGNORE INTO favorite_photos(folder, path) VALUES (?, ?)",
                           (folder, path))
            else:
                db.execute("DELETE FROM favorite_photos WHERE folder = ? AND path = ?",
                           (folder, path))
