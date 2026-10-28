import sqlite3
import threading
from pathlib import Path

import numpy as np
from image_cache_storage import _CacheConnection

from face_processing.interfaces import EmbeddingSimilarity, FaceDatabase
from face_processing.models import Match, Person, StoredEmbedding


class SQLiteFaceDatabase(FaceDatabase):
    _shared_lock = threading.RLock()
    _shared_epochs: dict[Path, int] = {}

    def __init__(
        self,
        db_path: str | Path,
        similarity: EmbeddingSimilarity,
    ):
        self._db_path = Path(db_path).resolve()
        self._similarity = similarity
        self._cache_lock = threading.RLock()
        self._people_cache: dict[int, Person] | None = None
        self._people_names_cache: list[str] | None = None
        self._embedding_cache: list[tuple[int, np.ndarray]] | None = None
        self._observed_epoch = 0

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def contains_people(self) -> bool:
        return bool(self._people_by_id())

    def cache_signature(self) -> str:
        with self._connect() as connection:
            person_row = connection.execute(
                """
                SELECT COUNT(*) AS count, COALESCE(MAX(id), 0) AS max_id
                FROM persons
                """
            ).fetchone()
            embedding_row = connection.execute(
                """
                SELECT COUNT(*) AS count, COALESCE(MAX(id), 0) AS max_id
                FROM embeddings
                """
            ).fetchone()

        return (
            f"{person_row['count']}:{person_row['max_id']}:"
            f"{embedding_row['count']}:{embedding_row['max_id']}"
        )

    def add_person(
        self,
        name: str,
    ) -> Person:
        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError("Person name cannot be empty.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO persons (name)
                VALUES (?)
                """,
                (normalized_name,),
            )

            row = connection.execute(
                """
                SELECT id, name
                FROM persons
                WHERE name = ?
                """,
                (normalized_name,),
            ).fetchone()

        if row is None:
            raise RuntimeError(f"Failed to load person record for {normalized_name!r}.")

        person = Person(
            id=row["id"],
            name=row["name"],
        )
        self._invalidate_caches()
        return person

    def add_embedding(
        self,
        person_id: int,
        embedding: np.ndarray,
    ) -> StoredEmbedding:
        serialized = self._serialize_embedding(embedding)

        with self._connect() as connection:
            person_exists = connection.execute(
                """
                SELECT 1
                FROM persons
                WHERE id = ?
                """,
                (person_id,),
            ).fetchone()

            if person_exists is None:
                raise ValueError(f"Unknown person id: {person_id}")

            cursor = connection.execute(
                """
                INSERT INTO embeddings (person_id, embedding)
                VALUES (?, ?)
                """,
                (person_id, serialized),
            )

        self._invalidate_caches()
        return StoredEmbedding(
            id=cursor.lastrowid,
            person_id=person_id,
            embedding=np.asarray(embedding, dtype=np.float32).copy(),
        )

    def assign_embeddings(self, embeddings, *, name: str | None = None,
                          person_id: int | None = None, canceled=None) -> Person:
        """Assign cached cluster vectors atomically to a new or existing person."""
        if (name is None) == (person_id is None):
            raise ValueError("Choose either a name or an existing person.")
        serialized = [self._serialize_embedding(value) for value in embeddings]
        if not serialized:
            raise ValueError("No cached face embeddings are available.")
        with self._connect() as connection:
            if person_id is None:
                normalized_name = name.strip()
                if not normalized_name:
                    raise ValueError("Person name cannot be empty.")
                # SQLite NOCASE covers ASCII. Python casefold also avoids duplicate
                # Unicode names that differ only in case.
                rows = connection.execute("SELECT id, name FROM persons").fetchall()
                row = next((item for item in rows
                            if item["name"].casefold() == normalized_name.casefold()), None)
                if row is None:
                    connection.execute("INSERT OR IGNORE INTO persons (name) VALUES (?)",
                                       (normalized_name,))
                    row = connection.execute(
                        "SELECT id, name FROM persons WHERE name = ? COLLATE NOCASE",
                        (normalized_name,)).fetchone()
            else:
                row = connection.execute("SELECT id, name FROM persons WHERE id=?",
                                         (person_id,)).fetchone()
            if row is None:
                raise ValueError("The selected known person is no longer available.")
            for value in serialized:
                if canceled is not None and canceled.is_set():
                    raise ValueError("People review was canceled.")
                connection.execute("INSERT INTO embeddings (person_id, embedding) VALUES (?,?)",
                                   (row["id"], value))
            if canceled is not None and canceled.is_set():
                raise ValueError("People review was canceled.")
        self._invalidate_caches()
        return Person(id=row["id"], name=row["name"])

    def rename_person(self, person_id: int, name: str, *, canceled=None) -> Person:
        normalized = name.strip()
        if not normalized:
            raise ValueError("Person name cannot be empty.")
        with self._connect() as connection:
            rows = connection.execute("SELECT id, name FROM persons").fetchall()
            if not any(row["id"] == person_id for row in rows):
                raise ValueError("The selected known person is no longer available.")
            if any(row["id"] != person_id and row["name"].casefold() == normalized.casefold()
                   for row in rows):
                raise ValueError("That person already exists.")
            if canceled is not None and canceled.is_set():
                raise ValueError("People review was canceled.")
            try:
                connection.execute("UPDATE persons SET name=? WHERE id=?", (normalized, person_id))
            except sqlite3.IntegrityError as exc:
                raise ValueError("That person already exists.") from exc
            if canceled is not None and canceled.is_set():
                raise ValueError("People review was canceled.")
        self._invalidate_caches()
        return Person(id=person_id, name=normalized)

    def delete_person(self, person_id: int, *, canceled=None) -> None:
        """Remove one identity and its embeddings; photo files are independent."""
        with self._connect() as connection:
            if canceled is not None and canceled.is_set():
                raise ValueError("People review was canceled.")
            cursor = connection.execute("DELETE FROM persons WHERE id=?", (person_id,))
            if cursor.rowcount != 1:
                raise ValueError("The selected known person is no longer available.")
            if canceled is not None and canceled.is_set():
                raise ValueError("People review was canceled.")
        self._invalidate_caches()

    def find_nearest_embedding(
        self,
        embedding: np.ndarray,
    ) -> Match | None:
        candidate = np.asarray(embedding, dtype=np.float32)
        cached_embeddings = self._embeddings()
        if not cached_embeddings:
            return None

        best_person_id = None
        best_score = float("-inf")

        for stored_person_id, stored_embedding in cached_embeddings:
            score = self._similarity.score(candidate, stored_embedding)

            if score > best_score:
                best_score = score
                best_person_id = stored_person_id

        if best_person_id is None:
            return None

        return Match(
            person_id=best_person_id,
            score=best_score,
        )

    def get_person(
        self,
        person_id: int,
    ) -> Person | None:
        return self._people_by_id().get(person_id)

    def list_people_names(self) -> list[str]:
        with self._cache_lock:
            if self._people_names_cache is None:
                self._people_by_id()
            return list(self._people_names_cache or [])

    def list_people(self) -> list[Person]:
        return list(self._people_by_id().values())

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, factory=_CacheConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _people_by_id(self) -> dict[int, Person]:
        with self._cache_lock:
            self._refresh_cache_epoch()
            if self._people_cache is not None:
                return self._people_cache

            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT id, name
                    FROM persons
                    ORDER BY name COLLATE NOCASE, id
                    """
                ).fetchall()

            people_cache = {
                row["id"]: Person(
                    id=row["id"],
                    name=row["name"],
                )
                for row in rows
            }
            self._people_cache = people_cache
            self._people_names_cache = [row["name"] for row in rows]
            return people_cache

    def _embeddings(self) -> list[tuple[int, np.ndarray]]:
        with self._cache_lock:
            self._refresh_cache_epoch()
            if self._embedding_cache is not None:
                return self._embedding_cache

            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT person_id, embedding
                    FROM embeddings
                    """
                ).fetchall()

            self._embedding_cache = [
                (row["person_id"], self._deserialize_embedding(row["embedding"]))
                for row in rows
            ]
            return self._embedding_cache

    def _refresh_cache_epoch(self) -> None:
        with self._shared_lock:
            epoch = self._shared_epochs.get(self._db_path, 0)
        if epoch != self._observed_epoch:
            self._people_cache = None
            self._people_names_cache = None
            self._embedding_cache = None
            self._observed_epoch = epoch

    def _invalidate_caches(self) -> None:
        with self._cache_lock:
            with self._shared_lock:
                epoch = self._shared_epochs.get(self._db_path, 0) + 1
                self._shared_epochs[self._db_path] = epoch
            self._observed_epoch = epoch
            self._people_cache = None
            self._people_names_cache = None
            self._embedding_cache = None

    @staticmethod
    def _serialize_embedding(
        embedding: np.ndarray,
    ) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _deserialize_embedding(
        payload: bytes,
    ) -> np.ndarray:
        return np.frombuffer(payload, dtype=np.float32).copy()
