"""Compatibility and project aggregates in the existing unified cache database."""
import hashlib
import json

from image_cache_storage import SharedImageCacheDatabase


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class ProjectStageCache:
    def __init__(self, path=None):
        self.storage = SharedImageCacheDatabase(path)
        self.storage.initialize()
        with self.storage.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS processor_compatibility (
                    path TEXT NOT NULL, stage TEXT NOT NULL, signature TEXT NOT NULL,
                    PRIMARY KEY(path, stage),
                    FOREIGN KEY(path) REFERENCES image_entries(path) ON DELETE CASCADE);
                CREATE TABLE IF NOT EXISTS project_stage_results (
                    folder TEXT NOT NULL, stage TEXT NOT NULL, signature TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(folder, stage));
            """)

    def compatible(self, path, stage, key):
        with self.storage.connect() as db:
            row = db.execute("SELECT signature FROM processor_compatibility WHERE path=? AND stage=?",
                             (str(path), stage)).fetchone()
        return row is not None and row[0] == key

    def mark(self, path, stage, key):
        with self.storage.connect() as db:
            db.execute("INSERT OR REPLACE INTO processor_compatibility VALUES (?,?,?)", (str(path), stage, key))

    def get(self, folder, stage, key):
        with self.storage.connect() as db:
            row = db.execute("SELECT payload FROM project_stage_results WHERE folder=? AND stage=? AND signature=?",
                             (str(folder), stage, key)).fetchone()
        try:
            return None if row is None else json.loads(row[0])
        except (ValueError, TypeError):
            return None

    def put(self, folder, stage, key, payload):
        with self.storage.connect() as db:
            db.execute("INSERT OR REPLACE INTO project_stage_results VALUES (?,?,?,?)",
                       (str(folder), stage, key, json.dumps(payload)))
