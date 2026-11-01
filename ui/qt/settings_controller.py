"""Durable preferences and the single resolver for Qt scan snapshots."""
import json

from PySide6.QtCore import QObject, Property, Signal, Slot

from app_state_database import AppStateDatabase
from automatic_scan.config import (AutomaticScanConfig, BlurPolicy,
    DuplicateSelectionConfig, FaceQualityPreset, PerformanceConfig)
from automatic_scan.decode_pool import available_cpu_cores, automatic_decode_workers, max_manual_decode_workers


DEFAULTS = {
    "worker_count": "automatic", "memory_usage": "automatic",
    "blur_policy": "keep_all", "face_quality_preset": "balanced",
    "keepers_per_group": 1, "prefer_open_eyes": True,
    "prioritize_primary_face_quality": True,
}


def valid(key, value):
    if key == "worker_count":
        return value == "automatic" or type(value) is int and 1 <= value <= 4
    if key == "memory_usage":
        return value in ("automatic", "conservative")
    if key == "blur_policy":
        return value in ("automatic", "review_borderline", "keep_all")
    if key == "face_quality_preset":
        return value in ("relaxed", "balanced", "strict")
    if key == "keepers_per_group":
        return type(value) is int and 1 <= value <= 5
    if key in ("prefer_open_eyes", "prioritize_primary_face_quality"):
        return type(value) is bool
    return False


class SettingsController(QObject):
    changed = Signal()

    def __init__(self, database=None, parent=None):
        super().__init__(parent)
        self.database = database or AppStateDatabase()

    def _read(self, table, project_id=None):
        with self.database.connect() as db:
            if project_id is None:
                rows = db.execute("SELECT key, value FROM global_preferences").fetchall()
            else:
                rows = db.execute("SELECT key, value FROM project_preference_overrides WHERE project_id=?",
                                  (project_id,)).fetchall()
        result = {}
        for row in rows:
            try:
                value = json.loads(row["value"])
                if valid(row["key"], value):
                    result[row["key"]] = value
            except (ValueError, TypeError):
                pass
        return result

    def resolve(self, project_id=None):
        values = {**DEFAULTS, **self._read("global_preferences")}
        if project_id:
            values.update(self._read("project_preference_overrides", project_id))
        return values

    def scan_config(self, project_id):
        values = self.resolve(project_id)
        workers = values["worker_count"]
        return AutomaticScanConfig(
            blur_policy=BlurPolicy(values["blur_policy"]),
            face_quality_preset=FaceQualityPreset(values["face_quality_preset"]),
            duplicate_selection=DuplicateSelectionConfig(
                keepers_per_group=values["keepers_per_group"],
                prefer_open_eyes=values["prefer_open_eyes"],
                prioritize_primary_face_quality=values["prioritize_primary_face_quality"]),
            performance=PerformanceConfig(worker_count=(automatic_decode_workers() if workers == "automatic" else min(workers, max_manual_decode_workers())),
                memory_conservative=values["memory_usage"] == "conservative", batch_size=32))

    @Slot(int, result=bool)
    def requiresCpuConfirmation(self, selected_workers):
        return type(selected_workers) is int and 1 <= selected_workers <= 4 and selected_workers / available_cpu_cores()[0] >= 0.75

    @Slot(str, result="QVariantMap")
    def pageData(self, project_id):
        global_values = {**DEFAULTS, **self._read("global_preferences")}
        overrides = self._read("project_preference_overrides", project_id) if project_id else {}
        limit = max_manual_decode_workers()
        for source in (global_values, overrides):
            if type(source.get("worker_count")) is int:
                source["worker_count"] = min(source["worker_count"], limit)
        values = {**global_values, **overrides}
        return {"values": values, "overrides": overrides,
                "globalValues": global_values, "defaults": DEFAULTS,
                "cpuCores": available_cpu_cores()[0], "cpuSource": available_cpu_cores()[1],
                "maxWorkers": limit}

    @Slot(str, "QVariantMap", result=bool)
    def save(self, project_id, values):
        if any(not valid(key, value) or key == "worker_count" and type(value) is int
               and value > max_manual_decode_workers() for key, value in values.items()):
            return False
        with self.database.connect() as db:
            if project_id:
                if not db.execute("SELECT 1 FROM projects WHERE project_id=?", (project_id,)).fetchone():
                    return False
                db.execute("DELETE FROM project_preference_overrides WHERE project_id=?", (project_id,))
                db.executemany("INSERT INTO project_preference_overrides VALUES (?, ?, ?)",
                    [(project_id, key, json.dumps(value)) for key, value in values.items()])
            else:
                db.execute("DELETE FROM global_preferences")
                db.executemany("INSERT INTO global_preferences VALUES (?, ?)",
                    [(key, json.dumps(value)) for key, value in values.items()])
        self.changed.emit()
        return True

    @Slot(str)
    def reset(self, project_id):
        self.save(project_id, {})
