from __future__ import annotations

import tempfile
import unittest
import json
import sqlite3
from pathlib import Path

from contextlib import closing

from automatic_scan.models import ManualOverrideKind, PipelineImageState, PipelineImageStatus
from automatic_scan.state_store import (AutomaticStateStore, FolderAutomaticState,
                                        DeletionTransactionRecord, PersistedImageDecision)


class AutomaticStateStoreTests(unittest.TestCase):
    def test_sqlite_restart_preserves_all_folder_details_without_writing_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = str((Path(temp_dir) / "photos").resolve())
            image = str(Path(folder) / "one.jpg")
            store = AutomaticStateStore(temp_dir)
            state = FolderAutomaticState(folder, current_tab="People",
                ignored_unknown_cluster_ids={"cluster-1"},
                image_decisions={image: PersistedImageDecision(image, 3, 42, True, False,
                    False, None, "duplicate-1", False, "REMOVE")},
                deletion_transactions=[DeletionTransactionRecord("2026-01-01T00:00:00Z", [image], ["missing.jpg"])])
            store.save(state)
            self.assertEqual([], list(Path(temp_dir).glob("*.json")))
            self.assertTrue((Path(temp_dir) / "room36_app.sqlite3").is_file())
            reopened = AutomaticStateStore(temp_dir).load(folder)
            self.assertEqual("People", reopened.current_tab)
            self.assertEqual({"cluster-1"}, reopened.ignored_unknown_cluster_ids)
            self.assertEqual("REMOVE", reopened.image_decisions[image].review_state)
            self.assertTrue(reopened.image_decisions[image].selected_for_deletion)
            self.assertEqual([image], reopened.deletion_transactions[0].paths)
            self.assertEqual(["missing.jpg"], reopened.deletion_transactions[0].failed_paths)

    def test_legacy_json_migrates_once_and_is_kept_as_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = str((Path(temp_dir) / "photos").resolve())
            image = str(Path(folder) / "one.jpg")
            path = AutomaticStateStore(temp_dir)._state_path(folder)
            payload = {"version": 1, "folder": folder, "current_tab": "Faces",
                "ignored_unknown_cluster_ids": ["cluster-2"],
                "image_decisions": [{"path": image, "file_size": 1, "mtime_ns": 10,
                    "selected_for_deletion": False, "manually_restored": True,
                    "manually_kept": False, "manual_override": ManualOverrideKind.RESTORE.value,
                    "review_state": "KEEP"}],
                "deletion_transactions": [{"created_at_utc": "now", "paths": [image], "failed_paths": []}]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            store = AutomaticStateStore(temp_dir)
            loaded = store.load(folder)
            self.assertEqual("Faces", loaded.current_tab)
            self.assertEqual("KEEP", loaded.image_decisions[image].review_state)
            self.assertEqual({"cluster-2"}, loaded.ignored_unknown_cluster_ids)
            self.assertEqual([image], loaded.deletion_transactions[0].paths)
            self.assertTrue(path.is_file())
            with closing(sqlite3.connect(Path(temp_dir) / "room36_app.sqlite3")) as db:
                self.assertEqual(1, db.execute("SELECT COUNT(*) FROM image_decisions").fetchone()[0])
            payload["current_tab"] = "Stale JSON"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual("Faces", AutomaticStateStore(temp_dir).load(folder).current_tab)

    def test_malformed_legacy_state_does_not_crash_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = str((Path(temp_dir) / "photos").resolve())
            store = AutomaticStateStore(temp_dir)
            legacy = store._state_path(folder)
            legacy.write_text("[]", encoding="utf-8")
            reopened = AutomaticStateStore(temp_dir)
            self.assertEqual(folder, reopened.load(folder).folder)
            self.assertEqual({}, reopened.load(folder).image_decisions)
            self.assertTrue(legacy.exists())

    def test_persists_restore_and_deletion_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomaticStateStore(temp_dir)
            folder_state = FolderAutomaticState(folder="/tmp/example")
            image = PipelineImageState(
                path="/tmp/example/a.jpg",
                file_size=1,
                mtime_ns=10,
                extension=".jpg",
                orientation="landscape",
                capture_timestamp=None,
                width=100,
                height=80,
                status=PipelineImageStatus.MANUALLY_RESTORED,
                selected_for_deletion=False,
                manually_restored=True,
                manual_override=ManualOverrideKind.RESTORE,
            )
            store.capture(folder_state, [image])
            store.save(folder_state)

            loaded = store.load("/tmp/example")
            restored = PipelineImageState(
                path="/tmp/example/a.jpg",
                file_size=1,
                mtime_ns=10,
                extension=".jpg",
                orientation="landscape",
                capture_timestamp=None,
                width=100,
                height=80,
            )
            store.apply(loaded, [restored])

            self.assertTrue(restored.manually_restored)
            self.assertEqual(restored.status, PipelineImageStatus.MANUALLY_RESTORED)
            self.assertFalse(restored.selected_for_deletion)

    def test_file_change_invalidates_saved_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomaticStateStore(temp_dir)
            folder_state = FolderAutomaticState(folder="/tmp/example")
            image = PipelineImageState(
                path="/tmp/example/a.jpg",
                file_size=1,
                mtime_ns=10,
                extension=".jpg",
                orientation="landscape",
                capture_timestamp=None,
                width=100,
                height=80,
                selected_for_deletion=True,
            )
            store.capture(folder_state, [image])
            store.save(folder_state)

            loaded = store.load("/tmp/example")
            changed = PipelineImageState(
                path="/tmp/example/a.jpg",
                file_size=2,
                mtime_ns=11,
                extension=".jpg",
                orientation="landscape",
                capture_timestamp=None,
                width=100,
                height=80,
            )
            store.apply(loaded, [changed])

            self.assertFalse(changed.selected_for_deletion)
            self.assertNotIn(changed.path, loaded.image_decisions)


if __name__ == "__main__":
    unittest.main()
