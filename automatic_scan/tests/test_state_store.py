from __future__ import annotations

import tempfile
import unittest

from automatic_scan.models import ManualOverrideKind, PipelineImageState, PipelineImageStatus
from automatic_scan.state_store import AutomaticStateStore, FolderAutomaticState


class AutomaticStateStoreTests(unittest.TestCase):
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
