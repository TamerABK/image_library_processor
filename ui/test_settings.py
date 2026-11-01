import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app_state_database import AppStateDatabase
from automatic_scan.decode_pool import automatic_decode_workers, max_manual_decode_workers
from ui.qt.settings_controller import SettingsController, valid


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.cpu_patch = patch("automatic_scan.decode_pool.available_cpu_cores",
                               return_value=(8, "physical"))
        self.cpu_patch.start()
        self.addCleanup(self.cpu_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.db = AppStateDatabase(Path(self.temp.name) / "state.sqlite3")
        with self.db.connect() as db:
            for project in ("A", "B"):
                db.execute("INSERT INTO projects(project_id,folder,name,created_at) VALUES(?,?,?,0)",
                           (project, project, project))
        self.settings = SettingsController(self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_inheritance_persistence_and_reset(self):
        self.assertTrue(self.settings.save("", {"worker_count": 3}))
        self.assertEqual(SettingsController(self.db).resolve("A")["worker_count"], 3)
        self.assertTrue(self.settings.save("A", {"worker_count": 2}))
        self.assertTrue(self.settings.save("", {"worker_count": 4}))
        self.assertEqual(self.settings.resolve("A")["worker_count"], 2)
        self.assertEqual(self.settings.resolve("B")["worker_count"], 4)
        self.settings.reset("A")
        self.assertEqual(self.settings.resolve("A")["worker_count"], 4)
        self.settings.save("A", {"worker_count": 2})
        self.settings.reset("")
        self.assertEqual(self.settings.resolve("A")["worker_count"], 2)
        self.assertEqual(self.settings.resolve("B")["worker_count"], "automatic")

    def test_invalid_values_and_snapshot(self):
        self.assertFalse(valid("worker_count", 0))
        self.assertFalse(self.settings.save("", {"worker_count": 0}))
        with self.db.connect() as db:
            db.execute("INSERT INTO global_preferences VALUES(?,?)", ("worker_count", json.dumps(12)))
            db.execute("INSERT INTO global_preferences VALUES(?,?)", ("obsolete", json.dumps("legacy")))
        self.assertEqual(self.settings.scan_config("B").performance.worker_count,
                         min(12, max_manual_decode_workers()))
        before = self.settings.scan_config("A")
        self.settings.save("", {"worker_count": 2})
        self.assertEqual(before.performance.worker_count, automatic_decode_workers())
        self.assertEqual(self.settings.scan_config("A").performance.worker_count, 2)

    def test_project_deletion_cascades(self):
        self.settings.save("A", {"keepers_per_group": 3})
        with self.db.connect() as db:
            db.execute("DELETE FROM projects WHERE project_id='A'")
            self.assertEqual(db.execute("SELECT count(*) FROM project_preference_overrides").fetchone()[0], 0)

    def test_cpu_warning_boundary(self):
        from unittest.mock import patch
        with patch("ui.qt.settings_controller.available_cpu_cores", return_value=(4, "physical")):
            self.assertFalse(self.settings.requiresCpuConfirmation(2))
            self.assertTrue(self.settings.requiresCpuConfirmation(3))
            self.assertTrue(self.settings.requiresCpuConfirmation(4))

    def test_worker_options_follow_ninety_percent_without_four_worker_ceiling(self):
        from unittest.mock import patch
        for cores, expected in ((1, 1), (2, 1), (4, 3), (8, 7), (16, 14)):
            with self.subTest(cores=cores), patch("automatic_scan.decode_pool.available_cpu_cores",
                                                  return_value=(cores, "physical")):
                self.assertEqual(expected, self.settings.pageData("")["maxWorkers"])
                self.assertFalse(self.settings.save("", {"worker_count": expected + 1}))


if __name__ == "__main__":
    unittest.main()
