import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app_state_database import AppStateDatabase
from automatic_scan.decode_pool import normal_decode_workers, performance_decode_workers, performance_mode_available
from ui.qt.settings_controller import SettingsController, valid


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = AppStateDatabase(Path(self.temp.name) / 'app.sqlite3')
        with self.db.connect() as db:
            for project in ('A', 'B'):
                db.execute('INSERT INTO projects(project_id,folder,name,created_at) VALUES(?,?,?,0)',
                           (project, project, project))
        self.settings = SettingsController(self.db)
        topology = patch('ui.qt.settings_controller.available_cpu_topology', return_value=(12, 6, 'physical cores'))
        topology.start()
        self.addCleanup(topology.stop)

    def test_mode_calculation_and_eligibility(self):
        cases = [(2, 4, 2, 4, False), (4, 4, 2, 4, False), (4, 6, 2, 5, True),
                 (4, 8, 3, 6, True), (6, 12, 4, 9, True), (8, 16, 6, 12, True),
                 (12, 24, 8, 18, True)]
        for physical, logical, normal, performance, eligible in cases:
            with self.subTest(logical=logical):
                self.assertEqual(normal_decode_workers(logical), normal)
                self.assertEqual(performance_decode_workers(logical), performance)
                self.assertEqual(performance_mode_available(logical, physical), eligible)
        self.assertEqual(normal_decode_workers(1), 2)
        self.assertFalse(performance_mode_available(12, None))
        self.assertFalse(performance_mode_available(5, None))
        self.assertFalse(performance_mode_available(12, 2))

    def test_default_and_fixed_memory(self):
        data = self.settings.pageData('A')
        self.assertEqual(data['values']['performance_mode'], 'normal')
        self.assertEqual(data['normalWorkers'], 4)
        self.assertEqual(data['performanceWorkers'], 9)
        self.assertTrue(data['performanceAvailable'])
        config = self.settings.scan_config('A')
        self.assertEqual(config.performance.worker_count, 4)
        self.assertEqual(config.performance.decode_memory_budget_mib, 2048)
        self.assertFalse(config.performance.memory_conservative)

    def test_inheritance_restart_and_reset(self):
        self.assertTrue(self.settings.save('', {'performance_mode': 'performance'}))
        self.assertEqual(SettingsController(self.db).scan_config('B').performance.worker_count, 9)
        self.settings.save('A', {'performance_mode': 'normal'})
        self.assertEqual(self.settings.scan_config('A').performance.worker_count, 4)
        self.settings.reset('')
        self.assertEqual(self.settings.scan_config('B').performance.worker_count, 4)
        self.settings.save('', {'performance_mode': 'performance'})
        self.assertEqual(self.settings.scan_config('A').performance.worker_count, 4)
        self.settings.reset('A')
        self.assertEqual(self.settings.scan_config('A').performance.worker_count, 9)

    def test_saved_performance_falls_back_without_deleting_preference(self):
        self.settings.save('A', {'performance_mode': 'performance'})
        with patch('ui.qt.settings_controller.available_cpu_topology', return_value=(12, None, 'unknown')):
            self.assertFalse(self.settings.pageData('A')['performanceAvailable'])
            self.assertEqual(self.settings.scan_config('A').performance.worker_count, 4)
        with patch('ui.qt.settings_controller.available_cpu_topology', return_value=(4, 2, 'unknown')):
            self.assertEqual(self.settings.scan_config('A').performance.worker_count, 2)
            self.assertEqual(self.settings.pageData('A')['effectiveMode'], 'normal')
            self.assertEqual(self.settings.pageData('A')['values']['performance_mode'], 'performance')
            self.assertFalse(self.settings.save('B', {'performance_mode': 'performance'}))
        self.assertEqual(self.settings.scan_config('A').performance.worker_count, 9)

    def test_legacy_values_are_ignored(self):
        self.assertFalse(valid('worker_count', 4))
        self.assertFalse(valid('decode_memory_budget_mib', 4096))
        with self.db.connect() as db:
            db.execute('INSERT INTO global_preferences VALUES (?,?)', ('worker_count', json.dumps(10)))
            db.execute('INSERT INTO global_preferences VALUES (?,?)', ('memory_usage', json.dumps('conservative')))
            db.execute('INSERT INTO global_preferences VALUES (?,?)', ('decode_memory_budget_mib', json.dumps(4096)))
        self.assertEqual(self.settings.scan_config('B').performance.worker_count, 4)
        self.assertEqual(self.settings.scan_config('B').performance.decode_memory_budget_mib, 2048)


if __name__ == '__main__': unittest.main()
