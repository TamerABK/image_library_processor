import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.tools.benchmark_decoder_sweep import (_csv, _read, _plan, _fine_plan,
    _summary, RUNS, SUMMARY)


class DecoderSweepTests(unittest.TestCase):
    def test_screen_matrix_and_deterministic_order(self):
        args = SimpleNamespace(mode="comprehensive", workers=None, memory=None,
                               seed=36, screen_repetitions=2)
        one = _plan(args, "subset", "full")
        self.assertEqual(len(one), 96)
        self.assertEqual(one, _plan(args, "subset", "full"))
        self.assertEqual(len({(w, m) for _, _, w, m, _ in one}), 48)

    def test_fine_grid_includes_baseline_and_winners(self):
        args = SimpleNamespace(exhaustive=False, seed=36, full_repetitions=3)
        summary = [{"dataset_id": "subset", "worker_count": "10",
                    "memory_budget_mib": "3072", "median_wall_seconds": "100"}]
        plan = _fine_plan(args, summary, "full")
        self.assertEqual({r for _, r, _, _, _ in plan}, {1, 2, 3})
        for repetition in (1, 2, 3):
            self.assertIn(("B", repetition, 4, 512, "full"), plan)
            self.assertIn(("B", repetition, 10, 3072, "full"), plan)

    def test_csv_checkpoint_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            baseline = {"run_id": "a", "dataset_id": "dataset", "requested_workers": "4",
                        "requested_memory_mib": "512", "status": "complete", "wall_seconds": "10",
                        "images_per_second": "1"}
            faster = {**baseline, "run_id": "b", "requested_workers": "6", "wall_seconds": "8"}
            _csv(path, RUNS, [baseline, faster])
            self.assertEqual([r["run_id"] for r in _read(path)], ["a", "b"])
            summary = _summary(_read(path), [])
            self.assertEqual(len(summary), 2)
            winning = next(r for r in summary if r["worker_count"] == "6")
            self.assertEqual(winning["speedup_vs_baseline"], 1.25)
            self.assertEqual(set(SUMMARY) - set(winning), set())


if __name__ == "__main__": unittest.main()
