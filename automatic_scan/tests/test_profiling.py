from __future__ import annotations

import json
import gc
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from automatic_scan.perf import AutomaticScanProfiler, active_profiler
from automatic_scan.profiling import HierarchicalProfiler, profile_scope
from automatic_scan.profiling_resources import ResourceSampler
from automatic_scan.profiling_gpu_windows import WindowsGpuProcessSampler
from automatic_scan.profiling_disk_windows import WindowsPhysicalDiskSampler, _matching_instance
from automatic_scan.decode_diagnostic import _measure
from automatic_scan.storage_report import stage_resource_summary
from automatic_scan.benchmark import main as benchmark_main, _run_pipeline


class ProfilingTests(unittest.TestCase):
    def test_stage_disk_correlation_keeps_process_and_device_distinct(self):
        trace = {"traceEvents": [
            {"name": "stage.blur", "ph": "X", "ts": 0, "dur": 2_000_000},
            {"name": "resources", "ph": "C", "ts": 0, "args": {
                "physical_disk_read_mb_s": 10, "read_bytes": 100}},
            {"name": "resources", "ph": "C", "ts": 1_000_000, "args": {
                "physical_disk_read_mb_s": 20, "read_bytes": 200}},
        ]}
        row = stage_resource_summary({}, trace)["stages"]["stage.blur"]
        self.assertEqual(row["physical_read_bytes_estimated"], 20_000_000)
        self.assertEqual(row["process_logical_read_bytes_approx"], 100)

    def test_physical_disk_instance_requires_unique_dataset_drive(self):
        self.assertEqual(_matching_instance(["0 C:", "1 D:"], "C:"), "0 C:")
        self.assertIsNone(_matching_instance(["0 C:", "1 C:"], "C:"))
        self.assertIsNone(_matching_instance(["0 D:"], "C:"))

    def test_physical_disk_sampler_does_not_infer_process_io(self):
        class Pdh:
            @staticmethod
            def PdhCollectQueryData(_query):
                return 0
        sampler = object.__new__(WindowsPhysicalDiskSampler)
        sampler.available, sampler.error, sampler.instance, sampler.drive = True, None, None, "C:"
        sampler._pdh, sampler._query = Pdh(), object()
        sampler._counters = {"read_mb_s": "read", "queue_depth": "queue"}
        sampler._values = lambda counter: [("0 C:", 20_000_000 if counter == "read" else 2.0),
                                            ("1 D:", 900_000_000)]
        sample = sampler.sample()
        self.assertEqual(sample["physical_disk_read_mb_s"], 20)
        self.assertEqual(sample["physical_disk_queue_depth"], 2)
        self.assertNotIn("read_bytes", sample)

    def test_stage_boundaries_survive_leaf_trace_overflow(self):
        profiler = HierarchicalProfiler(max_events=10)
        with active_profiler(profiler):
            with profile_scope("stage.blur", detail="stages"):
                pass
            for _ in range(30):
                with profile_scope("image.cv2_imread"):
                    pass
            with profile_scope("stage.faces", detail="stages"):
                pass
        names = [e["name"] for e in profiler.trace()["traceEvents"]]
        self.assertIn("stage.blur", names)
        self.assertIn("stage.faces", names)
        self.assertLessEqual(len(names), 10)
        self.assertGreater(profiler.report()["trace_dropped_events"], 0)

    def test_split_jpeg_decode_matches_production_flags(self):
        import cv2
        with tempfile.TemporaryDirectory() as root:
            photo = Path(root) / "oriented.jpg"
            exif = Image.Exif()
            exif[274] = 6  # Rotate 90 degrees on decode.
            Image.new("RGB", (64, 32), (20, 40, 60)).save(photo, exif=exif)
            for flag in (cv2.IMREAD_COLOR, cv2.IMREAD_REDUCED_COLOR_2):
                result = _measure(photo, flag, check=True)
                self.assertTrue(result["equivalent"])
                self.assertEqual((result["width"], result["height"]),
                                 (32, 64) if flag == cv2.IMREAD_COLOR else (16, 32))
                self.assertGreater(result["read_ns"], 0)
                self.assertGreater(result["imdecode_ns"], 0)

    def test_gpu_process_counters_filter_other_processes_and_keep_engine_types(self):
        class Pdh:
            @staticmethod
            def PdhCollectQueryData(_query):
                return 0

        sampler = object.__new__(WindowsGpuProcessSampler)
        sampler.available = True
        sampler.error = None
        sampler._pdh = Pdh()
        sampler._query = object()
        sampler._counters = {key: key for key in ("engine", "dedicated", "shared")}
        pid = os.getpid()
        values = {
            "engine": [(f"pid_{pid}_luid_0x0_0x0_phys_0_eng_0_engtype_3D", 7.5),
                       (f"pid_{pid}_luid_0x0_0x0_phys_0_eng_1_engtype_Compute_0", 1.25),
                       (f"pid_{pid+1}_luid_0x0_0x0_phys_0_eng_0_engtype_3D", 50.0)],
            "dedicated": [(f"pid_{pid}_luid_0x0_0x0_phys_0", 1024.0),
                          (f"pid_{pid+1}_luid_0x0_0x0_phys_0", 9999.0)],
            "shared": [(f"pid_{pid}_luid_0x0_0x0_phys_0", 256.0)],
        }
        sampler._values = lambda kind: values[kind]
        sample = sampler.sample()
        self.assertEqual(sample["gpu_process_engine_percent_sum"],
                         {"3D": 7.5, "Compute_0": 1.25})
        self.assertEqual(sample["gpu_process_compute_engine_percent"], 1.25)
        self.assertEqual(sample["gpu_process_dedicated_bytes"], 1024)

    def test_nested_inclusive_and_exclusive_nanoseconds(self):
        with patch("automatic_scan.profiling.time.perf_counter_ns", side_effect=[0, 10, 20, 30, 50, 60]):
            profiler = HierarchicalProfiler()
            with active_profiler(profiler):
                with profile_scope("outer"):
                    with profile_scope("inner"):
                        pass
            report = profiler.report(image_count=2, face_count=1)
        self.assertEqual(report["timings"]["outer"]["inclusive_seconds"], 40 / 1e9)
        self.assertEqual(report["timings"]["outer"]["exclusive_seconds"], 30 / 1e9)
        self.assertEqual(report["timings"]["inner"]["inclusive_seconds"], 10 / 1e9)
        self.assertEqual(report["timings"]["outer"]["p95_seconds"], 40 / 1e9)
        self.assertEqual(report["images_per_second"], 2 / (60 / 1e9))

    def test_disabled_scope_is_shared_noop_and_legacy_profiler_still_works(self):
        with patch("automatic_scan.profiling.time.perf_counter_ns") as timer:
            one = profile_scope("disabled")
            two = profile_scope("disabled")
            self.assertIs(one, two)
            with one:
                pass
            with active_profiler(AutomaticScanProfiler()):
                with profile_scope("legacy"):
                    pass
            timer.assert_not_called()

    def test_thread_labels_bounds_and_no_private_paths(self):
        profiler = HierarchicalProfiler(max_events=1)
        def work():
            with active_profiler(profiler):
                with profile_scope("decode", width=100, height=80):
                    pass
        threads = [threading.Thread(target=work, name=f"worker-{i}") for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(profiler.report()["timings"]["decode"]["calls"], 2)
        self.assertEqual(len(profiler.trace()["traceEvents"]), 1)
        self.assertEqual(profiler.report()["trace_dropped_events"], 1)
        self.assertNotIn("photo.jpg", json.dumps(profiler.report()))
        self.assertIn("worker", profiler.trace()["traceEvents"][0]["args"])

    def test_sampler_stops_and_is_bounded(self):
        sampler = ResourceSampler(interval=.05, max_samples=2)
        sampler.start()
        gc.collect()
        sampler.stop()
        self.assertFalse(sampler._thread.is_alive())
        self.assertLessEqual(sampler.report()["sample_count"], 2)
        self.assertGreaterEqual(sampler.report()["gc_collections"], 1)
        self.assertNotIn(sampler._on_gc, gc.callbacks)

    def test_image_benchmark_reads_sources_and_writes_reports_elsewhere(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            dataset = root / "images"
            output = root / "reports"
            dataset.mkdir()
            source = dataset / "sample.jpg"
            Image.new("RGB", (32, 20), (10, 20, 30)).save(source)
            before = source.read_bytes()
            self.assertEqual(benchmark_main(["--dataset", str(dataset), "--out", str(output),
                "--scenario", "image", "--size", "50"]), 0)
            self.assertEqual(source.read_bytes(), before)
            report = json.loads((output / "run-1.json").read_text())
            self.assertEqual(report["profile"]["image_count"], 1)
            self.assertIn("image.decode_and_preprocess", report["profile"]["timings"])
            self.assertFalse((dataset / "analysis.sqlite3").exists())

    def test_blur_scenario_stops_real_pipeline_before_models(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            photo = root / "sample.jpg"
            Image.new("RGB", (32, 20), (10, 20, 30)).save(photo)
            result = _run_pipeline([photo], root / "isolated", folder=root,
                scenario="blur", detail="normal", batch_size=None, ort_profile=False)
            self.assertEqual({"blur": "COMPLETE"},
                             {key: value["state"] for key, value in result["stages"].items()})
            self.assertIsNone(result["profile"]["face_count"])
            self.assertTrue(photo.is_file())

    def test_bounded_image_workers_finish_and_show_overlap(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = []
            for index in range(8):
                photo = root / f"{index}.jpg"
                Image.new("RGB", (32, 20), (10, 20, 30)).save(photo)
                paths.append(photo)
            result = _run_pipeline(paths, root / "isolated", folder=root,
                scenario="image", detail="normal", batch_size=None,
                ort_profile=False, concurrency=2)
            self.assertEqual(8, result["profile"]["timings"]["image.decode_and_preprocess"]["calls"])
            self.assertLessEqual(result["profile"]["trace_dropped_events"], 50000)


if __name__ == "__main__":
    unittest.main()
