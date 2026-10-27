"""Correlate bounded profiler stage spans with sampled physical-disk activity.

Input contains no paths; output intentionally includes no photo identifiers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def stage_resource_summary(report: dict, trace: dict) -> dict:
    events = trace.get("traceEvents", [])
    stages = [e for e in events if e.get("ph") == "X" and
              (e.get("name", "").startswith("stage.") or e.get("name") == "project.metadata")]
    samples = sorted((e for e in events if e.get("ph") == "C" and e.get("name") == "resources"),
                     key=lambda e: e["ts"])
    result = {}
    for stage in stages:
        start, stop = stage["ts"], stage["ts"] + stage["dur"]
        selected = [s for s in samples if start <= s["ts"] < stop]
        def average(field):
            values = [float(s["args"][field]) for s in selected if field in s.get("args", {})]
            return sum(values) / len(values) if values else None
        physical_bytes = 0.0
        logical_bytes = None
        for old, new in zip(selected, selected[1:]):
            rate = new.get("args", {}).get("physical_disk_read_mb_s")
            if rate is not None:
                physical_bytes += max(0, new["ts"] - old["ts"]) / 1e6 * rate * 1_000_000
        if len(selected) >= 2:
            first, last = selected[0].get("args", {}), selected[-1].get("args", {})
            if "read_bytes" in first and "read_bytes" in last:
                logical_bytes = max(0, last["read_bytes"] - first["read_bytes"])
        result[stage["name"]] = {"seconds": stage["dur"] / 1e6,
            "sample_count": len(selected),
            "physical_read_mb_s_mean": average("physical_disk_read_mb_s"),
            "physical_read_iops_mean": average("physical_disk_read_iops"),
            "physical_read_latency_ms_mean": (lambda x: x * 1000 if x is not None else None)(
                average("physical_disk_read_latency_s")),
            "physical_queue_depth_mean": average("physical_disk_queue_depth"),
            "physical_activity_percent_mean": average("physical_disk_activity_percent"),
            "physical_read_bytes_estimated": physical_bytes if selected else None,
            "process_logical_read_bytes_approx": logical_bytes,
            "rss_bytes_mean": average("rss_bytes"),
            "gpu_process_compute_engine_percent_mean": average("gpu_process_compute_engine_percent")}
    return {"stages": result,
        "physical_disk_note": "Device-wide sampled rates include other processes; integrated bytes are approximate.",
        "logical_io_note": "Process logical reads may be served from filesystem cache."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    summary = stage_resource_summary(report, trace)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
