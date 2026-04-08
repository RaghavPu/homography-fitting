#!/usr/bin/env python3
"""Benchmark FPS: run the pipeline N times and report timing statistics.

Usage
-----
    python scripts/benchmark_fps.py --config configs/default.yaml --runs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from banner_pipeline.pipeline import load_config, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline FPS")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    parser.add_argument("--output", default=None, help="Save benchmark results to JSON")
    args = parser.parse_args()

    config = load_config(args.config)

    # Require pre-defined prompts (no interactive UI in benchmark mode).
    if not config["input"].get("prompts"):
        print("ERROR: Benchmarking requires pre-defined prompts in the config.")
        print("       Run interactively first to save prompts, then benchmark.")
        sys.exit(1)

    all_metrics: list[dict] = []

    for i in range(args.runs):
        print(f"\n=== Run {i + 1}/{args.runs} ===")
        t_start = time.perf_counter()
        results = run_pipeline(config)
        t_total = time.perf_counter() - t_start

        m = results.get("metrics", {})
        m["run_total_s"] = t_total
        all_metrics.append(m)

    # Aggregate statistics.
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    keys = ["load_frame_s", "segment_s", "fit_s", "composite_s", "total_s", "run_total_s"]
    report: dict = {}
    for key in keys:
        values = [m.get(key, 0) for m in all_metrics if key in m]
        if values:
            report[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            print(
                f"  {key:18s}: "
                f"mean={report[key]['mean']:.4f}s  "
                f"std={report[key]['std']:.4f}s  "
                f"min={report[key]['min']:.4f}s  "
                f"max={report[key]['max']:.4f}s"
            )

    total_times = [m.get("run_total_s", 0) for m in all_metrics]
    if total_times:
        fps_values = [1.0 / t for t in total_times if t > 0]
        print(f"\n  FPS (single frame): mean={np.mean(fps_values):.2f}  "
              f"min={np.min(fps_values):.2f}  max={np.max(fps_values):.2f}")
        report["fps"] = {
            "mean": float(np.mean(fps_values)),
            "min": float(np.min(fps_values)),
            "max": float(np.max(fps_values)),
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"runs": args.runs, "stages": report, "raw": all_metrics}, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
