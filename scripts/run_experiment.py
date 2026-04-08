#!/usr/bin/env python3
"""Run an experiment: execute the pipeline, save outputs + metrics.

Usage
-----
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/experiments/lp_oriented.yaml --name lp_test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from banner_pipeline.pipeline import load_config, run


def main():
    parser = argparse.ArgumentParser(description="Run an experiment and save results")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--name", default=None, help="Experiment name (default: auto from config)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Build experiment directory name.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fitter_type = config["pipeline"]["fitter"]["type"]
    compositor_type = config["pipeline"]["compositor"]["type"]
    exp_name = args.name or f"{fitter_type}_{compositor_type}"
    exp_dir = os.path.join(config["output"]["dir"], f"{timestamp}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)
    outputs_dir = os.path.join(exp_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Freeze a copy of the config.
    shutil.copy2(args.config, os.path.join(exp_dir, "config.yaml"))

    print(f"[experiment] {exp_dir}")

    mode = config.get("pipeline", {}).get("mode", "image")
    output_video_path = os.path.join(outputs_dir, "output.mp4")

    # Run the pipeline.
    results = run(config, config_path=args.config, output_path=output_video_path)

    # Save outputs.
    if mode == "video":
        if results.get("output_path"):
            print(f"  Video: {results['output_path']}")
    else:
        if results.get("composited") is not None:
            out_path = os.path.join(outputs_dir, "composited.png")
            cv2.imwrite(out_path, results["composited"])
            print(f"  Saved: {out_path}")

        if results.get("frame") is not None:
            out_path = os.path.join(outputs_dir, "frame.png")
            cv2.imwrite(out_path, results["frame"])

        # Save masks.
        for obj_id, mask in results.get("masks", {}).items():
            mask_path = os.path.join(outputs_dir, f"mask_obj{obj_id}.png")
            cv2.imwrite(mask_path, (mask > 0).astype("uint8") * 255)

    # Save metrics.
    metrics = results.get("metrics", {})
    metrics["experiment_name"] = exp_name
    metrics["config_file"] = args.config
    metrics["timestamp"] = timestamp
    metrics["fitter"] = fitter_type
    metrics["compositor"] = compositor_type
    metrics["segmenter"] = config["pipeline"]["segmenter"]["type"]

    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    print(f"\n[experiment] Done → {exp_dir}")


if __name__ == "__main__":
    main()
