#!/usr/bin/env python3
"""Run the banner pipeline on a Modal GPU.

Usage
-----
    # First time: authenticate with Modal
    uv run modal setup

    # Run on T4 (cheapest)
    uv run modal run scripts/modal_run.py --gpu T4 --config configs/default.yaml

    # Run on A10G
    uv run modal run scripts/modal_run.py --gpu A10G --config configs/default.yaml

    # Run on A100
    uv run modal run scripts/modal_run.py --gpu A100 --config configs/default.yaml

    # Benchmark (5 runs)
    uv run modal run scripts/modal_run.py --gpu T4 --config configs/default.yaml --benchmark 5
"""

from __future__ import annotations

import sys

import modal

# ---------------------------------------------------------------------------
# Parse --gpu from CLI before decorators run (Modal evaluates decorators at
# import time, so the GPU must be known before @app.function is reached).
# ---------------------------------------------------------------------------

_GPU = "T4"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _GPU = sys.argv[i + 1]
        break

# ---------------------------------------------------------------------------
# Modal image: Linux + CUDA torch + SAM2 + our pipeline code
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git", "build-essential")
    .pip_install(
        "torch>=2.0",
        "torchvision>=0.15",
        "opencv-python-headless>=4.8",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "Pillow>=10.0",
        "scipy>=1.11",
        "pyyaml>=6.0",
    )
    # Install SAM2 from source with C extension compiled.
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /tmp/sam2",
        "cd /tmp/sam2 && pip install -e '.[all]'",
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("banner-pipeline", image=image)

# Persistent volume for SAM2 checkpoints (so we don't re-download every run).
checkpoints_volume = modal.Volume.from_name(
    "banner-pipeline-checkpoints",
    create_if_missing=True,
)


# ---------------------------------------------------------------------------
# Remote GPU function
# ---------------------------------------------------------------------------


@app.function(
    gpu=_GPU,
    volumes={"/checkpoints": checkpoints_volume},
    timeout=600,
)
def run_on_gpu(
    config_dict: dict,
    video_bytes: bytes,
    logo_bytes: bytes | None,
    benchmark_runs: int = 1,
) -> dict:
    """Run the pipeline on a GPU. Returns metrics + output image bytes."""
    import os
    import sys
    import tempfile
    import time

    import cv2
    import numpy as np
    import torch

    # Make our source code importable.
    sys.path.insert(0, "/root/src")

    from banner_pipeline.pipeline import run

    # --- Download SAM2 checkpoint if not cached ---
    checkpoint_path = config_dict["pipeline"]["segmenter"]["checkpoint"]
    checkpoint_filename = os.path.basename(checkpoint_path)
    cached_checkpoint = f"/checkpoints/{checkpoint_filename}"

    if not os.path.exists(cached_checkpoint):
        print(f"Downloading checkpoint: {checkpoint_filename} …")
        _download_checkpoint(checkpoint_filename, cached_checkpoint)
        checkpoints_volume.commit()
        print(f"Cached checkpoint: {cached_checkpoint}")
    else:
        print(f"Using cached checkpoint: {cached_checkpoint}")

    # Point the config at the cached checkpoint.
    config_dict["pipeline"]["segmenter"]["checkpoint"] = cached_checkpoint

    # --- Write input files to temp dir ---
    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, "input.mp4")
    with open(video_path, "wb") as f:
        f.write(video_bytes)
    config_dict["input"]["video"] = video_path

    if logo_bytes:
        logo_path = os.path.join(tmpdir, "logo.png")
        with open(logo_path, "wb") as f:
            f.write(logo_bytes)
        config_dict["input"]["logo"] = logo_path

    # --- Report GPU info ---
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        gpu_name = "none"
        gpu_mem = 0
        print("WARNING: No CUDA GPU detected!")

    mode = config_dict.get("pipeline", {}).get("mode", "image")

    # --- Run pipeline (with optional benchmark) ---
    all_metrics = []
    output_bytes = None
    output_ext = ".mp4" if mode == "video" else ".png"

    for i in range(benchmark_runs):
        if benchmark_runs > 1:
            print(f"\n=== Run {i + 1}/{benchmark_runs} ===")

        t_start = time.perf_counter()
        output_video_path = os.path.join(tmpdir, "output.mp4")
        results = run(config_dict, output_path=output_video_path)
        t_total = time.perf_counter() - t_start

        m = results.get("metrics", {})
        m["run_total_s"] = t_total
        m["gpu"] = gpu_name
        m["gpu_memory_gb"] = round(gpu_mem, 1)
        m["mode"] = mode
        all_metrics.append(m)

        # Save output from last run.
        if mode == "video" and results.get("output_path"):
            with open(results["output_path"], "rb") as f:
                output_bytes = f.read()
        elif results.get("composited") is not None:
            _, buf = cv2.imencode(".png", results["composited"])
            output_bytes = buf.tobytes()

    # --- Aggregate benchmark stats ---
    report = {"runs": benchmark_runs, "gpu": gpu_name, "gpu_memory_gb": round(gpu_mem, 1)}
    if benchmark_runs > 1:
        for key in [
            "load_frame_s",
            "segment_s",
            "fit_s",
            "composite_s",
            "total_s",
            "run_total_s",
        ]:
            values = [m.get(key, 0) for m in all_metrics if key in m]
            if values:
                report[key] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4),
                    "min": round(float(np.min(values)), 4),
                    "max": round(float(np.max(values)), 4),
                }
        total_times = [m.get("run_total_s", 0) for m in all_metrics]
        fps_values = [1.0 / t for t in total_times if t > 0]
        if fps_values:
            report["fps"] = {
                "mean": round(float(np.mean(fps_values)), 2),
                "min": round(float(np.min(fps_values)), 2),
                "max": round(float(np.max(fps_values)), 2),
            }
    else:
        report.update(all_metrics[0])

    return {
        "metrics": report,
        "output_bytes": output_bytes,
        "output_ext": output_ext,
    }


def _download_checkpoint(filename: str, dest: str) -> None:
    """Download a SAM2 checkpoint from the official release."""
    import os
    import urllib.request

    base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
    url = f"{base_url}/{filename}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {url} …")
    urllib.request.urlretrieve(url, dest)


# ---------------------------------------------------------------------------
# Local entrypoint (runs on your Mac, dispatches to Modal)
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    config: str = "configs/default.yaml",
    gpu: str = "T4",
    mode: str = "",
    benchmark: int = 1,
    name: str = "",
):
    import json
    import os
    from datetime import datetime

    import yaml

    # Load config.
    with open(config) as f:
        config_dict = yaml.safe_load(f)

    # Override mode if specified via CLI.
    if mode:
        config_dict.setdefault("pipeline", {})["mode"] = mode

    # Read input files as bytes.
    video_path = config_dict["input"]["video"]
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    print(f"Video: {video_path} ({len(video_bytes) / 1024:.0f} KB)")

    logo_bytes = None
    logo_path = config_dict["input"].get("logo")
    if logo_path and os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_bytes = f.read()
        print(f"Logo: {logo_path} ({len(logo_bytes) / 1024:.0f} KB)")

    print(f"GPU: {gpu}")
    print(f"Benchmark runs: {benchmark}")

    # Call the remote function.
    result = run_on_gpu.remote(
        config_dict=config_dict,
        video_bytes=video_bytes,
        logo_bytes=logo_bytes,
        benchmark_runs=benchmark,
    )

    # --- Save results locally ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fitter_type = config_dict["pipeline"]["fitter"]["type"]
    exp_name = name or f"{fitter_type}_{gpu}"
    exp_dir = os.path.join(config_dict["output"]["dir"], f"{timestamp}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save frozen config.
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # Save metrics.
    metrics = result["metrics"]
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save output (image or video).
    if result.get("output_bytes"):
        out_dir = os.path.join(exp_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        ext = result.get("output_ext", ".png")
        out_name = "composited" + ext
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(result["output_bytes"])
        print(f"Saved: {out_path}")

    # Print results.
    print(f"\n{'=' * 50}")
    print(f"RESULTS — {gpu}")
    print(f"{'=' * 50}")
    print(json.dumps(metrics, indent=2))
    print(f"\nExperiment saved: {exp_dir}")
