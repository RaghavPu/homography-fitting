#!/usr/bin/env python3
"""Analyze benchmark matrix results and generate plots + summary report.

Reads every ``experiments/*/metrics.json`` and produces:
  - analysis/plots/fps_vs_gpu.png       — FPS scaling across GPUs
  - analysis/plots/fps_vs_prompts.png    — FPS vs prompt count
  - analysis/plots/stage_breakdown.png   — per-stage time breakdown
  - analysis/plots/cost_efficiency.png   — FPS per $/hr
  - analysis/report.md                   — summary table + key findings

Usage
-----
    uv run python scripts/analyze_matrix.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# GPU pricing per hour (from Modal docs).
GPU_COST_PER_HR = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
}

# GPU display order (cheapest → most expensive).
GPU_ORDER = ["T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100", "H200", "B200"]

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ANALYSIS_DIR = REPO_ROOT / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"


def _mean_of(metric):
    """Extract the mean value from a metric (which may be a dict or scalar)."""
    if isinstance(metric, dict):
        return metric.get("mean", 0)
    return metric or 0


def _gpu_short_name(full_name: str) -> str:
    """Map a torch device name like 'NVIDIA H100 80GB HBM3' → 'H100'."""
    if not full_name:
        return "?"
    upper = full_name.upper()
    # Order matters: check more specific names first.
    if "B200" in upper:
        return "B200"
    if "H200" in upper:
        return "H200"
    if "H100" in upper:
        return "H100"
    if "A100" in upper:
        return "A100-80GB" if "80GB" in upper else "A100"
    if "L40S" in upper:
        return "L40S"
    if "A10G" in upper:
        return "A10G"
    if "L4" in upper:
        return "L4"
    if "T4" in upper:
        return "T4"
    return full_name


def _requested_gpu_from_dir(dir_name: str) -> str:
    """Extract requested GPU from a directory name like '..._11prompts_H100'."""
    # Last token after the last underscore.
    return dir_name.rsplit("_", 1)[-1]


def load_experiments() -> list[dict]:
    """Read every metrics.json in experiments/ and return a list of records."""
    records = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        actual_gpu = _gpu_short_name(m.get("gpu", ""))
        requested_gpu = _requested_gpu_from_dir(d.name)
        records.append(
            {
                "experiment": d.name,
                "gpu": actual_gpu,
                "requested_gpu": requested_gpu,
                "gpu_mismatch": actual_gpu != requested_gpu,
                "num_prompts": m.get("num_prompts", 0),
                "num_frames": m.get("num_frames", 0),
                "duration_s": m.get("duration_s", 0),
                "output_fps": _mean_of(m.get("output_fps")),
                "total_s": _mean_of(m.get("total_s")),
                "segment_total_s": _mean_of(m.get("segment_total_s")),
                "fit_mean_ms": _mean_of(m.get("fit_mean_ms")),
                "composite_mean_ms": _mean_of(m.get("composite_mean_ms")),
                "write_video_s": _mean_of(m.get("write_video_s")),
            }
        )
    return records


def plot_fps_vs_gpu(records: list[dict]) -> None:
    """One line per prompt count, x = GPU, y = FPS."""
    prompt_counts = sorted({r["num_prompts"] for r in records if r["num_prompts"]})
    gpus_present = [g for g in GPU_ORDER if any(r["gpu"] == g for r in records)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for npc in prompt_counts:
        xs, ys = [], []
        for gpu in gpus_present:
            match = [r for r in records if r["num_prompts"] == npc and r["gpu"] == gpu]
            if match:
                xs.append(gpu)
                ys.append(match[0]["output_fps"])
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"{npc} prompt(s)")

    ax.set_xlabel("GPU")
    ax.set_ylabel("Output FPS")
    ax.set_title("Pipeline FPS by GPU")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fps_vs_gpu.png", dpi=150)
    plt.close(fig)


def plot_fps_vs_prompts(records: list[dict]) -> None:
    """One line per GPU, x = prompt count, y = FPS."""
    gpus_present = [g for g in GPU_ORDER if any(r["gpu"] == g for r in records)]
    prompt_counts = sorted({r["num_prompts"] for r in records if r["num_prompts"]})

    fig, ax = plt.subplots(figsize=(8, 5))
    for gpu in gpus_present:
        xs, ys = [], []
        for npc in prompt_counts:
            match = [r for r in records if r["num_prompts"] == npc and r["gpu"] == gpu]
            if match:
                xs.append(npc)
                ys.append(match[0]["output_fps"])
        ax.plot(xs, ys, marker="o", linewidth=2, label=gpu)

    ax.set_xlabel("Number of prompts (objects)")
    ax.set_ylabel("Output FPS")
    ax.set_title("FPS vs prompt count")
    ax.set_xticks(prompt_counts)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fps_vs_prompts.png", dpi=150)
    plt.close(fig)


def plot_stage_breakdown(records: list[dict]) -> None:
    """Stacked bar chart: per-experiment time per stage."""
    records_sorted = sorted(
        records,
        key=lambda r: (
            r["num_prompts"],
            GPU_ORDER.index(r["gpu"]) if r["gpu"] in GPU_ORDER else 99,
        ),
    )

    labels = [f"{r['num_prompts']}p\n{r['gpu']}" for r in records_sorted]
    n = len(records_sorted)
    num_frames = max(r["num_frames"] for r in records_sorted) or 1

    segment_s = [r["segment_total_s"] for r in records_sorted]
    fit_s = [r["fit_mean_ms"] * num_frames / 1000 for r in records_sorted]
    composite_s = [r["composite_mean_ms"] * num_frames / 1000 for r in records_sorted]
    write_s = [r["write_video_s"] for r in records_sorted]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 6))
    x = np.arange(n)
    bottom = np.zeros(n)
    for label, vals, color in [
        ("segment (SAM2)", segment_s, "#4C72B0"),
        ("fit", fit_s, "#55A868"),
        ("composite", composite_s, "#C44E52"),
        ("write video", write_s, "#8172B2"),
    ]:
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-stage time breakdown")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "stage_breakdown.png", dpi=150)
    plt.close(fig)


def plot_cost_efficiency(records: list[dict]) -> None:
    """FPS per $/hr — bigger is better."""
    prompt_counts = sorted({r["num_prompts"] for r in records if r["num_prompts"]})
    gpus_present = [g for g in GPU_ORDER if any(r["gpu"] == g for r in records)]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(prompt_counts)
    for i, npc in enumerate(prompt_counts):
        xs, ys = [], []
        for j, gpu in enumerate(gpus_present):
            match = [r for r in records if r["num_prompts"] == npc and r["gpu"] == gpu]
            if match and gpu in GPU_COST_PER_HR:
                xs.append(j + i * width)
                ys.append(match[0]["output_fps"] / GPU_COST_PER_HR[gpu])
        ax.bar(xs, ys, width=width, label=f"{npc} prompt(s)")

    ax.set_xticks(np.arange(len(gpus_present)) + width * (len(prompt_counts) - 1) / 2)
    ax.set_xticklabels(gpus_present)
    ax.set_xlabel("GPU")
    ax.set_ylabel("FPS per $/hr (higher = better value)")
    ax.set_title("Cost efficiency: FPS per dollar-hour")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cost_efficiency.png", dpi=150)
    plt.close(fig)


def write_report(records: list[dict]) -> None:
    """Write a markdown report with the summary table."""
    records_sorted = sorted(
        records,
        key=lambda r: (
            r["num_prompts"],
            GPU_ORDER.index(r["gpu"]) if r["gpu"] in GPU_ORDER else 99,
        ),
    )

    lines = []
    lines.append("# Benchmark Matrix Report")
    lines.append("")
    if records_sorted:
        sample = records_sorted[0]
        lines.append(
            f"Input: **{sample['num_frames']} frames** "
            f"({sample['duration_s']:.1f}s) at the original resolution.",
        )
        lines.append("")
        lines.append("Each row is the mean across 3 benchmark runs.")
        lines.append("")

    lines.append("## Summary table")
    lines.append("")
    header = (
        "| Prompts | GPU (actual) | $/hr | Total time (s) | Output FPS"
        " | Segment (s) | Fit (ms/frame) | Composite (ms/frame) |"
    )
    lines.append(header)
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in records_sorted:
        cost = GPU_COST_PER_HR.get(r["gpu"], 0)
        gpu_label = r["gpu"]
        if r["gpu_mismatch"]:
            gpu_label = f"{r['gpu']} (req {r['requested_gpu']})"
        lines.append(
            f"| {r['num_prompts']} | {gpu_label} | ${cost:.2f} | "
            f"{r['total_s']:.1f} | **{r['output_fps']:.2f}** | "
            f"{r['segment_total_s']:.1f} | {r['fit_mean_ms']:.1f} | "
            f"{r['composite_mean_ms']:.1f} |",
        )
    lines.append("")

    mismatches = [r for r in records_sorted if r["gpu_mismatch"]]
    if mismatches:
        lines.append("> **Note:** Modal allocated a different GPU than requested for some runs:")
        for r in mismatches:
            lines.append(
                f"> - Requested `{r['requested_gpu']}`, got `{r['gpu']}`"
                f" ({r['num_prompts']} prompts)"
            )
        lines.append("")

    # Best FPS overall
    best = max(records_sorted, key=lambda r: r["output_fps"])
    lines.append("## Key findings")
    lines.append("")
    lines.append(
        f"- **Fastest run:** {best['output_fps']:.2f} FPS — "
        f"{best['gpu']} with {best['num_prompts']} prompt(s)",
    )
    if records_sorted and records_sorted[0]["num_frames"]:
        target_fps = records_sorted[0]["num_frames"] / records_sorted[0]["duration_s"]
        gap = target_fps / best["output_fps"]
        lines.append(
            f"- **Real-time target:** {target_fps:.0f} FPS "
            f"(input video framerate). Best run is {gap:.1f}x slower than real-time.",
        )

    # Best cost efficiency
    cost_eff = [
        (r, r["output_fps"] / GPU_COST_PER_HR[r["gpu"]])
        for r in records_sorted
        if r["gpu"] in GPU_COST_PER_HR and r["output_fps"] > 0
    ]
    if cost_eff:
        best_value = max(cost_eff, key=lambda x: x[1])
        lines.append(
            f"- **Best value:** {best_value[0]['gpu']} with "
            f"{best_value[0]['num_prompts']} prompt(s) — "
            f"{best_value[1]:.3f} FPS per $/hr",
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("![FPS vs GPU](plots/fps_vs_gpu.png)")
    lines.append("")
    lines.append("![FPS vs prompts](plots/fps_vs_prompts.png)")
    lines.append("")
    lines.append("![Stage breakdown](plots/stage_breakdown.png)")
    lines.append("")
    lines.append("![Cost efficiency](plots/cost_efficiency.png)")
    lines.append("")

    (ANALYSIS_DIR / "report.md").write_text("\n".join(lines))


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    records = load_experiments()
    if not records:
        print(f"No experiments found in {EXPERIMENTS_DIR}")
        return

    print(f"Loaded {len(records)} experiments")

    plot_fps_vs_gpu(records)
    plot_fps_vs_prompts(records)
    plot_stage_breakdown(records)
    plot_cost_efficiency(records)
    write_report(records)

    # Print summary table to stdout.
    print()
    print(f"{'Prompts':>8} {'GPU':>10} {'FPS':>8} {'Total(s)':>10} {'Segment(s)':>12}")
    print("-" * 55)
    for r in sorted(records, key=lambda r: (r["num_prompts"], r["gpu"])):
        print(
            f"{r['num_prompts']:>8} {r['gpu']:>10} "
            f"{r['output_fps']:>8.2f} {r['total_s']:>10.1f} "
            f"{r['segment_total_s']:>12.1f}",
        )

    print()
    print(f"Plots saved to {PLOTS_DIR}/")
    print(f"Report saved to {ANALYSIS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
