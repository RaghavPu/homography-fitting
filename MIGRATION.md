# Migration Guide

## What changed

The flat, single-file scripts were restructured into a modular pipeline with swappable components, YAML-based experiment configs, and shared utilities. No core algorithms were changed — only reorganized.

## File mapping

| Old file | New location(s) | Notes |
|---|---|---|
| `banner_segment.py` | `segment/sam2_image.py`, `fitting/pca_fit.py`, `composite/inpaint.py`, `ui.py`, `io.py` | Split into components; SAM2 loading and click UI extracted to shared utils |
| `region_overlay.py` | `segment/sam2_image.py`, `fitting/lp_fit.py`, `composite/alpha.py`, `homography/camera.py` | Camera-aware homography moved to its own module; LP fitter separated |
| `court_homography.py` | `fitting/hull_fit.py`, `homography/court.py`, `geometry.py` | Corner deduction logic preserved in hull fitter; line math shared |
| `video_masker.py` | `segment/sam2_video.py`, `device.py`, `io.py`, `viz.py` | VideoMasker class preserved; helpers extracted to shared modules |
| `find_diff_region.py` | `diff.py` | Moved as-is with minor cleanup |
| `test_fit.py` | `archive/` | Throwaway demo; algorithm preserved in `fitting/pca_fit.py` |
| `martina_sam2_boards.py` | `archive/` | Colab notebook; patterns captured in `segment/sam2_video.py` |

All original files are preserved in `archive/` for reference.

## What's new

- **Config system** — YAML configs in `configs/` drive the pipeline; swap components by changing `type` fields
- **Experiment tracking** — `scripts/run_experiment.py` saves frozen config, outputs, and metrics per run
- **FPS benchmarking** — `scripts/benchmark_fps.py` measures per-stage timing
- **Interactive → replay** — first run collects clicks interactively and saves coordinates to config; subsequent runs replay headless
- **Pre-commit hooks** — ruff (lint + format) + mypy (type checking) on every commit
- **uv** — `uv sync` for reproducible dependency installation

## How to run (quick reference)

```bash
# Interactive pipeline
python scripts/run_pipeline.py --config configs/default.yaml --save result.png

# Experiment with saved results
python scripts/run_experiment.py --config configs/default.yaml

# Benchmark FPS
python scripts/benchmark_fps.py --config configs/default.yaml --runs 5
```
