# Banner Pipeline

Video banner/logo replacement using SAM2 segmentation. Detects billboard regions in video frames, tracks them across all frames, fits perspective-aware quadrilaterals, and composites new logos with correct aspect ratio and luminosity matching.

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd homography-fitting

# 2. Install all dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Authenticate with Modal (one-time, for GPU runs)
uv run modal setup
```

SAM2 setup is only needed for **local** runs. Modal builds SAM2 from source automatically.

```bash
# Only if running locally (not needed for Modal)
git clone https://github.com/facebookresearch/sam2.git
pip install -e ./sam2
cd sam2/checkpoints && ./download_ckpts.sh && cd ../..
```

## Running the pipeline

Two-step process: collect clicks locally, then run on a remote GPU.

### Step 1: Select banner regions (local, no GPU needed)

```bash
uv run python scripts/collect_prompts.py --config configs/default.yaml
```

This opens the first frame of the video. Click on the banner region(s) you want to track. The coordinates are saved into the config file automatically.

### Step 2: Run on a GPU via Modal

```bash
# Video mode (processes all frames, outputs .mp4)
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu T4 --mode video

# Image mode (processes single frame, outputs .png)
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu T4 --mode image
```

### Available GPUs

Pass any of these to `--gpu`:

| GPU | VRAM | Cost/hr |
|-----|------|---------|
| `T4` | 16 GB | $0.59 |
| `L4` | 24 GB | $0.80 |
| `A10G` | 24 GB | $1.10 |
| `L40S` | 48 GB | $1.95 |
| `A100` | 40 GB | $2.10 |
| `A100-80GB` | 80 GB | $2.50 |
| `H100` | 80 GB | $3.95 |
| `H200` | 141 GB | $4.54 |
| `B200` | 192 GB | $6.25 |

### Benchmarking across GPUs

```bash
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu T4 --mode video --benchmark 5
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu A100 --mode video --benchmark 5
```

## Metrics

Each run produces a `metrics.json` in the experiment directory. Example output (video mode, T4):

```json
{
  "gpu": "Tesla T4",
  "gpu_memory_gb": 14.6,
  "mode": "video",
  "num_frames": 202,
  "input_fps": 25.0,
  "segment_total_s": 95.68,
  "fit_mean_ms": 10.25,
  "composite_mean_ms": 202.53,
  "write_video_s": 2.54,
  "total_s": 141.21,
  "output_fps": 1.43
}
```

| Metric | Description |
|--------|-------------|
| `num_frames` | Total frames in the video |
| `input_fps` | Original video framerate |
| `segment_total_s` | Time for SAM2 to track objects across all frames |
| `fit_mean_ms` | Average time to fit a quad per frame |
| `composite_mean_ms` | Average time to composite logo per frame |
| `write_video_s` | Time to encode the output video |
| `total_s` | End-to-end wall time |
| `output_fps` | Processing speed (`num_frames / total_s`) — compare this to `input_fps` to gauge how far from real-time |

## Experiments and reproducibility

Each run saves to `experiments/<timestamp>_<name>/`:

```
experiments/2026-04-07_20-38-28_pca_T4/
  config.yaml      # frozen config with exact click coordinates + all settings
  metrics.json      # timing, FPS, GPU info
  outputs/
    composited.mp4   # output video (or .png for image mode)
```

Everything is tracked in git — configs, metrics, and outputs. For long videos that exceed GitHub's file size limit, the output will be rejected by git; in that case, just add the specific output to `.gitignore` and let teammates reproduce it from the saved config:

```bash
# Reproduce an experiment exactly
uv run modal run scripts/modal_run.py --config experiments/2026-04-07_20-38-28_pca_T4/config.yaml --gpu T4

# Reuse same coordinates with different settings
cp experiments/2026-04-07_20-38-28_pca_T4/config.yaml configs/experiments/my_test.yaml
# edit fitter.type, compositor.type, etc.
uv run modal run scripts/modal_run.py --config configs/experiments/my_test.yaml --gpu A100 --mode video
```

## Running locally

```bash
# Interactive (opens UI for clicking + runs SAM2 locally)
uv run python scripts/run_pipeline.py --config configs/default.yaml --save result.png

# Run experiment with saved outputs + metrics
uv run python scripts/run_experiment.py --config configs/default.yaml
```

## Swapping components

Change the config to use different algorithms:

```yaml
pipeline:
  fitter:
    type: lp           # pca | lp | hull
  compositor:
    type: alpha         # inpaint | alpha
```

| Fitter | Algorithm | Best for |
|--------|-----------|----------|
| `pca` | Weighted PCA with Hann windows | Rectangular banners |
| `lp` | Linear programming supporting lines | Tight convex bounds |
| `hull` | Hull vertex deduction | Regions extending off-screen |

## Adding a new segmentation model

1. Create `src/banner_pipeline/segment/sam3_image.py`
2. Implement the `SegmentationModel` interface (see `segment/base.py`)
3. Register it in `pipeline.py`: `SEGMENTERS["sam3"] = SAM3ImageSegmenter`
4. Set `segmenter.type: sam3` in your config

## Project structure

```
src/banner_pipeline/
  io.py, device.py, geometry.py, ui.py, viz.py   # shared utilities
  segment/    sam2_image.py, sam2_video.py         # segmentation models
  fitting/    pca_fit.py, lp_fit.py, hull_fit.py   # quad fitting algorithms
  homography/ camera.py, court.py                  # camera intrinsics
  composite/  inpaint.py, alpha.py                 # compositing strategies
  pipeline.py                                      # orchestration + config
configs/      default.yaml, experiments/            # experiment configs
scripts/      collect_prompts.py, modal_run.py,     # main workflow
              run_pipeline.py, run_experiment.py,    # local alternatives
              benchmark_fps.py
```

See [MIGRATION.md](MIGRATION.md) for how the old files map to this structure.
