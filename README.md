# Banner Pipeline

Video banner/logo replacement using SAM2 segmentation. Detects billboard regions in video frames, fits perspective-aware quadrilaterals, and composites new logos with correct aspect ratio and luminosity matching.

## Quick start

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd homography-fitting

# 2. Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# 3. Install SAM2
git clone https://github.com/facebookresearch/sam2.git
pip install -e ./sam2

# 4. Download a SAM2 checkpoint
cd sam2/checkpoints && ./download_ckpts.sh && cd ../..

# 5. Install pre-commit hooks
uv run pre-commit install

# 6. Run the pipeline (interactive — click on banner regions)
python scripts/run_pipeline.py --config configs/default.yaml --save result.png
```

## Pipeline stages

```
Input frame → [Segment] → [Fit quad] → [Composite] → Output frame
                SAM2        PCA/LP/Hull   Inpaint/Alpha
```

Each stage is swappable via the YAML config.

## Running experiments

```bash
# Run and save outputs + metrics
python scripts/run_experiment.py --config configs/default.yaml

# Benchmark FPS (requires pre-defined prompts in config)
python scripts/benchmark_fps.py --config configs/default.yaml --runs 10
```

Experiment results are saved to `experiments/<timestamp>_<name>/` with frozen config, output images, and `metrics.json`.

## Swapping components

Change the config to use different algorithms:

```yaml
# configs/experiments/lp_oriented.yaml
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
scripts/      run_pipeline.py, run_experiment.py, benchmark_fps.py
```

See [MIGRATION.md](MIGRATION.md) for how the old files map to this structure.
