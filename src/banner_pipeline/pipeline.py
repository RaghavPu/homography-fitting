"""Pipeline orchestration — config loading, factory functions, and run_pipeline()."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from banner_pipeline.composite.alpha import AlphaCompositor
from banner_pipeline.composite.base import Compositor
from banner_pipeline.composite.inpaint import InpaintCompositor
from banner_pipeline.fitting.base import QuadFitter
from banner_pipeline.fitting.hull_fit import HullFitter
from banner_pipeline.fitting.lp_fit import LPFitter
from banner_pipeline.fitting.pca_fit import PCAFitter
from banner_pipeline.homography.camera import compute_oriented_homography, estimate_camera_matrix
from banner_pipeline.io import load_frame
from banner_pipeline.segment.base import ObjectPrompt, SegmentationModel
from banner_pipeline.segment.sam2_image import SAM2ImageSegmenter
from banner_pipeline.ui import collect_clicks

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

SEGMENTERS: dict[str, type] = {
    "sam2_image": SAM2ImageSegmenter,
}

FITTERS: dict[str, type[QuadFitter]] = {
    "pca": PCAFitter,
    "lp": LPFitter,
    "hull": HullFitter,
}

COMPOSITORS: dict[str, type[Compositor]] = {
    "inpaint": InpaintCompositor,
    "alpha": AlphaCompositor,
}

# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_segmenter(cfg: dict) -> SegmentationModel:
    cls = SEGMENTERS[cfg["type"]]
    kwargs = {}
    if "checkpoint" in cfg:
        kwargs["checkpoint"] = cfg["checkpoint"]
    if "model_cfg" in cfg:
        kwargs["model_cfg"] = cfg["model_cfg"]
    if "device" in cfg:
        kwargs["device"] = cfg["device"]
    return cls(**kwargs)


def build_fitter(cfg: dict) -> QuadFitter:
    return FITTERS[cfg["type"]]()


def build_compositor(cfg: dict) -> Compositor:
    return COMPOSITORS[cfg["type"]]()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load a YAML config and return the dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _prompts_from_config(prompts_cfg: list[dict]) -> list[ObjectPrompt]:
    """Convert a list of prompt dicts from YAML to ObjectPrompt instances."""
    out = []
    for p in prompts_cfg:
        pts = np.array(p["points"], dtype=np.float32)
        labels = np.ones(len(pts), dtype=np.int32)
        if "labels" in p:
            labels = np.array(p["labels"], dtype=np.int32)
        out.append(ObjectPrompt(
            obj_id=p["obj_id"],
            points=pts,
            labels=labels,
            frame_idx=p.get("frame_idx", 0),
        ))
    return out


def _clicks_to_prompts(click_groups: list[list[tuple[int, int]]]) -> list[ObjectPrompt]:
    """Convert interactive click groups to ObjectPrompt list."""
    prompts = []
    for idx, group in enumerate(click_groups):
        obj_id = idx + 1
        pts = np.array(group, dtype=np.float32)
        labels = np.ones(len(group), dtype=np.int32)
        prompts.append(ObjectPrompt(obj_id=obj_id, points=pts, labels=labels))
    return prompts


def _save_prompts_to_config(
    config: dict,
    prompts: list[ObjectPrompt],
    config_path: str,
) -> None:
    """Write collected prompts back into the config YAML for replay."""
    prompts_list = []
    for p in prompts:
        entry: dict[str, Any] = {
            "obj_id": p.obj_id,
            "points": p.points.tolist(),
        }
        if p.frame_idx != 0:
            entry["frame_idx"] = p.frame_idx
        prompts_list.append(entry)

    config["input"]["prompts"] = prompts_list
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Prompts saved to: {config_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    config: dict,
    config_path: str | None = None,
) -> dict:
    """Execute the full banner-replacement pipeline.

    Parameters
    ----------
    config : dict
        Parsed YAML config.
    config_path : str, optional
        Path to the config file (used for auto-saving prompts on interactive runs).

    Returns
    -------
    dict with keys: frame, masks, corners_map, composited, metrics
    """
    metrics: dict[str, Any] = {}
    pipeline_cfg = config["pipeline"]
    input_cfg = config["input"]

    # --- Load frame ---
    t0 = time.perf_counter()
    frame = load_frame(input_cfg["video"])
    metrics["load_frame_s"] = time.perf_counter() - t0
    print(f"[pipeline] Frame: {frame.shape[1]}x{frame.shape[0]}")

    # --- Get prompts (interactive or from config) ---
    prompts_cfg = input_cfg.get("prompts")
    if prompts_cfg:
        prompts = _prompts_from_config(prompts_cfg)
        print(f"[pipeline] Loaded {len(prompts)} prompts from config")
    else:
        print("[pipeline] Interactive mode — collecting clicks …")
        click_groups = collect_clicks(frame)
        if not click_groups:
            print("[pipeline] No clicks — exiting.")
            return {"frame": frame, "masks": {}, "corners_map": {}, "composited": None, "metrics": metrics}
        prompts = _clicks_to_prompts(click_groups)
        # Save prompts back to config for replay.
        if config_path:
            _save_prompts_to_config(config, prompts, config_path)

    # --- Segment ---
    t0 = time.perf_counter()
    segmenter = build_segmenter(pipeline_cfg["segmenter"])
    masks = segmenter.segment(frame, prompts)
    metrics["segment_s"] = time.perf_counter() - t0
    print(f"[pipeline] Segmented {len(masks)} objects in {metrics['segment_s']:.2f}s")

    # --- Fit quads ---
    t0 = time.perf_counter()
    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})
    corners_map: dict[int, np.ndarray] = {}
    for obj_id, mask in masks.items():
        corners = fitter.fit(mask, **fitter_params)
        if corners is not None:
            corners_map[obj_id] = corners
    metrics["fit_s"] = time.perf_counter() - t0
    print(f"[pipeline] Fitted {len(corners_map)} quads in {metrics['fit_s']:.2f}s")

    # --- Composite ---
    composited = None
    logo_path = input_cfg.get("logo")
    if logo_path and corners_map:
        overlay = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise RuntimeError(f"Could not read logo: {logo_path}")

        t0 = time.perf_counter()
        compositor = build_compositor(pipeline_cfg["compositor"])
        compositor_params = pipeline_cfg["compositor"].get("params", {})
        composited = frame.copy()

        # Camera matrix for alpha compositor.
        focal_length = pipeline_cfg.get("camera", {}).get("focal_length")
        K = estimate_camera_matrix(frame.shape, focal_length=focal_length)

        for obj_id in sorted(corners_map):
            extra_kw = dict(compositor_params)
            if compositor.name == "alpha":
                homo = compute_oriented_homography(corners_map[obj_id], K)
                extra_kw["homo"] = homo
            composited = compositor.composite(
                composited,
                corners_map[obj_id],
                overlay,
                mask=masks.get(obj_id),
                **extra_kw,
            )
        metrics["composite_s"] = time.perf_counter() - t0
        print(f"[pipeline] Composited in {metrics['composite_s']:.2f}s")

    metrics["total_s"] = sum(
        v for k, v in metrics.items() if k.endswith("_s")
    )
    return {
        "frame": frame,
        "masks": masks,
        "corners_map": corners_map,
        "composited": composited,
        "metrics": metrics,
    }
