"""Pipeline orchestration — config loading, factory functions, and run_pipeline()."""

from __future__ import annotations

import time
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
from banner_pipeline.io import get_video_fps, load_frame, write_video
from banner_pipeline.segment.base import ObjectPrompt, SegmentationModel
from banner_pipeline.segment.sam2_image import SAM2ImageSegmenter
from banner_pipeline.segment.sam2_video import SAM2VideoSegmenter
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
        out.append(
            ObjectPrompt(
                obj_id=p["obj_id"],
                points=pts,
                labels=labels,
                frame_idx=p.get("frame_idx", 0),
            )
        )
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
            return {
                "frame": frame,
                "masks": {},
                "corners_map": {},
                "composited": None,
                "metrics": metrics,
            }
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

    metrics["total_s"] = sum(v for k, v in metrics.items() if k.endswith("_s"))
    return {
        "frame": frame,
        "masks": masks,
        "corners_map": corners_map,
        "composited": composited,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Video pipeline
# ---------------------------------------------------------------------------


def build_video_segmenter(cfg: dict) -> SAM2VideoSegmenter:
    kwargs = {}
    if "checkpoint" in cfg:
        kwargs["checkpoint"] = cfg["checkpoint"]
    if "model_cfg" in cfg:
        kwargs["model_cfg"] = cfg["model_cfg"]
    if "device" in cfg:
        kwargs["device"] = cfg["device"]
    return SAM2VideoSegmenter(**kwargs)


def run_pipeline_video(
    config: dict,
    output_path: str = "output.mp4",
    config_path: str | None = None,
) -> dict:
    """Execute the full video banner-replacement pipeline.

    Tracks objects across all frames, fits quads per frame, composites
    per frame, and writes an output video.

    Returns
    -------
    dict with keys: output_path, metrics
    """
    import os
    import shutil

    metrics: dict[str, Any] = {}
    pipeline_cfg = config["pipeline"]
    input_cfg = config["input"]
    video_path = input_cfg["video"]

    # --- Get prompts ---
    prompts_cfg = input_cfg.get("prompts")
    if prompts_cfg:
        prompts = _prompts_from_config(prompts_cfg)
        print(f"[video] Loaded {len(prompts)} prompts from config")
    else:
        print("[video] Interactive mode — collecting clicks …")
        frame = load_frame(video_path)
        click_groups = collect_clicks(frame)
        if not click_groups:
            print("[video] No clicks — exiting.")
            return {"output_path": None, "metrics": metrics}
        prompts = _clicks_to_prompts(click_groups)
        if config_path:
            _save_prompts_to_config(config, prompts, config_path)

    # --- Input video info ---
    input_fps = get_video_fps(video_path)
    metrics["input_fps"] = input_fps

    # --- Segment + track across all frames ---
    t0 = time.perf_counter()
    video_segmenter = build_video_segmenter(pipeline_cfg["segmenter"])
    video_segments, frame_dir, frame_names = video_segmenter.segment_video(
        video_path,
        prompts,
    )
    metrics["segment_total_s"] = time.perf_counter() - t0
    metrics["num_frames"] = len(frame_names)
    print(
        f"[video] Tracked {len(frame_names)} frames in {metrics['segment_total_s']:.2f}s",
    )

    # --- Per-frame: fit + composite ---
    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})

    overlay = None
    logo_path = input_cfg.get("logo")
    if logo_path:
        overlay = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise RuntimeError(f"Could not read logo: {logo_path}")

    compositor = build_compositor(pipeline_cfg["compositor"]) if overlay is not None else None
    compositor_params = pipeline_cfg["compositor"].get("params", {}) if overlay is not None else {}
    focal_length = pipeline_cfg.get("camera", {}).get("focal_length")

    output_frames: list[np.ndarray] = []
    fit_times: list[float] = []
    composite_times: list[float] = []

    try:
        for frame_idx, fname in enumerate(frame_names):
            frame_bgr = cv2.imread(os.path.join(frame_dir, fname))
            if frame_bgr is None:
                raise RuntimeError(f"Could not read frame {frame_idx}: {fname}")

            masks_for_frame = video_segments.get(frame_idx, {})

            # Squeeze masks to 2D (SAM2 video outputs may have extra dims).
            masks_2d: dict[int, np.ndarray] = {
                obj_id: mask.squeeze() for obj_id, mask in masks_for_frame.items()
            }

            # Fit quads for this frame.
            t_fit = time.perf_counter()
            corners_map: dict[int, np.ndarray] = {}
            for obj_id, mask_2d in masks_2d.items():
                corners = fitter.fit(mask_2d, **fitter_params)
                if corners is not None:
                    corners_map[obj_id] = corners
            fit_times.append(time.perf_counter() - t_fit)

            # Composite for this frame.
            if overlay is not None and compositor is not None and corners_map:
                t_comp = time.perf_counter()
                K = estimate_camera_matrix(frame_bgr.shape, focal_length=focal_length)
                for obj_id in sorted(corners_map):
                    extra_kw = dict(compositor_params)
                    if compositor.name == "alpha":
                        homo = compute_oriented_homography(corners_map[obj_id], K)
                        extra_kw["homo"] = homo
                    frame_bgr = compositor.composite(
                        frame_bgr,
                        corners_map[obj_id],
                        overlay,
                        mask=masks_2d.get(obj_id),
                        **extra_kw,
                    )
                composite_times.append(time.perf_counter() - t_comp)

            output_frames.append(frame_bgr)

            if (frame_idx + 1) % 50 == 0 or frame_idx == len(frame_names) - 1:
                print(f"[video] Processed frame {frame_idx + 1}/{len(frame_names)}")

    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    # --- Write output video ---
    t0 = time.perf_counter()
    write_video(output_frames, output_path, fps=input_fps)
    metrics["write_video_s"] = time.perf_counter() - t0
    print(f"[video] Wrote {len(output_frames)} frames → {output_path}")

    # --- Aggregate metrics ---
    fit_arr = np.array(fit_times) * 1000  # ms
    metrics["fit_mean_ms"] = round(float(fit_arr.mean()), 2)
    metrics["fit_std_ms"] = round(float(fit_arr.std()), 2)

    if composite_times:
        comp_arr = np.array(composite_times) * 1000
        metrics["composite_mean_ms"] = round(float(comp_arr.mean()), 2)
        metrics["composite_std_ms"] = round(float(comp_arr.std()), 2)

    metrics["total_s"] = round(
        metrics["segment_total_s"]
        + sum(fit_times)
        + sum(composite_times)
        + metrics["write_video_s"],
        4,
    )
    metrics["output_fps"] = round(len(frame_names) / metrics["total_s"], 2)

    return {
        "output_path": output_path,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def run(
    config: dict,
    config_path: str | None = None,
    output_path: str = "output.mp4",
) -> dict:
    """Dispatch to single-frame or video pipeline based on config ``mode``."""
    mode = config.get("pipeline", {}).get("mode", "image")
    if mode == "video":
        return run_pipeline_video(config, output_path=output_path, config_path=config_path)
    return run_pipeline(config, config_path=config_path)
