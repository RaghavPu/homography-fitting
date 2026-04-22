# -*- coding: utf-8 -*-
"""
precompute_masks.py
-------------------
Run SAM2 video propagation for banners AND players once, and save all
per-frame masks + metadata to a cache directory.  Later scripts can load
the cache instantly without re-running SAM2.

Usage
-----
    python precompute_masks.py tennis-clip.mp4 --cache-dir cache/

Output structure (inside --cache-dir):
    meta.json             — clicks, obj IDs, fps, frame size, axis, …
    frames/00000.jpg …    — extracted JPEG frames (kept for debug video)
    banner_masks.npz      — frame-0 banner masks  {obj1: (H,W), …}
    player_masks/00000.npz … — per-frame player masks {obj_id: (H,W) bool}
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reuse helpers from banner_segment
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from banner_segment import (
    collect_clicks,
    extract_frames,
    get_video_fps,
    _collect_player_clicks,
)


def _sam2_setup(checkpoint: str, model_cfg: str):
    _repo = os.path.join(SCRIPT_DIR, "sam2")
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    import torch
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}", flush=True)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    return device


def main():
    ap = argparse.ArgumentParser(description="Precompute SAM2 masks for banners + players")
    ap.add_argument("video", help="Input video file")
    ap.add_argument("--cache-dir", default="cache", help="Where to save everything")
    ap.add_argument("--checkpoint", default="sam2/checkpoints/sam2.1_hiera_tiny.pt")
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    args = ap.parse_args()

    cache = args.cache_dir
    os.makedirs(cache, exist_ok=True)
    frames_sub = os.path.join(cache, "frames")
    player_sub = os.path.join(cache, "player_masks")

    # ---- 1. Extract frames ----
    print("[1/5] Extracting frames …", flush=True)
    if os.path.isdir(frames_sub) and glob.glob(os.path.join(frames_sub, "*.jpg")):
        print("  Frames already extracted, reusing.")
    else:
        extract_frames(args.video, out_dir=frames_sub)

    frame_names = sorted(
        [p for p in os.listdir(frames_sub) if p.lower().endswith((".jpg", ".jpeg"))],
        key=lambda p: int(os.path.splitext(p)[0]),
    )
    n_frames = len(frame_names)
    fps = get_video_fps(args.video)
    frame0 = cv2.imread(os.path.join(frames_sub, frame_names[0]))
    h, w = frame0.shape[:2]
    print(f"  {n_frames} frames, {w}x{h} @ {fps:.1f} fps")

    # ---- 2. Collect clicks (long-axis banners, short-axis banners, players) ----
    print("[2/6] Click on LONG-AXIS banners (Esc to skip) …", flush=True)
    long_clicks = collect_clicks(frame0)
    print(f"  {len(long_clicks)} long-axis banner(s)")

    print("[3/6] Click on SHORT-AXIS banners (Esc to skip) …", flush=True)
    short_clicks = collect_clicks(frame0)
    print(f"  {len(short_clicks)} short-axis banner(s)")

    banner_clicks = long_clicks + short_clicks
    if not banner_clicks:
        print("No banner clicks — exiting.")
        return

    print("[4/6] Click on PLAYERS (Esc to skip) …", flush=True)
    player_clicks = _collect_player_clicks(frame0)
    print(f"  {len(player_clicks)} player group(s)")

    all_clicks = banner_clicks + player_clicks
    n_long = len(long_clicks)
    n_short = len(short_clicks)
    banner_long_ids = list(range(1, n_long + 1))
    banner_short_ids = list(range(n_long + 1, n_long + n_short + 1))
    banner_ids = banner_long_ids + banner_short_ids
    player_ids = list(range(len(banner_clicks) + 1, len(all_clicks) + 1))

    # ---- 3. SAM2 video propagation (banners + players together) ----
    print("[4/5] Running SAM2 video predictor …", flush=True)
    device = _sam2_setup(args.checkpoint, args.model_cfg)

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=device)

    inference_state = predictor.init_state(video_path=frames_sub)

    for idx, group in enumerate(all_clicks):
        obj_id = idx + 1
        points = np.array(group, dtype=np.float32)
        labels = np.ones(len(group), dtype=np.int32)
        _, _, logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        kind = "banner" if obj_id in banner_ids else "player"
        coverage = float((logits[0] > 0.0).float().mean())
        print(f"  {kind} obj {obj_id}: {len(group)} pts, coverage={coverage:.4f}", flush=True)

    print("  Propagating …", flush=True)
    video_segments: dict[int, dict[int, np.ndarray]] = {}
    for fi, out_obj_ids, out_logits in predictor.propagate_in_video(inference_state):
        video_segments[fi] = {
            int(oid): (out_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, oid in enumerate(out_obj_ids)
        }
        if (fi + 1) % 100 == 0:
            print(f"    frame {fi + 1}/{n_frames}", flush=True)
    print(f"  Done — {len(video_segments)} frames propagated.")

    # ---- 4. Save to cache ----
    print("[5/5] Saving cache …", flush=True)

    # Banner masks from frame 0
    banner_mask_dict = {}
    seg0 = video_segments.get(0, {})
    for bid in banner_ids:
        m = seg0.get(bid)
        if m is not None:
            banner_mask_dict[f"obj{bid}"] = m.astype(np.uint8)
    np.savez_compressed(os.path.join(cache, "banner_masks.npz"), **banner_mask_dict)
    print(f"  Saved banner_masks.npz ({len(banner_mask_dict)} masks)")

    # Per-frame player masks
    os.makedirs(player_sub, exist_ok=True)
    for fi in range(n_frames):
        seg = video_segments.get(fi, {})
        frame_dict = {}
        for pid in player_ids:
            m = seg.get(pid)
            if m is not None:
                frame_dict[f"obj{pid}"] = m.astype(np.uint8)
        # Also store banner masks per frame for debugging
        for bid in banner_ids:
            m = seg.get(bid)
            if m is not None:
                frame_dict[f"obj{bid}"] = m.astype(np.uint8)
        np.savez_compressed(os.path.join(player_sub, f"{fi:05d}.npz"), **frame_dict)

    print(f"  Saved {n_frames} per-frame mask files to {player_sub}/")

    # Metadata
    meta = {
        "video": os.path.abspath(args.video),
        "n_frames": n_frames,
        "fps": fps,
        "width": w,
        "height": h,
        "banner_clicks": banner_clicks,
        "banner_long_clicks": long_clicks,
        "banner_short_clicks": short_clicks,
        "player_clicks": player_clicks,
        "banner_ids": banner_ids,
        "banner_long_ids": banner_long_ids,
        "banner_short_ids": banner_short_ids,
        "player_ids": player_ids,
    }
    with open(os.path.join(cache, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved meta.json")

    print(f"\nAll done. Cache directory: {cache}/")
    print(f"  frames/         — {n_frames} JPEGs")
    print(f"  banner_masks.npz — {len(banner_ids)} banner mask(s) from frame 0")
    print(f"  player_masks/    — per-frame masks ({len(player_ids)} players + {len(banner_ids)} banners)")
    print(f"  meta.json        — clicks, IDs, video info")


if __name__ == "__main__":
    main()
