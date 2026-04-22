# -*- coding: utf-8 -*-
"""
render_median.py
----------------
Two-pass: fit all corners, then use a centered median filter on each corner
coordinate. Median is much better than EMA at rejecting jitter outliers.

Usage:
    python render_median.py --cache-dir cache/ --logo redbull_white.png \
           --axis short --window 21 --out output_median.mp4
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from banner_segment import fit_quadrilateral, composite_logo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--logo", required=True)
    ap.add_argument("--axis", choices=["short", "long"], default="short")
    ap.add_argument("--window", type=int, default=21,
                    help="Median filter window size (odd, centered)")
    ap.add_argument("--out", default="output_median.mp4")
    args = ap.parse_args()

    cache = args.cache_dir
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n_frames, fps = meta["n_frames"], meta["fps"]
    w, h = meta["width"], meta["height"]
    banner_ids, player_ids = meta["banner_ids"], meta["player_ids"]
    frames_dir = os.path.join(cache, "frames")
    masks_dir = os.path.join(cache, "player_masks")

    # --- Pass 1: fit corners on every frame ---
    print("[Pass 1] Fitting corners …", flush=True)
    # raw_corners[bid][fi] = (4,2) or None
    raw_corners: dict[int, list[np.ndarray | None]] = {bid: [] for bid in banner_ids}

    for fi in range(n_frames):
        data = np.load(os.path.join(masks_dir, f"{fi:05d}.npz"))
        for bid in banner_ids:
            key = f"obj{bid}"
            if key not in data:
                raw_corners[bid].append(None)
                continue
            bmask = data[key].astype(np.uint8)
            if bmask.sum() < 50:
                raw_corners[bid].append(None)
                continue
            c = fit_quadrilateral(bmask, axis=args.axis, debug=False, verbose=False)
            raw_corners[bid].append(c)
        if (fi + 1) % 100 == 0:
            print(f"  {fi + 1}/{n_frames}", flush=True)

    # --- Pass 2: median filter on corners ---
    print(f"[Pass 2] Median filtering (window={args.window}) …", flush=True)
    half = args.window // 2
    smooth_corners: dict[int, list[np.ndarray | None]] = {}

    for bid in banner_ids:
        raw = raw_corners[bid]
        n_valid = sum(1 for c in raw if c is not None)
        print(f"  Banner {bid}: {n_valid}/{n_frames} valid fits")

        # Fill gaps with nearest valid
        filled = list(raw)
        last = None
        for i in range(n_frames):
            if filled[i] is not None:
                last = filled[i]
            elif last is not None:
                filled[i] = last
        last = None
        for i in range(n_frames - 1, -1, -1):
            if filled[i] is not None:
                last = filled[i]
            elif last is not None:
                filled[i] = last

        smoothed = []
        for i in range(n_frames):
            if filled[i] is None:
                smoothed.append(None)
                continue
            # Gather window
            window_corners = []
            for j in range(max(0, i - half), min(n_frames, i + half + 1)):
                if filled[j] is not None:
                    window_corners.append(filled[j])
            if not window_corners:
                smoothed.append(filled[i])
                continue
            # Stack (N, 4, 2) and take per-coordinate median
            stacked = np.stack(window_corners, axis=0)  # (N, 4, 2)
            med = np.median(stacked, axis=0)  # (4, 2)
            smoothed.append(med.astype(np.float32))

        smooth_corners[bid] = smoothed

    # --- Render ---
    print("[Render] Compositing …", flush=True)
    raw_out = args.out.rsplit(".", 1)[0] + "_raw.mp4"
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for fi in range(n_frames):
        frame = cv2.imread(os.path.join(frames_dir, f"{fi:05d}.jpg"))
        data = np.load(os.path.join(masks_dir, f"{fi:05d}.npz"))

        occ = np.zeros((h, w), dtype=bool)
        for pid in player_ids:
            pk = f"obj{pid}"
            if pk in data:
                occ |= data[pk].astype(bool)

        result = frame
        for bid in banner_ids:
            corners = smooth_corners[bid][fi]
            if corners is None:
                continue
            key = f"obj{bid}"
            bmask = data.get(key, np.zeros((h, w), dtype=np.uint8)).astype(np.uint8)

            result = composite_logo(
                result, corners, args.logo,
                mask=bmask,
                occlusion_mask=occ if occ.any() else None,
                save_path=None, debug=False, verbose=False,
            )

        writer.write(result)
        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            print(f"  {fi + 1}/{n_frames}", flush=True)

    writer.release()
    subprocess.run(
        ["ffmpeg", "-i", raw_out, "-c:v", "libx264", "-preset", "fast",
         "-crf", "23", "-pix_fmt", "yuv420p", args.out, "-y", "-loglevel", "error"],
        check=True,
    )
    os.remove(raw_out)
    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
