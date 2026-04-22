# -*- coding: utf-8 -*-
"""
render_hsmooth.py
-----------------
Two-pass homography-space stabilisation:
  Pass 1: fit quad on every frame → compute per-frame H (ref→frame)
  Pass 2: smooth H with a centered temporal window (look-ahead + look-back)
           + spike rejection + hold-last-good
  Render: composite logo using smoothed H

Usage:
    python render_hsmooth.py --cache-dir cache/ --logo redbull_white.png \
           --axis short --window 11 --out output_hsmooth.mp4
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


def corners_to_H(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    H, _ = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64))
    if H is None:
        return None
    H /= H[2, 2]
    return H


def warp_corners(H: np.ndarray, src: np.ndarray) -> np.ndarray:
    pts = np.hstack([src.astype(np.float64), np.ones((4, 1))]).T
    warped = (H @ pts).T
    return warped[:, :2] / warped[:, 2:3]


def smooth_homographies(H_list: list[np.ndarray | None], window: int,
                        spike_px: float = 8.0) -> list[np.ndarray]:
    """Centered temporal smoothing of homographies with outlier rejection.

    Uses a Gaussian-weighted average over a window of frames.
    Frames where H is None are skipped. If a frame's H deviates too much
    from its neighbours, it's excluded from the average (spike rejection).
    """
    n = len(H_list)
    half = window // 2
    sigma = half / 2.0

    # Gaussian weights
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    gauss = np.exp(-0.5 * (offsets / max(sigma, 0.5)) ** 2)

    # Fill gaps with nearest valid H
    filled = list(H_list)
    last_good = None
    for i in range(n):
        if filled[i] is not None:
            last_good = filled[i]
        elif last_good is not None:
            filled[i] = last_good
    # backward pass for leading Nones
    last_good = None
    for i in range(n - 1, -1, -1):
        if filled[i] is not None:
            last_good = filled[i]
        elif last_good is not None:
            filled[i] = last_good

    if filled[0] is None:
        filled = [np.eye(3)] * n

    # For spike rejection: precompute projected corners from each H
    # (we just use identity as src since we only care about relative displacement)
    ref_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64) * 100

    smoothed = []
    for i in range(n):
        H_acc = np.zeros((3, 3), dtype=np.float64)
        w_sum = 0.0

        center_proj = warp_corners(filled[i], ref_pts)

        for j_off in range(-half, half + 1):
            j = i + j_off
            if j < 0 or j >= n:
                continue
            Hj = filled[j]

            # Spike rejection: skip if this neighbour is too far from center
            proj_j = warp_corners(Hj, ref_pts)
            disp = np.mean(np.linalg.norm(proj_j - center_proj, axis=1))
            if disp > spike_px:
                continue

            wt = gauss[j_off + half]
            H_acc += wt * Hj
            w_sum += wt

        if w_sum > 0:
            H_avg = H_acc / w_sum
            H_avg /= H_avg[2, 2]
            smoothed.append(H_avg)
        else:
            smoothed.append(filled[i])

    return smoothed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--logo", required=True)
    ap.add_argument("--axis", choices=["short", "long"], default="short")
    ap.add_argument("--window", type=int, default=11,
                    help="Temporal smoothing window (centered, odd number)")
    ap.add_argument("--spike-px", type=float, default=8.0,
                    help="Spike rejection threshold in pixels")
    ap.add_argument("--out", default="output_hsmooth.mp4")
    args = ap.parse_args()

    cache = args.cache_dir
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n_frames, fps = meta["n_frames"], meta["fps"]
    w, h = meta["width"], meta["height"]
    banner_ids, player_ids = meta["banner_ids"], meta["player_ids"]
    frames_dir = os.path.join(cache, "frames")
    masks_dir = os.path.join(cache, "player_masks")

    # --- Pass 1: fit quads on every frame, compute per-frame H ---
    print("[Pass 1] Fitting quads …", flush=True)
    data0 = np.load(os.path.join(masks_dir, "00000.npz"))

    ref_src: dict[int, np.ndarray] = {}
    raw_H: dict[int, list[np.ndarray | None]] = {}

    for bid in banner_ids:
        key = f"obj{bid}"
        if key not in data0:
            continue
        bmask0 = data0[key].astype(np.uint8)
        c0 = fit_quadrilateral(bmask0, axis=args.axis, debug=False, verbose=False)
        if c0 is None:
            continue
        ref_src[bid] = c0.astype(np.float64)
        raw_H[bid] = []
        print(f"  Banner {bid}: ref corners set")

    for fi in range(n_frames):
        data = np.load(os.path.join(masks_dir, f"{fi:05d}.npz"))
        for bid in ref_src:
            key = f"obj{bid}"
            if key not in data:
                raw_H[bid].append(None)
                continue
            bmask = data[key].astype(np.uint8)
            if bmask.sum() < 50:
                raw_H[bid].append(None)
                continue
            corners = fit_quadrilateral(bmask, axis=args.axis,
                                        debug=False, verbose=False)
            if corners is None:
                raw_H[bid].append(None)
                continue
            H = corners_to_H(ref_src[bid], corners.astype(np.float64))
            raw_H[bid].append(H)
        if (fi + 1) % 100 == 0:
            print(f"    {fi + 1}/{n_frames}", flush=True)

    # --- Pass 2: smooth homographies with centered window ---
    print(f"[Pass 2] Smoothing (window={args.window}, spike={args.spike_px}px) …",
          flush=True)
    smooth_H: dict[int, list[np.ndarray]] = {}
    for bid in ref_src:
        smooth_H[bid] = smooth_homographies(raw_H[bid], args.window, args.spike_px)
        n_valid = sum(1 for x in raw_H[bid] if x is not None)
        print(f"  Banner {bid}: {n_valid}/{n_frames} valid fits")

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
        for bid in ref_src:
            Hs = smooth_H[bid][fi]
            corners = warp_corners(Hs, ref_src[bid]).astype(np.float32)

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
