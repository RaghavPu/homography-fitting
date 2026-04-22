# -*- coding: utf-8 -*-
"""
render_logo_from_cache.py
-------------------------
Load precomputed SAM2 masks from cache/ and composite a logo onto banners.

Tracking strategy:
  1. Frame 0: fit quad from SAM2 mask → get 4 corners
  2. Detect good feature points (Shi-Tomasi) inside the banner region
  3. Each frame: track features with LK optical flow → compute homography
     from correspondences (RANSAC) → warp the corners
  4. EMA smooth the corners
  5. Re-detect features periodically to avoid drift

Usage:
    python render_logo_from_cache.py --cache-dir cache/ --logo redbull_white.png \
           --axis short --ema-alpha 0.1 --out output_flow.mp4
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


LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)

REDETECT_INTERVAL = 30   # re-detect features every N frames to fight drift
MIN_FEATURES = 15        # re-detect if tracked points drop below this


def detect_features(gray: np.ndarray, mask_u8: np.ndarray) -> np.ndarray | None:
    """Find good features inside the mask region (dilated a bit for margin)."""
    roi = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    pts = cv2.goodFeaturesToTrack(gray, mask=roi, **FEATURE_PARAMS)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--logo", default=None,
                    help="Fallback logo for all banners")
    ap.add_argument("--logo-long", default=None,
                    help="Logo for long-axis banners")
    ap.add_argument("--logo-short", default=None,
                    help="Logo for short-axis banners")
    ap.add_argument("--axis", choices=["short", "long"], default="short",
                    help="Fallback axis if cache has no long/short split")
    ap.add_argument("--ema-alpha", type=float, default=0.15,
                    help="EMA smoothing on corners (0=frozen, 1=no smoothing)")
    ap.add_argument("--eps", type=float, default=5.0,
                    help="Deadzone: ignore corner updates smaller than this (pixels)")
    ap.add_argument("--alpha-fast", type=float, default=0.6,
                    help="EMA alpha for large motion (camera pan)")
    ap.add_argument("--motion-thresh", type=float, default=15.0,
                    help="Mean corner displacement (px) to trigger fast alpha")
    ap.add_argument("--skip-ids", type=int, nargs="*", default=[],
                    help="Banner IDs to skip (e.g. --skip-ids 5 6)")
    ap.add_argument("--lum-strength", type=float, default=0.3,
                    help="Luminosity matching strength (0=keep original, 1=full remap)")
    ap.add_argument("--out", default="output_flow.mp4")
    args = ap.parse_args()

    logo_long = args.logo_long or args.logo
    logo_short = args.logo_short or args.logo
    if not logo_long and not logo_short:
        print("ERROR: provide --logo or --logo-long / --logo-short")
        return

    cache = args.cache_dir
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n_frames = meta["n_frames"]
    fps = meta["fps"]
    w, h = meta["width"], meta["height"]
    banner_ids = meta["banner_ids"]
    player_ids = meta["player_ids"]

    long_ids = set(meta.get("banner_long_ids", []))
    short_ids = set(meta.get("banner_short_ids", []))
    if not long_ids and not short_ids:
        short_ids = set(banner_ids)

    def axis_for(bid):
        return "long" if bid in long_ids else "short"

    def logo_for(bid):
        return logo_long if bid in long_ids else logo_short

    skip_ids = set(args.skip_ids)
    print(f"  Long-axis banners: {sorted(long_ids)} → {logo_long}")
    print(f"  Short-axis banners: {sorted(short_ids)} → {logo_short}")
    if skip_ids:
        print(f"  Skipping banner IDs: {sorted(skip_ids)}")

    frames_dir = os.path.join(cache, "frames")
    masks_dir = os.path.join(cache, "player_masks")

    alpha = args.ema_alpha

    # --- Frame-0 setup per banner ---
    frame0 = cv2.imread(os.path.join(frames_dir, "00000.jpg"))
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    data0 = np.load(os.path.join(masks_dir, "00000.npz"))

    # Per-banner state
    ref_corners: dict[int, np.ndarray] = {}
    cur_corners: dict[int, np.ndarray] = {}
    ema_corners: dict[int, np.ndarray] = {}
    prev_pts: dict[int, np.ndarray] = {}
    prev_gray: dict[int, np.ndarray] = {}
    frames_since_detect: dict[int, int] = {}
    ref_lum_stats: dict[int, tuple[float, float]] = {}

    active_banner_ids = [bid for bid in banner_ids if bid not in skip_ids]

    frame0_lab = cv2.cvtColor(frame0, cv2.COLOR_BGR2LAB).astype(np.float32)

    for bid in active_banner_ids:
        key = f"obj{bid}"
        if key not in data0:
            continue
        bmask = data0[key].astype(np.uint8) * 255
        corners = fit_quadrilateral(data0[key].astype(np.uint8), axis=axis_for(bid),
                                    debug=False, verbose=False)
        if corners is None:
            continue

        ref_corners[bid] = corners.astype(np.float64)
        cur_corners[bid] = corners.astype(np.float64)
        ema_corners[bid] = corners.astype(np.float64)

        mask_bool = data0[key].astype(bool)
        if mask_bool.any():
            l_vals = frame0_lab[mask_bool, 0]
            ref_lum_stats[bid] = (
                float(np.percentile(l_vals, 10)),
                float(np.percentile(l_vals, 90)),
            )
            print(f"  Banner {bid}: ref L range [{ref_lum_stats[bid][0]:.0f}, {ref_lum_stats[bid][1]:.0f}]")

        pts = detect_features(gray0, bmask)
        if pts is None or len(pts) < 4:
            print(f"  Banner {bid}: not enough features on frame 0, skipping")
            continue
        prev_pts[bid] = pts
        prev_gray[bid] = gray0.copy()
        frames_since_detect[bid] = 0
        print(f"  Banner {bid}: {len(pts)} features, corners fitted")

    # --- Video writer ---
    raw_out = args.out.rsplit(".", 1)[0] + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (w, h))

    for fi in range(n_frames):
        frame = cv2.imread(os.path.join(frames_dir, f"{fi:05d}.jpg"))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_path = os.path.join(masks_dir, f"{fi:05d}.npz")
        data = np.load(mask_path) if os.path.exists(mask_path) else {}

        # Player occlusion
        occ_mask = np.zeros((h, w), dtype=bool)
        for pid in player_ids:
            pkey = f"obj{pid}"
            if pkey in data:
                occ_mask |= data[pkey].astype(bool)

        result = frame
        for bid in active_banner_ids:
            if bid not in ref_corners or bid not in prev_pts:
                continue
            key = f"obj{bid}"

            if fi > 0:
                # Track features from previous frame
                old_pts = prev_pts[bid]
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray[bid], gray, old_pts, None, **LK_PARAMS)

                # Forward-backward check
                back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(
                    gray, prev_gray[bid], new_pts, None, **LK_PARAMS)
                fb_err = np.linalg.norm(
                    (old_pts - back_pts).reshape(-1, 2), axis=1)

                good = (status.ravel() == 1) & (status_b.ravel() == 1) & (fb_err < 1.5)

                old_good = old_pts[good].reshape(-1, 2)
                new_good = new_pts[good].reshape(-1, 2)

                if len(old_good) >= 4:
                    H, inliers = cv2.findHomography(old_good, new_good, cv2.RANSAC, 3.0)
                    if H is not None:
                        pts_h = np.hstack([cur_corners[bid],
                                           np.ones((4, 1))]).T  # 3x4
                        warped = (H @ pts_h).T
                        warped = warped[:, :2] / warped[:, 2:3]
                        cur_corners[bid] = warped

                    prev_pts[bid] = new_good.reshape(-1, 1, 2).astype(np.float32)
                else:
                    prev_pts[bid] = new_pts[status.ravel() == 1].reshape(-1, 1, 2)

                frames_since_detect[bid] += 1

                # Re-detect features periodically or when too few remain
                need_redetect = (frames_since_detect[bid] >= REDETECT_INTERVAL
                                 or len(prev_pts[bid]) < MIN_FEATURES)
                if need_redetect and key in data:
                    bmask_u8 = data[key].astype(np.uint8) * 255
                    new_feats = detect_features(gray, bmask_u8)
                    if new_feats is not None and len(new_feats) >= 4:
                        prev_pts[bid] = new_feats
                        frames_since_detect[bid] = 0

            prev_gray[bid] = gray.copy()

            # Adaptive EMA: 0 below deadzone, base alpha normal, fast alpha for big motion
            delta = np.mean(np.linalg.norm(cur_corners[bid] - ema_corners[bid], axis=1))
            if delta <= args.eps:
                a = 0.0
            elif delta >= args.motion_thresh:
                a = args.alpha_fast
            else:
                a = alpha
            if a > 0:
                ema_corners[bid] = (a * cur_corners[bid]
                                    + (1 - a) * ema_corners[bid])
            smooth = ema_corners[bid].astype(np.float32)

            bmask = data.get(key, np.zeros((h, w), dtype=np.uint8))

            bl = logo_for(bid)
            if bl is None:
                continue
            result = composite_logo(
                result, smooth, bl,
                mask=bmask,
                occlusion_mask=occ_mask if occ_mask.any() else None,
                save_path=None,
                debug=False,
                verbose=False,
                lum_strength=args.lum_strength,
                ref_lum=ref_lum_stats.get(bid),
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
