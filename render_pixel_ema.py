# -*- coding: utf-8 -*-
"""
render_pixel_ema.py
-------------------
EMA smooth the actual warped logo overlay in pixel space, not just corners.

Per frame:
  1. Fit quad from mask → get corners
  2. Warp logo onto frame-sized canvas → get warped_rgb + warped_alpha
  3. EMA smooth the warped_rgb and warped_alpha across frames
  4. Blend the smoothed overlay onto the current frame

This smooths the actual composited image, not just corner positions.

Usage:
    python render_pixel_ema.py --cache-dir cache/ --logo redbull_white.png \
           --axis short --ema-alpha 0.03 --out output_pixel_ema.mp4
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

from banner_segment import fit_quadrilateral


def warp_logo(logo_bgra: np.ndarray, corners: np.ndarray,
              frame_h: int, frame_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Warp logo into frame space. Returns (warped_rgb, warped_alpha) both float32."""
    h_logo, w_logo = logo_bgra.shape[:2]

    # Compute canvas from corner edge lengths
    w_top = np.linalg.norm(corners[1] - corners[0])
    w_bot = np.linalg.norm(corners[2] - corners[3])
    h_left = np.linalg.norm(corners[3] - corners[0])
    h_right = np.linalg.norm(corners[2] - corners[1])
    avg_w = (w_top + w_bot) / 2
    avg_h = (h_left + h_right) / 2
    scale_up = max(1.0, 500 / max(avg_w, avg_h))
    canvas_w = max(int(avg_w * scale_up), 1)
    canvas_h = max(int(avg_h * scale_up), 1)

    # Place logo on canvas with padding
    rgb_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    pad_frac = 0.05
    pad_w = int(canvas_w * pad_frac)
    pad_h = int(canvas_h * pad_frac)
    scale = min((canvas_w - 2 * pad_w) / w_logo, (canvas_h - 2 * pad_h) / h_logo)
    new_w, new_h = int(w_logo * scale), int(h_logo * scale)
    resized = cv2.resize(logo_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2
    rgb_canvas[y0:y0 + new_h, x0:x0 + new_w] = resized[:, :, :3]
    alpha_canvas[y0:y0 + new_h, x0:x0 + new_w] = resized[:, :, 3]

    # Homography: canvas → frame
    src = np.array([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]], dtype=np.float32)
    H, _ = cv2.findHomography(src, corners.astype(np.float32))
    warped_rgb = cv2.warpPerspective(rgb_canvas, H, (frame_w, frame_h)).astype(np.float32)
    warped_alpha = cv2.warpPerspective(alpha_canvas, H, (frame_w, frame_h)).astype(np.float32) / 255.0

    return warped_rgb, warped_alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--logo", required=True)
    ap.add_argument("--axis", choices=["short", "long"], default="short")
    ap.add_argument("--ema-alpha", type=float, default=0.03,
                    help="EMA on the warped overlay pixels (lower = smoother)")
    ap.add_argument("--out", default="output_pixel_ema.mp4")
    args = ap.parse_args()

    cache = args.cache_dir
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n_frames, fps = meta["n_frames"], meta["fps"]
    w, h = meta["width"], meta["height"]
    banner_ids, player_ids = meta["banner_ids"], meta["player_ids"]
    frames_dir = os.path.join(cache, "frames")
    masks_dir = os.path.join(cache, "player_masks")
    alpha = args.ema_alpha

    # Load logo once
    logo_bgra = cv2.imread(args.logo, cv2.IMREAD_UNCHANGED)
    if logo_bgra.shape[2] == 3:
        logo_bgra = cv2.cvtColor(logo_bgra, cv2.COLOR_BGR2BGRA)

    # EMA accumulators per banner: smoothed warped overlay
    ema_rgb: dict[int, np.ndarray] = {}    # float32 (H, W, 3)
    ema_alpha: dict[int, np.ndarray] = {}  # float32 (H, W)

    raw_out = args.out.rsplit(".", 1)[0] + "_raw.mp4"
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for fi in range(n_frames):
        frame = cv2.imread(os.path.join(frames_dir, f"{fi:05d}.jpg"))
        data = np.load(os.path.join(masks_dir, f"{fi:05d}.npz"))

        # Player occlusion
        occ = np.zeros((h, w), dtype=np.float32)
        for pid in player_ids:
            pk = f"obj{pid}"
            if pk in data:
                occ = np.maximum(occ, data[pk].astype(np.float32))
        # Soften occlusion edges
        if occ.any():
            occ_u8 = (occ * 255).astype(np.uint8)
            occ_u8 = cv2.dilate(occ_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            occ = cv2.GaussianBlur(occ_u8, (11, 11), 3.0).astype(np.float32) / 255.0

        # Inpaint old logos from all banner masks first
        inpainted = frame.copy()
        for bid in banner_ids:
            key = f"obj{bid}"
            if key not in data:
                continue
            bmask = data[key].astype(np.uint8)
            if bmask.sum() < 50:
                continue
            mask_u8 = bmask * 255
            mask_u8 = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            inpainted = cv2.inpaint(inpainted, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        result = inpainted.astype(np.float32)

        for bid in banner_ids:
            key = f"obj{bid}"
            if key not in data:
                continue
            bmask = data[key].astype(np.uint8)
            if bmask.sum() < 50:
                continue

            corners = fit_quadrilateral(bmask, axis=args.axis,
                                        debug=False, verbose=False)
            if corners is None:
                if bid in ema_rgb:
                    w_rgb = ema_rgb[bid]
                    w_a = ema_alpha[bid]
                else:
                    continue
            else:
                w_rgb, w_a = warp_logo(logo_bgra, corners, h, w)

                if bid not in ema_rgb:
                    ema_rgb[bid] = w_rgb.copy()
                    ema_alpha[bid] = w_a.copy()
                else:
                    ema_rgb[bid] = alpha * w_rgb + (1 - alpha) * ema_rgb[bid]
                    ema_alpha[bid] = alpha * w_a + (1 - alpha) * ema_alpha[bid]

                w_rgb = ema_rgb[bid]
                w_a = ema_alpha[bid]

            # Subtract player occlusion from alpha
            effective_a = np.clip(w_a - occ, 0, 1)[..., None]

            # Blend smoothed overlay onto inpainted frame
            result = w_rgb * effective_a + result * (1 - effective_a)

        writer.write(result.clip(0, 255).astype(np.uint8))
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
