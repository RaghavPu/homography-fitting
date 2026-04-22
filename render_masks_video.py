# -*- coding: utf-8 -*-
"""
render_masks_video.py
---------------------
Load precomputed masks from cache/ and render an overlay video showing
all segmented objects (banners + players) with colour-coded masks.

Usage:
    python render_masks_video.py --cache-dir cache/ --out masks_overlay.mp4
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess

import cv2
import numpy as np

COLORS_BGR = [
    (0, 200, 0),    # obj 1 — green  (banner)
    (0, 100, 255),   # obj 2 — orange (player 1)
    (255, 50, 50),   # obj 3 — blue   (player 2)
    (0, 255, 255),   # obj 4
    (255, 0, 255),   # obj 5
    (255, 255, 0),   # obj 6
]

ALPHA = 0.45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="masks_overlay.mp4")
    args = ap.parse_args()

    cache = args.cache_dir
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n_frames = meta["n_frames"]
    fps = meta["fps"]
    w, h = meta["width"], meta["height"]
    all_ids = meta["banner_ids"] + meta["player_ids"]

    frames_dir = os.path.join(cache, "frames")
    masks_dir = os.path.join(cache, "player_masks")

    raw_out = args.out.rsplit(".", 1)[0] + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (w, h))

    for fi in range(n_frames):
        frame = cv2.imread(os.path.join(frames_dir, f"{fi:05d}.jpg"))
        mask_path = os.path.join(masks_dir, f"{fi:05d}.npz")
        masks = np.load(mask_path) if os.path.exists(mask_path) else {}

        vis = frame.copy()
        for oid in all_ids:
            key = f"obj{oid}"
            if key not in masks:
                continue
            m = masks[key].astype(bool)
            if not m.any():
                continue
            color = np.array(COLORS_BGR[(oid - 1) % len(COLORS_BGR)], dtype=np.uint8)
            vis[m] = (vis[m].astype(np.float32) * (1 - ALPHA)
                      + color.astype(np.float32) * ALPHA).astype(np.uint8)

            # Draw contour outline
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, tuple(int(c) for c in color), 2, cv2.LINE_AA)

        # Label in top-left
        for i, oid in enumerate(all_ids):
            kind = "banner" if oid in meta["banner_ids"] else "player"
            color = COLORS_BGR[(oid - 1) % len(COLORS_BGR)]
            y = 30 + i * 28
            cv2.putText(vis, f"obj{oid} ({kind})", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.putText(vis, f"frame {fi}", (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(vis)
        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            print(f"  {fi + 1}/{n_frames}", flush=True)

    writer.release()

    # Re-encode to H264
    subprocess.run(
        ["ffmpeg", "-i", raw_out, "-c:v", "libx264", "-preset", "fast",
         "-crf", "23", "-pix_fmt", "yuv420p", args.out, "-y", "-loglevel", "error"],
        check=True,
    )
    os.remove(raw_out)
    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
