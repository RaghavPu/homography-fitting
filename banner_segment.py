# -*- coding: utf-8 -*-
"""
banner_segment.py
-----------------
Interactive banner segmentation + parallelogram fitting.

1. Show frame 0 of a video
2. Click on banner regions (left-click = add point, Enter/Space = done)
3. SAM2 segments each clicked object
4. Fit a parallelogram to each mask
5. Visualize results

Usage
-----
    python banner_segment.py video.mp4
"""

from __future__ import annotations
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Click UI
# ---------------------------------------------------------------------------

def collect_clicks(frame: np.ndarray) -> list[tuple[int, int]]:
    """Show frame in an OpenCV window; collect left-clicks. Press Enter/Space
    to finish, Escape to cancel."""
    clicks: list[tuple[int, int]] = []
    display = frame.copy()
    win = "Click on banners (Enter=done, Esc=cancel)"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.drawMarker(display, (x, y), (0, 255, 0),
                           cv2.MARKER_STAR, 20, 2)
            cv2.putText(display, str(len(clicks)), (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win, display)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, display)

    print("[UI] Left-click on each banner. Press Enter/Space when done, Esc to cancel.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):  # Enter or Space
            break
        if key == 27:  # Escape
            clicks.clear()
            break

    cv2.destroyAllWindows()
    return clicks


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame0(video_path: str) -> np.ndarray:
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vframes", "1", "-q:v", "2",
         tmp, "-y", "-loglevel", "error"],
        check=True,
    )
    frame = cv2.imread(tmp)
    os.unlink(tmp)
    if frame is None:
        raise RuntimeError(f"Could not read frame from {video_path}")
    return frame


# ---------------------------------------------------------------------------
# SAM2 (image predictor — fast, single-frame)
# ---------------------------------------------------------------------------

def run_sam2(frame_bgr: np.ndarray, clicks: list[tuple[int, int]],
             checkpoint: str = "sam2/checkpoints/sam2.1_hiera_tiny.pt",
             model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"):
    """Run SAM2 image predictor: one positive click per object.
    Returns dict[obj_id] -> binary mask (H, W) for the given frame."""
    import sys
    # The sam2 repo dir shadows the installed package — point Python inside it
    _repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  SAM2 device: {device}", flush=True)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print("  Loading model …", flush=True)
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print("  Setting image …", flush=True)
    predictor.set_image(frame_rgb)

    masks_out: dict[int, np.ndarray] = {}
    for idx, (x, y) in enumerate(clicks):
        obj_id = idx + 1
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best = masks[np.argmax(scores)]
        masks_out[obj_id] = best
        print(f"  Click ({x},{y}) → obj {obj_id}, score={scores.max():.3f}", flush=True)

    print(f"  Got {len(masks_out)} masks", flush=True)
    return masks_out


# ---------------------------------------------------------------------------
# Parallelogram fitting (reuses court_homography)
# ---------------------------------------------------------------------------

from court_homography import get_hull_vertices, classify_vertices, find_corners


def fit_parallelogram(mask_bool: np.ndarray) -> np.ndarray | None:
    """Given a boolean mask, return 4 corners [TL, TR, BR, BL] or None."""
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    try:
        pts = get_hull_vertices(mask_u8)
        labels = classify_vertices(pts, mask_u8.shape)
        corners = find_corners(pts, labels)
        return corners
    except RuntimeError as e:
        print(f"  [warn] parallelogram fit failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

OBJ_COLORS = [
    (0, 200, 0), (200, 0, 0), (0, 0, 200),
    (200, 200, 0), (200, 0, 200), (0, 200, 200),
]


def visualize(frame: np.ndarray,
              masks: dict[int, np.ndarray],
              corners_map: dict[int, np.ndarray],
              save_path: str = "banner_result.png"):
    vis = frame.copy()

    # Overlay masks
    for obj_id, mask in masks.items():
        col = np.array(OBJ_COLORS[obj_id % len(OBJ_COLORS)], dtype=np.uint8)
        m = mask.astype(bool)
        vis[m] = (vis[m].astype(np.float32) * 0.55 + col.astype(np.float32) * 0.45).astype(np.uint8)

    # Draw parallelogram quads
    corner_labels = ["TL", "TR", "BR", "BL"]
    for obj_id, corners in corners_map.items():
        col = OBJ_COLORS[obj_id % len(OBJ_COLORS)]
        quad = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [quad], isClosed=True, color=(255, 255, 255), thickness=2)
        for pt, clbl in zip(corners, corner_labels):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(vis, (cx, cy), 6, col, -1)
            cv2.putText(vis, f"{obj_id}-{clbl}", (cx + 8, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, vis)
    print(f"  Saved: {save_path}")

    # Warped top-down views
    warpeds = []
    for obj_id in sorted(corners_map):
        c = corners_map[obj_id]
        dst_w, dst_h = 300, 450
        dst = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
        H, _ = cv2.findHomography(c, dst)
        if H is not None:
            warpeds.append((obj_id, cv2.warpPerspective(frame, H, (dst_w, dst_h))))

    ncols = 1 + len(warpeds)
    ratios = [3] + [1] * len(warpeds)
    fig, axes = plt.subplots(1, ncols, figsize=(8 + 4 * len(warpeds), 6),
                             gridspec_kw={"width_ratios": ratios})
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Masks + parallelogram fit ({len(corners_map)} objects)")
    axes[0].axis("off")

    for i, (oid, warped) in enumerate(warpeds):
        axes[1 + i].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1 + i].set_title(f"Obj {oid} top-down")
        axes[1 + i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path.rsplit(".", 1)[0] + "_full.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive banner segmentation + parallelogram fit")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--save", default="banner_result.png", help="Output image path")
    parser.add_argument("--checkpoint", default="sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--mask-dir", default="masks", help="Directory to save masks + frame (default: masks/)")
    args = parser.parse_args()

    print("[1/4] Extracting frame 0 …", flush=True)
    frame = extract_frame0(args.video)

    print("[2/4] Collecting clicks …", flush=True)
    clicks = collect_clicks(frame)
    if not clicks:
        print("No clicks — exiting.")
        return
    print(f"  {len(clicks)} click(s): {clicks}", flush=True)

    print("[3/4] Running SAM2 (image mode) …", flush=True)
    masks = run_sam2(frame, clicks,
                     checkpoint=args.checkpoint, model_cfg=args.model_cfg)

    # Save masks + original frame for offline experimentation
    mask_dir = args.mask_dir
    os.makedirs(mask_dir, exist_ok=True)
    frame_path = os.path.join(mask_dir, "frame0.png")
    cv2.imwrite(frame_path, frame)
    print(f"  Saved frame: {frame_path}", flush=True)
    for obj_id, mask in masks.items():
        mask_path = os.path.join(mask_dir, f"mask_obj{obj_id}.png")
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
        print(f"  Saved mask: {mask_path}", flush=True)

    print("[4/4] Fitting parallelograms …", flush=True)
    corners_map = {}
    for obj_id, mask in masks.items():
        print(f"  Object {obj_id}:")
        corners = fit_parallelogram(mask)
        if corners is not None:
            corners_map[obj_id] = corners
            for lbl, pt in zip(["TL", "TR", "BR", "BL"], corners):
                print(f"    {lbl}: ({int(pt[0])}, {int(pt[1])})")

    print(f"  Visualizing ({len(corners_map)} parallelograms) …", flush=True)
    visualize(frame, masks, corners_map, save_path=args.save)
    print("Done.")


if __name__ == "__main__":
    main()
