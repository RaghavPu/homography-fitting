# -*- coding: utf-8 -*-
"""
banner_segment.py
-----------------
Interactive banner segmentation + parallelogram fitting + video output.

Single-frame mode (default):
    python banner_segment.py video.mp4

Video mode (with logo replacement):
    python banner_segment.py video.mp4 --logo sponsor.png --video-out output.mp4

Workflow:
1. Show frame 0 of a video
2. Click on banner regions (left-click = add point, N = next object, Enter/Space = done)
3. Optionally click on players to track for occlusion masking (P = start player clicks)
4. SAM2 video predictor propagates masks across all frames
5. Fit a parallelogram to each banner mask per frame (with EMA stabilisation)
6. Composite sponsor logo with player occlusion
7. Write output video
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Click UI
# ---------------------------------------------------------------------------

OBJ_COLORS_UI = [
    (0, 255, 0), (0, 100, 255), (0, 0, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0),
]


def collect_clicks(frame: np.ndarray) -> list[list[tuple[int, int]]]:
    """Show frame in an OpenCV window; collect grouped clicks.

    - Left-click : add point to current object
    - N          : finish current object, start a new one
    - Enter/Space: finish all
    - Escape     : cancel

    Returns a list of groups, e.g. [[(x1,y1),(x2,y2)], [(x3,y3)], ...]
    """
    groups: list[list[tuple[int, int]]] = [[]]
    display = frame.copy()
    win = "Click banners (N=next object, Enter=done, Esc=cancel)"

    def current_color():
        return OBJ_COLORS_UI[(len(groups) - 1) % len(OBJ_COLORS_UI)]

    def redraw_status():
        obj_idx = len(groups)
        n_pts = len(groups[-1])
        label = f"Object {obj_idx}  ({n_pts} pts)  |  N=next  Enter=done"
        cv2.rectangle(display, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
        cv2.putText(display, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, display)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            groups[-1].append((x, y))
            col = current_color()
            pt_idx = len(groups[-1])
            cv2.drawMarker(display, (x, y), col, cv2.MARKER_STAR, 20, 2)
            cv2.putText(display, f"{len(groups)}.{pt_idx}", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
            redraw_status()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, on_mouse)
    redraw_status()

    print("[UI] Left-click to add points. N = next object. Enter/Space = done. Esc = cancel.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):  # Enter or Space
            break
        if key == 27:  # Escape
            groups.clear()
            break
        if key in (ord('n'), ord('N')):
            if groups[-1]:  # only advance if current group has points
                print(f"  Object {len(groups)} done ({len(groups[-1])} pts). Starting object {len(groups) + 1}…")
                groups.append([])
                redraw_status()

    cv2.destroyAllWindows()
    # Drop trailing empty group (if user pressed N then immediately Enter)
    return [g for g in groups if g]


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


def extract_frames(video_path: str, out_dir: str | None = None) -> str:
    """Extract all frames from *video_path* as numbered JPEGs.
    Returns the directory path containing the frames."""
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="banner_frames_")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-q:v", "2", "-start_number", "0",
         os.path.join(out_dir, "%05d.jpg"), "-y", "-loglevel", "error"],
        check=True,
    )
    n = len(glob.glob(os.path.join(out_dir, "*.jpg")))
    print(f"  Extracted {n} frames to {out_dir}")
    return out_dir


def get_video_fps(video_path: str, default: float = 30.0) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1 or np.isnan(fps):
        return default
    return float(fps)


def reencode_to_h264(src: str, dst: str | None = None) -> str:
    """Re-encode an mp4v file to H264 for browser/player compatibility."""
    if dst is None:
        dst = src.rsplit(".", 1)[0] + "_h264.mp4"
    subprocess.run(
        ["ffmpeg", "-i", src, "-c:v", "libx264", "-preset", "fast",
         "-crf", "23", "-pix_fmt", "yuv420p", dst, "-y", "-loglevel", "error"],
        check=True,
    )
    return dst


# ---------------------------------------------------------------------------
# SAM2 (image predictor — fast, single-frame)
# ---------------------------------------------------------------------------

def run_sam2(frame_bgr: np.ndarray, click_groups: list[list[tuple[int, int]]],
             checkpoint: str = "sam2/checkpoints/sam2.1_hiera_tiny.pt",
             model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"):
    """Run SAM2 image predictor: one group of positive clicks per object.
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
    for idx, group in enumerate(click_groups):
        obj_id = idx + 1
        point_coords = np.array(group, dtype=np.float32)
        point_labels = np.ones(len(group), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best = masks[np.argmax(scores)]
        masks_out[obj_id] = best
        pts_str = ", ".join(f"({x},{y})" for x, y in group)
        print(f"  Obj {obj_id}: {len(group)} pts [{pts_str}] → score={scores.max():.3f}", flush=True)

    print(f"  Got {len(masks_out)} masks", flush=True)
    return masks_out


# ---------------------------------------------------------------------------
# SAM2 (video predictor — propagates masks across frames)
# ---------------------------------------------------------------------------

def _sam2_device_and_model(checkpoint: str, model_cfg: str, *, video: bool = False):
    """Shared setup: device selection + model build for SAM2."""
    import sys
    _repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
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
    print(f"  SAM2 device: {device}", flush=True)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    if video:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    else:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(model_cfg, checkpoint, device=device)
        predictor = SAM2ImagePredictor(model)

    return device, predictor


def run_sam2_video_players(
    frames_dir: str,
    player_clicks: list[list[tuple[int, int]]],
    checkpoint: str = "sam2/checkpoints/sam2.1_hiera_tiny.pt",
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
) -> tuple[dict[int, dict[int, np.ndarray]], list[int]]:
    """Run SAM2 video predictor to propagate **player** masks across all frames.

    Returns (video_segments, player_obj_ids) where
    video_segments[frame_idx][obj_id] = bool mask (H, W).
    """
    import torch

    _, predictor = _sam2_device_and_model(checkpoint, model_cfg, video=True)

    print("  Initialising video state …", flush=True)
    inference_state = predictor.init_state(video_path=frames_dir)

    player_obj_ids: list[int] = []
    for idx, group in enumerate(player_clicks):
        obj_id = idx + 1
        player_obj_ids.append(obj_id)
        points = np.array(group, dtype=np.float32)
        labels = np.ones(len(group), dtype=np.int32)
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        score_approx = float((out_mask_logits[0] > 0.0).float().mean())
        print(f"  Player obj {obj_id}: {len(group)} pts, mask coverage={score_approx:.4f}", flush=True)

    print("  Propagating player masks …", flush=True)
    video_segments: dict[int, dict[int, np.ndarray]] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            int(out_obj_id): (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    n_frames = len(video_segments)
    print(f"  Propagated player masks to {n_frames} frames "
          f"({len(player_obj_ids)} players)", flush=True)
    return video_segments, player_obj_ids


# ---------------------------------------------------------------------------
# Perspective-aware quadrilateral fitting
# ---------------------------------------------------------------------------

def _fit_line_pts(pts: np.ndarray, weights: np.ndarray | None = None):
    """Fit a line to 2D points. Returns (point_on_line, unit_direction).

    If *weights* is given, uses weighted PCA instead of cv2.fitLine.
    """
    if weights is None:
        vx, vy, cx, cy = cv2.fitLine(pts.reshape(-1, 1, 2).astype(np.float32),
                                      cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return np.array([cx, cy], dtype=np.float64), np.array([vx, vy], dtype=np.float64)

    w = weights / weights.sum()
    centroid = (pts * w[:, None]).sum(axis=0)
    centered = pts - centroid
    cov = (centered * w[:, None]).T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, -1].astype(np.float64)
    return centroid.astype(np.float64), direction


def _line_intersect(p1, d1, p2, d2):
    """Intersect lines p1+t*d1 and p2+s*d2. Returns the intersection point."""
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return (p1 + p2) / 2.0
    dp = p2 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
    return p1 + t * d1


def fit_quadrilateral(mask: np.ndarray, axis: str = "short",
                      debug: bool = True, verbose: bool = True) -> np.ndarray | None:
    """Fit a perspective-aware quadrilateral to a binary mask.

    *axis* controls which pair of edges gets independent line fitting:
      - "short" (default): split along short axis → top/bottom edges can converge
      - "long": split along long axis → left/right edges can converge

    Returns [TL, TR, BR, BL] or None on failure.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    # Keep only the largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    if n_labels > 2:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_u8 = ((labels == largest_label) * 255).astype(np.uint8)
        if verbose:
            print(f"  Kept largest component ({stats[largest_label, cv2.CC_STAT_AREA]}px), "
                  f"dropped {n_labels - 2} small blob(s)")

    # Smooth edges
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kern)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kern)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(np.float64)

    rect = cv2.minAreaRect(largest)
    center = np.array(rect[0])
    angle_deg = rect[2]
    w, h = rect[1]
    if w < h:
        angle_deg += 90

    angle_rad = np.deg2rad(angle_deg)
    long_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    short_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

    rel = pts - center

    if axis == "long":
        proj_long = rel @ long_dir
        proj_short = rel @ short_dir
        short_bins = np.round(proj_short).astype(int)
        unique_bins = np.unique(short_bins)
        # Trim top/bottom 15% of rows to avoid corner-rounding points
        n_trim = max(1, int(len(unique_bins) * 0.15))
        unique_bins = unique_bins[n_trim:-n_trim]
        left_idx, right_idx = [], []
        for b in unique_bins:
            in_bin = np.where(short_bins == b)[0]
            left_idx.append(in_bin[np.argmin(proj_long[in_bin])])
            right_idx.append(in_bin[np.argmax(proj_long[in_bin])])
        group_a = pts[left_idx]
        group_b = pts[right_idx]
        labels = ("Left", "Right")

        # Hann-window weights: middle of edge → 1, top/bottom → 0
        def _hann_weights(grp):
            s = (grp - center) @ short_dir
            t = (s - s.min()) / max(s.max() - s.min(), 1e-9)
            return np.clip(0.5 * (1 - np.cos(2 * np.pi * t)), 0.01, 1.0)

        weights_a = _hann_weights(group_a)
        weights_b = _hann_weights(group_b)
    else:
        proj_short = rel @ short_dir
        group_a = pts[proj_short >= 0]
        group_b = pts[proj_short < 0]
        labels = ("Top", "Bottom")

        # Hann weights along the long axis to downweight left/right corners
        def _hann_weights_long(grp):
            s = (grp - center) @ long_dir
            t = (s - s.min()) / max(s.max() - s.min(), 1e-9)
            return np.clip(0.5 * (1 - np.cos(2 * np.pi * t)), 0.01, 1.0)

        weights_a = _hann_weights_long(group_a)
        weights_b = _hann_weights_long(group_b)

    if verbose:
        print(f"  {labels[0]} edge: {len(group_a)} pts, {labels[1]} edge: {len(group_b)} pts")

    if len(group_a) < 5 or len(group_b) < 5:
        if verbose:
            print("  [warn] Not enough points, falling back to minAreaRect")
        return cv2.boxPoints(rect).astype(np.float32)

    pt_a, dir_a = _fit_line_pts(group_a, weights_a)
    pt_b, dir_b = _fit_line_pts(group_b, weights_b)

    if dir_a @ dir_b < 0:
        dir_b = -dir_b

    # Find the two extent edges of the minAreaRect perpendicular to the
    # fitted lines, then intersect to get corners.
    box_corners = cv2.boxPoints(rect).astype(np.float64)
    if axis == "long":
        # Fitted lines ≈ short dir; extent edges ≈ long dir (top/bottom of rect)
        extent_dir = short_dir
    else:
        # Fitted lines ≈ long dir; extent edges ≈ short dir (left/right of rect)
        extent_dir = long_dir

    proj_box = (box_corners - center) @ extent_dir
    order = np.argsort(proj_box)
    edge1_pts = box_corners[order[:2]]  # low-projection edge
    edge2_pts = box_corners[order[2:]]  # high-projection edge
    edge1_dir = edge1_pts[1] - edge1_pts[0]
    edge1_dir /= np.linalg.norm(edge1_dir)
    edge2_dir = edge2_pts[1] - edge2_pts[0]
    edge2_dir /= np.linalg.norm(edge2_dir)

    c_a1 = _line_intersect(pt_a, dir_a, edge1_pts[0], edge1_dir)
    c_a2 = _line_intersect(pt_a, dir_a, edge2_pts[0], edge2_dir)
    c_b1 = _line_intersect(pt_b, dir_b, edge1_pts[0], edge1_dir)
    c_b2 = _line_intersect(pt_b, dir_b, edge2_pts[0], edge2_dir)

    corners = np.array([c_a1, c_a2, c_b1, c_b2], dtype=np.float32)

    if verbose:
        angle_a = np.degrees(np.arctan2(dir_a[1], dir_a[0]))
        angle_b = np.degrees(np.arctan2(dir_b[1], dir_b[0]))
        print(f"  {labels[0]} line angle: {angle_a:.2f}°, {labels[1]} line angle: {angle_b:.2f}°")
        print(f"  Convergence: {abs(angle_a - angle_b):.2f}°")

    if debug:
        h_img, w_img = mask.shape[:2]
        dbg = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        dbg[mask_u8 > 0] = (60, 60, 60)
        box_pts = cv2.boxPoints(rect).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(dbg, [box_pts], True, (255, 255, 255), 1, cv2.LINE_AA)
        arrow_len = 60
        ct = tuple(center.astype(int))
        cv2.arrowedLine(dbg, ct, tuple((center + arrow_len * long_dir).astype(int)),
                        (0, 200, 200), 1, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(dbg, "long", tuple((center + arrow_len * long_dir + 5).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
        cv2.arrowedLine(dbg, ct, tuple((center + arrow_len * short_dir).astype(int)),
                        (200, 200, 0), 1, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(dbg, "short", tuple((center + arrow_len * short_dir + 5).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)
        for p in pts.astype(int):
            cv2.circle(dbg, tuple(p), 1, (120, 120, 120), -1)
        for p in group_a.astype(int):
            cv2.circle(dbg, tuple(p), 3, (255, 255, 0), -1)
        for p in group_b.astype(int):
            cv2.circle(dbg, tuple(p), 3, (255, 0, 255), -1)
        line_len = max(h_img, w_img)
        la1 = (pt_a - line_len * dir_a).astype(int)
        la2 = (pt_a + line_len * dir_a).astype(int)
        lb1 = (pt_b - line_len * dir_b).astype(int)
        lb2 = (pt_b + line_len * dir_b).astype(int)
        cv2.line(dbg, tuple(la1), tuple(la2), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dbg, tuple(lb1), tuple(lb2), (0, 0, 255), 1, cv2.LINE_AA)
        e1a, e1b = edge1_pts[0].astype(int), edge1_pts[1].astype(int)
        e2a, e2b = edge2_pts[0].astype(int), edge2_pts[1].astype(int)
        cv2.line(dbg, tuple(e1a), tuple(e1b), (0, 180, 0), 1, cv2.LINE_AA)
        cv2.line(dbg, tuple(e2a), tuple(e2b), (0, 180, 0), 1, cv2.LINE_AA)
        for i, c in enumerate(corners):
            cv2.circle(dbg, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)
            cv2.putText(dbg, str(i), (int(c[0]) + 8, int(c[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) > 0:
            pad = 60
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w_img)
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h_img)
            dbg = dbg[y0:y1, x0:x1]
        cv2.imwrite("debug_fit.png", dbg)
        print("  Debug image saved: debug_fit.png")

    # Sort into TL/TR/BR/BL
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    return np.array([
        corners[np.argmin(s)],
        corners[np.argmax(d)],
        corners[np.argmax(s)],
        corners[np.argmin(d)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Corner stabilisation (EMA)
# ---------------------------------------------------------------------------

class CornerTracker:
    """Track quad corners across frames using Lucas-Kanade optical flow.

    SAM2 segments each banner only on frame 0; subsequent frames use sparse
    optical flow to propagate the 4 corner points, which is both faster and
    more temporally stable than re-segmenting every frame.
    """

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self._corners: dict[int, np.ndarray] = {}   # obj_id → (4, 2) float32
        self._smoothed: dict[int, np.ndarray] = {}
        self._prev_gray: np.ndarray | None = None

    def init(self, obj_id: int, corners: np.ndarray, frame_gray: np.ndarray):
        """Seed an object's corners from the frame-0 quad fit."""
        c = corners.astype(np.float32).reshape(4, 2)
        self._corners[obj_id] = c
        self._smoothed[obj_id] = c.copy()
        self._prev_gray = frame_gray

    def update(self, frame_gray: np.ndarray) -> dict[int, np.ndarray]:
        """Track all objects to a new frame. Returns {obj_id: corners}."""
        if self._prev_gray is None or not self._corners:
            self._prev_gray = frame_gray
            return {oid: c.copy() for oid, c in self._smoothed.items()}

        all_pts = []
        obj_ids = []
        for oid, c in self._corners.items():
            all_pts.append(c)
            obj_ids.append(oid)

        pts_old = np.vstack(all_pts).reshape(-1, 1, 2)
        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, frame_gray, pts_old, None, **self.LK_PARAMS
        )

        # Reverse check for robustness
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            frame_gray, self._prev_gray, pts_new, None, **self.LK_PARAMS
        )
        fb_dist = np.linalg.norm(
            (pts_old - pts_back).reshape(-1, 2), axis=1
        )
        good = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_dist < 2.0)

        idx = 0
        result: dict[int, np.ndarray] = {}
        for oid in obj_ids:
            new_c = np.empty((4, 2), dtype=np.float32)
            for j in range(4):
                if good[idx + j]:
                    new_c[j] = pts_new[idx + j].ravel()
                else:
                    new_c[j] = self._corners[oid][j]
            self._corners[oid] = new_c
            # EMA smooth
            sm = self.ema_alpha * new_c + (1 - self.ema_alpha) * self._smoothed[oid]
            self._smoothed[oid] = sm
            result[oid] = sm.copy()
            idx += 4

        self._prev_gray = frame_gray
        return result


# ---------------------------------------------------------------------------
# Visualisation / compositing
# ---------------------------------------------------------------------------

def composite_logo(frame: np.ndarray, corners: np.ndarray, logo_path: str,
                   mask: np.ndarray,
                   occlusion_mask: np.ndarray | None = None,
                   save_path: str | None = "sponsor_morph_result.png",
                   debug: bool = True,
                   verbose: bool = True,
                   lum_strength: float = 1.0,
                   ref_lum: tuple[float, float] | None = None) -> np.ndarray:
    """Inpaint the old logo away, then warp only the opaque pixels of the
    new sponsor logo into the detected quad region.

    *occlusion_mask*: if provided, pixels where this mask is True are excluded
    from the logo overlay (e.g. a player walking in front of the banner).
    """
    logo_bgra = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo_bgra is None:
        raise RuntimeError(f"Could not read logo: {logo_path}")
    if logo_bgra.shape[2] == 3:
        logo_bgra = cv2.cvtColor(logo_bgra, cv2.COLOR_BGR2BGRA)

    h, w = frame.shape[:2]

    # Step 1: Inpaint the masked region to erase the old logo
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    dilate_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.dilate(mask_u8, dilate_kern)
    inpainted = cv2.inpaint(frame, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    if verbose:
        print("  Inpainted old logo region")

    # Sample bg color for luminosity matching
    border_mask = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) & ~mask_u8
    bg_color = np.median(frame[border_mask > 0], axis=0).astype(np.uint8)

    # Step 2: Build logo + alpha canvases. Use average screen-space edge
    # lengths (best available estimate without world-space info).
    w_top = np.linalg.norm(corners[1] - corners[0])
    w_bot = np.linalg.norm(corners[2] - corners[3])
    h_left = np.linalg.norm(corners[3] - corners[0])
    h_right = np.linalg.norm(corners[2] - corners[1])
    avg_w = (w_top + w_bot) / 2
    avg_h = (h_left + h_right) / 2
    # Scale up to a minimum resolution for quality
    scale_up = max(1.0, 500 / max(avg_w, avg_h))
    canvas_w = max(int(avg_w * scale_up), 1)
    canvas_h = max(int(avg_h * scale_up), 1)
    if verbose:
        print(f"  Canvas {canvas_w}x{canvas_h} (aspect {avg_w/max(avg_h,1):.2f})")

    rgb_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    logo_h, logo_w = logo_bgra.shape[:2]
    pad_frac = 0.05
    pad_w = int(canvas_w * pad_frac)
    pad_h = int(canvas_h * pad_frac)
    scale = min((canvas_w - 2 * pad_w) / logo_w, (canvas_h - 2 * pad_h) / logo_h)
    new_w, new_h = int(logo_w * scale), int(logo_h * scale)
    logo_resized = cv2.resize(logo_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2
    rgb_canvas[y0:y0 + new_h, x0:x0 + new_w] = logo_resized[:, :, :3]
    alpha_canvas[y0:y0 + new_h, x0:x0 + new_w] = logo_resized[:, :, 3]

    if debug:
        dbg_canvas = rgb_canvas.copy()
        dbg_canvas[alpha_canvas == 0] = (40, 40, 40)
        cv2.imwrite("debug_canvas.png", dbg_canvas)
        src_rect = np.array([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]], dtype=np.float32)
        H_rect, _ = cv2.findHomography(corners.astype(np.float32), src_rect)
        if H_rect is not None:
            rectified = cv2.warpPerspective(frame, H_rect, (canvas_w, canvas_h))
            cv2.imwrite("debug_rectified.png", rectified)
            print(f"  Debug rectified original saved: debug_rectified.png")
        print(f"  Debug canvas saved: debug_canvas.png ({canvas_w}x{canvas_h}, logo {new_w}x{new_h})")

    # Warp both canvases into frame space
    src = np.array([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]], dtype=np.float32)
    H, _ = cv2.findHomography(src, corners.astype(np.float32))
    warped_rgb = cv2.warpPerspective(rgb_canvas, H, (w, h))
    warped_alpha = cv2.warpPerspective(alpha_canvas, H, (w, h))

    # Match new logo luminosity to the original banner region.
    # When ref_lum is provided, use those fixed stats (from frame 0) for
    # temporal consistency; otherwise compute from this frame.
    logo_pixels = warped_alpha > 0
    if logo_pixels.any():
        if ref_lum is not None:
            orig_l_lo, orig_l_hi = ref_lum
        else:
            orig_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            orig_mask_l = orig_lab[mask > 0, 0]
            orig_l_lo, orig_l_hi = np.percentile(orig_mask_l, [10, 90])

        new_lab = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
        new_l = new_lab[logo_pixels, 0]
        new_l_lo, new_l_hi = np.percentile(new_l, [10, 90])

        if new_l_hi - new_l_lo > 1:
            scale = (orig_l_hi - orig_l_lo) / (new_l_hi - new_l_lo)
        else:
            scale = 1.0
        remapped_l = np.clip((new_l - new_l_lo) * scale + orig_l_lo, 0, 255)
        new_lab[logo_pixels, 0] = new_l * (1 - lum_strength) + remapped_l * lum_strength
        warped_rgb = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        if verbose:
            print(f"  Remapped L [{new_l_lo:.0f},{new_l_hi:.0f}] → [{orig_l_lo:.0f},{orig_l_hi:.0f}]")

    # Soften alpha edges for smoother blending
    warped_alpha = cv2.GaussianBlur(warped_alpha, (5, 5), 1.0)

    # Subtract player/occlusion mask so players appear on top of the logo
    if occlusion_mask is not None:
        occ = (np.squeeze(occlusion_mask) > 0).astype(np.uint8) * 255
        occ_dilated = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        occ_soft = cv2.GaussianBlur(occ_dilated, (11, 11), 3.0)
        warped_alpha = np.clip(
            warped_alpha.astype(np.float32) - occ_soft.astype(np.float32), 0, 255
        ).astype(np.uint8)

    # Blend: only opaque logo pixels are painted onto the inpainted frame
    a = (warped_alpha.astype(np.float32) / 255.0)[..., None]
    result = (warped_rgb.astype(np.float32) * a
              + inpainted.astype(np.float32) * (1.0 - a)).astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, result)
        print(f"  Saved composited result: {save_path}")
    return result


OBJ_COLORS = [
    (0, 200, 0), (200, 0, 0), (0, 0, 200),
    (200, 200, 0), (200, 0, 200), (0, 200, 200),
]


def visualize(frame: np.ndarray,
              masks: dict[int, np.ndarray],
              corners_map: dict[int, np.ndarray],
              composited: np.ndarray | None = None,
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

    ncols = 2 if composited is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Masks + quad fit ({len(corners_map)} objects)")
    axes[0].axis("off")

    if composited is not None:
        axes[1].imshow(cv2.cvtColor(composited, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Composited logo")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path.rsplit(".", 1)[0] + "_full.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _collect_player_clicks(frame: np.ndarray) -> list[list[tuple[int, int]]]:
    """Second click phase: collect clicks on players for occlusion masking.

    Same UI as collect_clicks but with a different window title / colour.
    Returns list of groups (one per player). Empty list = skip player tracking.
    """
    groups: list[list[tuple[int, int]]] = [[]]
    display = frame.copy()
    win = "Click PLAYERS for occlusion (N=next, Enter=done, Esc=skip)"

    player_color = (0, 0, 255)  # red

    def redraw_status():
        n_players = len(groups)
        n_pts = len(groups[-1])
        label = f"Player {n_players}  ({n_pts} pts)  |  N=next  Enter=done  Esc=skip"
        cv2.rectangle(display, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
        cv2.putText(display, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, display)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            groups[-1].append((x, y))
            pt_idx = len(groups[-1])
            cv2.drawMarker(display, (x, y), player_color, cv2.MARKER_STAR, 20, 2)
            cv2.putText(display, f"P{len(groups)}.{pt_idx}", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, player_color, 2, cv2.LINE_AA)
            redraw_status()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, on_mouse)
    redraw_status()

    print("[UI] Click on PLAYERS that may occlude banners. N = next player. "
          "Enter = done. Esc = skip player tracking.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):  # Enter / Space
            break
        if key == 27:  # Escape = skip
            groups.clear()
            break
        if key in (ord('n'), ord('N')):
            if groups[-1]:
                print(f"  Player {len(groups)} done ({len(groups[-1])} pts). "
                      f"Starting player {len(groups) + 1}…")
                groups.append([])
                redraw_status()

    cv2.destroyAllWindows()
    return [g for g in groups if g]


# ---------------------------------------------------------------------------
# Single-frame mode
# ---------------------------------------------------------------------------

def _run_single_frame(args):
    """Original single-frame workflow."""
    print("[1/4] Extracting frame 0 …", flush=True)
    frame = extract_frame0(args.video)

    print("[2/4] Collecting clicks …", flush=True)
    clicks = collect_clicks(frame)
    if not clicks:
        print("No clicks — exiting.")
        return
    print(f"  {len(clicks)} object(s): {clicks}", flush=True)

    print("[3/4] Running SAM2 (image mode) …", flush=True)
    masks = run_sam2(frame, clicks,
                     checkpoint=args.checkpoint, model_cfg=args.model_cfg)

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
        corners = fit_quadrilateral(mask, axis=args.axis)
        if corners is not None:
            corners_map[obj_id] = corners
            for lbl, pt in zip(["TL", "TR", "BR", "BL"], corners):
                print(f"    {lbl}: ({int(pt[0])}, {int(pt[1])})")

    composited = None
    if args.logo and corners_map:
        print("[5/5] Compositing logo …", flush=True)
        composited = frame
        for obj_id in sorted(corners_map):
            print(f"  Object {obj_id}:")
            composited = composite_logo(composited, corners_map[obj_id], args.logo,
                                        mask=masks[obj_id],
                                        save_path="sponsor_morph_result.png")

    print(f"  Visualizing ({len(corners_map)} parallelograms) …", flush=True)
    visualize(frame, masks, corners_map, composited=composited, save_path=args.save)
    print("Done.")


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------

def _run_video(args):
    """Full video pipeline:
    - Banners: SAM2 image predictor on frame 0 only → quad fit → optical flow
      tracking of the 4 corners across all subsequent frames.
    - Players: SAM2 video predictor propagates masks every frame (for occlusion).
    """
    video_out = args.video_out
    logo_path = args.logo
    if not logo_path:
        print("ERROR: --logo is required for video mode.")
        return

    # --- 1. Extract frames ---
    print("[1/7] Extracting frames …", flush=True)
    frames_dir = extract_frames(args.video)
    frame_names = sorted(
        [p for p in os.listdir(frames_dir) if p.lower().endswith((".jpg", ".jpeg"))],
        key=lambda p: int(os.path.splitext(p)[0]),
    )
    n_frames = len(frame_names)
    fps = get_video_fps(args.video)
    frame0 = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    h, w = frame0.shape[:2]
    print(f"  {n_frames} frames, {w}x{h} @ {fps:.1f} fps")

    # --- 2. Collect clicks (banners + optional players) ---
    print("[2/7] Collecting banner clicks …", flush=True)
    banner_clicks = collect_clicks(frame0)
    if not banner_clicks:
        print("No banner clicks — exiting.")
        shutil.rmtree(frames_dir, ignore_errors=True)
        return
    print(f"  {len(banner_clicks)} banner(s): {banner_clicks}")

    print("[2b/7] Collecting player clicks (for occlusion) …", flush=True)
    player_clicks = _collect_player_clicks(frame0)
    if player_clicks:
        print(f"  {len(player_clicks)} player(s): {player_clicks}")
    else:
        print("  No player clicks — skipping occlusion masking.")

    # --- 3. SAM2 on frame 0 for banners (image predictor, one-shot) ---
    print("[3/7] Running SAM2 on frame 0 for banners …", flush=True)
    banner_masks = run_sam2(frame0, banner_clicks,
                            checkpoint=args.checkpoint, model_cfg=args.model_cfg)

    # Fit quads on frame 0 and seed optical flow tracker
    print("[4/7] Fitting quads on frame 0 + seeding optical flow tracker …", flush=True)
    tracker = CornerTracker(ema_alpha=args.ema_alpha)
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    banner_f0_masks: dict[int, np.ndarray] = {}
    for obj_id, mask in banner_masks.items():
        corners = fit_quadrilateral(mask, axis=args.axis)
        if corners is not None:
            tracker.init(obj_id, corners, gray0)
            banner_f0_masks[obj_id] = mask
            for lbl, pt in zip(["TL", "TR", "BR", "BL"], corners):
                print(f"    Banner {obj_id} {lbl}: ({int(pt[0])}, {int(pt[1])})")

    if not tracker._corners:
        print("No valid banner quads — exiting.")
        shutil.rmtree(frames_dir, ignore_errors=True)
        return

    # --- 4. SAM2 video propagation for players only ---
    player_segments: dict[int, dict[int, np.ndarray]] = {}
    player_ids: list[int] = []
    if player_clicks:
        print("[5/7] Running SAM2 video propagation for players …", flush=True)
        player_segments, player_ids = run_sam2_video_players(
            frames_dir, player_clicks,
            checkpoint=args.checkpoint, model_cfg=args.model_cfg,
        )
    else:
        print("[5/7] Skipping player propagation (no player clicks).")

    # --- 5. Process each frame ---
    print("[6/7] Processing frames (optical flow + composite) …", flush=True)
    raw_out = video_out.rsplit(".", 1)[0] + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (w, h))

    for fi, fname in enumerate(frame_names):
        frame = cv2.imread(os.path.join(frames_dir, fname))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track banner corners via optical flow
        if fi == 0:
            tracked = {oid: tracker._smoothed[oid].copy()
                       for oid in tracker._corners}
        else:
            tracked = tracker.update(gray)

        # Merge all player masks into one occlusion mask
        occ_mask = np.zeros((h, w), dtype=bool)
        if player_ids:
            seg = player_segments.get(fi, {})
            for pid in player_ids:
                pm = seg.get(pid)
                if pm is not None:
                    occ_mask |= np.squeeze(pm).astype(bool)

        # Composite logo at tracked corners for each banner
        result = frame
        for obj_id, corners in tracked.items():
            result = composite_logo(
                result, corners, logo_path,
                mask=banner_f0_masks[obj_id],
                occlusion_mask=occ_mask if occ_mask.any() else None,
                save_path=None,
                debug=False,
                verbose=False,
            )

        writer.write(result)
        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            print(f"  Frame {fi + 1}/{n_frames}", flush=True)

    writer.release()
    print(f"  Raw video: {raw_out}")

    # --- 6. Re-encode to H264 ---
    print("[7/7] Re-encoding to H264 …", flush=True)
    final_path = reencode_to_h264(raw_out, video_out)
    os.remove(raw_out)
    print(f"  Final video: {final_path}")

    # Cleanup temp frames
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"Done. Output: {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive banner segmentation + logo replacement (single-frame or video)")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--save", default="banner_result.png",
                        help="Output image path (single-frame mode)")
    parser.add_argument("--checkpoint", default="sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--axis", choices=["short", "long"], default="short",
                        help="Which axis to split contour for line fitting")
    parser.add_argument("--logo", default=None,
                        help="Path to sponsor logo PNG to composite")
    parser.add_argument("--mask-dir", default="masks",
                        help="Directory to save masks + frame (default: masks/)")
    parser.add_argument("--video-out", default=None,
                        help="Output video path. If set, runs full video pipeline "
                             "instead of single-frame mode.")
    parser.add_argument("--ema-alpha", type=float, default=0.3,
                        help="EMA smoothing factor for corner stabilisation (0=static, 1=none)")
    args = parser.parse_args()

    if args.video_out:
        _run_video(args)
    else:
        _run_single_frame(args)


if __name__ == "__main__":
    main()
