# -*- coding: utf-8 -*-
"""
banner_segment.py
-----------------
Interactive banner segmentation + parallelogram fitting.

1. Show frame 0 of a video
2. Click on banner regions (left-click = add point, N = next object, Enter/Space = done)
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


def fit_quadrilateral(mask: np.ndarray, axis: str = "short") -> np.ndarray | None:
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

    print(f"  {labels[0]} edge: {len(group_a)} pts, {labels[1]} edge: {len(group_b)} pts")

    if len(group_a) < 5 or len(group_b) < 5:
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

    angle_a = np.degrees(np.arctan2(dir_a[1], dir_a[0]))
    angle_b = np.degrees(np.arctan2(dir_b[1], dir_b[0]))
    print(f"  {labels[0]} line angle: {angle_a:.2f}°, {labels[1]} line angle: {angle_b:.2f}°")
    print(f"  Convergence: {abs(angle_a - angle_b):.2f}°")

    # --- Debug visualisation ---
    h_img, w_img = mask.shape[:2]
    dbg = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    dbg[mask_u8 > 0] = (60, 60, 60)
    # minAreaRect (white dashed-ish)
    box_pts = cv2.boxPoints(rect).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(dbg, [box_pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    # Long / short axis arrows from center
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
    # All contour points in grey
    for p in pts.astype(int):
        cv2.circle(dbg, tuple(p), 1, (120, 120, 120), -1)
    # group_a points (cyan) and group_b points (magenta)
    for p in group_a.astype(int):
        cv2.circle(dbg, tuple(p), 3, (255, 255, 0), -1)
    for p in group_b.astype(int):
        cv2.circle(dbg, tuple(p), 3, (255, 0, 255), -1)
    # Fitted lines — extend far in both directions
    line_len = max(h_img, w_img)
    la1 = (pt_a - line_len * dir_a).astype(int)
    la2 = (pt_a + line_len * dir_a).astype(int)
    lb1 = (pt_b - line_len * dir_b).astype(int)
    lb2 = (pt_b + line_len * dir_b).astype(int)
    cv2.line(dbg, tuple(la1), tuple(la2), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(dbg, tuple(lb1), tuple(lb2), (0, 0, 255), 1, cv2.LINE_AA)
    # Extent edges (green) — the minAreaRect edges used for intersection
    e1a, e1b = edge1_pts[0].astype(int), edge1_pts[1].astype(int)
    e2a, e2b = edge2_pts[0].astype(int), edge2_pts[1].astype(int)
    cv2.line(dbg, tuple(e1a), tuple(e1b), (0, 180, 0), 1, cv2.LINE_AA)
    cv2.line(dbg, tuple(e2a), tuple(e2b), (0, 180, 0), 1, cv2.LINE_AA)
    # Corners
    for i, c in enumerate(corners):
        cv2.circle(dbg, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)
        cv2.putText(dbg, str(i), (int(c[0]) + 8, int(c[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Crop to region of interest (mask bounding box + padding)
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
# Visualisation
# ---------------------------------------------------------------------------

def composite_logo(frame: np.ndarray, corners: np.ndarray, logo_path: str,
                   mask: np.ndarray,
                   save_path: str = "sponsor_morph_result.png") -> np.ndarray:
    """Inpaint the old logo away, then warp only the opaque pixels of the
    new sponsor logo into the detected quad region."""
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

    # Debug: save pre-warp canvas + rectified original region
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

    # Match new logo luminosity to the original logo in the frame.
    # Measure the original's L range and remap the new logo's L to match,
    # plus a subtle color tint so it looks painted on the surface.
    logo_pixels = warped_alpha > 0
    if logo_pixels.any():
        orig_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        orig_mask_l = orig_lab[mask > 0, 0]
        orig_l_lo, orig_l_hi = np.percentile(orig_mask_l, [10, 90])
        bg_lab = cv2.cvtColor(np.full((1, 1, 3), bg_color, dtype=np.uint8), cv2.COLOR_BGR2LAB)
        orig_bg_l = float(bg_lab[0, 0, 0])
        print(f"  Original region L: bg={orig_bg_l:.0f}, logo range=[{orig_l_lo:.0f}, {orig_l_hi:.0f}]")

        new_lab = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
        new_l = new_lab[logo_pixels, 0]
        new_l_lo, new_l_hi = np.percentile(new_l, [10, 90])

        # Remap new logo L range → original logo L range
        if new_l_hi - new_l_lo > 1:
            scale = (orig_l_hi - orig_l_lo) / (new_l_hi - new_l_lo)
        else:
            scale = 1.0
        new_lab[logo_pixels, 0] = np.clip(
            (new_l - new_l_lo) * scale + orig_l_lo, 0, 255)
        warped_rgb = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        print(f"  Remapped L [{new_l_lo:.0f},{new_l_hi:.0f}] → [{orig_l_lo:.0f},{orig_l_hi:.0f}]")

    # Soften alpha edges for smoother blending
    warped_alpha = cv2.GaussianBlur(warped_alpha, (5, 5), 1.0)

    # Blend: only opaque logo pixels are painted onto the inpainted frame
    a = (warped_alpha.astype(np.float32) / 255.0)[..., None]
    result = (warped_rgb.astype(np.float32) * a
              + inpainted.astype(np.float32) * (1.0 - a)).astype(np.uint8)

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

def main():
    parser = argparse.ArgumentParser(description="Interactive banner segmentation + parallelogram fit")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--save", default="banner_result.png", help="Output image path")
    parser.add_argument("--checkpoint", default="sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--axis", choices=["short", "long"], default="short",
                        help="Which axis to split contour for line fitting: "
                             "'short' (default) fits top/bottom edges independently; "
                             "'long' fits left/right edges independently")
    parser.add_argument("--logo", default=None, help="Path to sponsor logo PNG to composite into the detected region")
    parser.add_argument("--mask-dir", default="masks", help="Directory to save masks + frame (default: masks/)")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
