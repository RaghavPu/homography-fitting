# -*- coding: utf-8 -*-
"""
region_overlay.py
-----------------
Select any region in an image/video frame, optionally segment it with SAM2,
compute the oriented homography using the camera focal point, and overlay an image.

Key idea
--------
A standard findHomography gives a 2-D projective warp, but ignores the camera
focal point.  Here we also decompose the homography with the camera matrix K to:
  * recover the 3-D plane orientation (R, t, normal)
  * measure the TRUE physical aspect ratio of the selected region
  * warp the overlay to that exact aspect ratio before compositing

Without K-aware rectification the overlay gets squashed or stretched whenever
the banner is viewed at an oblique angle.

Selection modes
---------------
  poly  - click 4+ corners manually (right-click or Enter to close)
  sam2  - click seed points; SAM2 segments the object, quad is fitted to mask

Usage
-----
    python region_overlay.py video.mp4
    python region_overlay.py video.mp4 --logo sponsor_logo.png
    python region_overlay.py image.jpg --logo logo.png --mode poly
    python region_overlay.py video.mp4 --logo logo.png --save out.png
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def estimate_camera_matrix(frame_shape: tuple, focal_length: float | None = None) -> np.ndarray:
    """Build K = [[f,0,cx],[0,f,cy],[0,0,1]].

    If focal_length is None, default to max(w, h) — a reasonable estimate for
    a standard lens (~53° diagonal FoV in the larger dimension).
    """
    h, w = frame_shape[:2]
    f = focal_length if focal_length is not None else float(max(h, w))
    cx, cy = w / 2.0, h / 2.0
    return np.array([[f, 0.0, cx],
                     [0.0, f, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Region selection — polygon mode
# ---------------------------------------------------------------------------

def select_polygon(frame: np.ndarray) -> np.ndarray | None:
    """Interactive polygon selection.

    Left-click to add a vertex.
    Right-click or press Enter/Space to close and accept.
    Press Escape to cancel.

    Returns an (N, 2) float32 array of clicked points, or None.
    """
    pts: list[tuple[int, int]] = []
    display = frame.copy()
    win = "Click corners (right-click/Enter=done, Esc=cancel)"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.drawMarker(display, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            if len(pts) > 1:
                cv2.line(display, pts[-2], pts[-1], (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(display, str(len(pts)), (x + 8, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(win, display)
        elif event == cv2.EVENT_RBUTTONDOWN:
            _close()

    def _close():
        if len(pts) >= 3:
            cv2.line(display, pts[-1], pts[0], (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, display)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, display)

    print("[UI] Left-click to add polygon vertices.")
    print("     Right-click or Enter/Space to finish | Esc to cancel")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):  # Enter / Space
            break
        if key == 27:  # Escape
            pts.clear()
            break

    cv2.destroyAllWindows()

    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Region selection — SAM2 mode
# ---------------------------------------------------------------------------

def collect_sam2_clicks(frame: np.ndarray) -> list[list[tuple[int, int]]]:
    """Same click UI as banner_segment.py — returns groups of seed points."""
    groups: list[list[tuple[int, int]]] = [[]]
    display = frame.copy()
    win = "Click seeds (n=next obj, Enter=done, Esc=cancel)"
    COLORS = [(0, 255, 0), (255, 100, 0), (0, 100, 255), (255, 255, 0)]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            gid = len(groups) - 1
            groups[gid].append((x, y))
            col = COLORS[gid % len(COLORS)]
            cv2.drawMarker(display, (x, y), col, cv2.MARKER_STAR, 18, 2)
            cv2.putText(display, f"{gid+1}.{len(groups[gid])}", (x+10, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.imshow(win, display)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, display)

    print("[UI] Left-click seed points | n=next object | Enter=done | Esc=cancel")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):
            break
        if key == 27:
            groups.clear()
            break
        if key == ord("n") and groups and groups[-1]:
            groups.append([])

    cv2.destroyAllWindows()
    return [g for g in groups if g]


def run_sam2(frame_bgr: np.ndarray,
             click_groups: list[list[tuple[int, int]]],
             checkpoint: str = "sam2/checkpoints/sam2.1_hiera_tiny.pt",
             model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml") -> dict[int, np.ndarray]:
    """Run SAM2 image predictor; return dict obj_id → binary mask (H, W)."""
    _repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"  SAM2 device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    masks_out: dict[int, np.ndarray] = {}
    for idx, group in enumerate(click_groups):
        obj_id = idx + 1
        coords = np.array(group, dtype=np.float32)
        labels = np.ones(len(group), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=coords, point_labels=labels, multimask_output=True)
        masks_out[obj_id] = masks[np.argmax(scores)]
        print(f"  obj {obj_id}: score={scores.max():.3f}")
    return masks_out


# ---------------------------------------------------------------------------
# Quad fitting (from mask)
# ---------------------------------------------------------------------------

from scipy.optimize import linprog


def _intersect_lines(pt1, d1, pt2, d2):
    det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(det) < 1e-9:
        return None
    b = pt2 - pt1
    t = (b[0] * (-d2[1]) - b[1] * (-d2[0])) / det
    return pt1 + t * d1


def fit_quadrilateral(mask: np.ndarray) -> np.ndarray | None:
    """Fit a tight perspective quad to a binary mask via 4 supporting-line LPs."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    hull = cv2.convexHull(largest).reshape(-1, 2).astype(np.float64)
    xs, ys = hull[:, 0], hull[:, 1]
    ones = np.ones(len(xs))
    bnd = [(None, None), (None, None)]

    res = linprog([0, -1], A_ub=np.c_[xs, ones],   b_ub=ys,   bounds=bnd)
    if not res.success: return cv2.boxPoints(rect).astype(np.float32)
    a, b = res.x; pt_t = np.array([0., b]); d_t = np.array([1., a])

    res = linprog([0,  1], A_ub=np.c_[-xs, -ones], b_ub=-ys,  bounds=bnd)
    if not res.success: return cv2.boxPoints(rect).astype(np.float32)
    a, b = res.x; pt_b = np.array([0., b]); d_b = np.array([1., a])

    res = linprog([0, -1], A_ub=np.c_[ys, ones],   b_ub=xs,   bounds=bnd)
    if not res.success: return cv2.boxPoints(rect).astype(np.float32)
    c, d = res.x; pt_l = np.array([d, 0.]); d_l = np.array([c, 1.])

    res = linprog([0,  1], A_ub=np.c_[-ys, -ones], b_ub=-xs,  bounds=bnd)
    if not res.success: return cv2.boxPoints(rect).astype(np.float32)
    c, d = res.x; pt_r = np.array([d, 0.]); d_r = np.array([c, 1.])

    for v in [d_t, d_b, d_l, d_r]:
        v /= np.linalg.norm(v)

    def _ang(a, b):
        return np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))

    def _avg(a, b):
        if np.dot(a, b) < 0:
            b = -b
        v = a + b
        return v / np.linalg.norm(v)

    if _ang(d_t, d_b) <= _ang(d_l, d_r):
        d_t = d_b = _avg(d_t, d_b)
    else:
        d_l = d_r = _avg(d_l, d_r)

    c_tl = _intersect_lines(pt_t, d_t, pt_l, d_l)
    c_tr = _intersect_lines(pt_t, d_t, pt_r, d_r)
    c_br = _intersect_lines(pt_b, d_b, pt_r, d_r)
    c_bl = _intersect_lines(pt_b, d_b, pt_l, d_l)

    if any(c is None for c in [c_tl, c_tr, c_br, c_bl]):
        return cv2.boxPoints(rect).astype(np.float32)

    corners = np.array([c_tl, c_tr, c_br, c_bl], dtype=np.float32)
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    return np.array([corners[np.argmin(s)], corners[np.argmax(d)],
                     corners[np.argmax(s)], corners[np.argmin(d)]], dtype=np.float32)


def polygon_to_quad(pts: np.ndarray) -> np.ndarray:
    """Order N polygon points as TL, TR, BR, BL quad via convex hull + centroid."""
    hull_idx = cv2.convexHull(pts.astype(np.float32), returnPoints=False).flatten()
    hull_pts = pts[hull_idx].astype(np.float32)

    # Approximate to 4 corners
    peri = cv2.arcLength(hull_pts, True)
    approx = cv2.approxPolyDP(hull_pts, 0.02 * peri, True)
    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        # Fallback: minAreaRect
        rect = cv2.minAreaRect(hull_pts)
        corners = cv2.boxPoints(rect).astype(np.float32)

    # Sort: TL, TR, BR, BL
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    return np.array([corners[np.argmin(s)], corners[np.argmax(d)],
                     corners[np.argmax(s)], corners[np.argmin(d)]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Oriented homography
# ---------------------------------------------------------------------------

def compute_oriented_homography(
    corners: np.ndarray,
    K: np.ndarray,
) -> dict:
    """Compute the oriented homography for a planar region using camera intrinsics.

    Parameters
    ----------
    corners : (4, 2) float32
        Quad corners in image pixels, ordered TL, TR, BR, BL.
    K : (3, 3) float64
        Camera intrinsic matrix.

    Returns
    -------
    dict with keys:
        H           : (3,3) homography mapping a dst_rect (0..dst_w, 0..dst_h) to image
        dst_w, dst_h: canonical destination size preserving true physical aspect ratio
        aspect      : physical width / height ratio
        R, t, normal: best decomposition (camera-frame rotation, translation, plane normal)
        pts_3d      : (4,3) 3-D positions of the four corners (up to a depth scale)
    """
    corners_f64 = corners.astype(np.float64)

    # --- Step 1: H from a unit square to image corners ---
    unit_src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    H_unit, _ = cv2.findHomography(unit_src, corners_f64)
    if H_unit is None:
        raise ValueError("findHomography failed — degenerate corners?")

    # --- Step 2: decompose H with respect to K ---
    # cv2.decomposeHomographyMat expects the homography that maps from a plane
    # (normalized image coords) to the camera image, and the camera matrix.
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H_unit, K)

    # Pick the decomposition where the plane is in FRONT of the camera
    # (translation z > 0) and the normal points TOWARD the camera (n_z < 0).
    best = 0
    for i in range(num):
        n = Ns[i].flatten()
        t = Ts[i].flatten()
        if t[2] > 0 and n[2] < 0:
            best = i
            break

    R = Rs[best]
    t = Ts[best].flatten()
    normal = Ns[best].flatten()   # plane normal in camera coords (points toward cam)

    # --- Step 3: unproject image corners to 3D using the plane normal ---
    # The convention from decomposeHomographyMat: plane equation is n^T * X = 1
    # For each image point p_i: ray direction r_i = K^{-1} * [x,y,1]^T
    # Depth: lambda_i = 1 / (n^T * r_i)
    K_inv = np.linalg.inv(K)
    corners_h = np.hstack([corners_f64, np.ones((4, 1))])    # (4, 3)
    rays = (K_inv @ corners_h.T).T                            # (4, 3) normalized rays

    dot = rays @ normal                                        # (4,)
    # Guard against near-zero (degenerate view angle)
    eps = 1e-6
    dot = np.where(np.abs(dot) < eps, eps, dot)
    depths = 1.0 / dot                                        # (4,) depths

    # Force positive depths (flip normal sign if needed)
    if depths.mean() < 0:
        normal = -normal
        depths = -depths

    pts_3d = rays * depths[:, np.newaxis]                     # (4, 3)

    # --- Step 4: true physical aspect ratio ---
    # Average the two widths (top and bottom edges) and two heights (left, right)
    w_top  = np.linalg.norm(pts_3d[1] - pts_3d[0])
    w_bot  = np.linalg.norm(pts_3d[2] - pts_3d[3])
    h_left = np.linalg.norm(pts_3d[3] - pts_3d[0])
    h_rgt  = np.linalg.norm(pts_3d[2] - pts_3d[1])

    phys_w = (w_top + w_bot) / 2.0
    phys_h = (h_left + h_rgt) / 2.0
    aspect = phys_w / phys_h if phys_h > 1e-6 else 1.0

    # --- Step 5: canonical destination rectangle (preserves aspect ratio) ---
    DST_H = 256
    DST_W = max(1, int(round(DST_H * aspect)))
    dst_rect = np.array([[0, 0], [DST_W, 0], [DST_W, DST_H], [0, DST_H]],
                        dtype=np.float32)

    # Final homography: dst_rect → image corners
    H_final, _ = cv2.findHomography(dst_rect, corners.astype(np.float32))

    return dict(
        H=H_final,
        dst_w=DST_W,
        dst_h=DST_H,
        dst_rect=dst_rect,
        aspect=aspect,
        phys_w=phys_w,
        phys_h=phys_h,
        R=R,
        t=t,
        normal=normal,
        pts_3d=pts_3d,
    )


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def composite_overlay(frame: np.ndarray,
                      corners: np.ndarray,
                      overlay_img: np.ndarray,
                      homo: dict,
                      padding: float = 0.05) -> np.ndarray:
    """Warp overlay_img into the banner region using the oriented homography.

    overlay_img can be RGBA (alpha compositing) or RGB.
    """
    dst_w, dst_h = homo["dst_w"], homo["dst_h"]
    H_final = homo["H"]

    # Scale overlay to fit dst rect with padding, centered
    avail_w = int(dst_w * (1 - 2 * padding))
    avail_h = int(dst_h * (1 - 2 * padding))

    ov_h, ov_w = overlay_img.shape[:2]
    scale = min(avail_w / ov_w, avail_h / ov_h)
    new_w = max(1, int(round(ov_w * scale)))
    new_h = max(1, int(round(ov_h * scale)))
    ov_resized = cv2.resize(overlay_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Sample background color from frame's banner region
    H_to_rect, _ = cv2.findHomography(corners, homo["dst_rect"])
    warped_orig = cv2.warpPerspective(frame, H_to_rect, (dst_w, dst_h))
    bg_color = tuple(int(c) for c in cv2.mean(warped_orig)[:3])

    canvas = np.full((dst_h, dst_w, 3), bg_color, dtype=np.uint8)
    ox = (dst_w - new_w) // 2
    oy = (dst_h - new_h) // 2

    if ov_resized.ndim == 3 and ov_resized.shape[2] == 4:
        rgb   = ov_resized[:, :, :3].astype(np.float32)
        alpha = ov_resized[:, :, 3:].astype(np.float32) / 255.0
        patch = canvas[oy:oy+new_h, ox:ox+new_w].astype(np.float32)
        canvas[oy:oy+new_h, ox:ox+new_w] = (rgb * alpha + patch * (1 - alpha)).astype(np.uint8)
    else:
        canvas[oy:oy+new_h, ox:ox+new_w] = ov_resized[:, :, :3]

    # Warp canvas back into the original frame perspective
    warped_canvas = cv2.warpPerspective(canvas, H_final, (frame.shape[1], frame.shape[0]))

    # Mask = convex hull of corners
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)

    result = frame.copy()
    result[mask > 0] = warped_canvas[mask > 0]
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(frame: np.ndarray,
              regions: list[dict],           # each: {corners, homo, mask(optional)}
              composited: np.ndarray | None,
              save_path: str = "region_result.png"):
    """Draw quads, vanishing lines, and show composited + rectified views."""
    COLORS = [(0, 200, 0), (200, 80, 0), (0, 80, 200), (200, 200, 0)]

    vis = frame.copy()

    for i, reg in enumerate(regions):
        corners = reg["corners"]
        homo    = reg["homo"]
        col     = COLORS[i % len(COLORS)]
        col_bgr = col[::-1] if len(col) == 3 else col   # already BGR

        # Draw mask overlay if available
        if "mask" in reg and reg["mask"] is not None:
            m = reg["mask"].astype(bool)
            c = np.array(col, dtype=np.uint8)
            vis[m] = (vis[m].astype(np.float32) * 0.55 + c.astype(np.float32) * 0.45).astype(np.uint8)

        # Draw quad
        quad = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [quad], True, (255, 255, 255), 2)
        for lbl, pt in zip(["TL", "TR", "BR", "BL"], corners):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(vis, (cx, cy), 6, col, -1)
            cv2.putText(vis, f"{i+1}-{lbl}", (cx+8, cy-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Vanishing lines for each edge pair
        tl, tr, br, bl = corners.astype(np.float64)
        h_img, w_img = vis.shape[:2]
        for label, p1a, p1b, p2a, p2b, vc in [
            ("TB", tl, tr, bl, br, (0, 255, 255)),
            ("LR", tl, bl, tr, br, (255, 0, 255)),
        ]:
            d1 = p1b - p1a;  d2 = p2b - p2a
            for pa, dv in [(p1a, d1), (p2a, d2)]:
                n = np.linalg.norm(dv)
                if n < 1e-6:
                    continue
                du = dv / n * max(w_img, h_img) * 3
                cv2.line(vis, tuple((pa - du).astype(int)),
                         tuple((pa + du).astype(int)), vc, 1, cv2.LINE_AA)
            vp = _intersect_lines(p1a, d1, p2a, d2)
            if vp is not None:
                vx, vy = int(vp[0]), int(vp[1])
                if -w_img < vx < 2 * w_img and -h_img < vy < 2 * h_img:
                    cv2.drawMarker(vis, (vx, vy), vc, cv2.MARKER_DIAMOND, 18, 2)
                    cv2.putText(vis, f"VP-{label}", (vx+12, vy-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, vc, 1, cv2.LINE_AA)

        # Plane normal annotation
        n = homo["normal"]
        centroid = corners.mean(axis=0).astype(int)
        arrow_len = int(min(vis.shape[:2]) * 0.12)
        # Project normal direction to 2D (ignore z component for display)
        nx_2d = int(centroid[0] + n[0] * arrow_len)
        ny_2d = int(centroid[1] + n[1] * arrow_len)
        cv2.arrowedLine(vis, tuple(centroid), (nx_2d, ny_2d),
                        (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(vis, f"n={n[0]:.2f},{n[1]:.2f},{n[2]:.2f}",
                    (centroid[0]+4, centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        # Aspect ratio annotation
        cv2.putText(vis,
                    f"AR={homo['aspect']:.2f}  "
                    f"({homo['phys_w']:.0f}x{homo['phys_h']:.0f} depth-units)",
                    (centroid[0] - 60, centroid[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 0), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, vis)
    print(f"  Saved: {save_path}")

    if composited is not None:
        comp_path = save_path.rsplit(".", 1)[0] + "_composited.png"
        cv2.imwrite(comp_path, composited)
        print(f"  Saved: {comp_path}")

    # ---- matplotlib figure ----
    warpeds = []
    for reg in regions:
        c = reg["corners"]
        dw, dh = reg["homo"]["dst_w"], reg["homo"]["dst_h"]
        dst = np.array([[0,0],[dw,0],[dw,dh],[0,dh]], dtype=np.float32)
        H, _ = cv2.findHomography(c, dst)
        if H is not None:
            warpeds.append(cv2.warpPerspective(frame, H, (dw, dh)))

    has_comp = composited is not None
    ncols = 1 + (1 if has_comp else 0) + len(warpeds)
    ratios = [3] + ([3] if has_comp else []) + [1] * len(warpeds)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5),
                             gridspec_kw={"width_ratios": ratios})
    if ncols == 1:
        axes = [axes]

    ci = 0
    axes[ci].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[ci].set_title(f"Oriented quads ({len(regions)} region(s))")
    axes[ci].axis("off");  ci += 1

    if has_comp:
        axes[ci].imshow(cv2.cvtColor(composited, cv2.COLOR_BGR2RGB))
        axes[ci].set_title("Overlay composited")
        axes[ci].axis("off");  ci += 1

    for j, w in enumerate(warpeds):
        ar = regions[j]["homo"]["aspect"]
        axes[ci + j].imshow(cv2.cvtColor(w, cv2.COLOR_BGR2RGB))
        axes[ci + j].set_title(f"Obj {j+1} rectified\nAR={ar:.2f}")
        axes[ci + j].axis("off")

    plt.tight_layout()
    full_path = save_path.rsplit(".", 1)[0] + "_full.png"
    plt.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {full_path}")


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_frame(path: str) -> np.ndarray:
    """Load a single frame from an image or the first frame of a video."""
    # Try as image first
    frame = cv2.imread(path)
    if frame is not None:
        return frame
    # Try as video
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-i", path, "-vframes", "1", "-q:v", "2", tmp, "-y", "-loglevel", "error"],
        check=True,
    )
    frame = cv2.imread(tmp)
    os.unlink(tmp)
    if frame is None:
        raise RuntimeError(f"Could not load frame from: {path}")
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select region → oriented homography (focal-point-aware) → overlay")
    parser.add_argument("input", help="Image or video file")
    parser.add_argument("--logo", default=None,
                        help="Image to warp into the selected region(s)")
    parser.add_argument("--mode", choices=["poly", "sam2"], default="sam2",
                        help="Region selection mode (default: sam2)")
    parser.add_argument("--focal-length", type=float, default=None,
                        help="Camera focal length in pixels (default: max(w,h))")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="Overlay padding fraction (default: 0.05)")
    parser.add_argument("--save", default="region_result.png",
                        help="Output image path")
    parser.add_argument("--checkpoint",
                        default="sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg",
                        default="configs/sam2.1/sam2.1_hiera_t.yaml")
    args = parser.parse_args()

    # Load overlay image
    overlay = None
    if args.logo:
        overlay = cv2.imread(args.logo, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise RuntimeError(f"Could not read logo: {args.logo}")
        print(f"  Loaded logo: {args.logo} ({overlay.shape[1]}×{overlay.shape[0]})")

    # Load input frame
    print("[1] Loading frame …")
    frame = load_frame(args.input)
    print(f"  Frame: {frame.shape[1]}×{frame.shape[0]}")

    # Camera matrix
    K = estimate_camera_matrix(frame.shape, focal_length=args.focal_length)
    print(f"  Camera K: f={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # --- Region selection ---
    regions: list[dict] = []

    if args.mode == "poly":
        print("[2] Select region(s) via polygon …")
        while True:
            pts = select_polygon(frame)
            if pts is None or len(pts) < 3:
                if not regions:
                    print("No region selected — exiting.")
                    return
                break
            corners = polygon_to_quad(pts)
            homo = compute_oriented_homography(corners, K)
            regions.append({"corners": corners, "homo": homo})
            print(f"  Region {len(regions)}: aspect={homo['aspect']:.3f}  "
                  f"normal={homo['normal'].round(3)}")
            ans = input("  Add another region? [y/N]: ").strip().lower()
            if ans != "y":
                break

    else:  # sam2
        print("[2] Collecting SAM2 seed clicks …")
        click_groups = collect_sam2_clicks(frame)
        if not click_groups:
            print("No clicks — exiting.")
            return
        print("[3] Running SAM2 …")
        masks = run_sam2(frame, click_groups,
                         checkpoint=args.checkpoint, model_cfg=args.model_cfg)
        print("[4] Fitting quads …")
        for obj_id, mask in masks.items():
            corners = fit_quadrilateral(mask)
            if corners is None:
                print(f"  obj {obj_id}: quad fitting failed, skipping")
                continue
            homo = compute_oriented_homography(corners, K)
            regions.append({"corners": corners, "homo": homo, "mask": mask})
            print(f"  obj {obj_id}: aspect={homo['aspect']:.3f}  "
                  f"normal={homo['normal'].round(3)}")

    if not regions:
        print("No valid regions — exiting.")
        return

    # --- Compositing ---
    composited = None
    if overlay is not None:
        print("[*] Compositing overlay …")
        composited = frame.copy()
        for reg in regions:
            composited = composite_overlay(
                composited, reg["corners"], overlay, reg["homo"],
                padding=args.padding)

    # --- Visualize ---
    print("[*] Visualizing …")
    visualize(frame, regions, composited=composited, save_path=args.save)
    print("Done.")


if __name__ == "__main__":
    main()
