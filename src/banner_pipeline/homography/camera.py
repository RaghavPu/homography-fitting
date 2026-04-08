"""Camera intrinsics and oriented homography decomposition."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_camera_matrix(
    frame_shape: tuple,
    focal_length: float | None = None,
) -> np.ndarray:
    """Build K = ``[[f,0,cx],[0,f,cy],[0,0,1]]``.

    If *focal_length* is ``None``, defaults to ``max(w, h)`` — a reasonable
    estimate for a standard lens (~53° diagonal FoV).
    """
    h, w = frame_shape[:2]
    f = focal_length if focal_length is not None else float(max(h, w))
    cx, cy = w / 2.0, h / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def compute_oriented_homography(
    corners: np.ndarray,
    K: np.ndarray,
) -> dict:
    """Compute the oriented homography for a planar region using camera intrinsics.

    Parameters
    ----------
    corners : (4, 2) float32
        Quad corners ordered TL, TR, BR, BL.
    K : (3, 3) float64
        Camera intrinsic matrix.

    Returns
    -------
    dict with keys:
        H, dst_w, dst_h, dst_rect, aspect, phys_w, phys_h, R, t, normal, pts_3d
    """
    corners_f64 = corners.astype(np.float64)

    # H from unit square to image corners.
    unit_src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    H_unit, _ = cv2.findHomography(unit_src, corners_f64)
    if H_unit is None:
        raise ValueError("findHomography failed — degenerate corners?")

    # Decompose H with respect to K.
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H_unit, K)

    # Pick decomposition where plane is in front (t_z > 0) and normal
    # points toward camera (n_z < 0).
    best = 0
    for i in range(num):
        n = Ns[i].flatten()
        t = Ts[i].flatten()
        if t[2] > 0 and n[2] < 0:
            best = i
            break

    R = Rs[best]
    t = Ts[best].flatten()
    normal = Ns[best].flatten()

    # Unproject image corners to 3D using plane equation n^T X = 1.
    K_inv = np.linalg.inv(K)
    corners_h = np.hstack([corners_f64, np.ones((4, 1))])
    rays = (K_inv @ corners_h.T).T

    dot = rays @ normal
    eps = 1e-6
    dot = np.where(np.abs(dot) < eps, eps, dot)
    depths = 1.0 / dot

    if depths.mean() < 0:
        normal = -normal
        depths = -depths

    pts_3d = rays * depths[:, np.newaxis]

    # True physical aspect ratio.
    w_top = np.linalg.norm(pts_3d[1] - pts_3d[0])
    w_bot = np.linalg.norm(pts_3d[2] - pts_3d[3])
    h_left = np.linalg.norm(pts_3d[3] - pts_3d[0])
    h_rgt = np.linalg.norm(pts_3d[2] - pts_3d[1])

    phys_w = (w_top + w_bot) / 2.0
    phys_h = (h_left + h_rgt) / 2.0
    aspect = phys_w / phys_h if phys_h > 1e-6 else 1.0

    # Canonical destination rectangle (preserves aspect ratio).
    DST_H = 256
    DST_W = max(1, int(round(DST_H * aspect)))
    dst_rect = np.array(
        [[0, 0], [DST_W, 0], [DST_W, DST_H], [0, DST_H]],
        dtype=np.float32,
    )

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
