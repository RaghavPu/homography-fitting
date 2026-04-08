"""Weighted-PCA quad fitter (from banner_segment.py).

Splits the mask contour along an axis, fits independent weighted-PCA lines
to each edge (Hann-window weighting), and intersects with minAreaRect
extent edges to recover 4 perspective-aware corners.
"""

from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline.fitting.base import QuadFitter
from banner_pipeline.geometry import intersect_parametric, sort_corners_tlbr


def _fit_line_pts(
    pts: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a line to 2-D points.  Returns ``(point_on_line, unit_direction)``.

    If *weights* is given, uses weighted PCA instead of ``cv2.fitLine``.
    """
    if weights is None:
        vx, vy, cx, cy = cv2.fitLine(
            pts.reshape(-1, 1, 2).astype(np.float32),
            cv2.DIST_L2, 0, 0.01, 0.01,
        ).flatten()
        return np.array([cx, cy], dtype=np.float64), np.array([vx, vy], dtype=np.float64)

    w = weights / weights.sum()
    centroid = (pts * w[:, None]).sum(axis=0)
    centered = pts - centroid
    cov = (centered * w[:, None]).T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, -1].astype(np.float64)
    return centroid.astype(np.float64), direction


class PCAFitter(QuadFitter):
    """Perspective-aware quad fitting via Hann-weighted PCA on split contour edges."""

    @property
    def name(self) -> str:
        return "pca"

    def fit(self, mask: np.ndarray, **kwargs) -> np.ndarray | None:
        """Fit a quadrilateral to *mask*.

        Keyword arguments
        -----------------
        axis : ``"short"`` (default) or ``"long"``
            Which axis to split the contour along for independent edge fitting.
        """
        axis: str = kwargs.get("axis", "short")
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        # Keep only the largest connected component.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
        if n_labels > 2:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_u8 = ((labels == largest_label) * 255).astype(np.uint8)

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
            proj_short = rel @ short_dir
            short_bins = np.round(proj_short).astype(int)
            unique_bins = np.unique(short_bins)
            n_trim = max(1, int(len(unique_bins) * 0.15))
            unique_bins = unique_bins[n_trim:-n_trim]
            proj_long = rel @ long_dir
            left_idx, right_idx = [], []
            for b in unique_bins:
                in_bin = np.where(short_bins == b)[0]
                left_idx.append(in_bin[np.argmin(proj_long[in_bin])])
                right_idx.append(in_bin[np.argmax(proj_long[in_bin])])
            group_a = pts[left_idx]
            group_b = pts[right_idx]

            def _hann(grp):
                s = (grp - center) @ short_dir
                t = (s - s.min()) / max(s.max() - s.min(), 1e-9)
                return np.clip(0.5 * (1 - np.cos(2 * np.pi * t)), 0.01, 1.0)

            weights_a, weights_b = _hann(group_a), _hann(group_b)
        else:  # short
            proj_short = rel @ short_dir
            group_a = pts[proj_short >= 0]
            group_b = pts[proj_short < 0]

            def _hann_long(grp):
                s = (grp - center) @ long_dir
                t = (s - s.min()) / max(s.max() - s.min(), 1e-9)
                return np.clip(0.5 * (1 - np.cos(2 * np.pi * t)), 0.01, 1.0)

            weights_a, weights_b = _hann_long(group_a), _hann_long(group_b)

        if len(group_a) < 5 or len(group_b) < 5:
            return cv2.boxPoints(rect).astype(np.float32)

        pt_a, dir_a = _fit_line_pts(group_a, weights_a)
        pt_b, dir_b = _fit_line_pts(group_b, weights_b)

        if dir_a @ dir_b < 0:
            dir_b = -dir_b

        # Extent edges of the minAreaRect.
        box_corners = cv2.boxPoints(rect).astype(np.float64)
        extent_dir = short_dir if axis == "long" else long_dir
        proj_box = (box_corners - center) @ extent_dir
        order = np.argsort(proj_box)
        edge1_pts = box_corners[order[:2]]
        edge2_pts = box_corners[order[2:]]
        edge1_dir = edge1_pts[1] - edge1_pts[0]
        edge1_dir /= np.linalg.norm(edge1_dir)
        edge2_dir = edge2_pts[1] - edge2_pts[0]
        edge2_dir /= np.linalg.norm(edge2_dir)

        c_a1 = intersect_parametric(pt_a, dir_a, edge1_pts[0], edge1_dir)
        c_a2 = intersect_parametric(pt_a, dir_a, edge2_pts[0], edge2_dir)
        c_b1 = intersect_parametric(pt_b, dir_b, edge1_pts[0], edge1_dir)
        c_b2 = intersect_parametric(pt_b, dir_b, edge2_pts[0], edge2_dir)

        if any(c is None for c in [c_a1, c_a2, c_b1, c_b2]):
            return cv2.boxPoints(rect).astype(np.float32)

        corners = np.array([c_a1, c_a2, c_b1, c_b2], dtype=np.float32)
        return sort_corners_tlbr(corners)
