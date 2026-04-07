"""LP-based quad fitter (from region_overlay.py).

Fits a tight perspective quadrilateral by solving four linear programmes
to find supporting lines (top, bottom, left, right), then intersects them.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linprog

from banner_pipeline.fitting.base import QuadFitter
from banner_pipeline.geometry import intersect_parametric, sort_corners_tlbr


class LPFitter(QuadFitter):
    """Tight quad fitting via 4 supporting-line linear programmes."""

    @property
    def name(self) -> str:
        return "lp"

    def fit(self, mask: np.ndarray, **kwargs) -> np.ndarray | None:
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

        def _fallback():
            return cv2.boxPoints(rect).astype(np.float32)

        # Top supporting line (min intercept s.t. all points below).
        res = linprog([0, -1], A_ub=np.c_[xs, ones], b_ub=ys, bounds=bnd)
        if not res.success:
            return _fallback()
        a, b = res.x
        pt_t, d_t = np.array([0.0, b]), np.array([1.0, a])

        # Bottom supporting line.
        res = linprog([0, 1], A_ub=np.c_[-xs, -ones], b_ub=-ys, bounds=bnd)
        if not res.success:
            return _fallback()
        a, b = res.x
        pt_b, d_b = np.array([0.0, b]), np.array([1.0, a])

        # Left supporting line (rotated coordinates).
        res = linprog([0, -1], A_ub=np.c_[ys, ones], b_ub=xs, bounds=bnd)
        if not res.success:
            return _fallback()
        c, d = res.x
        pt_l, d_l = np.array([d, 0.0]), np.array([c, 1.0])

        # Right supporting line.
        res = linprog([0, 1], A_ub=np.c_[-ys, -ones], b_ub=-xs, bounds=bnd)
        if not res.success:
            return _fallback()
        c, d = res.x
        pt_r, d_r = np.array([d, 0.0]), np.array([c, 1.0])

        for v in [d_t, d_b, d_l, d_r]:
            v /= np.linalg.norm(v)

        # If one pair is more parallel, average them.
        def _ang(va, vb):
            return np.degrees(np.arccos(np.clip(abs(np.dot(va, vb)), 0, 1)))

        def _avg(va, vb):
            if np.dot(va, vb) < 0:
                vb = -vb
            v = va + vb
            return v / np.linalg.norm(v)

        if _ang(d_t, d_b) <= _ang(d_l, d_r):
            d_t = d_b = _avg(d_t, d_b)
        else:
            d_l = d_r = _avg(d_l, d_r)

        c_tl = intersect_parametric(pt_t, d_t, pt_l, d_l)
        c_tr = intersect_parametric(pt_t, d_t, pt_r, d_r)
        c_br = intersect_parametric(pt_b, d_b, pt_r, d_r)
        c_bl = intersect_parametric(pt_b, d_b, pt_l, d_l)

        if any(c is None for c in [c_tl, c_tr, c_br, c_bl]):
            return _fallback()

        corners = np.array([c_tl, c_tr, c_br, c_bl], dtype=np.float32)
        return sort_corners_tlbr(corners)
