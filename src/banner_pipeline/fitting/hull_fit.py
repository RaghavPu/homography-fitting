"""Hull-based quad fitter (from court_homography.py).

Simplifies the convex hull to ≤6 vertices, classifies each as "internal"
or "boundary" (near the frame edge), and deduces the 4 parallelogram
corners — even when 1–2 corners are off-screen.
"""

from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline.fitting.base import QuadFitter
from banner_pipeline.geometry import intersect_implicit, line_from_points, sort_corners_tlbr


# ---------------------------------------------------------------------------
# Hull extraction + vertex classification
# ---------------------------------------------------------------------------

def get_hull_vertices(mask: np.ndarray) -> np.ndarray:
    """Largest contour → convex hull → simplified to ≤6 vertices."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in mask.")
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)

    perim = cv2.arcLength(hull, True)
    pts = hull
    for factor in np.linspace(0.01, 0.5, 1000):
        approx = cv2.approxPolyDP(hull, factor * perim, True)
        if len(approx) <= 6:
            pts = approx
            break
    return pts.reshape(-1, 2).astype(np.float32)


def classify_vertices(
    pts: np.ndarray,
    img_shape: tuple,
    margin: int = 15,
) -> list[str]:
    """Label each vertex as ``"boundary"`` or ``"internal"``."""
    h, w = img_shape[:2]
    labels = []
    for p in pts:
        near_edge = p[0] < margin or p[0] > w - margin or p[1] < margin or p[1] > h - margin
        labels.append("boundary" if near_edge else "internal")
    return labels


# ---------------------------------------------------------------------------
# Corner deduction helpers
# ---------------------------------------------------------------------------

def _are_adjacent(i0: int, i1: int, n: int, labels: list[str]) -> bool:
    j = (i0 + 1) % n
    while j != i1:
        if labels[j] == "boundary":
            return False
        j = (j + 1) % n
    return True


def _corners_4(pts: np.ndarray, internal_idxs: list[int]) -> np.ndarray:
    return sort_corners_tlbr(pts[internal_idxs].copy())


def _corners_3(
    pts: np.ndarray,
    labels: list[str],
    internal_idxs: list[int],
) -> np.ndarray:
    n = len(pts)
    for mid_pos in range(3):
        ia = internal_idxs[mid_pos - 1]
        ib = internal_idxs[mid_pos]
        ic = internal_idxs[(mid_pos + 1) % 3]
        if _are_adjacent(ia, ib, n, labels) and _are_adjacent(ib, ic, n, labels):
            a, b, c = pts[ia], pts[ib], pts[ic]
            d = a + c - b
            return sort_corners_tlbr(np.array([a, b, c, d], dtype=np.float32))
    raise RuntimeError("Could not identify middle vertex among 3 internals")


def _corners_2_adjacent(
    pts: np.ndarray,
    labels: list[str],
    i0: int,
    i1: int,
    fwd_is_direct: bool,
) -> np.ndarray:
    n = len(pts)
    if not fwd_is_direct:
        i0, i1 = i1, i0

    b0 = (i0 - 1) % n
    b1 = (i1 + 1) % n

    side_line_0 = line_from_points(pts[i0], pts[b0])
    side_line_1 = line_from_points(pts[i1], pts[b1])
    known_side = line_from_points(pts[i0], pts[i1])

    boundary_pts = [pts[i] for i in range(n) if labels[i] == "boundary"]
    a, b, c = known_side
    norm = np.hypot(a, b)
    ref_dist = (a * float(pts[i0][0]) + b * float(pts[i0][1]) + c) / norm
    farthest = max(
        boundary_pts,
        key=lambda p: abs((a * float(p[0]) + b * float(p[1]) + c) / norm - ref_dist),
    )
    c_opp = -(a * float(farthest[0]) + b * float(farthest[1]))
    opp_line = (a, b, c_opp)

    c0 = intersect_implicit(side_line_0, opp_line)
    c1 = intersect_implicit(side_line_1, opp_line)
    if c0 is None or c1 is None:
        raise RuntimeError("Side lines parallel to opposite side — cannot intersect.")
    return sort_corners_tlbr(np.array([pts[i0], pts[i1], c0, c1], dtype=np.float32))


def _corners_2_opposite(
    pts: np.ndarray,
    labels: list[str],
    i0: int,
    i1: int,
) -> np.ndarray:
    n = len(pts)
    b0_next = (i0 + 1) % n
    b1_prev = (i1 - 1) % n
    b1_next = (i1 + 1) % n
    b0_prev = (i0 - 1) % n

    line_a = line_from_points(pts[i0], pts[b0_next])
    line_b = line_from_points(pts[b1_prev], pts[i1])
    corner_fwd = intersect_implicit(line_a, line_b)

    line_c = line_from_points(pts[i1], pts[b1_next])
    line_d = line_from_points(pts[b0_prev], pts[i0])
    corner_bwd = intersect_implicit(line_c, line_d)

    if corner_fwd is None or corner_bwd is None:
        raise RuntimeError("Opposite-side lines are parallel — cannot intersect.")
    return sort_corners_tlbr(
        np.array([pts[i0], pts[i1], corner_fwd, corner_bwd], dtype=np.float32),
    )


def find_corners(pts: np.ndarray, labels: list[str]) -> np.ndarray:
    """Derive 4 parallelogram corners from hull vertices + labels.

    Handles 2, 3, or 4+ visible (internal) vertices.
    Returns ``(4, 2)`` float32 array ordered ``[TL, TR, BR, BL]``.
    """
    internal_idxs = [i for i in range(len(pts)) if labels[i] == "internal"]
    num = len(internal_idxs)

    if num >= 4:
        return _corners_4(pts, internal_idxs[:4])
    if num == 3:
        return _corners_3(pts, labels, internal_idxs)
    if num == 2:
        n = len(pts)
        i0, i1 = internal_idxs
        adj_fwd = _are_adjacent(i0, i1, n, labels)
        adj_bwd = _are_adjacent(i1, i0, n, labels)
        if adj_fwd or adj_bwd:
            return _corners_2_adjacent(pts, labels, i0, i1, adj_fwd)
        return _corners_2_opposite(pts, labels, i0, i1)
    raise RuntimeError(f"Need ≥2 internal vertices to fit a parallelogram, got {num}")


# ---------------------------------------------------------------------------
# QuadFitter implementation
# ---------------------------------------------------------------------------

class HullFitter(QuadFitter):
    """Hull vertex deduction fitter — works when corners extend off-screen."""

    @property
    def name(self) -> str:
        return "hull"

    def fit(self, mask: np.ndarray, **kwargs) -> np.ndarray | None:
        """Fit a quad.  Keyword ``img_shape`` is required (frame dimensions)."""
        img_shape: tuple | None = kwargs.get("img_shape")
        margin: int = kwargs.get("margin", 15)

        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        pts = get_hull_vertices(mask)
        if img_shape is None:
            img_shape = mask.shape[:2]
        labels = classify_vertices(pts, img_shape, margin=margin)
        return find_corners(pts, labels)
