"""Line intersection, corner sorting, and other geometry helpers."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Parametric line intersection  (p + t*d form)
# ---------------------------------------------------------------------------

def intersect_parametric(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray | None:
    """Intersect lines  p1 + t*d1  and  p2 + s*d2.

    Returns the intersection point, or ``None`` if the lines are parallel.
    """
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    dp = p2 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
    return p1 + t * d1


# ---------------------------------------------------------------------------
# Implicit line intersection  (ax + by + c = 0 form)
# ---------------------------------------------------------------------------

def line_from_points(
    p1: np.ndarray,
    p2: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(a, b, c)`` for the line  ax + by + c = 0  through *p1*, *p2*."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c


def intersect_implicit(
    l1: tuple[float, float, float],
    l2: tuple[float, float, float],
) -> np.ndarray | None:
    """Intersect two ``(a, b, c)`` lines.  Returns ``(x, y)`` or ``None``."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float32)


# ---------------------------------------------------------------------------
# Corner sorting
# ---------------------------------------------------------------------------

def sort_corners_tlbr(corners: np.ndarray) -> np.ndarray:
    """Sort 4 points into ``[TL, TR, BR, BL]`` order.

    Uses the sum/difference heuristic (works for quads that are roughly
    axis-aligned or seen at moderate perspective).
    """
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    return np.array(
        [
            corners[np.argmin(s)],  # TL: smallest  x+y
            corners[np.argmax(d)],  # TR: largest   x-y
            corners[np.argmax(s)],  # BR: largest   x+y
            corners[np.argmin(d)],  # BL: smallest  x-y
        ],
        dtype=np.float32,
    )
