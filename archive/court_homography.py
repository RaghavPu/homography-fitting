# -*- coding: utf-8 -*-
"""
court_homography.py
-------------------
1. Diff original vs masked frame → clean binary mask
2. Largest blob → convex hull → simplified vertices
3. Classify each vertex as "internal" or "boundary" (near frame edge)
4. Draw + label them

Usage
-----
    python court_homography.py tennis-clip.mp4 output.mp4
"""

from __future__ import annotations
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Frame / mask helpers
# ---------------------------------------------------------------------------

def _grab_frame(video_path: str) -> np.ndarray:
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


def extract_mask(original: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Pixels that changed between original and colour-overlay frame."""
    diff = np.abs(original.astype(np.float32) - masked.astype(np.float32)).max(axis=2)
    _, mask = cv2.threshold(diff.astype(np.uint8), 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


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


def classify_vertices(pts: np.ndarray, img_shape: tuple, margin: int = 15) -> list[str]:
    """Label each vertex as 'boundary' (within margin of frame edge) or 'internal'."""
    H_img, W_img = img_shape[:2]
    labels = []
    for p in pts:
        near_edge = (p[0] < margin or p[0] > W_img - margin or
                     p[1] < margin or p[1] > H_img - margin)
        labels.append("boundary" if near_edge else "internal")
    return labels


# ---------------------------------------------------------------------------
# Line math
# ---------------------------------------------------------------------------

def _line_from_pts(p1, p2) -> tuple[float, float, float]:
    """Return (a, b, c) for the line ax + by + c = 0 through p1 and p2."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c


def _intersect(l1: tuple[float, float, float],
               l2: tuple[float, float, float]) -> np.ndarray | None:
    """Intersect two (a, b, c) lines. Returns (x, y) or None if parallel."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float32)


# ---------------------------------------------------------------------------
# Corner extraction
# ---------------------------------------------------------------------------

def _sort_corners(corners: np.ndarray) -> np.ndarray:
    """Sort 4 points into [TL, TR, BR, BL] order."""
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    return np.array([
        corners[np.argmin(s)],   # TL: smallest x+y
        corners[np.argmax(d)],   # TR: largest  x-y
        corners[np.argmax(s)],   # BR: largest  x+y
        corners[np.argmin(d)],   # BL: smallest x-y
    ], dtype=np.float32)


def _are_adjacent(i0: int, i1: int, n: int, labels: list[str]) -> bool:
    """Check if two internal hull indices are adjacent (no boundary between them
    when walking forward from i0 to i1)."""
    j = (i0 + 1) % n
    while j != i1:
        if labels[j] == "boundary":
            return False
        j = (j + 1) % n
    return True


def _corners_4(pts: np.ndarray, internal_idxs: list[int]) -> np.ndarray:
    """All 4 corners visible — just return them sorted."""
    corners = pts[internal_idxs].copy()
    print(f"  All 4 corners internal — using directly")
    return _sort_corners(corners)


def _corners_3(pts: np.ndarray, labels: list[str],
               internal_idxs: list[int]) -> np.ndarray:
    """3 corners visible. The 4th = A + C - B where B is the middle
    vertex (adjacent to both A and C on the hull)."""
    n = len(pts)
    i0, i1, i2 = internal_idxs  # hull order

    # Find the middle: the one that has no boundary gap to either neighbour
    for mid_pos in range(3):
        ia = internal_idxs[mid_pos - 1]
        ib = internal_idxs[mid_pos]
        ic = internal_idxs[(mid_pos + 1) % 3]
        if (_are_adjacent(ia, ib, n, labels) and
                _are_adjacent(ib, ic, n, labels)):
            a, b, c = pts[ia], pts[ib], pts[ic]
            d = a + c - b
            print(f"  Middle vertex {ib}, deduced 4th: ({int(d[0])},{int(d[1])})")
            return _sort_corners(np.array([a, b, c, d], dtype=np.float32))

    raise RuntimeError("Could not identify middle vertex among 3 internals")


def _corners_2(pts: np.ndarray, labels: list[str],
               internal_idxs: list[int]) -> np.ndarray:
    """2 corners visible. Detect adjacent vs opposite and solve accordingly."""
    n = len(pts)
    i0, i1 = internal_idxs  # hull order

    adjacent_fwd = _are_adjacent(i0, i1, n, labels)
    adjacent_bwd = _are_adjacent(i1, i0, n, labels)

    if adjacent_fwd or adjacent_bwd:
        return _corners_2_adjacent(pts, labels, i0, i1,
                                   adjacent_fwd)
    else:
        return _corners_2_opposite(pts, labels, i0, i1)


def _corners_2_adjacent(pts: np.ndarray, labels: list[str],
                        i0: int, i1: int,
                        fwd_is_direct: bool) -> np.ndarray:
    """Two internal vertices share a direct hull edge (one side known).
    The opposite side is parallel; side lines come from boundary neighbours."""
    n = len(pts)
    if not fwd_is_direct:
        i0, i1 = i1, i0

    # i0 → i1 is the known side (no boundary between them going forward)
    # Boundary neighbour of i0 going backward
    b0 = (i0 - 1) % n
    # Boundary neighbour of i1 going forward
    b1 = (i1 + 1) % n

    side_line_0 = _line_from_pts(pts[i0], pts[b0])
    side_line_1 = _line_from_pts(pts[i1], pts[b1])
    known_side = _line_from_pts(pts[i0], pts[i1])

    # Parallel opposite side through the farthest boundary vertex
    boundary_pts = [pts[i] for i in range(n) if labels[i] == "boundary"]

    # "Farthest" = max signed distance from the known side
    a, b, c = known_side
    norm = np.hypot(a, b)
    ref_dist = (a * float(pts[i0][0]) + b * float(pts[i0][1]) + c) / norm
    farthest = max(boundary_pts,
                   key=lambda p: abs((a * float(p[0]) + b * float(p[1]) + c) / norm - ref_dist))
    c_opp = -(a * float(farthest[0]) + b * float(farthest[1]))
    opp_line = (a, b, c_opp)

    c0 = _intersect(side_line_0, opp_line)
    c1 = _intersect(side_line_1, opp_line)
    if c0 is None or c1 is None:
        raise RuntimeError("Side lines parallel to opposite side — cannot intersect.")

    print(f"  Adjacent pair ({i0},{i1}), deduced corners: "
          f"({int(c0[0])},{int(c0[1])}), ({int(c1[0])},{int(c1[1])})")
    return _sort_corners(np.array([pts[i0], pts[i1], c0, c1], dtype=np.float32))


def _corners_2_opposite(pts: np.ndarray, labels: list[str],
                        i0: int, i1: int) -> np.ndarray:
    """Two internal vertices are diagonal (boundary between them both ways).
    Intersect adjacent side-direction lines to find the missing corners."""
    n = len(pts)

    # Boundary neighbours of i0
    b0_prev = (i0 - 1) % n
    b0_next = (i0 + 1) % n
    # Boundary neighbours of i1
    b1_prev = (i1 - 1) % n
    b1_next = (i1 + 1) % n

    # Walking i0 → ... → i1 forward, the gap contains one missing corner.
    # That corner = intersection of line(i0, b0_next) and line(b1_prev, i1).
    line_a = _line_from_pts(pts[i0], pts[b0_next])
    line_b = _line_from_pts(pts[b1_prev], pts[i1])
    corner_fwd = _intersect(line_a, line_b)

    # Walking i1 → ... → i0 forward, the other gap.
    line_c = _line_from_pts(pts[i1], pts[b1_next])
    line_d = _line_from_pts(pts[b0_prev], pts[i0])
    corner_bwd = _intersect(line_c, line_d)

    if corner_fwd is None or corner_bwd is None:
        raise RuntimeError("Opposite-side lines are parallel — cannot intersect.")

    print(f"  Opposite pair ({i0},{i1}), deduced corners: "
          f"({int(corner_fwd[0])},{int(corner_fwd[1])}), "
          f"({int(corner_bwd[0])},{int(corner_bwd[1])})")
    return _sort_corners(
        np.array([pts[i0], pts[i1], corner_fwd, corner_bwd], dtype=np.float32))


def find_corners(pts: np.ndarray, labels: list[str]) -> np.ndarray:
    """Derive 4 parallelogram corners from hull vertices + internal/boundary labels.

    Handles 2, 3, or 4 visible (internal) vertices:
      4 internal → use directly
      3 internal → deduce 4th via D = A + C - B
      2 internal adjacent → parallel opposite side + side-line intersections
      2 internal opposite → intersect adjacent side-direction lines

    Returns (4,2) float32 array ordered [TL, TR, BR, BL].
    """
    n = len(pts)
    internal_idxs = [i for i in range(n) if labels[i] == "internal"]
    num = len(internal_idxs)
    print(f"  {num} internal vertices: {internal_idxs}")

    if num >= 4:
        return _corners_4(pts, internal_idxs[:4])
    elif num == 3:
        return _corners_3(pts, labels, internal_idxs)
    elif num == 2:
        return _corners_2(pts, labels, internal_idxs)
    else:
        raise RuntimeError(
            f"Need ≥2 internal vertices to fit a parallelogram, got {num}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

EDGE_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
]


def _extend_line(p1, p2, scale=4000):
    """Return two points that extend the segment p1→p2 far in both directions."""
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return p1, p2
    ux, uy = dx / length, dy / length
    a = (int(p1[0] - ux * scale), int(p1[1] - uy * scale))
    b = (int(p2[0] + ux * scale), int(p2[1] + uy * scale))
    return a, b


def draw_vertices(original: np.ndarray, pts: np.ndarray, labels: list[str],
                  corners: np.ndarray | None = None,
                  save_path: str = "debug_vertices.png", padding: int = 1200):
    h, w = original.shape[:2]
    canvas_h, canvas_w = h + 2 * padding, w + 2 * padding
    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

    canvas[padding:padding + h, padding:padding + w] = original

    cv2.rectangle(canvas, (padding, padding), (padding + w - 1, padding + h - 1),
                  (80, 80, 80), 2)

    offset = np.array([padding, padding], dtype=np.float32)
    shifted = pts + offset

    # Draw each hull edge as an extrapolated infinite line
    n = len(shifted)
    for i in range(n):
        p1, p2 = shifted[i], shifted[(i + 1) % n]
        col = EDGE_COLORS[i % len(EDGE_COLORS)]
        a, b = _extend_line(p1, p2)
        cv2.line(canvas, a, b, col, 1, cv2.LINE_AA)

    # Draw the synthetic bottom line if we have corners
    if corners is not None:
        shifted_corners = corners + offset
        # BL=index 3, BR=index 2 in [TL, TR, BR, BL] order
        bl, br = shifted_corners[3], shifted_corners[2]
        a, b = _extend_line(bl, br)
        cv2.line(canvas, a, b, (0, 180, 255), 2, cv2.LINE_AA)

    # Hull polygon
    poly = shifted.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [poly], isClosed=True, color=(0, 255, 255), thickness=2)

    # Hull vertices
    COLOR_INTERNAL = (0, 220, 0)
    COLOR_BOUNDARY = (0, 0, 220)
    for i, (p, lbl) in enumerate(zip(shifted, labels)):
        cx, cy = int(p[0]), int(p[1])
        col = COLOR_INTERNAL if lbl == "internal" else COLOR_BOUNDARY
        cv2.circle(canvas, (cx, cy), 10, col, -1)
        tag = f"{i} {lbl}"
        cv2.putText(canvas, tag, (cx + 14, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    # Draw derived 4 corners as white diamonds + quad
    if corners is not None:
        shifted_corners = corners + offset
        corner_labels = ["TL", "TR", "BR", "BL"]

        quad = shifted_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [quad], isClosed=True, color=(255, 255, 255), thickness=2)

        for (pt, clbl) in zip(shifted_corners, corner_labels):
            cx, cy = int(pt[0]), int(pt[1])
            diamond = np.array([
                [cx, cy - 14], [cx + 14, cy], [cx, cy + 14], [cx - 14, cy]
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [diamond], (255, 255, 255))
            cv2.putText(canvas, clbl, (cx + 18, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Warp to top-down if we have corners
    warped = None
    if corners is not None:
        dst_w, dst_h = 400, 600
        dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
        H, _ = cv2.findHomography(corners, dst_pts)
        warped = cv2.warpPerspective(original, H, (dst_w, dst_h))

    cv2.imwrite(save_path, canvas)
    if warped is not None:
        warped_path = save_path.rsplit(".", 1)[0] + "_warped.png"
        cv2.imwrite(warped_path, warped)
        print(f"  Saved: {warped_path}")
    print(f"  Saved: {save_path}")

    ncols = 2 if warped is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(24 if ncols == 2 else 16, 10),
                             gridspec_kw={"width_ratios": [3, 1]} if ncols == 2 else None)
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    title = (f"Hull vertices ({len(pts)} pts) — green=internal, red=boundary\n"
             f"Lines extrapolated; grey rect = original frame")
    if corners is not None:
        title += "\nWhite diamonds = derived 4 corners; orange = synthetic bottom"
    axes[0].set_title(title)
    axes[0].axis("off")

    if warped is not None:
        axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Top-down (homography warp)")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(original_path: str, masked_path: str, save_path: str = "debug_vertices.png"):
    print("[Court] Loading frames …")
    original = _grab_frame(original_path)
    masked   = _grab_frame(masked_path)

    print("[Court] Extracting mask …")
    mask = extract_mask(original, masked)

    print("[Court] Getting hull vertices …")
    pts = get_hull_vertices(mask)
    labels = classify_vertices(pts, mask.shape)

    for i, (p, lbl) in enumerate(zip(pts, labels)):
        print(f"    [{i}] x={int(p[0])}, y={int(p[1])}  ({lbl})")

    print("[Court] Finding 4 corners …")
    corners = find_corners(pts, labels)
    corner_labels = ["TL", "TR", "BR", "BL"]
    for clbl, pt in zip(corner_labels, corners):
        print(f"    {clbl}: ({int(pt[0])}, {int(pt[1])})")

    draw_vertices(original, pts, labels, corners=corners, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original", help="Original video, e.g. tennis-clip.mp4")
    parser.add_argument("masked",   help="Masked output video, e.g. output.mp4")
    parser.add_argument("--save", default="debug_vertices.png",
                        help="Output image path (default: debug_vertices.png)")
    args = parser.parse_args()
    run(args.original, args.masked, args.save)
