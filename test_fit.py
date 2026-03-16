"""Fit a perspective-aware quadrilateral by fitting the 2 long sides
independently (no forced parallelism) and clipping at min/max extent."""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def _fit_line_pts(pts: np.ndarray):
    """Fit a line to 2D points. Returns (point_on_line, unit_direction)."""
    vx, vy, cx, cy = cv2.fitLine(pts.reshape(-1, 1, 2).astype(np.float32),
                                  cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.array([cx, cy], dtype=np.float64), np.array([vx, vy], dtype=np.float64)


def fit_quadrilateral(mask: np.ndarray) -> np.ndarray:
    """Fit a perspective-aware quadrilateral to the contour.

    1. minAreaRect → approximate orientation for splitting top/bottom
    2. Split contour into top-edge and bottom-edge points
    3. Fit a line to each independently (each keeps its natural tilt)
    4. Slide each line to the left/right contour extents using its own
       direction → 4 corners that naturally capture perspective convergence
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(np.float64)

    rect = cv2.minAreaRect(largest)
    center = np.array(rect[0])
    angle_deg = rect[2]
    w, h = rect[1]
    if w < h:
        angle_deg += 90
        w, h = h, w

    angle_rad = np.deg2rad(angle_deg)
    short_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

    rel = pts - center
    proj_short = rel @ short_dir

    top_pts = pts[proj_short >= 0]
    bot_pts = pts[proj_short < 0]
    print(f"  Top edge: {len(top_pts)} pts, Bottom edge: {len(bot_pts)} pts")

    if len(top_pts) < 5 or len(bot_pts) < 5:
        print("  [warn] Not enough points, falling back to minAreaRect")
        return cv2.boxPoints(rect).astype(np.float32)

    # Fit lines to the two long edges — each keeps its natural tilt
    pt_top, dir_top = _fit_line_pts(top_pts)
    pt_bot, dir_bot = _fit_line_pts(bot_pts)

    if dir_top @ dir_bot < 0:
        dir_bot = -dir_bot

    # Average direction only for determining left/right extent
    d_avg = (dir_top + dir_bot) / 2.0
    d_avg /= np.linalg.norm(d_avg)

    proj_long = pts @ d_avg
    p_min = proj_long.min()
    p_max = proj_long.max()

    # Evaluate each fitted line at p_min/p_max using its OWN direction.
    # Find t such that (pt + t * d) · d_avg = target
    def point_on_line(pt_on, d, target):
        denom = d @ d_avg
        if abs(denom) < 1e-9:
            return pt_on
        t = (target - pt_on @ d_avg) / denom
        return pt_on + t * d

    tl = point_on_line(pt_top, dir_top, p_min)
    tr = point_on_line(pt_top, dir_top, p_max)
    br = point_on_line(pt_bot, dir_bot, p_max)
    bl = point_on_line(pt_bot, dir_bot, p_min)

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    angle_top = np.degrees(np.arctan2(dir_top[1], dir_top[0]))
    angle_bot = np.degrees(np.arctan2(dir_bot[1], dir_bot[0]))
    print(f"  Top line angle: {angle_top:.2f}°, Bottom line angle: {angle_bot:.2f}°")
    print(f"  Convergence: {abs(angle_top - angle_bot):.2f}°")

    # Sort into TL/TR/BR/BL
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    return np.array([
        corners[np.argmin(s)],
        corners[np.argmax(d)],
        corners[np.argmax(s)],
        corners[np.argmin(d)],
    ], dtype=np.float32)


# --- Run ---
mask = cv2.imread("masks/mask_obj1.png", cv2.IMREAD_GRAYSCALE)
frame = cv2.imread("masks/frame0.png")
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

corners = fit_quadrilateral(mask)
print("\nFitted parallelogram:")
for lbl, pt in zip(["TL", "TR", "BR", "BL"], corners):
    print(f"  {lbl}: ({int(pt[0])}, {int(pt[1])})")

# --- Warp sponsor logo INTO the banner region ---
logo = cv2.imread("sponsor_logo.png", cv2.IMREAD_UNCHANGED)
if logo is None:
    raise RuntimeError("Could not read sponsor_logo.png")

# Warp banner to top-down rectangle first (we know this works)
dst_w, dst_h = 400, 120
dst_rect = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
H_forward, _ = cv2.findHomography(corners, dst_rect)
H_inverse = np.linalg.inv(H_forward)

# Sample the background color from the original banner region
warped_orig = cv2.warpPerspective(frame, H_forward, (dst_w, dst_h))
bg_color = cv2.mean(warped_orig)[:3]  # BGR mean
bg_color = tuple(int(c) for c in bg_color)
print(f"  Banner background color (BGR): {bg_color}")

# Fit a single logo tile with padding, preserving aspect ratio
logo_h, logo_w = logo.shape[:2]
PAD_V = 0.08
tile_h = int(dst_h * (1 - 2 * PAD_V))
logo_scale = tile_h / logo_h
tile_w = int(round(logo_w * logo_scale))
tile_resized = cv2.resize(logo, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

# Build canvas with background color, tile logos across it
canvas = np.full((dst_h, dst_w, 3), bg_color, dtype=np.uint8)
GAP = int(tile_w * 0.3)
total_tile = tile_w + GAP
n_tiles = max(1, (dst_w + GAP) // total_tile + 1)
start_x = (dst_w - (n_tiles * tile_w + (n_tiles - 1) * GAP)) // 2
oy = (dst_h - tile_h) // 2

for i in range(n_tiles):
    ox = start_x + i * total_tile
    if ox + tile_w > dst_w:
        break
    if ox < 0:
        continue
    if tile_resized.shape[2] == 4:
        rgb = tile_resized[:, :, :3]
        alpha = tile_resized[:, :, 3:].astype(np.float32) / 255.0
        patch = canvas[oy:oy + tile_h, ox:ox + tile_w].astype(np.float32)
        canvas[oy:oy + tile_h, ox:ox + tile_w] = (
            rgb.astype(np.float32) * alpha + patch * (1 - alpha)
        ).astype(np.uint8)
    else:
        canvas[oy:oy + tile_h, ox:ox + tile_w] = tile_resized

print(f"  Top-down rect: {dst_w}x{dst_h}, tile: {tile_w}x{tile_h}, n_tiles: {n_tiles}")

# Warp canvas back into the frame using the inverse homography
H_logo = H_inverse

# Warp the logo onto a full-size canvas
warped_logo = cv2.warpPerspective(canvas, H_logo, (frame.shape[1], frame.shape[0]))

# Build a mask for the warped region
logo_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
cv2.fillConvexPoly(logo_mask, corners.astype(np.int32), 255)

# Composite: replace the banner region with the warped logo
composited = frame.copy()
region = logo_mask > 0
composited[region] = warped_logo[region]

# Also draw the parallelogram outline on the composited image
cv2.polylines(composited, [corners.astype(np.int32).reshape(-1, 1, 2)],
              True, (0, 255, 255), 2)

cv2.imwrite("composited.png", composited)
print("Saved: composited.png")

# --- Draw ---
# Zoom region
all_x, all_y = corners[:, 0], corners[:, 1]
pad = 80
x0 = max(0, int(all_x.min()) - pad)
x1 = min(frame.shape[1], int(all_x.max()) + pad)
y0 = max(0, int(all_y.min()) - pad)
y1 = min(frame.shape[0], int(all_y.max()) + pad)

# Top-down warp of the original banner
dst_w, dst_h = 400, 120
dst_rect = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
H_td, _ = cv2.findHomography(corners, dst_rect)
warped_topdown = cv2.warpPerspective(frame, H_td, (dst_w, dst_h))

fig, axes = plt.subplots(1, 3, figsize=(22, 6),
                         gridspec_kw={"width_ratios": [2, 2, 1]})
# Original zoomed with outline
vis = frame.copy()
cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
cv2.polylines(vis, [corners.astype(np.int32).reshape(-1, 1, 2)],
              True, (0, 255, 255), 2)
axes[0].imshow(cv2.cvtColor(vis[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
axes[0].set_title("Original + fit")
axes[0].axis("off")

# Composited zoomed
axes[1].imshow(cv2.cvtColor(composited, cv2.COLOR_BGR2RGB))
axes[1].set_title("Logo warped into banner region")
axes[1].axis("off")

# Top-down
axes[2].imshow(cv2.cvtColor(warped_topdown, cv2.COLOR_BGR2RGB))
axes[2].set_title("Top-down view")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("test_fit_compare.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: test_fit_compare.png")
