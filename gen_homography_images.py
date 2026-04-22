"""
Generate the 4 images for the Homography & Perspective Geometry slide.

1. Opens the original banner image
2. User clicks 4 corners: TL → TR → BR → BL
3. Computes homography and generates:
   - 1_original.png   (original frame)
   - 2_bbox.png        (frame + parallelogram overlay)
   - 3_rectified.png   (perspective-corrected flat banner)
   - 4_overlay.png     (new logo warped into original perspective)

Usage:
    venv/bin/python gen_homography_images.py
"""

import os
import cv2
import numpy as np
from banner_segment import composite_logo

ORIGINAL_PATH = "/Users/praghav/.cursor/projects/Users-praghav-roi-tracking-sam-test/assets/Screenshot_2026-03-18_at_8.59.09_PM-7dc1aa73-a3de-4bdc-8992-82af94536866.png"
LOGO_PATH = "ferrari_white.png"
OUT_DIR = "../midterm-pres/public/homography"

CORNER_LABELS = ["TL", "TR", "BR", "BL"]


def collect_corners(frame):
    """Show image, let user click 4 corners. Returns np.array of shape (4,2)."""
    corners = []
    display = frame.copy()
    win = "Click 4 corners: TL -> TR -> BR -> BL  (Enter=done, Esc=cancel)"

    def redraw():
        img = frame.copy()
        for i, (x, y) in enumerate(corners):
            cv2.circle(img, (x, y), 5, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(img, CORNER_LABELS[i], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        if len(corners) > 1:
            for i in range(len(corners) - 1):
                cv2.line(img, corners[i], corners[i + 1], (0, 220, 255), 2, cv2.LINE_AA)
        if len(corners) == 4:
            cv2.line(img, corners[3], corners[0], (0, 220, 255), 2, cv2.LINE_AA)
        n = len(corners)
        label = f"Click {CORNER_LABELS[n]}" if n < 4 else "Press Enter to confirm"
        cv2.rectangle(img, (0, 0), (frame.shape[1], 25), (30, 30, 30), -1)
        cv2.putText(img, label, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            print(f"  {CORNER_LABELS[len(corners)-1]}: ({x}, {y})")
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1] * 3, 1200), min(frame.shape[0] * 3, 800))
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    print("[UI] Click 4 corners: TL → TR → BR → BL. Press Enter when done.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32) and len(corners) == 4:
            break
        if key == 27:
            corners.clear()
            break

    cv2.destroyAllWindows()
    if not corners:
        return None
    return np.array(corners, dtype=np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    frame = cv2.imread(ORIGINAL_PATH)
    if frame is None:
        raise RuntimeError(f"Could not read: {ORIGINAL_PATH}")
    h, w = frame.shape[:2]
    print(f"Input: {w}x{h}")

    # Pass --corners TLx,TLy TRx,TRy BRx,BRy BLx,BLy to skip clicking
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--corners" and len(sys.argv) >= 6:
        pts = []
        for s in sys.argv[2:6]:
            x, y = s.split(",")
            pts.append([float(x), float(y)])
        corners = np.array(pts, dtype=np.float32)
        print(f"Using provided corners: {corners.tolist()}")
    else:
        corners = collect_corners(frame)
        if corners is None:
            print("Cancelled.")
            return
        print(f"\nCorners: {corners.tolist()}")
        print(f"Re-run with: --corners {' '.join(f'{int(c[0])},{int(c[1])}' for c in corners)}")

    # --- Image 1: Original ---
    cv2.imwrite(os.path.join(OUT_DIR, "1_original.png"), frame)
    print("Saved 1_original.png")

    # --- Image 2: Frame + bounding box ---
    bbox_frame = frame.copy()
    quad = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(bbox_frame, [quad], True, (0, 220, 255), 2, cv2.LINE_AA)
    for i, pt in enumerate(corners):
        cv2.circle(bbox_frame, (int(pt[0]), int(pt[1])), 5, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT_DIR, "2_bbox.png"), bbox_frame)
    print("Saved 2_bbox.png")

    # --- Image 3: Rectified (flattened) via homography ---
    w_top = np.linalg.norm(corners[1] - corners[0])
    w_bot = np.linalg.norm(corners[2] - corners[3])
    h_left = np.linalg.norm(corners[3] - corners[0])
    h_right = np.linalg.norm(corners[2] - corners[1])
    rect_w = int((w_top + w_bot) / 2)
    rect_h = int((h_left + h_right) / 2)

    dst_rect = np.array([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]], dtype=np.float32)
    H_rect, _ = cv2.findHomography(corners, dst_rect)
    rectified = cv2.warpPerspective(frame, H_rect, (rect_w, rect_h))
    cv2.imwrite(os.path.join(OUT_DIR, "3_rectified.png"), rectified)
    print(f"Saved 3_rectified.png ({rect_w}x{rect_h})")

    # --- Image 4: Meta logo composited via composite_logo (inpaint + warp + luminosity) ---
    # Build a filled mask from the clicked corners
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)

    result = composite_logo(
        frame, corners, LOGO_PATH, mask=mask.astype(bool),
        save_path=os.path.join(OUT_DIR, "4_overlay.png"),
    )
    print("Saved 4_overlay.png (via composite_logo)")

    print("\nDone! All 4 images in", OUT_DIR)


if __name__ == "__main__":
    main()
