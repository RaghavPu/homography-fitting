"""
find_diff_region.py

Compares two videos frame-by-frame and outputs the regions that differ.
Groups contiguous diff blobs into clusters and tracks each across frames.

Usage:
    python find_diff_region.py outputs/boards.mp4 outputs/players.mp4 outputs/diff_region.mp4
"""

import argparse
import cv2
import numpy as np
from collections import defaultdict


def build_diff_mask(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int = 30) -> np.ndarray:
    diff = cv2.absdiff(frame_a, frame_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def find_diff_region(video_a: str, video_b: str, output_path: str, threshold: int = 30, min_area: int = 500):
    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)

    fps = cap_a.get(cv2.CAP_PROP_FPS)
    w = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Accumulate union bbox per region across frames
    region_bboxes = defaultdict(lambda: [np.inf, np.inf, 0, 0])  # x1,y1,x2,y2

    COLORS = [(0, 255, 100), (0, 100, 255), (255, 100, 0), (255, 0, 180)]

    frame_idx = 0
    while True:
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            break

        mask = build_diff_mask(frame_a, frame_b, threshold)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        # Sort by area descending — assign stable region IDs by vertical position of centroid
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        output_frame = frame_a.copy()

        # Overlay diff mask lightly
        colored_mask = np.zeros_like(frame_a)

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            cy = y + bh // 2

            # Region ID: bucket by vertical thirds of the frame
            region_id = int(cy / h * 3)  # 0=top, 1=mid, 2=bottom
            color = COLORS[region_id % len(COLORS)]

            cv2.drawContours(colored_mask, [contour], -1, color, -1)
            cv2.rectangle(output_frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(output_frame, f"R{region_id}", (x + 4, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Accumulate global bbox
            b = region_bboxes[region_id]
            b[0] = min(b[0], x)
            b[1] = min(b[1], y)
            b[2] = max(b[2], x + bw)
            b[3] = max(b[3], y + bh)

        output_frame = cv2.addWeighted(output_frame, 1.0, colored_mask, 0.4, 0)
        out.write(output_frame)
        frame_idx += 1

    cap_a.release()
    cap_b.release()
    out.release()

    print(f"Processed {frame_idx} frames\n")
    print("Diff regions (stable bounding boxes across all frames):")
    for rid, (x1, y1, x2, y2) in sorted(region_bboxes.items()):
        label = ["top", "middle", "bottom"][rid] if rid < 3 else str(rid)
        print(f"  Region {rid} ({label}): x={int(x1)}–{int(x2)}, y={int(y1)}–{int(y2)}, "
              f"size={int(x2-x1)}x{int(y2-y1)}")
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_a", help="First video (e.g. boards.mp4)")
    parser.add_argument("video_b", help="Second video (e.g. players.mp4)")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--threshold", type=int, default=30,
                        help="Pixel difference threshold (default: 30)")
    parser.add_argument("--min-area", type=int, default=500,
                        help="Minimum contour area to consider (default: 500)")
    args = parser.parse_args()

    find_diff_region(args.video_a, args.video_b, args.output, args.threshold, args.min_area)
