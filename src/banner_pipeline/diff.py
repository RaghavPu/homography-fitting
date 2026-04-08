"""Frame-diff detection between two videos (from find_diff_region.py)."""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np


def build_diff_mask(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    threshold: int = 30,
) -> np.ndarray:
    """Compute a binary diff mask between two frames."""
    diff = cv2.absdiff(frame_a, frame_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def find_diff_region(
    video_a: str,
    video_b: str,
    output_path: str,
    threshold: int = 30,
    min_area: int = 500,
) -> dict[int, list[float]]:
    """Compare two videos frame-by-frame and write a diff-overlay video.

    Returns a dict mapping ``region_id → [x1, y1, x2, y2]`` union bboxes.
    """
    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)

    fps = cap_a.get(cv2.CAP_PROP_FPS)
    w = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    region_bboxes: dict[int, list[float]] = defaultdict(
        lambda: [float("inf"), float("inf"), 0.0, 0.0],
    )
    COLORS = [(0, 255, 100), (0, 100, 255), (255, 100, 0), (255, 0, 180)]

    frame_idx = 0
    while True:
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            break

        mask = build_diff_mask(frame_a, frame_b, threshold)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(
            [c for c in contours if cv2.contourArea(c) >= min_area],
            key=cv2.contourArea,
            reverse=True,
        )

        output_frame = frame_a.copy()
        colored_mask = np.zeros_like(frame_a)

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            cy = y + bh // 2
            region_id = int(cy / h * 3)
            color = COLORS[region_id % len(COLORS)]

            cv2.drawContours(colored_mask, [contour], -1, color, -1)
            cv2.rectangle(output_frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                output_frame, f"R{region_id}", (x + 4, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

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

    print(f"Processed {frame_idx} frames → {output_path}")
    return dict(region_bboxes)
