"""Oriented-homography + alpha-blend compositor (from region_overlay.py).

Strategy: uses the camera-aware oriented homography to warp the overlay
with correct physical aspect ratio, then alpha-composites into the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline.composite.base import Compositor


class AlphaCompositor(Compositor):
    """Composites using oriented homography for aspect-ratio-correct warping."""

    @property
    def name(self) -> str:
        return "alpha"

    def composite(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Requires ``homo`` (oriented-homography dict) in *kwargs*."""
        homo: dict = kwargs["homo"]
        padding: float = kwargs.get("padding", 0.05)

        dst_w, dst_h = homo["dst_w"], homo["dst_h"]
        H_final = homo["H"]

        avail_w = int(dst_w * (1 - 2 * padding))
        avail_h = int(dst_h * (1 - 2 * padding))

        ov_h, ov_w = overlay.shape[:2]
        scale = min(avail_w / ov_w, avail_h / ov_h)
        new_w = max(1, int(round(ov_w * scale)))
        new_h = max(1, int(round(ov_h * scale)))
        ov_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Sample background colour from the frame's banner region.
        H_to_rect, _ = cv2.findHomography(corners, homo["dst_rect"])
        warped_orig = cv2.warpPerspective(frame, H_to_rect, (dst_w, dst_h))
        bg_color = tuple(int(c) for c in cv2.mean(warped_orig)[:3])

        canvas = np.full((dst_h, dst_w, 3), bg_color, dtype=np.uint8)
        ox = (dst_w - new_w) // 2
        oy = (dst_h - new_h) // 2

        if ov_resized.ndim == 3 and ov_resized.shape[2] == 4:
            rgb = ov_resized[:, :, :3].astype(np.float32)
            alpha = ov_resized[:, :, 3:].astype(np.float32) / 255.0
            patch = canvas[oy : oy + new_h, ox : ox + new_w].astype(np.float32)
            canvas[oy : oy + new_h, ox : ox + new_w] = (
                rgb * alpha + patch * (1 - alpha)
            ).astype(np.uint8)
        else:
            canvas[oy : oy + new_h, ox : ox + new_w] = ov_resized[:, :, :3]

        warped_canvas = cv2.warpPerspective(
            canvas, H_final, (frame.shape[1], frame.shape[0]),
        )

        region_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(region_mask, corners.astype(np.int32), 255)

        result = frame.copy()
        result[region_mask > 0] = warped_canvas[region_mask > 0]
        return result
