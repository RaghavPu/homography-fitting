"""Inpaint + luminosity-matching compositor (from banner_segment.py).

Strategy: inpaint old region → build logo canvas with aspect-aware
sizing → match luminosity in LAB space → alpha-blend with soft edges.
"""

from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline.composite.base import Compositor


class InpaintCompositor(Compositor):
    """Inpaints the old logo away, then warps the new one with luminosity matching."""

    @property
    def name(self) -> str:
        return "inpaint"

    def composite(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        padding: float = kwargs.get("padding", 0.05)

        # Ensure BGRA overlay.
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        h, w = frame.shape[:2]

        # --- Step 1: inpaint the masked region ---
        if mask is not None:
            mask_u8 = (mask > 0).astype(np.uint8) * 255
            dilate_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_u8 = cv2.dilate(mask_u8, dilate_kern)
            inpainted = cv2.inpaint(frame, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        else:
            inpainted = frame.copy()

        # --- Step 2: build logo + alpha canvases ---
        w_top = np.linalg.norm(corners[1] - corners[0])
        w_bot = np.linalg.norm(corners[2] - corners[3])
        h_left = np.linalg.norm(corners[3] - corners[0])
        h_right = np.linalg.norm(corners[2] - corners[1])
        avg_w = (w_top + w_bot) / 2
        avg_h = (h_left + h_right) / 2
        scale_up = max(1.0, 500.0 / max(float(avg_w), float(avg_h)))
        canvas_w = max(int(avg_w * scale_up), 1)
        canvas_h = max(int(avg_h * scale_up), 1)

        rgb_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        logo_h, logo_w = overlay.shape[:2]
        pad_w = int(canvas_w * padding)
        pad_h = int(canvas_h * padding)
        scale = min((canvas_w - 2 * pad_w) / logo_w, (canvas_h - 2 * pad_h) / logo_h)
        new_w, new_h = int(logo_w * scale), int(logo_h * scale)
        logo_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x0 = (canvas_w - new_w) // 2
        y0 = (canvas_h - new_h) // 2
        rgb_canvas[y0 : y0 + new_h, x0 : x0 + new_w] = logo_resized[:, :, :3]
        alpha_canvas[y0 : y0 + new_h, x0 : x0 + new_w] = logo_resized[:, :, 3]

        # --- Step 3: warp into frame space ---
        src = np.array(
            [[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(src, corners.astype(np.float32))
        warped_rgb = cv2.warpPerspective(rgb_canvas, H, (w, h))
        warped_alpha = cv2.warpPerspective(alpha_canvas, H, (w, h))

        # --- Step 4: luminosity matching (LAB) ---
        logo_pixels = warped_alpha > 0
        if logo_pixels.any() and mask is not None:
            orig_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            orig_mask_l = orig_lab[mask > 0, 0]
            orig_l_lo, orig_l_hi = np.percentile(orig_mask_l, [10, 90])

            new_lab = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
            new_l = new_lab[logo_pixels, 0]
            new_l_lo, new_l_hi = np.percentile(new_l, [10, 90])

            s = (orig_l_hi - orig_l_lo) / (new_l_hi - new_l_lo) if new_l_hi - new_l_lo > 1 else 1.0
            new_lab[logo_pixels, 0] = np.clip(
                (new_l - new_l_lo) * s + orig_l_lo,
                0,
                255,
            )
            warped_rgb = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # --- Step 5: soft-edge alpha blend ---
        warped_alpha = cv2.GaussianBlur(warped_alpha, (5, 5), 1.0)
        a = (warped_alpha.astype(np.float32) / 255.0)[..., None]
        result = (
            warped_rgb.astype(np.float32) * a + inpainted.astype(np.float32) * (1.0 - a)
        ).astype(np.uint8)

        return result
