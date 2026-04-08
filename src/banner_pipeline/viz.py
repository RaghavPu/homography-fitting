"""Shared visualisation helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def color_for_obj(obj_id: int) -> np.ndarray:
    """Return a stable BGR ``uint8`` colour for *obj_id* (tab10 colourmap)."""
    cmap = plt.get_cmap("tab10")
    r, g, b, _ = cmap(obj_id % 10)
    return np.array([int(b * 255), int(g * 255), int(r * 255)], dtype=np.uint8)


def overlay_masks(
    frame_bgr: np.ndarray,
    masks_by_obj: dict[int, np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend coloured object masks onto *frame_bgr* (returns a copy)."""
    out = frame_bgr.copy()
    for obj_id, mask in masks_by_obj.items():
        mask2d = mask.squeeze().astype(bool)
        if not np.any(mask2d):
            continue
        color = color_for_obj(obj_id)
        out[mask2d] = (
            out[mask2d].astype(np.float32) * (1 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
    return out
