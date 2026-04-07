"""Court-specific mask extraction from video frame diffs."""

from __future__ import annotations

import cv2
import numpy as np


def extract_mask(original: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Binary mask from pixel-diff between *original* and *masked* frames."""
    diff = np.abs(original.astype(np.float32) - masked.astype(np.float32)).max(axis=2)
    _, mask = cv2.threshold(diff.astype(np.uint8), 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask
