"""Base class for segmentation models and shared data types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectPrompt:
    """Defines a segmentation prompt for one tracked object.

    Attributes
    ----------
    obj_id:    Unique integer ID (any positive int).
    points:    (N, 2) float32 array of (x, y) click coordinates.
    labels:    (N,) int32 array; 1 = positive, 0 = negative click.
    frame_idx: Frame index these prompts apply to (default 0).
    box:       Optional (4,) float32 ``[x0, y0, x1, y1]`` bounding box.
    """

    obj_id: int
    points: np.ndarray
    labels: np.ndarray
    frame_idx: int = 0
    box: np.ndarray | None = None


class SegmentationModel(ABC):
    """Interface for single-frame segmentation models."""

    @abstractmethod
    def segment(
        self,
        frame_bgr: np.ndarray,
        prompts: list[ObjectPrompt],
    ) -> dict[int, np.ndarray]:
        """Segment *frame_bgr* given click/box prompts.

        Returns ``dict[obj_id → binary mask (H, W)]``.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for logging / metrics."""
        ...
