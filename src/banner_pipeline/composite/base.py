"""Base class for compositing strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Compositor(ABC):
    """Interface for logo/overlay compositing into a detected region."""

    @abstractmethod
    def composite(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Warp *overlay* into the region defined by *corners*.

        Returns the composited frame.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name for logging / metrics."""
        ...
