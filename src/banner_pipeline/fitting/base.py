"""Base class for quadrilateral fitting algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class QuadFitter(ABC):
    """Interface for algorithms that fit a quadrilateral to a binary mask."""

    @abstractmethod
    def fit(self, mask: np.ndarray, **kwargs) -> np.ndarray | None:
        """Fit a quadrilateral to *mask*.

        Returns a ``(4, 2)`` float32 array ordered ``[TL, TR, BR, BL]``,
        or ``None`` on failure.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name for logging / metrics."""
        ...
