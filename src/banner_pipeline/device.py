"""Torch device detection and SAM2 model loading."""

from __future__ import annotations

import os
import sys

import torch

# Apple MPS compatibility: fall back to CPU for unsupported ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def detect_device(override: str = "auto") -> torch.device:
    """Select the best available compute device.

    *override* can be ``"auto"`` (default), ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if override != "auto":
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_torch_backend(device: torch.device) -> None:
    """Enable performance optimisations for *device* (autocast, TF32, etc.)."""
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def _ensure_sam2_importable() -> None:
    """Make sure the ``sam2`` package is importable.

    First tries a regular import.  If that fails, checks for a local ``sam2/``
    directory and temporarily adjusts ``sys.path``.
    """
    try:
        import sam2  # noqa: F401

        return
    except ImportError:
        pass

    # Look for a local clone of the sam2 repo.
    repo = os.path.join(os.getcwd(), "sam2")
    if os.path.isdir(repo) and repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        import sam2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAM2 is not installed. Either:\n"
            "  1. pip install git+https://github.com/facebookresearch/sam2.git\n"
            "  2. Clone into ./sam2 and run: pip install -e ./sam2"
        ) from exc


def load_sam2_image_predictor(
    checkpoint: str,
    model_cfg: str,
    device: torch.device | None = None,
):
    """Build and return a ``SAM2ImagePredictor`` ready for inference."""
    _ensure_sam2_importable()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device is None:
        device = detect_device()
    setup_torch_backend(device)

    model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def load_sam2_video_predictor(
    checkpoint: str,
    model_cfg: str,
    device: torch.device | None = None,
):
    """Build and return a SAM2 video predictor ready for inference."""
    _ensure_sam2_importable()
    from sam2.build_sam import build_sam2_video_predictor

    if device is None:
        device = detect_device()
    setup_torch_backend(device)

    return build_sam2_video_predictor(model_cfg, checkpoint, device=device)


def _ensure_sam3_importable() -> None:
    """Make sure the ``sam3`` package is importable.

    Mirrors the SAM2 helper: tries a regular import first, then checks for a
    local ``sam3/`` directory and adjusts ``sys.path`` if needed.
    """
    try:
        import sam3  # noqa: F401

        return
    except ImportError:
        pass

    repo = os.path.join(os.getcwd(), "sam3")
    if os.path.isdir(repo) and repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        import sam3  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAM3 is not installed. Either:\n"
            "  1. pip install git+https://github.com/facebookresearch/sam3.git\n"
            "  2. Clone into ./sam3 and run: pip install -e ./sam3"
        ) from exc


def load_sam3_video_predictor(
    checkpoint: str,
    device: torch.device | None = None,
):
    """Build and return a SAM 3 video predictor ready for inference.

    SAM 3 does not take a separate model_cfg argument — the architecture
    is baked into the package and selected by the checkpoint file.
    """
    _ensure_sam3_importable()
    from sam3.model_builder import build_sam3_video_predictor

    if device is None:
        device = detect_device()
    setup_torch_backend(device)

    return build_sam3_video_predictor(checkpoint=checkpoint, device=str(device))
