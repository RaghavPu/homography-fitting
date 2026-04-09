"""SAM3 video predictor — multi-frame object tracking via SAM 3.1.

Drop-in replacement for SAM2VideoSegmenter.  The public ``segment_video``
method returns the same ``(video_segments, frame_dir, frame_names)`` tuple
so ``pipeline.py`` needs no changes.

SAM 3 uses a session-based ``handle_request`` API instead of SAM 2's
``init_state / add_new_points_or_box / propagate_in_video`` pattern.
Point prompts are passed the same way (x, y coords + labels), so the
existing ``ObjectPrompt`` dataclass is reused unchanged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from banner_pipeline.device import detect_device, load_sam3_video_predictor
from banner_pipeline.io import extract_all_frames
from banner_pipeline.segment.base import ObjectPrompt

# Default SAM 3.1 checkpoint / config.
# Assumes checkpoints are stored under sam3/checkpoints/ (same convention
# as the SAM2 setup in this repo).
DEFAULT_CHECKPOINT = "sam3/checkpoints/sam3p1.pt"


class SAM3VideoSegmenter:
    """Wraps SAM 3's video predictor for full-video object tracking.

    Uses SAM 3.1's Object Multiplex for joint multi-object tracking,
    which is significantly faster than tracking objects sequentially.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "auto",
    ) -> None:
        self._device = detect_device(device)
        print(f"[SAM3Video] Loading model on {self._device} …", flush=True)
        self._predictor = load_sam3_video_predictor(checkpoint, self._device)

    @property
    def name(self) -> str:
        return "sam3_video"

    # ------------------------------------------------------------------
    # Core propagation
    # ------------------------------------------------------------------

    def _propagate(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[dict[int, dict[int, np.ndarray]], str, list[str]]:
        """Start a SAM 3 session, add prompts, propagate masks.

        Returns
        -------
        video_segments : dict[frame_idx, dict[obj_id, np.ndarray]]
            Per-frame binary masks for each tracked object.
        frame_dir : str
            Temporary directory with extracted JPEG frames.
            **Caller is responsible for cleanup** (``shutil.rmtree``).
        frame_names : list[str]
            Sorted frame filenames within *frame_dir*.
        """
        video_path = str(Path(video_path).expanduser().resolve())

        # Extract frames (SAM3's video predictor also accepts a JPEG folder).
        frame_dir = tempfile.mkdtemp(prefix="sam3_frames_")
        print("[SAM3Video] Extracting frames …", flush=True)
        frame_names = extract_all_frames(video_path, frame_dir)
        print(f"[SAM3Video] {len(frame_names)} frames → {frame_dir}")

        # Start a session — SAM 3 accepts either an MP4 or a JPEG folder.
        print("[SAM3Video] Starting session …", flush=True)
        response = self._predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frame_dir,
            )
        )
        session_id = response["session_id"]

        # Add all prompts.  SAM 3 accepts points the same way SAM 2 does.
        for prompt in prompts:
            req: dict = dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt.frame_idx,
                obj_id=prompt.obj_id,
            )
            if prompt.points is not None and len(prompt.points) > 0:
                req["points"] = prompt.points.tolist()
                req["labels"] = prompt.labels.tolist()
            if prompt.box is not None:
                req["box"] = prompt.box.tolist()
            self._predictor.handle_request(request=req)
            print(f"[SAM3Video] Prompt obj_id={prompt.obj_id} frame={prompt.frame_idx}")

        # Propagate across all frames.
        print("[SAM3Video] Propagating masks …", flush=True)
        response = self._predictor.handle_request(
            request=dict(
                type="propagate",
                session_id=session_id,
            )
        )

        # Parse the response into the same structure SAM2VideoSegmenter returns:
        # dict[frame_idx -> dict[obj_id -> binary mask (H, W) uint8]]
        video_segments = _parse_propagate_response(response, len(frame_names))

        # Close the session to free GPU memory.
        self._predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

        return video_segments, frame_dir, frame_names

    # ------------------------------------------------------------------
    # Public API — identical signature to SAM2VideoSegmenter
    # ------------------------------------------------------------------

    def segment_video(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[dict[int, dict[int, np.ndarray]], str, list[str]]:
        """Track objects across all frames and return per-frame masks.

        Returns
        -------
        video_segments : dict[frame_idx, dict[obj_id, np.ndarray]]
            Per-frame binary masks.
        frame_dir : str
            Temporary directory with extracted JPEG frames.
            **Caller is responsible for cleanup** (``shutil.rmtree``).
        frame_names : list[str]
            Sorted frame filenames within *frame_dir*.
        """
        return self._propagate(video_path, prompts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_propagate_response(
    response: dict,
    num_frames: int,
) -> dict[int, dict[int, np.ndarray]]:
    """Convert SAM 3's propagate response into the SAM2-compatible format.

    SAM 3 returns masks in ``response["outputs"]``, structured as a list of
    per-frame dicts, each containing ``frame_index``, ``obj_id``, and
    ``mask`` (a boolean/float numpy array or tensor).

    We normalise everything to ``uint8`` binary masks (0 / 255) with shape
    ``(H, W)``, keyed by ``dict[frame_idx][obj_id]``.
    """
    video_segments: dict[int, dict[int, np.ndarray]] = {}

    outputs = response.get("outputs", [])

    for entry in outputs:
        frame_idx = int(entry["frame_index"])
        obj_id = int(entry["obj_id"])
        mask = entry["mask"]

        # Normalise tensor / ndarray → (H, W) uint8 binary mask.
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask)

        # Squeeze any extra dimensions (e.g. (1, H, W) or (1, 1, H, W)).
        mask = mask.squeeze()

        # Binarise (handles float logits, bool, or uint8 inputs).
        if mask.dtype != np.uint8:
            mask = (mask > 0.0).astype(np.uint8) * 255
        else:
            mask = (mask > 0).astype(np.uint8) * 255

        if frame_idx not in video_segments:
            video_segments[frame_idx] = {}
        video_segments[frame_idx][obj_id] = mask

    # Ensure every frame index has an entry (empty dict for frames with no masks).
    for i in range(num_frames):
        video_segments.setdefault(i, {})

    return video_segments
