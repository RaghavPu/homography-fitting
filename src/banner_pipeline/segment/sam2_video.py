"""SAM2 video predictor — multi-frame object tracking."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from banner_pipeline.device import detect_device, load_sam2_video_predictor
from banner_pipeline.io import extract_all_frames, get_video_fps
from banner_pipeline.segment.base import ObjectPrompt
from banner_pipeline.viz import overlay_masks

# Default checkpoint / config for the *large* model (video tracking).
DEFAULT_CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_large.pt"
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class SAM2VideoSegmenter:
    """Wraps SAM2's video predictor for full-video object tracking."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        model_cfg: str = DEFAULT_MODEL_CFG,
        device: str = "auto",
    ) -> None:
        self._device = detect_device(device)
        print(f"[SAM2Video] Loading model on {self._device} …", flush=True)
        self._predictor = load_sam2_video_predictor(
            checkpoint,
            model_cfg,
            self._device,
        )

    @property
    def name(self) -> str:
        return "sam2_video"

    # ------------------------------------------------------------------
    # Core propagation (shared by segment_video and mask_video)
    # ------------------------------------------------------------------

    def _propagate(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[dict[int, dict[int, np.ndarray]], str, list[str]]:
        """Extract frames, add prompts, propagate masks.

        Returns
        -------
        video_segments : dict[frame_idx, dict[obj_id, np.ndarray]]
            Per-frame binary masks for each tracked object.
        frame_dir : str
            Path to the temporary directory containing extracted JPEG frames.
        frame_names : list[str]
            Sorted list of frame filenames.
        """
        video_path = str(Path(video_path).expanduser().resolve())

        frame_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        print("[SAM2Video] Extracting frames …", flush=True)
        frame_names = extract_all_frames(video_path, frame_dir)
        print(f"[SAM2Video] {len(frame_names)} frames → {frame_dir}")

        # Init inference state.
        inference_state = self._predictor.init_state(video_path=frame_dir)

        # Add prompts.
        for prompt in prompts:
            kwargs: dict = dict(
                inference_state=inference_state,
                frame_idx=prompt.frame_idx,
                obj_id=prompt.obj_id,
            )
            if prompt.points is not None:
                kwargs["points"] = prompt.points
                kwargs["labels"] = prompt.labels
            if prompt.box is not None:
                kwargs["box"] = prompt.box
            self._predictor.add_new_points_or_box(**kwargs)
            print(f"[SAM2Video] Prompt obj_id={prompt.obj_id} frame={prompt.frame_idx}")

        # Propagate.
        print("[SAM2Video] Propagating masks …", flush=True)
        video_segments: dict[int, dict[int, np.ndarray]] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self._predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(out_obj_ids)
            }

        return video_segments, frame_dir, frame_names

    # ------------------------------------------------------------------
    # Public API
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

    def mask_video(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
        output_path: str,
        alpha: float = 0.45,
    ) -> str:
        """Segment, track, and write an overlay video.

        Returns the absolute path to the written output video.
        """
        output_path = str(Path(output_path).expanduser().resolve())

        video_segments, frame_dir, frame_names = self._propagate(video_path, prompts)
        try:
            print("[SAM2Video] Writing output …", flush=True)
            self._write_video(
                frame_names,
                frame_dir,
                video_segments,
                video_path,
                output_path,
                alpha,
            )
            print(f"[SAM2Video] Saved: {output_path}")
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

        return output_path

    # ------------------------------------------------------------------

    @staticmethod
    def _write_video(
        frame_names: list[str],
        frame_dir: str,
        video_segments: dict[int, dict[int, np.ndarray]],
        source_video: str,
        output_path: str,
        alpha: float,
    ) -> None:
        first_bgr = cv2.imread(os.path.join(frame_dir, frame_names[0]))
        if first_bgr is None:
            raise RuntimeError("Could not read first frame.")
        h, w = first_bgr.shape[:2]
        fps = get_video_fps(source_video)

        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame_idx, fname in enumerate(frame_names):
            frame_bgr = cv2.imread(os.path.join(frame_dir, fname))
            if frame_bgr is None:
                raise RuntimeError(f"Could not read frame {frame_idx}: {fname}")
            masks_by_obj = video_segments.get(frame_idx, {})
            frame_bgr = overlay_masks(frame_bgr, masks_by_obj, alpha)
            writer.write(frame_bgr)

        writer.release()
