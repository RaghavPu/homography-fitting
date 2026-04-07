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
    """Wraps SAM2's video predictor for full-video object tracking.

    This is *not* a :class:`SegmentationModel` (which is single-frame).
    It operates on an entire video and writes a masked output.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        model_cfg: str = DEFAULT_MODEL_CFG,
        device: str = "auto",
    ) -> None:
        self._device = detect_device(device)
        print(f"[SAM2Video] Loading model on {self._device} …", flush=True)
        self._predictor = load_sam2_video_predictor(
            checkpoint, model_cfg, self._device,
        )

    @property
    def name(self) -> str:
        return "sam2_video"

    def mask_video(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
        output_path: str,
        alpha: float = 0.45,
    ) -> str:
        """Segment and track objects throughout *video_path*.

        Returns the absolute path to the written output video.
        """
        video_path = str(Path(video_path).expanduser().resolve())
        output_path = str(Path(output_path).expanduser().resolve())

        tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        try:
            # 1. Extract frames
            print("[SAM2Video] Extracting frames …", flush=True)
            frame_names = extract_all_frames(video_path, tmp_dir)
            print(f"[SAM2Video] {len(frame_names)} frames → {tmp_dir}")

            # 2. Init inference state
            inference_state = self._predictor.init_state(video_path=tmp_dir)

            # 3. Add prompts
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

            # 4. Propagate
            print("[SAM2Video] Propagating masks …", flush=True)
            video_segments: dict[int, dict[int, np.ndarray]] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in (
                self._predictor.propagate_in_video(inference_state)
            ):
                video_segments[out_frame_idx] = {
                    obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, obj_id in enumerate(out_obj_ids)
                }

            # 5. Write output video
            print("[SAM2Video] Writing output …", flush=True)
            self._write_video(
                frame_names, tmp_dir, video_segments, video_path, output_path, alpha,
            )
            print(f"[SAM2Video] Saved: {output_path}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
