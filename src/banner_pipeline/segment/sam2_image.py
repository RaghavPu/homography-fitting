"""SAM2 image predictor — single-frame segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline.device import detect_device, load_sam2_image_predictor
from banner_pipeline.segment.base import ObjectPrompt, SegmentationModel

# Default checkpoint / config for the *tiny* model (fast, single-frame).
DEFAULT_CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


class SAM2ImageSegmenter(SegmentationModel):
    """Wraps ``SAM2ImagePredictor`` behind the :class:`SegmentationModel` interface."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        model_cfg: str = DEFAULT_MODEL_CFG,
        device: str = "auto",
    ) -> None:
        self._device = detect_device(device)
        print(f"[SAM2Image] Loading model on {self._device} …", flush=True)
        self._predictor = load_sam2_image_predictor(
            checkpoint, model_cfg, self._device,
        )

    # -- SegmentationModel interface -----------------------------------------

    @property
    def name(self) -> str:
        return "sam2_image"

    def segment(
        self,
        frame_bgr: np.ndarray,
        prompts: list[ObjectPrompt],
    ) -> dict[int, np.ndarray]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(frame_rgb)

        masks_out: dict[int, np.ndarray] = {}
        for prompt in prompts:
            masks, scores, _ = self._predictor.predict(
                point_coords=prompt.points,
                point_labels=prompt.labels,
                box=prompt.box,
                multimask_output=True,
            )
            masks_out[prompt.obj_id] = masks[np.argmax(scores)]
            print(
                f"  obj {prompt.obj_id}: score={scores.max():.3f}",
                flush=True,
            )
        return masks_out
