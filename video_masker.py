# -*- coding: utf-8 -*-
"""
video_masker.py
---------------
Simple module for SAM2-based video object masking.

Usage example
-------------
    from video_masker import VideoMasker, ObjectPrompt
    import numpy as np

    masker = VideoMasker(
        checkpoint="sam2/checkpoints/sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    )

    prompts = [
        ObjectPrompt(obj_id=1, points=np.array([[1000, 100]], dtype=np.float32), labels=np.array([1])),
        ObjectPrompt(obj_id=2, points=np.array([[1000, 910]], dtype=np.float32), labels=np.array([1])),
    ]

    masker.mask_video("input.mp4", prompts, "output.mp4")
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Apple MPS compatibility: fall back to CPU for unsupported ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ObjectPrompt:
    """Defines a segmentation prompt for one tracked object.

    Attributes:
        obj_id:    Unique integer ID for this object (any positive int).
        points:    (N, 2) float32 array of (x, y) click coordinates.
        labels:    (N,) int32 array; 1 = positive click, 0 = negative click.
        frame_idx: Which frame index these prompts are applied to (default 0).
        box:       Optional (4,) float32 array [x0, y0, x1, y1] bounding box.
                   If provided together with points, both are sent to SAM2.
    """
    obj_id: int
    points: np.ndarray
    labels: np.ndarray
    frame_idx: int = 0
    box: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_fps(video_path: str, default: float = 30.0) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or np.isnan(fps) or fps < 1e-3:
        return default
    return fps


def _extract_frames(video_path: str, out_dir: str) -> list[str]:
    """Extract JPEG frames from *video_path* into *out_dir*.

    Returns a sorted list of frame filenames (not full paths).
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-q:v", "2",
        "-start_number", "0",
        os.path.join(out_dir, "%05d.jpg"),
        "-y",           # overwrite without asking
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)

    frame_names = sorted(
        [p for p in os.listdir(out_dir) if Path(p).suffix.lower() in {".jpg", ".jpeg"}],
        key=lambda p: int(Path(p).stem),
    )
    if not frame_names:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return frame_names


def _color_for_obj(obj_id: int) -> np.ndarray:
    """Return a stable BGR uint8 color for *obj_id*."""
    cmap = plt.get_cmap("tab10")
    r, g, b, _ = cmap(obj_id % 10)
    return np.array([int(b * 255), int(g * 255), int(r * 255)], dtype=np.uint8)


def _overlay_masks(
    frame_bgr: np.ndarray,
    masks_by_obj: dict[int, np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend all object masks onto *frame_bgr* (in-place copy)."""
    out = frame_bgr.copy()
    for obj_id, mask in masks_by_obj.items():
        mask2d = mask.squeeze().astype(bool)
        if not np.any(mask2d):
            continue
        color = _color_for_obj(obj_id)
        out[mask2d] = (
            out[mask2d].astype(np.float32) * (1 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VideoMasker:
    """Wraps SAM2 video predictor for one-call video masking.

    Parameters
    ----------
    checkpoint:
        Path to the SAM2 model checkpoint file (.pt).
    model_cfg:
        Path to the SAM2 model config YAML (relative to the sam2 package
        root or absolute).  e.g. "configs/sam2.1/sam2.1_hiera_l.yaml"
    device:
        "auto" (default) detects CUDA → MPS → CPU, or pass "cuda" / "mps" / "cpu".
    """

    def __init__(
        self,
        checkpoint: str,
        model_cfg: str,
        device: str = "auto",
    ) -> None:
        # The sam2/ repo directory in the cwd would shadow the installed package
        # if '' or cwd is ahead of site-packages in sys.path.  Remove them temporarily.
        import sys
        _cwd = os.getcwd()
        _removed = [p for p in sys.path if p in ("", _cwd)]
        for _p in _removed:
            sys.path.remove(_p)
        try:
            from sam2.build_sam import build_sam2_video_predictor
        finally:
            sys.path.extend(_removed)  # restore so the rest of the program is unaffected

        self.device = _detect_device() if device == "auto" else torch.device(device)

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "[VideoMasker] MPS support is preliminary — SAM2 was trained with CUDA "
                "and may give slightly different results on Apple Silicon."
            )

        print(f"[VideoMasker] Using device: {self.device}")
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=self.device
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask_video(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
        output_path: str,
        alpha: float = 0.45,
        keep_frames: bool = False,
    ) -> str:
        """Segment and track objects throughout *video_path*.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        prompts:
            List of :class:`ObjectPrompt` instances that describe which objects
            to track and where the seed clicks (or boxes) are.
        output_path:
            Where to write the output MP4.
        alpha:
            Mask overlay transparency (0 = invisible, 1 = opaque solid color).
        keep_frames:
            If True, the temporary JPEG frame directory is not deleted after
            inference (useful for debugging).

        Returns
        -------
        str
            Absolute path to the written output video.
        """
        video_path = str(Path(video_path).expanduser().resolve())
        output_path = str(Path(output_path).expanduser().resolve())

        tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        try:
            # 1. Extract frames
            print("[VideoMasker] Extracting frames …")
            frame_names = _extract_frames(video_path, tmp_dir)
            print(f"[VideoMasker] {len(frame_names)} frames extracted to {tmp_dir}")

            # 2. Init SAM2 inference state
            print("[VideoMasker] Initialising SAM2 inference state …")
            inference_state = self.predictor.init_state(video_path=tmp_dir)

            # 3. Add all prompts
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

                self.predictor.add_new_points_or_box(**kwargs)
                print(f"[VideoMasker] Added prompt for obj_id={prompt.obj_id} on frame {prompt.frame_idx}")

            # 4. Propagate across video
            print("[VideoMasker] Propagating masks …")
            video_segments: dict[int, dict[int, np.ndarray]] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, obj_id in enumerate(out_obj_ids)
                }

            # 5. Write output video
            print("[VideoMasker] Writing output video …")
            self._write_video(
                frame_names, tmp_dir, video_segments, video_path, output_path, alpha
            )
            print(f"[VideoMasker] Saved: {output_path}")

        finally:
            if not keep_frames:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_video(
        self,
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
        H, W = first_bgr.shape[:2]
        fps = _get_fps(source_video)

        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        for frame_idx, fname in enumerate(frame_names):
            frame_bgr = cv2.imread(os.path.join(frame_dir, fname))
            if frame_bgr is None:
                raise RuntimeError(f"Could not read frame {frame_idx}: {fname}")
            masks_by_obj = video_segments.get(frame_idx, {})
            frame_bgr = _overlay_masks(frame_bgr, masks_by_obj, alpha)
            writer.write(frame_bgr)

        writer.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mask objects in a video using SAM2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Track two players by clicking on them in frame 0
  python video_masker.py tennis-clip.mp4 output.mp4 \\
      --point 1:370,600 \\
      --point 2:720,165

  # Use a bounding box instead of a point
  python video_masker.py clip.mp4 out.mp4 --box 1:300,500,450,700

  # Mix points and boxes, pick a different seed frame
  python video_masker.py clip.mp4 out.mp4 --point 1:370,600 --frame 1:5
        """,
    )
    parser.add_argument("input",  help="Path to input video file")
    parser.add_argument("output", help="Path for output video file")
    parser.add_argument(
        "--point", metavar="ID:X,Y", action="append", default=[],
        help="Add a positive click prompt.  Format: obj_id:x,y  (repeatable)",
    )
    parser.add_argument(
        "--box", metavar="ID:X0,Y0,X1,Y1", action="append", default=[],
        help="Add a bounding-box prompt.  Format: obj_id:x0,y0,x1,y1  (repeatable)",
    )
    parser.add_argument(
        "--frame", metavar="ID:N", action="append", default=[],
        help="Override the seed frame for an object.  Format: obj_id:frame_idx  (repeatable)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.45,
        help="Mask overlay opacity, 0–1  (default: 0.45)",
    )
    parser.add_argument(
        "--checkpoint",
        default="sam2/checkpoints/sam2.1_hiera_large.pt",
        help="Path to SAM2 checkpoint file",
    )
    parser.add_argument(
        "--model-cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 model config YAML",
    )
    args = parser.parse_args()

    if not args.point and not args.box:
        parser.error("Provide at least one --point or --box prompt.")

    # Parse per-object frame overrides
    frame_overrides: dict[int, int] = {}
    for token in args.frame:
        oid_str, fidx_str = token.split(":")
        frame_overrides[int(oid_str)] = int(fidx_str)

    # Build prompts dict keyed by obj_id
    obj_points:  dict[int, list] = {}
    obj_labels:  dict[int, list] = {}
    obj_boxes:   dict[int, np.ndarray] = {}

    for token in args.point:
        oid_str, xy_str = token.split(":")
        oid = int(oid_str)
        x, y = map(float, xy_str.split(","))
        obj_points.setdefault(oid, []).append([x, y])
        obj_labels.setdefault(oid, []).append(1)

    for token in args.box:
        oid_str, coords_str = token.split(":")
        oid = int(oid_str)
        x0, y0, x1, y1 = map(float, coords_str.split(","))
        obj_boxes[oid] = np.array([x0, y0, x1, y1], dtype=np.float32)

    all_ids = set(obj_points) | set(obj_boxes)
    prompts = []
    for oid in sorted(all_ids):
        pts = np.array(obj_points[oid], dtype=np.float32) if oid in obj_points else None
        lbs = np.array(obj_labels[oid], dtype=np.int32)   if oid in obj_points else None
        box = obj_boxes.get(oid)
        prompts.append(ObjectPrompt(
            obj_id=oid,
            points=pts,
            labels=lbs,
            box=box,
            frame_idx=frame_overrides.get(oid, 0),
        ))

    masker = VideoMasker(checkpoint=args.checkpoint, model_cfg=args.model_cfg)
    masker.mask_video(args.input, prompts, args.output, alpha=args.alpha)
