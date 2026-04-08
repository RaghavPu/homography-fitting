"""Frame extraction and video I/O utilities."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def load_frame(path: str, frame_idx: int = 0) -> np.ndarray:
    """Load a single frame from an image file or video.

    Tries ``cv2.imread`` first; falls back to ffmpeg extraction for videos.
    """
    if frame_idx == 0:
        img = cv2.imread(path)
        if img is not None:
            return img

    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg",
            "-i",
            path,
            "-vf",
            f"select=eq(n\\,{frame_idx})",
            "-vframes",
            "1",
            "-q:v",
            "2",
            tmp,
            "-y",
            "-loglevel",
            "error",
        ]
        subprocess.run(cmd, check=True)
        frame = cv2.imread(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    if frame is None:
        raise RuntimeError(f"Could not load frame {frame_idx} from: {path}")
    return frame


def extract_all_frames(video_path: str, out_dir: str) -> list[str]:
    """Extract every frame as a numbered JPEG into *out_dir*.

    Returns a sorted list of filenames (not full paths).
    """
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-q:v",
        "2",
        "-start_number",
        "0",
        os.path.join(out_dir, "%05d.jpg"),
        "-y",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True)

    frame_names = sorted(
        [p for p in os.listdir(out_dir) if Path(p).suffix.lower() in {".jpg", ".jpeg"}],
        key=lambda p: int(Path(p).stem),
    )
    if not frame_names:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return frame_names


def write_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float = 30.0,
) -> None:
    """Write a list of BGR frames as an MP4 video."""
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    os.makedirs(str(Path(output_path).parent), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def get_video_fps(video_path: str, default: float = 30.0) -> float:
    """Read the FPS from a video file via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or np.isnan(fps) or fps < 1e-3:
        return default
    return fps
