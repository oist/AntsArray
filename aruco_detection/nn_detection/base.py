"""Abstract detector interface for ArUco marker detection.

All detector backends (OpenCV, YOLO, DeepArUco++, RT-DETR) implement the
``ArucoDetector`` interface so that they can be swapped transparently in the
pipeline.  The downstream contract is:

    DataFrame with columns: Frame, Instance, X, Y, Confidence

which matches ``tracking/tracking_utils.py`` expectations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Detection:
    """A single ArUco marker detection."""

    marker_id: int
    x: float  # centre X
    y: float  # centre Y
    confidence: float  # 0.0 – 1.0
    corners: np.ndarray | None = None  # (4, 2) optional corner coords


class ArucoDetector(ABC):
    """Base class for all ArUco detection backends."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for a single frame (grayscale or BGR).

        Parameters
        ----------
        frame : np.ndarray
            Input image – may be grayscale (H, W) or BGR (H, W, 3).

        Returns
        -------
        list[Detection]
            Detected markers with positions, IDs, and confidences.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable detector name for logging / benchmark tables."""
        return self.__class__.__name__


def detections_to_dataframe(
    frame_idx: int, detections: list[Detection]
) -> pd.DataFrame:
    """Convert a list of :class:`Detection` objects into the standard DataFrame.

    Columns: Frame, Instance, X, Y, Confidence
    """
    if not detections:
        return pd.DataFrame(columns=["Frame", "Instance", "X", "Y", "Confidence"])

    return pd.DataFrame(
        {
            "Frame": np.int32(frame_idx),
            "Instance": np.array([d.marker_id for d in detections], dtype=np.int32),
            "X": np.array([d.x for d in detections], dtype=np.float32),
            "Y": np.array([d.y for d in detections], dtype=np.float32),
            "Confidence": np.array(
                [d.confidence for d in detections], dtype=np.float32
            ),
        }
    )


def run_detector_on_video(
    detector: ArucoDetector,
    video_path: str,
    dictionary_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Run *any* ``ArucoDetector`` on a video and return the standard outputs.

    Returns
    -------
    tracks : (num_frames, dictionary_size, 2) float32
    confidences : (num_frames, dictionary_size) float32
    df : pd.DataFrame   (Frame, Instance, X, Y, Confidence)
    """
    import cv2
    from tqdm import tqdm

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total if total > 0 else None, desc=f"{detector.name}")

    all_dfs: list[pd.DataFrame] = []
    frame_idx = 0
    detections_per_frame: list[list[Detection]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        detections_per_frame.append(dets)
        if dets:
            all_dfs.append(detections_to_dataframe(frame_idx, dets))
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    num_frames = len(detections_per_frame)
    tracks = np.zeros((num_frames, dictionary_size, 2), dtype=np.float32)
    confidences = np.zeros((num_frames, dictionary_size), dtype=np.float32)

    for f, dets in enumerate(detections_per_frame):
        for d in dets:
            if 0 <= d.marker_id < dictionary_size:
                tracks[f, d.marker_id] = [d.x, d.y]
                confidences[f, d.marker_id] = d.confidence

    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(
        columns=["Frame", "Instance", "X", "Y", "Confidence"]
    )
    return tracks, confidences, df
