#!/usr/bin/env python3
"""
Multi-camera viewer for debugging colony tracking outputs.

The viewer displays synchronized camera videos or image sequences and projects
panorama-space tracking results back into each camera with the inverse
homography used by the mapping pipeline.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.font as tkfont
from PIL import Image, ImageTk
from tkinter import ttk


VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CAM_RE = re.compile(r"(?:^|[_/\\.-])cam(?P<cam>\d+)(?:[_/\\.-]|$)")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")
SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
)


def parse_camera_index(path_or_name: str | Path) -> int:
    path = Path(path_or_name)
    name = path.name
    m = CAM_RE.search(name) or CAM_RE.search(str(path_or_name))
    if not m:
        raise ValueError(f"Could not parse camera id from {name!r}; expected camNN in the filename.")
    cam_one_based = int(m.group("cam"))
    if cam_one_based <= 0:
        raise ValueError(f"Camera id must be one-based and positive: {name!r}")
    return cam_one_based - 1


def parse_timestamp(path_or_name: str | Path) -> Optional[datetime]:
    m = TS_RE.search(Path(path_or_name).name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return None


def parse_int_list(raw: str | None) -> Optional[list[int]]:
    if raw is None or not raw.strip():
        return None
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(part))
    return values


def color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id) + 31)
    bgr = rng.integers(70, 255, size=3)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def load_homography_stack(path: Path) -> list[np.ndarray]:
    data = np.load(path)
    if "H" not in data:
        raise KeyError(f"{path} does not contain key 'H'")
    H = data["H"]
    if H.ndim != 3 or H.shape[1:] != (3, 3):
        raise ValueError(f"{path} key 'H' must have shape (n_cam, 3, 3); got {H.shape}")
    return [np.asarray(H[i], dtype=np.float64) for i in range(H.shape[0])]


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def apply_homography_points(xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    if xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy), dtype=np.float64)])
    proj = pts @ H.T
    valid = np.isfinite(proj).all(axis=1) & (np.abs(proj[:, 2]) > 1e-12)
    out = np.full((len(xy), 2), np.nan, dtype=np.float64)
    out[valid] = proj[valid, :2] / proj[valid, 2:3]
    return out


@dataclass
class CameraFrame:
    image: Optional[np.ndarray]
    local_frame: int
    ok: bool


@dataclass(frozen=True)
class OverlayStyle:
    display_scale: float
    text_scale: float
    text_thickness: int
    text_outline: int
    track_radius: int
    aruco_radius: int
    sleap_radius: int
    trail_radius: int
    bodypoint_radius: int
    circle_thickness: int
    skeleton_thickness: int
    margin: int
    label_gap: int


class Flag:
    def __init__(self, value: bool):
        self.value = bool(value)

    def get(self) -> bool:
        return self.value


class CameraSource:
    def __init__(self, path: Path, cam_index: int, *, timestamp: Optional[datetime] = None):
        self.path = Path(path)
        self.cam_index = int(cam_index)
        self.timestamp = timestamp
        self.frame_offset = 0

    @property
    def label(self) -> str:
        return f"cam{self.cam_index + 1:02d}"

    @property
    def frame_count(self) -> Optional[int]:
        return None

    @property
    def fps(self) -> Optional[float]:
        return None

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        return None

    def read(self, local_frame: int) -> CameraFrame:
        raise NotImplementedError

    def close(self) -> None:
        return


class VideoSource(CameraSource):
    def __init__(self, path: Path, cam_index: int, *, timestamp: Optional[datetime] = None):
        super().__init__(path, cam_index, timestamp=timestamp)
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_count: Optional[int] = None
        self._fps: Optional[float] = None
        self._frame_size: Optional[tuple[int, int]] = None
        self._decoded_frame: Optional[int] = None
        self._last_frame: Optional[int] = None
        self._last_image: Optional[np.ndarray] = None

    def _open(self) -> cv2.VideoCapture:
        if self.cap is None:
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video: {self.path}")
            self.cap = cap
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            self._frame_size = (width, height) if width > 0 and height > 0 else None
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            self._fps = fps if fps > 0 else None
        return self.cap

    @property
    def frame_count(self) -> Optional[int]:
        self._open()
        return self._frame_count

    @property
    def fps(self) -> Optional[float]:
        self._open()
        return self._fps

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        self._open()
        return self._frame_size

    def read(self, local_frame: int) -> CameraFrame:
        if local_frame < 0:
            return CameraFrame(None, local_frame, False)
        if self._last_frame == local_frame and self._last_image is not None:
            return CameraFrame(self._last_image.copy(), local_frame, True)
        cap = self._open()
        if self._frame_count is not None and local_frame >= self._frame_count:
            return CameraFrame(None, local_frame, False)
        need_seek = self._decoded_frame is None or int(local_frame) != self._decoded_frame + 1
        if need_seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(local_frame))
        ok, frame = cap.read()
        if (not ok or frame is None) and not need_seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(local_frame))
            ok, frame = cap.read()
        if not ok or frame is None:
            return CameraFrame(None, local_frame, False)
        self._decoded_frame = int(local_frame)
        self._last_frame = int(local_frame)
        self._last_image = frame
        return CameraFrame(frame.copy(), local_frame, True)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ImageSequenceSource(CameraSource):
    def __init__(self, path: Path, cam_index: int, *, timestamp: Optional[datetime] = None):
        super().__init__(path, cam_index, timestamp=timestamp)
        if path.is_dir():
            files = [
                p
                for p in sorted(path.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
        elif any(ch in str(path) for ch in "*?[]"):
            files = [Path(p) for p in sorted(glob.glob(str(path)))]
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            files = [path]
        else:
            raise ValueError(f"Expected image file, image directory, or glob: {path}")
        if not files:
            raise FileNotFoundError(f"No images found for {path}")
        self.files = files
        self._frame_size: Optional[tuple[int, int]] = None
        self._last_frame: Optional[int] = None
        self._last_image: Optional[np.ndarray] = None

    @property
    def frame_count(self) -> Optional[int]:
        return len(self.files)

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        if self._frame_size is None:
            img = cv2.imread(str(self.files[0]), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                self._frame_size = (int(w), int(h))
        return self._frame_size

    def read(self, local_frame: int) -> CameraFrame:
        if len(self.files) == 1:
            local_frame = 0
        if local_frame < 0 or local_frame >= len(self.files):
            return CameraFrame(None, local_frame, False)
        if self._last_frame == local_frame and self._last_image is not None:
            return CameraFrame(self._last_image.copy(), local_frame, True)
        img = cv2.imread(str(self.files[local_frame]), cv2.IMREAD_COLOR)
        if img is None:
            return CameraFrame(None, local_frame, False)
        self._last_frame = int(local_frame)
        self._last_image = img
        return CameraFrame(img.copy(), local_frame, True)


class TrackingStore:
    def __init__(
        self,
        paths: Iterable[Path],
        *,
        track_ids: Optional[set[int]],
        frame_offset: int,
        frame_col: str,
        track_col: str,
        track_x_col: str,
        track_y_col: str,
        aruco_x_col: str,
        aruco_y_col: str,
        sleap_x_col: str,
        sleap_y_col: str,
        x_col: str,
        y_col: str,
        bodypoint_col: str,
        max_rows: int,
    ):
        self.frame_col = frame_col
        self.track_col = track_col
        self.track_x_col = track_x_col
        self.track_y_col = track_y_col
        self.aruco_x_col = aruco_x_col
        self.aruco_y_col = aruco_y_col
        self.sleap_x_col = sleap_x_col
        self.sleap_y_col = sleap_y_col
        self.x_col = x_col
        self.y_col = y_col
        self.bodypoint_col = bodypoint_col
        self.paths = [Path(p) for p in paths]
        self.track_ids = None if track_ids is None else {int(x) for x in track_ids}
        self.frame_offset = int(frame_offset)
        self._path_columns: dict[Path, list[str]] = {}
        self._skeleton_cache: dict[int, pd.DataFrame] = {}
        self._skeleton_cache_limit = 256
        self.anchor_df = self._load(paths, track_ids=track_ids, frame_offset=frame_offset, max_rows=max_rows)
        self.df = self.anchor_df
        self._anchor_frames = self.anchor_df[frame_col].to_numpy(np.int64) if not self.anchor_df.empty else np.empty(0, np.int64)
        self._df_frames = self.df[frame_col].to_numpy(np.int64) if not self.df.empty else np.empty(0, np.int64)

    @classmethod
    def empty(cls) -> "TrackingStore":
        obj = cls.__new__(cls)
        obj.frame_col = "Frame"
        obj.track_col = "TrackID"
        obj.track_x_col = "TrackX"
        obj.track_y_col = "TrackY"
        obj.aruco_x_col = "ArucoX"
        obj.aruco_y_col = "ArucoY"
        obj.sleap_x_col = "SleapAnchorX"
        obj.sleap_y_col = "SleapAnchorY"
        obj.x_col = "X"
        obj.y_col = "Y"
        obj.bodypoint_col = "Bodypoint"
        obj.paths = []
        obj.track_ids = None
        obj.frame_offset = 0
        obj._path_columns = {}
        obj._skeleton_cache = {}
        obj._skeleton_cache_limit = 256
        obj.df = pd.DataFrame()
        obj.anchor_df = pd.DataFrame()
        obj._anchor_frames = np.empty(0, np.int64)
        obj._df_frames = np.empty(0, np.int64)
        return obj

    def _columns_for_path(self, path: Path) -> list[str]:
        path = Path(path)
        if path not in self._path_columns:
            self._path_columns[path] = parquet_columns(path)
        return self._path_columns[path]

    def _load(
        self,
        paths: Iterable[Path],
        *,
        track_ids: Optional[set[int]],
        frame_offset: int,
        max_rows: int,
    ) -> pd.DataFrame:
        paths = [Path(p) for p in paths]
        if not paths:
            return pd.DataFrame()

        anchor_cols = [
            self.frame_col,
            self.track_col,
            self.track_x_col,
            self.track_y_col,
            self.aruco_x_col,
            self.aruco_y_col,
            self.sleap_x_col,
            self.sleap_y_col,
        ]
        fallback_cols = [self.bodypoint_col, self.x_col, self.y_col]
        parts: list[pd.DataFrame] = []
        for path in paths:
            cols = self._columns_for_path(path)
            needs_xy_fallback = self.track_x_col not in cols or self.track_y_col not in cols
            requested = anchor_cols + (fallback_cols if needs_xy_fallback else [])
            existing = [c for c in requested if c in cols]
            filters = None
            if track_ids is not None and self.track_col in existing:
                ids = sorted(int(x) for x in track_ids)
                filters = [(self.track_col, "in", ids)]
            df = pd.read_parquet(path, engine="pyarrow", columns=existing, filters=filters)
            if df.empty:
                continue
            if self.track_col not in df.columns:
                df[self.track_col] = 0
            if self.bodypoint_col not in df.columns:
                df[self.bodypoint_col] = 0
            if needs_xy_fallback and self.bodypoint_col in df.columns:
                bp0 = df[pd.to_numeric(df[self.bodypoint_col], errors="coerce") == 0]
                if not bp0.empty:
                    df = bp0
            if self.track_x_col not in df.columns and self.x_col in df.columns:
                df[self.track_x_col] = df[self.x_col]
            if self.track_y_col not in df.columns and self.y_col in df.columns:
                df[self.track_y_col] = df[self.y_col]
            for col in anchor_cols:
                if col not in df.columns:
                    df[col] = np.nan
            parts.append(df[anchor_cols])

        if not parts:
            return pd.DataFrame(columns=anchor_cols)

        out = pd.concat(parts, ignore_index=True, copy=False)
        for col in anchor_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=[self.frame_col, self.track_col])
        out[self.frame_col] = out[self.frame_col].astype(np.int64) + int(frame_offset)
        out[self.track_col] = out[self.track_col].astype(np.int64)
        out = out.drop_duplicates([self.frame_col, self.track_col], keep="first")
        if max_rows > 0 and len(out) > max_rows:
            raise MemoryError(
                f"Loaded {len(out):,} anchor rows, above --max_tracking_rows={max_rows:,}. "
                "Use --track_ids or a smaller parquet for interactive viewing."
            )
        out = out.sort_values(self.frame_col, kind="mergesort").reset_index(drop=True)
        return out

    def _make_anchor_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        cols = [
            self.frame_col,
            self.track_col,
            self.track_x_col,
            self.track_y_col,
            self.aruco_x_col,
            self.aruco_y_col,
            self.sleap_x_col,
            self.sleap_y_col,
        ]
        available = [c for c in cols if c in df.columns]
        anchor = df[available].drop_duplicates([self.frame_col, self.track_col], keep="first")
        return anchor.sort_values(self.frame_col, kind="mergesort").reset_index(drop=True)

    @property
    def frame_range(self) -> tuple[int, int] | None:
        if self.df.empty:
            return None
        return int(self.df[self.frame_col].min()), int(self.df[self.frame_col].max())

    def rows_for_frame(self, frame: int, *, anchors_only: bool = False) -> pd.DataFrame:
        if not anchors_only:
            return self.skeleton_rows_for_frame(frame)
        table = self.anchor_df
        frames = self._anchor_frames
        if table.empty:
            return table
        left = int(np.searchsorted(frames, int(frame), side="left"))
        right = int(np.searchsorted(frames, int(frame), side="right"))
        if left == right:
            return table.iloc[0:0]
        return table.iloc[left:right]

    def rows_for_frame_range(self, start: int, end: int, *, anchors_only: bool = False) -> pd.DataFrame:
        if not anchors_only:
            frames = [self.skeleton_rows_for_frame(frame) for frame in range(int(start), int(end) + 1)]
            frames = [df for df in frames if not df.empty]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        table = self.anchor_df
        frames = self._anchor_frames
        if table.empty or end < start:
            return table.iloc[0:0]
        left = int(np.searchsorted(frames, int(start), side="left"))
        right = int(np.searchsorted(frames, int(end), side="right"))
        if left == right:
            return table.iloc[0:0]
        return table.iloc[left:right]

    def skeleton_rows_for_frame(self, frame: int) -> pd.DataFrame:
        frame = int(frame)
        if frame in self._skeleton_cache:
            return self._skeleton_cache[frame]

        columns = [self.frame_col, self.track_col, self.bodypoint_col, self.x_col, self.y_col]
        raw_frame = frame - self.frame_offset
        parts: list[pd.DataFrame] = []
        if raw_frame >= 0:
            for path in self.paths:
                cols = self._columns_for_path(path)
                existing = [c for c in columns if c in cols]
                if self.frame_col not in existing or self.x_col not in existing or self.y_col not in existing:
                    continue
                filters = [(self.frame_col, "==", int(raw_frame))]
                if self.track_ids is not None and self.track_col in existing:
                    filters.append((self.track_col, "in", sorted(self.track_ids)))
                df = pd.read_parquet(path, engine="pyarrow", columns=existing, filters=filters)
                if df.empty:
                    continue
                if self.track_col not in df.columns:
                    df[self.track_col] = 0
                if self.bodypoint_col not in df.columns:
                    df[self.bodypoint_col] = 0
                for col in columns:
                    if col not in df.columns:
                        df[col] = np.nan
                parts.append(df[columns])

        if parts:
            out = pd.concat(parts, ignore_index=True, copy=False)
            for col in columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.dropna(subset=[self.frame_col, self.track_col, self.x_col, self.y_col])
            out[self.frame_col] = out[self.frame_col].astype(np.int64) + self.frame_offset
            out[self.track_col] = out[self.track_col].astype(np.int64)
            out[self.bodypoint_col] = out[self.bodypoint_col].fillna(-1).astype(np.int64)
            out = out.sort_values([self.track_col, self.bodypoint_col], kind="mergesort").reset_index(drop=True)
        else:
            out = pd.DataFrame(columns=columns)

        self._skeleton_cache[frame] = out
        if len(self._skeleton_cache) > self._skeleton_cache_limit:
            self._skeleton_cache.pop(next(iter(self._skeleton_cache)))
        return out


def discover_media(video_dir: Path, cameras: Optional[list[int]]) -> list[Path]:
    files = [
        p
        for p in sorted(video_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and p.name.startswith("cam")
    ]
    if cameras is not None:
        wanted = {int(c) - 1 if int(c) > 0 else int(c) for c in cameras}
        files = [p for p in files if parse_camera_index(p) in wanted]
    return files


def make_source(path: Path) -> CameraSource:
    cam_index = parse_camera_index(path)
    ts = parse_timestamp(path)
    if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES or any(ch in str(path) for ch in "*?[]"):
        return ImageSequenceSource(path, cam_index, timestamp=ts)
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return VideoSource(path, cam_index, timestamp=ts)
    raise ValueError(f"Unsupported media source: {path}")


def project_rows_to_camera(
    rows: pd.DataFrame,
    *,
    invH: np.ndarray,
    x_col: str,
    y_col: str,
) -> np.ndarray:
    if rows.empty or x_col not in rows.columns or y_col not in rows.columns:
        return np.empty((0, 2), dtype=np.float64)
    xy = rows[[x_col, y_col]].to_numpy(np.float64)
    finite = np.isfinite(xy).all(axis=1)
    out = np.full((len(xy), 2), np.nan, dtype=np.float64)
    if np.any(finite):
        out[finite] = apply_homography_points(xy[finite], invH)
    return out


def projected_visible_mask(
    rows: pd.DataFrame,
    *,
    invH: np.ndarray,
    x_col: str,
    y_col: str,
    image_shape: tuple[int, int],
    margin: int = 0,
) -> np.ndarray:
    if rows.empty:
        return np.zeros(0, dtype=bool)
    h, w = image_shape
    pts = project_rows_to_camera(rows, invH=invH, x_col=x_col, y_col=y_col)
    if pts.size == 0:
        return np.zeros(len(rows), dtype=bool)
    x = pts[:, 0]
    y = pts[:, 1]
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= -margin)
        & (x < w + margin)
        & (y >= -margin)
        & (y < h + margin)
    )


def draw_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.75,
    thickness: int = 2,
    outline: Optional[int] = None,
) -> None:
    x, y = xy
    outline_thickness = int(outline if outline is not None else max(thickness + 2, thickness * 2))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def overlay_style_for_tile(img: np.ndarray, tile_width: int, tile_height: int) -> OverlayStyle:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        display_scale = 1.0
    else:
        display_scale = min(tile_width / w, tile_height / h)
    source_per_screen = 1.0 / max(display_scale, 1e-6)
    return OverlayStyle(
        display_scale=display_scale,
        text_scale=max(0.5, 0.9 * source_per_screen),
        text_thickness=max(1, int(round(2.0 * source_per_screen))),
        text_outline=max(2, int(round(5.0 * source_per_screen))),
        track_radius=max(5, int(round(10.0 * source_per_screen))),
        aruco_radius=max(4, int(round(8.0 * source_per_screen))),
        sleap_radius=max(4, int(round(8.0 * source_per_screen))),
        trail_radius=max(2, int(round(3.5 * source_per_screen))),
        bodypoint_radius=max(3, int(round(5.0 * source_per_screen))),
        circle_thickness=max(2, int(round(2.5 * source_per_screen))),
        skeleton_thickness=max(2, int(round(2.5 * source_per_screen))),
        margin=max(8, int(round(12.0 * source_per_screen))),
        label_gap=max(4, int(round(5.0 * source_per_screen))),
    )


def source_text_xy(screen_x: int, screen_y: int, style: OverlayStyle) -> tuple[int, int]:
    source_per_screen = 1.0 / max(style.display_scale, 1e-6)
    return int(round(screen_x * source_per_screen)), int(round(screen_y * source_per_screen))


def fit_to_tile(img: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    scale = min(tile_width / w, tile_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    x0 = (tile_width - new_w) // 2
    y0 = (tile_height - new_h) // 2
    tile[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return tile


class MultiCameraTrackingViewer:
    def __init__(
        self,
        *,
        sources: list[CameraSource],
        homographies: list[np.ndarray],
        tracks: TrackingStore,
        start_frame: int,
        fps: float,
        sync_by_timestamp: bool,
        tile_width: int,
        tile_height: int,
        trail: int,
        transition_preroll: int,
    ):
        if not sources:
            raise ValueError("At least one camera source is required.")
        self.sources = sorted(sources, key=lambda s: s.cam_index)
        self.homographies = homographies
        self.inv_homographies = [np.linalg.inv(H) for H in homographies]
        self.tracks = tracks
        self.frame = int(start_frame)
        self.playing = False
        self.target_fps = float(fps)
        self.tile_width = int(tile_width)
        self.tile_height = int(tile_height)
        self.trail = int(trail)
        self.transition_preroll = int(transition_preroll)
        self.last_tick = 0.0
        self.play_job: Optional[str] = None
        self.resize_job: Optional[str] = None
        self.slider_job: Optional[str] = None
        self.setting_slider = False
        self.playback_delay_ms = max(1, int(round(1000.0 / max(1e-6, self.target_fps))))
        self.tk_images: list[ImageTk.PhotoImage] = []
        self.last_camera_stats: list[dict[str, object]] = []
        self.status_note = ""
        self.transition_frames: Optional[np.ndarray] = None
        self.transition_masks: Optional[np.ndarray] = None

        for source in self.sources:
            if source.cam_index < 0 or source.cam_index >= len(homographies):
                raise ValueError(
                    f"{source.label} needs homography index {source.cam_index}, "
                    f"but only {len(homographies)} homographies were loaded."
                )

        self._set_sync_offsets(sync_by_timestamp)
        self.max_frame = self._infer_max_frame()
        self.present_frames = self._compute_present_frames()
        if self.tracks.frame_range is not None:
            lo, hi = self.tracks.frame_range
            self.frame = max(lo, min(self.frame, hi))
            self.max_frame = max(self.max_frame, hi)

        self.root = tk.Tk()
        self.root.title("Multi-camera tracking viewer")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        self._bind_keys()
        self.render()

    def _set_sync_offsets(self, sync_by_timestamp: bool) -> None:
        if not sync_by_timestamp:
            for source in self.sources:
                source.frame_offset = 0
            return
        timestamps = [source.timestamp for source in self.sources if source.timestamp is not None]
        if not timestamps:
            for source in self.sources:
                source.frame_offset = 0
            return
        ref = min(timestamps)
        fps_values = [source.fps for source in self.sources if source.fps is not None]
        fps = float(np.median(fps_values)) if fps_values else self.target_fps
        for source in self.sources:
            if source.timestamp is None:
                source.frame_offset = 0
            else:
                source.frame_offset = int(round((source.timestamp - ref).total_seconds() * fps))

    def _infer_max_frame(self) -> int:
        if self.tracks.frame_range is not None:
            return self.tracks.frame_range[1]
        counts = []
        for source in self.sources:
            if source.frame_count is not None:
                counts.append(int(source.frame_count) + int(source.frame_offset))
        return max(counts) - 1 if counts else max(0, self.frame)

    def _compute_present_frames(self) -> np.ndarray:
        anchors = self.tracks.anchor_df
        if anchors.empty:
            return np.empty(0, dtype=np.int64)
        frames = anchors[self.tracks.frame_col].to_numpy(np.int64, copy=False)
        return np.unique(frames) if len(frames) else np.empty(0, dtype=np.int64)

    def _frame_visible_in_selected_cameras(self, frame: int) -> bool:
        rows = self.tracks.rows_for_frame(frame, anchors_only=True)
        if rows.empty:
            return False
        for source in self.sources:
            local_frame = int(frame) - int(source.frame_offset)
            if local_frame < 0:
                continue
            if source.frame_count is not None and local_frame >= int(source.frame_count):
                continue
            size = source.frame_size
            if size is None:
                continue
            width, height = size
            if width <= 0 or height <= 0:
                continue
            mask = projected_visible_mask(
                rows,
                invH=self.inv_homographies[source.cam_index],
                x_col=self.tracks.track_x_col,
                y_col=self.tracks.track_y_col,
                image_shape=(height, width),
                margin=0,
            )
            if bool(mask.any()):
                return True
        return False

    def _camera_mask_labels(self, mask: int) -> str:
        labels = [source.label for bit, source in enumerate(self.sources) if int(mask) & (1 << bit)]
        return "+".join(labels) if labels else "none"

    def _ensure_camera_transitions(self) -> None:
        if self.transition_frames is not None:
            return
        anchors = self.tracks.anchor_df
        if anchors.empty:
            self.transition_frames = np.empty(0, dtype=np.int64)
            self.transition_masks = np.empty(0, dtype=np.uint64)
            return
        if len(self.sources) > 63:
            raise ValueError("Camera transition indexing supports up to 63 selected cameras.")

        self.status_note = "Indexing camera transitions"
        if hasattr(self, "status_var"):
            self.status_var.set("Indexing camera transitions ...")
            self.root.update_idletasks()

        row_masks = np.zeros(len(anchors), dtype=np.uint64)
        frame_values = anchors[self.tracks.frame_col].to_numpy(np.int64, copy=False)
        for bit, source in enumerate(self.sources):
            size = source.frame_size
            if size is None:
                continue
            width, height = size
            if width <= 0 or height <= 0:
                continue
            visible = projected_visible_mask(
                anchors,
                invH=self.inv_homographies[source.cam_index],
                x_col=self.tracks.track_x_col,
                y_col=self.tracks.track_y_col,
                image_shape=(height, width),
                margin=0,
            )
            local_frames = frame_values - int(source.frame_offset)
            valid_local = local_frames >= 0
            if source.frame_count is not None:
                valid_local &= local_frames < int(source.frame_count)
            visible &= valid_local
            row_masks[visible] |= np.uint64(1 << bit)

        unique_frames, starts = np.unique(frame_values, return_index=True)
        frame_masks = np.bitwise_or.reduceat(row_masks, starts)
        changed = np.zeros(len(unique_frames), dtype=bool)
        if len(unique_frames) > 1:
            changed[1:] = (frame_masks[1:] != frame_masks[:-1]) & (frame_masks[1:] != 0) & (frame_masks[:-1] != 0)
        self.transition_frames = unique_frames[changed]
        self.transition_masks = frame_masks[changed]
        self.status_note = f"Indexed {len(self.transition_frames):,} camera transitions"

    def _configure_styles(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont").copy()
        default_font.configure(size=max(12, default_font.cget("size")))
        heading_font = default_font.copy()
        heading_font.configure(size=max(13, default_font.cget("size")), weight="bold")
        style = ttk.Style(self.root)
        style.configure("TLabel", font=default_font)
        style.configure("TButton", font=default_font, padding=(8, 5))
        style.configure("TCheckbutton", font=default_font)
        style.configure("TLabelframe.Label", font=heading_font)
        self.root.option_add("*Font", default_font)

    def _build_ui(self) -> None:
        self.root.configure(bg="#101010")
        self._configure_styles()
        rows, cols = self._tile_layout()
        initial_w = max(900, cols * self.tile_width + 360)
        initial_h = max(640, rows * self.tile_height + 130)
        self.root.geometry(f"{initial_w}x{initial_h}")
        self.root.minsize(800, 520)

        controls = ttk.Frame(self.root, padding=(8, 6))
        controls.pack(fill=tk.X)

        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT)
        ttk.Button(controls, text="<", width=3, command=lambda: self.step(-1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(controls, text=">", width=3, command=lambda: self.step(1)).pack(side=tk.LEFT)
        ttk.Button(controls, text="-100", command=lambda: self.step(-100)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="+100", command=lambda: self.step(100)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Prev present", command=lambda: self.jump_present(prev=True)).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(controls, text="Next present", command=lambda: self.jump_present(prev=False)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(controls, text="Prev cam move", command=lambda: self.jump_camera_transition(prev=True)).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(controls, text="Next cam move", command=lambda: self.jump_camera_transition(prev=False)).pack(side=tk.LEFT, padx=(4, 0))

        self.frame_var = tk.IntVar(value=self.frame)
        self.frame_spin = ttk.Spinbox(
            controls,
            from_=0,
            to=max(0, self.max_frame),
            textvariable=self.frame_var,
            width=12,
            command=self.on_frame_entry,
        )
        self.frame_spin.pack(side=tk.LEFT, padx=(12, 6))
        self.frame_spin.bind("<Return>", lambda _event: self.on_frame_entry())

        self.status_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=(8, 0))

        slider_frame = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        slider_frame.pack(fill=tk.X)
        self.frame_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=max(0, self.max_frame),
            orient=tk.HORIZONTAL,
            command=self.on_slider_changed,
        )
        self.frame_slider.pack(fill=tk.X)
        self.setting_slider = True
        self.frame_slider.set(self.frame)
        self.setting_slider = False

        toggles = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        toggles.pack(fill=tk.X)
        self.show_track = tk.BooleanVar(value=True)
        self.show_aruco = tk.BooleanVar(value=False)
        self.show_sleap_anchor = tk.BooleanVar(value=False)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        for label, var in (
            ("Track", self.show_track),
            ("ArUco", self.show_aruco),
            ("SLEAP anchor", self.show_sleap_anchor),
            ("Skeleton", self.show_skeleton),
            ("Labels", self.show_labels),
        ):
            ttk.Checkbutton(toggles, text=label, variable=var, command=self.render).pack(side=tk.LEFT)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main, padding=(8, 0, 4, 8))
        right = ttk.Frame(main, padding=(4, 0, 8, 8), width=360)
        main.add(left, weight=5)
        main.add(right, weight=1)

        self.canvas = tk.Canvas(
            left,
            bg="#111111",
            highlightthickness=0,
            width=cols * self.tile_width,
            height=rows * self.tile_height,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        self.frame_stats_var = tk.StringVar(value="")
        self.track_stats_var = tk.StringVar(value="")
        self.camera_stats_var = tk.StringVar(value="")

        frame_box = ttk.LabelFrame(right, text="Frame")
        frame_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(frame_box, textvariable=self.frame_stats_var, justify=tk.LEFT, wraplength=330).pack(
            anchor=tk.W, padx=8, pady=6
        )

        track_box = ttk.LabelFrame(right, text="Tracks")
        track_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(track_box, textvariable=self.track_stats_var, justify=tk.LEFT, wraplength=330).pack(
            anchor=tk.W, padx=8, pady=6
        )

        camera_box = ttk.LabelFrame(right, text="Cameras")
        camera_box.pack(fill=tk.BOTH, expand=True)
        ttk.Label(camera_box, textvariable=self.camera_stats_var, justify=tk.LEFT, wraplength=330).pack(
            anchor=tk.NW, padx=8, pady=6
        )

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda _event: self.step(-1))
        self.root.bind("<Right>", lambda _event: self.step(1))
        self.root.bind("<Shift-Left>", lambda _event: self.step(-100))
        self.root.bind("<Shift-Right>", lambda _event: self.step(100))
        self.root.bind("p", lambda _event: self.jump_present(prev=True))
        self.root.bind("n", lambda _event: self.jump_present(prev=False))
        self.root.bind("[", lambda _event: self.jump_camera_transition(prev=True))
        self.root.bind("]", lambda _event: self.jump_camera_transition(prev=False))
        self.root.bind("q", lambda _event: self.close())

    def on_canvas_configure(self, _event: tk.Event) -> None:
        if not hasattr(self, "root") or not hasattr(self, "frame_stats_var"):
            return
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self._render_after_resize)

    def _render_after_resize(self) -> None:
        self.resize_job = None
        self.render()

    def sync_frame_controls(self) -> None:
        self.frame_var.set(int(self.frame))
        self.setting_slider = True
        self.frame_slider.set(int(self.frame))
        self.setting_slider = False

    def on_slider_changed(self, value: str) -> None:
        if self.setting_slider:
            return
        target = max(0, min(int(float(value)), self.max_frame))
        if self.slider_job is not None:
            self.root.after_cancel(self.slider_job)
        self.slider_job = self.root.after(60, lambda: self.goto_frame(target))

    def on_frame_entry(self) -> None:
        try:
            frame = int(self.frame_var.get())
        except tk.TclError:
            return
        self.goto_frame(frame)

    def toggle_play(self) -> None:
        if self.playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self) -> None:
        if self.playing:
            return
        self.status_note = "Playing"
        self.playing = True
        self.play_button.configure(text="Pause")
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
        self.play_job = self.root.after(1, self._play_tick)

    def pause_playback(self) -> None:
        self.playing = False
        self.status_note = "Paused"
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        self.play_button.configure(text="Play")
        self.refresh_status()

    def _play_tick(self) -> None:
        if not self.playing:
            return
        self.play_job = None
        started = time.perf_counter()
        if self.frame >= self.max_frame:
            self.pause_playback()
            return
        self.goto_frame(self.frame + 1)
        elapsed_ms = int(round((time.perf_counter() - started) * 1000.0))
        next_delay = max(1, self.playback_delay_ms - elapsed_ms)
        self.play_job = self.root.after(next_delay, self._play_tick)

    def step(self, delta: int) -> None:
        self.goto_frame(int(self.frame) + int(delta))

    def goto_frame(self, frame: int) -> None:
        if self.slider_job is not None:
            self.root.after_cancel(self.slider_job)
            self.slider_job = None
        self.frame = max(0, min(int(frame), self.max_frame))
        self.sync_frame_controls()
        self.render()

    def jump_present(self, *, prev: bool) -> None:
        frames = self.present_frames
        if len(frames) == 0:
            self.status_note = "No loaded track frames"
            self.refresh_status()
            return
        if prev:
            idx = int(np.searchsorted(frames, int(self.frame), side="left")) - 1
            step = -1
        else:
            idx = int(np.searchsorted(frames, int(self.frame), side="right"))
            step = 1
        while 0 <= idx < len(frames):
            target = int(frames[idx])
            if self._frame_visible_in_selected_cameras(target):
                self.status_note = "Present in selected cameras"
                self.goto_frame(target)
                return
            idx += step
        self.status_note = "No previous selected-camera frame" if prev else "No next selected-camera frame"
        self.refresh_status()

    def jump_camera_transition(self, *, prev: bool) -> None:
        self._ensure_camera_transitions()
        frames = self.transition_frames if self.transition_frames is not None else np.empty(0, dtype=np.int64)
        masks = self.transition_masks if self.transition_masks is not None else np.empty(0, dtype=np.uint64)
        if len(frames) == 0:
            self.status_note = "No selected-camera transitions"
            self.refresh_status()
            return
        search_frame = int(self.frame) + max(0, self.transition_preroll)
        if prev:
            idx = int(np.searchsorted(frames, search_frame, side="left")) - 1
        else:
            idx = int(np.searchsorted(frames, search_frame, side="right"))
        if idx < 0 or idx >= len(frames):
            self.status_note = "No previous camera transition" if prev else "No next camera transition"
            self.refresh_status()
            return
        transition_frame = int(frames[idx])
        jump_frame = max(0, transition_frame - max(0, self.transition_preroll))
        mask_text = self._camera_mask_labels(int(masks[idx])) if idx < len(masks) else "unknown"
        self.status_note = f"Camera transition at {transition_frame} -> {mask_text}"
        self.goto_frame(jump_frame)

    def _tile_layout(self) -> tuple[int, int]:
        n = len(self.sources)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / cols))
        return rows, cols

    def render_camera(
        self,
        source: CameraSource,
        *,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        tile_width = int(tile_width if tile_width is not None else self.tile_width)
        tile_height = int(tile_height if tile_height is not None else self.tile_height)
        local_frame = int(self.frame) - int(source.frame_offset)
        frame = source.read(local_frame)
        stats: dict[str, object] = {
            "label": source.label,
            "local_frame": local_frame,
            "ok": bool(frame.ok),
            "track_visible": 0,
            "aruco_visible": 0,
            "sleap_anchor_visible": 0,
            "bodypoints_visible": 0,
            "visible_ids": [],
        }
        if frame.ok and frame.image is not None:
            img = frame.image
            style = overlay_style_for_tile(img, tile_width, tile_height)
            invH = self.inv_homographies[source.cam_index]
            stats.update(self._draw_overlays(img, invH, style))
        else:
            img = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            style = overlay_style_for_tile(img, tile_width, tile_height)
            draw_text(
                img,
                "no frame",
                source_text_xy(16, 34, style),
                (220, 220, 220),
                scale=style.text_scale,
                thickness=style.text_thickness,
                outline=style.text_outline,
            )

        draw_text(
            img,
            f"{source.label}  local={local_frame}",
            source_text_xy(16, 28, style),
            (240, 240, 240),
            scale=style.text_scale,
            thickness=style.text_thickness,
            outline=style.text_outline,
        )
        if source.frame_offset:
            draw_text(
                img,
                f"offset={source.frame_offset}",
                source_text_xy(16, 58, style),
                (180, 220, 255),
                scale=style.text_scale,
                thickness=style.text_thickness,
                outline=style.text_outline,
            )
        return fit_to_tile(img, tile_width, tile_height), stats

    def _draw_overlays(self, img: np.ndarray, invH: np.ndarray, style: OverlayStyle) -> dict[str, object]:
        h, w = img.shape[:2]
        margin = style.margin
        anchor_rows = self.tracks.rows_for_frame(self.frame, anchors_only=True)
        skel_rows = self.tracks.rows_for_frame(self.frame, anchors_only=False) if self.show_skeleton.get() else pd.DataFrame()
        track_mask = projected_visible_mask(
            anchor_rows,
            invH=invH,
            x_col=self.tracks.track_x_col,
            y_col=self.tracks.track_y_col,
            image_shape=(h, w),
            margin=margin,
        )
        aruco_mask = projected_visible_mask(
            anchor_rows,
            invH=invH,
            x_col=self.tracks.aruco_x_col,
            y_col=self.tracks.aruco_y_col,
            image_shape=(h, w),
            margin=margin,
        )
        sleap_anchor_mask = projected_visible_mask(
            anchor_rows,
            invH=invH,
            x_col=self.tracks.sleap_x_col,
            y_col=self.tracks.sleap_y_col,
            image_shape=(h, w),
            margin=margin,
        )
        bodypoint_mask = projected_visible_mask(
            skel_rows,
            invH=invH,
            x_col=self.tracks.x_col,
            y_col=self.tracks.y_col,
            image_shape=(h, w),
            margin=margin,
        )
        visible_ids = (
            sorted(int(x) for x in anchor_rows.loc[track_mask, self.tracks.track_col].dropna().unique())
            if len(track_mask)
            else []
        )
        stats: dict[str, object] = {
            "track_visible": int(track_mask.sum()) if len(track_mask) else 0,
            "aruco_visible": int(aruco_mask.sum()) if len(aruco_mask) else 0,
            "sleap_anchor_visible": int(sleap_anchor_mask.sum()) if len(sleap_anchor_mask) else 0,
            "bodypoints_visible": int(bodypoint_mask.sum()) if len(bodypoint_mask) else 0,
            "visible_ids": visible_ids,
        }
        if self.trail > 0:
            trail_rows = self.tracks.rows_for_frame_range(
                max(0, self.frame - self.trail),
                self.frame - 1,
                anchors_only=True,
            )
        else:
            trail_rows = pd.DataFrame()

        if self.show_skeleton.get():
            self._draw_skeleton_rows(img, skel_rows, invH, margin=margin, style=style)

        if self.show_track.get() and not trail_rows.empty:
            pts = project_rows_to_camera(
                trail_rows,
                invH=invH,
                x_col=self.tracks.track_x_col,
                y_col=self.tracks.track_y_col,
            )
            for row, (x, y) in zip(trail_rows.itertuples(index=False), pts):
                if np.isfinite(x) and np.isfinite(y) and -margin <= x < w + margin and -margin <= y < h + margin:
                    tid = int(getattr(row, self.tracks.track_col))
                    cv2.circle(
                        img,
                        (int(round(x)), int(round(y))),
                        style.trail_radius,
                        color_for_id(tid),
                        -1,
                        cv2.LINE_AA,
                    )

        if self.show_track.get():
            self._draw_anchor_points(
                img,
                anchor_rows,
                invH,
                self.tracks.track_x_col,
                self.tracks.track_y_col,
                radius=style.track_radius,
                color_override=None,
                label_prefix="",
                margin=margin,
                style=style,
            )
        if self.show_aruco.get():
            self._draw_anchor_points(
                img,
                anchor_rows,
                invH,
                self.tracks.aruco_x_col,
                self.tracks.aruco_y_col,
                radius=style.aruco_radius,
                color_override=(0, 255, 255),
                label_prefix="A",
                margin=margin,
                style=style,
            )
        if self.show_sleap_anchor.get():
            self._draw_anchor_points(
                img,
                anchor_rows,
                invH,
                self.tracks.sleap_x_col,
                self.tracks.sleap_y_col,
                radius=style.sleap_radius,
                color_override=(255, 0, 255),
                label_prefix="S",
                margin=margin,
                style=style,
            )
        return stats

    def _draw_anchor_points(
        self,
        img: np.ndarray,
        rows: pd.DataFrame,
        invH: np.ndarray,
        x_col: str,
        y_col: str,
        *,
        radius: int,
        color_override: Optional[tuple[int, int, int]],
        label_prefix: str,
        margin: int,
        style: OverlayStyle,
    ) -> None:
        if rows.empty:
            return
        h, w = img.shape[:2]
        pts = project_rows_to_camera(rows, invH=invH, x_col=x_col, y_col=y_col)
        for row, (x, y) in zip(rows.itertuples(index=False), pts):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if not (-margin <= x < w + margin and -margin <= y < h + margin):
                continue
            tid = int(getattr(row, self.tracks.track_col))
            color = color_override or color_for_id(tid)
            center = (int(round(x)), int(round(y)))
            cv2.circle(img, center, radius, color, style.circle_thickness, cv2.LINE_AA)
            if self.show_labels.get():
                label = f"{label_prefix}{tid}" if label_prefix else str(tid)
                draw_text(
                    img,
                    label,
                    (center[0] + radius + style.label_gap, center[1] - radius - style.label_gap),
                    color,
                    scale=style.text_scale,
                    thickness=style.text_thickness,
                    outline=style.text_outline,
                )

    def _draw_skeleton_rows(
        self,
        img: np.ndarray,
        rows: pd.DataFrame,
        invH: np.ndarray,
        *,
        margin: int,
        style: OverlayStyle,
    ) -> None:
        if rows.empty:
            return
        h, w = img.shape[:2]
        rows = rows.dropna(subset=[self.tracks.x_col, self.tracks.y_col, self.tracks.bodypoint_col])
        if rows.empty:
            return
        pts = project_rows_to_camera(rows, invH=invH, x_col=self.tracks.x_col, y_col=self.tracks.y_col)
        rows = rows.copy()
        rows["cx"] = pts[:, 0]
        rows["cy"] = pts[:, 1]
        for tid, group in rows.groupby(self.tracks.track_col, sort=False):
            color = color_for_id(int(tid))
            node_xy: dict[int, tuple[int, int]] = {}
            for r in group.itertuples(index=False):
                x = float(getattr(r, "cx"))
                y = float(getattr(r, "cy"))
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                if not (-margin <= x < w + margin and -margin <= y < h + margin):
                    continue
                bp = int(getattr(r, self.tracks.bodypoint_col))
                node_xy[bp] = (int(round(x)), int(round(y)))
                cv2.circle(img, node_xy[bp], style.bodypoint_radius, color, -1, cv2.LINE_AA)
            for a, b in SKELETON_EDGES:
                if a in node_xy and b in node_xy:
                    cv2.line(img, node_xy[a], node_xy[b], color, style.skeleton_thickness, cv2.LINE_AA)

    def render(self) -> None:
        rows, cols = self._tile_layout()
        self.canvas.delete("all")
        self.tk_images.clear()
        self.last_camera_stats = []
        canvas_w = int(self.canvas.winfo_width())
        canvas_h = int(self.canvas.winfo_height())
        if canvas_w <= 1:
            canvas_w = cols * self.tile_width
        if canvas_h <= 1:
            canvas_h = rows * self.tile_height
        tile_w = max(1, canvas_w // cols)
        tile_h = max(1, canvas_h // rows)
        for idx, source in enumerate(self.sources):
            r = idx // cols
            c = idx % cols
            img, stats = self.render_camera(source, tile_width=tile_w, tile_height=tile_h)
            self.last_camera_stats.append(stats)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.tk_images.append(photo)
            self.canvas.create_image(c * tile_w, r * tile_h, anchor=tk.NW, image=photo)
        self.refresh_status()

    def refresh_status(self) -> None:
        anchor_rows = self.tracks.rows_for_frame(self.frame, anchors_only=True)
        skel_rows = self.tracks._skeleton_cache.get(int(self.frame))
        skel_count_text = str(len(skel_rows)) if skel_rows is not None else "not loaded"
        visible_bodypoints = sum(int(stats.get("bodypoints_visible", 0)) for stats in self.last_camera_stats)
        frame_ids = sorted(int(x) for x in anchor_rows[self.tracks.track_col].dropna().unique()) if not anchor_rows.empty else []
        ok_cams = sum(1 for s in self.last_camera_stats if bool(s.get("ok")))
        visible_ids = sorted(
            {
                int(track_id)
                for stats in self.last_camera_stats
                for track_id in stats.get("visible_ids", [])
            }
        )
        track_frame_count = len(self.present_frames)
        present_idx = int(np.searchsorted(self.present_frames, int(self.frame), side="left")) if track_frame_count else -1
        if track_frame_count and present_idx < track_frame_count and int(self.present_frames[present_idx]) == int(self.frame):
            present_text = f"{present_idx + 1}/{track_frame_count}"
        elif track_frame_count:
            present_text = f"not a loaded track frame ({track_frame_count} total)"
        else:
            present_text = "none"
        transition_text = (
            "not indexed"
            if self.transition_frames is None
            else f"{len(self.transition_frames):,} transitions, preroll {self.transition_preroll}"
        )
        note = f" | {self.status_note}" if self.status_note else ""
        self.status_var.set(
            f"Frame {self.frame + 1}/{self.max_frame + 1} | "
            f"anchor rows {len(self.tracks.anchor_df):,} | "
            f"visible IDs {len(visible_ids)} | "
            f"cams {ok_cams}/{len(self.sources)}{note}"
        )
        self.frame_stats_var.set(
            f"Frame: {self.frame}\n"
            f"Display: {self.frame + 1} / {self.max_frame + 1}\n"
            f"Current track anchors: {len(anchor_rows)}\n"
            f"Current bodypoint rows: {skel_count_text}\n"
            f"Visible bodypoints across cameras: {visible_bodypoints}\n"
            f"Loaded-track frame index: {present_text}\n"
            f"Visible in selected cameras: {'yes' if visible_ids else 'no'}\n"
            f"Camera transitions: {transition_text}"
        )
        frame_id_text = ", ".join(str(x) for x in frame_ids[:18])
        if len(frame_ids) > 18:
            frame_id_text += ", ..."
        visible_id_text = ", ".join(str(x) for x in visible_ids[:18])
        if len(visible_ids) > 18:
            visible_id_text += ", ..."
        self.track_stats_var.set(
            f"Loaded anchor rows: {len(self.tracks.anchor_df):,}\n"
            f"Frame TrackIDs: {frame_id_text or 'none'}\n"
            f"Visible TrackIDs: {visible_id_text or 'none'}"
        )
        camera_lines = []
        for stats in self.last_camera_stats:
            visible = stats.get("visible_ids", [])
            visible_text = ",".join(str(x) for x in visible[:8]) if visible else "none"
            if len(visible) > 8:
                visible_text += ",..."
            camera_lines.append(
                f"{stats.get('label')} local {stats.get('local_frame')} "
                f"{'ok' if stats.get('ok') else 'no frame'} | "
                f"track {stats.get('track_visible')} "
                f"aruco {stats.get('aruco_visible')} "
                f"sleap {stats.get('sleap_anchor_visible')} "
                f"bp {stats.get('bodypoints_visible')} | IDs {visible_text}"
            )
        self.camera_stats_var.set("\n".join(camera_lines) if camera_lines else "No cameras")

    def close(self) -> None:
        for source in self.sources:
            source.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def render_smoke_frame(
    sources: list[CameraSource],
    homographies: list[np.ndarray],
    tracks: TrackingStore,
    *,
    frame: int,
    out_path: Path,
    tile_width: int,
    tile_height: int,
) -> None:
    viewer = MultiCameraTrackingViewer.__new__(MultiCameraTrackingViewer)
    viewer.sources = sorted(sources, key=lambda s: s.cam_index)
    viewer.homographies = homographies
    viewer.inv_homographies = [np.linalg.inv(H) for H in homographies]
    viewer.tracks = tracks
    viewer.frame = int(frame)
    viewer.tile_width = int(tile_width)
    viewer.tile_height = int(tile_height)
    viewer.trail = 0
    viewer.show_track = Flag(True)
    viewer.show_aruco = Flag(True)
    viewer.show_sleap_anchor = Flag(True)
    viewer.show_skeleton = Flag(True)
    viewer.show_labels = Flag(True)
    rows, cols = viewer._tile_layout()
    canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)
    for idx, source in enumerate(viewer.sources):
        r = idx // cols
        c = idx % cols
        img, _stats = viewer.render_camera(source, tile_width=tile_width, tile_height=tile_height)
        canvas[r * tile_height : (r + 1) * tile_height, c * tile_width : (c + 1) * tile_width] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    for source in sources:
        source.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="View synchronized camera media with panorama tracking overlays.")
    p.add_argument("--hmats", type=Path, required=True, help="Homography .npz with key H.")
    p.add_argument("--video_dir", type=Path, default=None, help="Directory containing camNN video files.")
    p.add_argument("--media", nargs="+", type=Path, default=None, help="Explicit camera video/image/image-directory paths.")
    p.add_argument("--cameras", type=str, default=None, help="One-based camera list/ranges, e.g. 4,5,9 or 1-7.")
    p.add_argument("--tracks", nargs="+", type=Path, default=None, help="Tracking parquet file(s).")
    p.add_argument("--track_ids", type=str, default=None, help="TrackID list/ranges to load, e.g. 17 or 17,23.")
    p.add_argument("--track_frame_offset", type=int, default=0, help="Add this offset to tracking Frame values.")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--sync_by_timestamp", action="store_true", help="Align camera local frames by timestamps in filenames.")
    p.add_argument("--tile_width", type=int, default=640)
    p.add_argument("--tile_height", type=int, default=484)
    p.add_argument("--trail", type=int, default=0, help="Draw previous N frames of TrackX/TrackY as a trail.")
    p.add_argument(
        "--transition_preroll",
        type=int,
        default=48,
        help="Frames before a selected-camera transition to seek to.",
    )
    p.add_argument("--max_tracking_rows", type=int, default=25_000_000)
    p.add_argument("--frame_col", default="Frame")
    p.add_argument("--track_col", default="TrackID")
    p.add_argument("--bodypoint_col", default="Bodypoint")
    p.add_argument("--x_col", default="X")
    p.add_argument("--y_col", default="Y")
    p.add_argument("--track_x_col", default="TrackX")
    p.add_argument("--track_y_col", default="TrackY")
    p.add_argument("--aruco_x_col", default="ArucoX")
    p.add_argument("--aruco_y_col", default="ArucoY")
    p.add_argument("--sleap_x_col", default="SleapAnchorX")
    p.add_argument("--sleap_y_col", default="SleapAnchorY")
    p.add_argument("--smoke_out", type=Path, default=None, help="Render one frame to this PNG and exit.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cameras = parse_int_list(args.cameras)
    media_paths = list(args.media or [])
    if args.video_dir is not None:
        media_paths.extend(discover_media(args.video_dir, cameras))
    if not media_paths:
        raise ValueError("Provide --media or --video_dir.")

    if args.media is not None and cameras is not None and args.video_dir is None:
        wanted = {int(c) - 1 if int(c) > 0 else int(c) for c in cameras}
        media_paths = [p for p in media_paths if parse_camera_index(p) in wanted]

    sources = [make_source(path) for path in media_paths]
    homographies = load_homography_stack(args.hmats)
    track_ids_raw = parse_int_list(args.track_ids)
    track_ids = None if track_ids_raw is None else {int(x) for x in track_ids_raw}
    tracks = (
        TrackingStore(
            args.tracks or [],
            track_ids=track_ids,
            frame_offset=args.track_frame_offset,
            frame_col=args.frame_col,
            track_col=args.track_col,
            track_x_col=args.track_x_col,
            track_y_col=args.track_y_col,
            aruco_x_col=args.aruco_x_col,
            aruco_y_col=args.aruco_y_col,
            sleap_x_col=args.sleap_x_col,
            sleap_y_col=args.sleap_y_col,
            x_col=args.x_col,
            y_col=args.y_col,
            bodypoint_col=args.bodypoint_col,
            max_rows=args.max_tracking_rows,
        )
        if args.tracks
        else TrackingStore.empty()
    )

    if args.smoke_out is not None:
        render_smoke_frame(
            sources,
            homographies,
            tracks,
            frame=args.start_frame,
            out_path=args.smoke_out,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
        )
        return

    app = MultiCameraTrackingViewer(
        sources=sources,
        homographies=homographies,
        tracks=tracks,
        start_frame=args.start_frame,
        fps=args.fps,
        sync_by_timestamp=args.sync_by_timestamp,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        trail=args.trail,
        transition_preroll=args.transition_preroll,
    )
    app.run()


if __name__ == "__main__":
    main()
