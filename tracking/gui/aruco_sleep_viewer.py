#!/usr/bin/env python3
"""Focused video viewer for tuning sleep labels from cached SLEAP motion."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Optional

import cv2
import h5py
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.sleep_motion_utils import (  # noqa: E402
    ANTENNA_BODYPOINT_IDS,
    BODY_BODYPOINT_IDS,
    SleepMotionCache,
    aggregate_bodypoint_group_slice,
    classify_body_antenna_quiet,
    load_sleep_motion_cache,
)
from tracking.gui.aruco_curation import (  # noqa: E402
    ArucoDetectionStore,
    ChunkSpec,
    OpenCvVideoReader,
    SLEAP_SKELETON_EDGES,
    SleapSkeletonStore,
    color_for_id,
    matching_aruco_tracks_path,
    parse_camera_index,
    resolve_chunk_specs,
)


SLEEP_COLOR = (20, 180, 255)
WAKE_COLOR = (60, 220, 80)
UNKNOWN_COLOR = (175, 175, 175)
SLEAP_UNMATCHED_COLOR = (150, 150, 150)


@dataclass(frozen=True)
class SleepParameters:
    body_threshold_mm_s: float = 0.50
    antenna_threshold_mm_s: float = 0.70
    window_seconds: float = 10.0
    quiet_fraction: float = 0.90
    min_valid_frame_fraction: float = 0.50
    body_percentile: float = 75.0
    antenna_percentile: float = 75.0
    min_valid_body_fraction: float = 0.75
    min_valid_antenna_fraction: float = 0.50
    max_gap_frames: int = 5
    max_speed_mm_s: float = 20.0

    def validate(self) -> "SleepParameters":
        nonnegative = {
            "Body threshold": self.body_threshold_mm_s,
            "Antenna threshold": self.antenna_threshold_mm_s,
            "Tracking-jump cutoff": self.max_speed_mm_s,
        }
        for label, value in nonnegative.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{label} must be a finite nonnegative number")
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("History must be a positive number of seconds")
        for label, value in (
            ("Required low-motion fraction", self.quiet_fraction),
            ("Required valid-frame fraction", self.min_valid_frame_fraction),
            ("Required bodypoint fraction", self.min_valid_body_fraction),
            ("Required antenna-point fraction", self.min_valid_antenna_fraction),
        ):
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{label} must be between 0 and 1")
        for label, value in (
            ("Body percentile", self.body_percentile),
            ("Antenna percentile", self.antenna_percentile),
        ):
            if not 0 <= float(value) <= 100:
                raise ValueError(f"{label} must be between 0 and 100")
        if int(self.max_gap_frames) < 1:
            raise ValueError("Maximum point gap must be at least one frame")
        return self

    def key(self) -> tuple[object, ...]:
        return (
            self.body_threshold_mm_s,
            self.antenna_threshold_mm_s,
            self.window_seconds,
            self.quiet_fraction,
            self.min_valid_frame_fraction,
            self.body_percentile,
            self.antenna_percentile,
            self.min_valid_body_fraction,
            self.min_valid_antenna_fraction,
            self.max_gap_frames,
            self.max_speed_mm_s,
        )


@dataclass
class MotionRecord:
    side: str
    track_id: int
    cache_dir: Path
    cache: Optional[SleepMotionCache] = None

    def load(self) -> SleepMotionCache:
        if self.cache is None:
            self.cache = load_sleep_motion_cache(self.cache_dir)
        return self.cache


@dataclass
class StateBlock:
    start_frame: int
    stop_frame: int
    body_motion: np.ndarray
    antenna_motion: np.ndarray
    state: np.ndarray
    quiet_fraction: np.ndarray
    valid_fraction: np.ndarray
    mean_body_speed: np.ndarray
    mean_antenna_speed: np.ndarray


@dataclass
class MotionBlock:
    start_frame: int
    stop_frame: int
    body_motion: np.ndarray
    antenna_motion: np.ndarray


class SleepMotionStore:
    def __init__(self, root: Path, *, block_cache_size: int = 96):
        root = Path(root)
        per_track = root if root.name == "per_track" else root / "per_track"
        metadata_paths = sorted(per_track.glob("*/sleep_motion_metadata.json"))
        if not metadata_paths:
            raise FileNotFoundError(f"No sleep-motion caches found under {root}")

        self.records: dict[tuple[str, int], MotionRecord] = {}
        self.sides_by_track: dict[int, list[str]] = {}
        for metadata_path in metadata_paths:
            try:
                metadata = json.loads(metadata_path.read_text())
            except Exception:
                continue
            track_id = metadata.get("track_id")
            if track_id is None:
                match = re.search(r"TrackID_(\d+)", metadata_path.parent.name)
                if match is None:
                    continue
                track_id = int(match.group(1))
            side = str(metadata.get("side") or "").strip().lower()
            if side not in {"left", "right"}:
                match = re.search(r"(?:^|_)(left|right)(?:_|$)", metadata_path.parent.name.lower())
                side = match.group(1) if match else ""
            if not side:
                continue
            record = MotionRecord(side=side, track_id=int(track_id), cache_dir=metadata_path.parent)
            self.records[(side, int(track_id))] = record
            self.sides_by_track.setdefault(int(track_id), []).append(side)

        if not self.records:
            raise FileNotFoundError(f"No side-aware sleep-motion caches found under {root}")
        for track_id in self.sides_by_track:
            self.sides_by_track[track_id] = sorted(set(self.sides_by_track[track_id]))
        self.block_cache_size = max(1, int(block_cache_size))
        self.blocks: OrderedDict[tuple[object, ...], StateBlock] = OrderedDict()
        self.motion_blocks: OrderedDict[tuple[object, ...], MotionBlock] = OrderedDict()

    def clear_blocks(self) -> None:
        self.blocks.clear()

    def only_side_for_track(self, track_id: int) -> Optional[str]:
        sides = self.sides_by_track.get(int(track_id), [])
        return sides[0] if len(sides) == 1 else None

    def _build_block(
        self,
        record: MotionRecord,
        *,
        block_start: int,
        block_stop: int,
        params: SleepParameters,
    ) -> StateBlock:
        cache = record.load()
        if params.max_gap_frames > cache.cache_max_gap_frames:
            raise ValueError(
                f"Maximum point gap {params.max_gap_frames} exceeds the cached maximum "
                f"of {cache.cache_max_gap_frames}"
            )
        window_frames = max(1, int(round(params.window_seconds * cache.fps)))
        start_frame = max(cache.frame_min, int(block_start) - window_frames + 1)
        stop_frame = min(cache.frame_max + 1, int(block_stop))
        if stop_frame <= start_frame:
            empty = np.empty(0, dtype=np.float32)
            return StateBlock(
                start_frame=start_frame,
                stop_frame=stop_frame,
                body_motion=empty,
                antenna_motion=empty,
                state=np.empty(0, dtype=np.int8),
                quiet_fraction=empty,
                valid_fraction=empty,
                mean_body_speed=empty,
                mean_antenna_speed=empty,
            )

        motion_key = (
            record.side,
            record.track_id,
            start_frame,
            stop_frame,
            params.body_percentile,
            params.antenna_percentile,
            params.min_valid_body_fraction,
            params.min_valid_antenna_fraction,
            params.max_gap_frames,
            params.max_speed_mm_s,
        )
        motion = self.motion_blocks.get(motion_key)
        if motion is None:
            start_index = start_frame - cache.frame_min
            stop_index = stop_frame - cache.frame_min
            common = {
                "max_gap_frames": params.max_gap_frames,
                "max_bodypoint_speed_mm_s": params.max_speed_mm_s,
            }
            body = aggregate_bodypoint_group_slice(
                cache,
                start_index,
                stop_index,
                bodypoint_ids=BODY_BODYPOINT_IDS,
                bodypoint_percentile=params.body_percentile,
                min_valid_bodypoint_fraction=params.min_valid_body_fraction,
                **common,
            )
            antenna = aggregate_bodypoint_group_slice(
                cache,
                start_index,
                stop_index,
                bodypoint_ids=ANTENNA_BODYPOINT_IDS,
                bodypoint_percentile=params.antenna_percentile,
                min_valid_bodypoint_fraction=params.min_valid_antenna_fraction,
                **common,
            )
            motion = MotionBlock(start_frame, stop_frame, body, antenna)
            self.motion_blocks[motion_key] = motion
            while len(self.motion_blocks) > self.block_cache_size:
                self.motion_blocks.popitem(last=False)
        else:
            self.motion_blocks.move_to_end(motion_key)
        classified = classify_body_antenna_quiet(
            motion.body_motion,
            motion.antenna_motion,
            fps=cache.fps,
            body_speed_threshold_mm_s=params.body_threshold_mm_s,
            antenna_speed_threshold_mm_s=params.antenna_threshold_mm_s,
            window_seconds=params.window_seconds,
            quiet_fraction_threshold=params.quiet_fraction,
            min_valid_fraction=params.min_valid_frame_fraction,
            require_full_window=True,
        )
        return StateBlock(
            start_frame=start_frame,
            stop_frame=stop_frame,
            body_motion=motion.body_motion,
            antenna_motion=motion.antenna_motion,
            state=classified["state"],
            quiet_fraction=classified["quiet_fraction"],
            valid_fraction=classified["valid_fraction"],
            mean_body_speed=classified["mean_body_speed_mm_s"],
            mean_antenna_speed=classified["mean_antenna_speed_mm_s"],
        )

    def state_for(
        self,
        *,
        side: str,
        track_id: int,
        frame: int,
        block_start: int,
        block_stop: int,
        params: SleepParameters,
    ) -> dict[str, object]:
        record = self.records.get((str(side), int(track_id)))
        if record is None:
            return {"label": "no cache", "side": side}
        cache = record.load()
        evaluation_frames = max(1, int(round(300.0 * cache.fps)))
        evaluation_start = int(block_start) + (
            (int(frame) - int(block_start)) // evaluation_frames
        ) * evaluation_frames
        evaluation_stop = min(int(block_stop), evaluation_start + evaluation_frames)
        cache_key = (
            str(side),
            int(track_id),
            evaluation_start,
            evaluation_stop,
            *params.key(),
        )
        block = self.blocks.get(cache_key)
        if block is None:
            block = self._build_block(
                record,
                block_start=evaluation_start,
                block_stop=evaluation_stop,
                params=params,
            )
            self.blocks[cache_key] = block
            while len(self.blocks) > self.block_cache_size:
                self.blocks.popitem(last=False)
        else:
            self.blocks.move_to_end(cache_key)

        index = int(frame) - block.start_frame
        if index < 0 or index >= len(block.state):
            return {"label": "no data", "side": side}
        state = int(block.state[index])
        valid_fraction = float(block.valid_fraction[index])
        if state == 1:
            label = "sleep"
        elif state == 0:
            label = "wake"
        elif int(frame) - cache.frame_min + 1 < max(1, int(round(params.window_seconds * cache.fps))):
            label = "warmup"
        elif valid_fraction < params.min_valid_frame_fraction:
            label = "no data"
        else:
            label = "warmup"
        return {
            "label": label,
            "side": side,
            "body_speed_mm_s": float(block.body_motion[index]),
            "antenna_speed_mm_s": float(block.antenna_motion[index]),
            "mean_body_speed_mm_s": float(block.mean_body_speed[index]),
            "mean_antenna_speed_mm_s": float(block.mean_antenna_speed[index]),
            "quiet_fraction": float(block.quiet_fraction[index]),
            "valid_fraction": valid_fraction,
        }


@dataclass(frozen=True)
class ArucoPoint:
    track_id: int
    x: float
    y: float
    confidence: float


class ChunkLabelReader:
    """Keep only the current chunk's H5 handles and a small frame cache open."""

    def __init__(self, specs: list[ChunkSpec]):
        self.specs = list(specs)
        self._stops = np.asarray([spec.stop for spec in specs], dtype=np.int64)
        self.spec: Optional[ChunkSpec] = None
        self.aruco_h5: Optional[h5py.File] = None
        self.sleap_h5: Optional[h5py.File] = None
        self.aruco_store: Optional[ArucoDetectionStore] = None
        self.sleap_store: Optional[SleapSkeletonStore] = None
        self.sleap_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.sleap_last_frame: Optional[int] = None
        self.sleap_last_stop = 0

    def spec_for_frame(self, frame: int) -> ChunkSpec:
        index = int(np.searchsorted(self._stops, int(frame), side="right"))
        return self.specs[min(max(index, 0), len(self.specs) - 1)]

    def _open(self, spec: ChunkSpec) -> None:
        if self.spec == spec:
            return
        self.close_handles()
        self.spec = spec
        dense_path = spec.aruco_path if "_aruco_tracks" in spec.aruco_path.stem else matching_aruco_tracks_path(spec.aruco_path)
        if dense_path is not None and dense_path.exists():
            self.aruco_h5 = h5py.File(dense_path, "r")
        else:
            self.aruco_store = ArucoDetectionStore.from_path(spec.aruco_path)
        if spec.sleap_path is not None:
            if spec.sleap_path.suffix.lower() in {".h5", ".hdf5"}:
                self.sleap_h5 = h5py.File(spec.sleap_path, "r")
            else:
                self.sleap_store = SleapSkeletonStore.from_path(spec.sleap_path)

    def aruco_for_frame(self, frame: int) -> tuple[ChunkSpec, list[ArucoPoint]]:
        spec = self.spec_for_frame(frame)
        self._open(spec)
        local_frame = spec.to_local(frame)
        if self.aruco_h5 is not None:
            tracks = np.asarray(self.aruco_h5["aruco_tracks"][local_frame], dtype=np.float32)
            confidence = (
                np.asarray(self.aruco_h5["aruco_confidences"][local_frame], dtype=np.float32)
                if "aruco_confidences" in self.aruco_h5
                else np.ones(len(tracks), dtype=np.float32)
            )
            valid = np.isfinite(tracks).all(axis=1) & ((tracks[:, 0] != 0) | (tracks[:, 1] != 0))
            points = [
                ArucoPoint(int(track_id), float(tracks[track_id, 0]), float(tracks[track_id, 1]), float(confidence[track_id]))
                for track_id in np.flatnonzero(valid)
            ]
            return spec, points
        assert self.aruco_store is not None
        points = [
            ArucoPoint(int(det.instance), float(det.x), float(det.y), float(det.confidence))
            for det in self.aruco_store.get_frame_detections(local_frame)
        ]
        return spec, points

    @staticmethod
    def _lower_bound(ds: h5py.Dataset, frame: int) -> int:
        lo, hi = 0, int(ds.shape[0])
        while lo < hi:
            mid = (lo + hi) // 2
            if int(ds[mid]["Frame"]) < int(frame):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _sleap_h5_rows(self, local_frame: int) -> np.ndarray:
        cached = self.sleap_cache.get(int(local_frame))
        if cached is not None:
            self.sleap_cache.move_to_end(int(local_frame))
            return cached
        assert self.sleap_h5 is not None
        ds = self.sleap_h5["sleap_data"]
        if self.sleap_last_frame is not None and int(local_frame) == self.sleap_last_frame + 1:
            start = self.sleap_last_stop
        else:
            start = self._lower_bound(ds, int(local_frame))
        if start >= int(ds.shape[0]) or int(ds[start]["Frame"]) != int(local_frame):
            rows = np.empty((0, 5), dtype=np.float32)
            self.sleap_last_frame = int(local_frame)
            self.sleap_last_stop = start
        else:
            pos = start
            parts: list[np.ndarray] = []
            stop = start
            while pos < int(ds.shape[0]):
                raw = ds[pos : min(int(ds.shape[0]), pos + 2048)]
                frames = raw["Frame"]
                different = np.flatnonzero(frames != int(local_frame))
                take = int(different[0]) if len(different) else len(raw)
                if take:
                    parts.append(raw[:take])
                stop = pos + take
                if take < len(raw):
                    break
                pos += len(raw)
            raw_rows = np.concatenate(parts) if len(parts) > 1 else parts[0]
            rows = np.column_stack(
                [
                    raw_rows["Instance"],
                    raw_rows["Bodypoint"],
                    raw_rows["X"],
                    raw_rows["Y"],
                    raw_rows["Score_node"],
                ]
            ).astype(np.float32, copy=False)
            self.sleap_last_frame = int(local_frame)
            self.sleap_last_stop = stop
        self.sleap_cache[int(local_frame)] = rows
        while len(self.sleap_cache) > 96:
            self.sleap_cache.popitem(last=False)
        return rows

    def sleap_for_frame(self, frame: int) -> np.ndarray:
        spec = self.spec_for_frame(frame)
        self._open(spec)
        local_frame = spec.to_local(frame)
        if self.sleap_h5 is not None:
            return self._sleap_h5_rows(local_frame)
        if self.sleap_store is not None:
            return self.sleap_store.rows_for_frame(local_frame)
        return np.empty((0, 5), dtype=np.float32)

    def close_handles(self) -> None:
        if self.aruco_h5 is not None:
            self.aruco_h5.close()
        if self.sleap_h5 is not None:
            self.sleap_h5.close()
        self.aruco_h5 = None
        self.sleap_h5 = None
        self.aruco_store = None
        self.sleap_store = None
        self.sleap_cache.clear()
        self.sleap_last_frame = None
        self.sleap_last_stop = 0

    def close(self) -> None:
        self.close_handles()


def infer_hmats_path(block_dir: Path) -> Optional[Path]:
    calibration_root = Path(block_dir).parent.parent / "cameraArray_calib"
    candidates = list(calibration_root.glob("*/frame0/aruco_stitch/aruco_H_mats.npz"))
    if not candidates:
        return None
    dataset_match = re.search(r"(20\d{6})", Path(block_dir).parent.name)
    dataset_date = int(dataset_match.group(1)) if dataset_match else None

    def date_key(path: Path) -> int:
        match = re.search(r"(20\d{6})", str(path))
        return int(match.group(1)) if match else -1

    if dataset_date is not None:
        preceding = [path for path in candidates if 0 < date_key(path) <= dataset_date]
        if preceding:
            return max(preceding, key=date_key)
    return max(candidates, key=date_key)


def infer_x_split(block_dir: Path, fallback: float = 2475.0) -> float:
    path = Path(block_dir) / "stitched" / "grid_bounds_from_tracks.json"
    if path.exists():
        try:
            return float(json.loads(path.read_text())["x_split_px"])
        except Exception:
            pass
    return float(fallback)


class ArucoSleepViewer:
    def __init__(
        self,
        root: tk.Tk,
        *,
        video_path: Path,
        specs: list[ChunkSpec],
        motion_store: SleepMotionStore,
        params: SleepParameters,
        start_frame: int,
        playback_fps: float,
        side: str,
        panorama_h: Optional[np.ndarray],
        x_split_px: float,
    ):
        self.root = root
        self.video_path = Path(video_path)
        self.specs = specs
        self.labels = ChunkLabelReader(specs)
        self.motion_store = motion_store
        self.params = params.validate()
        self.video = OpenCvVideoReader(video_path)
        self.overlay_scale = max(1.0, float(self.video.width) / 1600.0)
        label_frames = int(specs[-1].stop)
        reported_frames = int(self.video.frame_count())
        self.frame_count = min(label_frames, reported_frames) if reported_frames > 0 else label_frames
        self.current_frame = min(max(0, int(start_frame)), self.frame_count - 1)
        self.playback_fps = max(0.25, float(playback_fps))
        self.is_playing = False
        self.play_job: Optional[str] = None
        self.slider_job: Optional[str] = None
        self.resize_job: Optional[str] = None
        self.setting_slider = False
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.canvas_image: Optional[int] = None
        self.current_annotated_rgb: Optional[np.ndarray] = None
        self.panorama_h = panorama_h
        self.x_split_px = float(x_split_px)
        self.match_distance_px = 160.0

        self.frame_var = tk.StringVar(value=str(self.current_frame))
        self.side_var = tk.StringVar(value=str(side))
        self.show_aruco_var = tk.BooleanVar(value=True)
        self.show_sleap_var = tk.BooleanVar(value=True)
        self.show_metrics_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="")
        self.parameter_status_var = tk.StringVar(value="")
        self.param_vars = {
            "body_threshold_mm_s": tk.DoubleVar(value=params.body_threshold_mm_s),
            "antenna_threshold_mm_s": tk.DoubleVar(value=params.antenna_threshold_mm_s),
            "window_seconds": tk.DoubleVar(value=params.window_seconds),
            "quiet_fraction": tk.DoubleVar(value=params.quiet_fraction),
            "min_valid_frame_fraction": tk.DoubleVar(value=params.min_valid_frame_fraction),
            "body_percentile": tk.DoubleVar(value=params.body_percentile),
            "antenna_percentile": tk.DoubleVar(value=params.antenna_percentile),
            "min_valid_body_fraction": tk.DoubleVar(value=params.min_valid_body_fraction),
            "min_valid_antenna_fraction": tk.DoubleVar(value=params.min_valid_antenna_fraction),
            "max_gap_frames": tk.IntVar(value=params.max_gap_frames),
            "max_speed_mm_s": tk.DoubleVar(value=params.max_speed_mm_s),
        }

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self.render)

    def _build_ui(self) -> None:
        self.root.title(f"Sleep classifier viewer - {self.video_path.name}")
        self.root.geometry("1500x920")
        self.root.minsize(980, 650)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(8, 7))
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(toolbar, text="|<", width=4, command=lambda: self.step(-24)).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="<", width=4, command=lambda: self.step(-1)).pack(side=tk.LEFT, padx=(4, 0))
        self.play_button = ttk.Button(toolbar, text="Play", width=7, command=self.toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text=">", width=4, command=lambda: self.step(1)).pack(side=tk.LEFT)
        ttk.Button(toolbar, text=">|", width=4, command=lambda: self.step(24)).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(toolbar, text="Frame").pack(side=tk.LEFT)
        frame_entry = ttk.Entry(toolbar, textvariable=self.frame_var, width=10)
        frame_entry.pack(side=tk.LEFT, padx=4)
        frame_entry.bind("<Return>", lambda _event: self.goto_from_entry())
        ttk.Button(toolbar, text="Go", command=self.goto_from_entry).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(toolbar, text="Playback fps").pack(side=tk.LEFT)
        fps_box = ttk.Combobox(toolbar, values=(2, 5, 10, 15, 24), width=5, state="readonly")
        fps_box.set(f"{self.playback_fps:g}")
        fps_box.pack(side=tk.LEFT, padx=(4, 14))
        fps_box.bind("<<ComboboxSelected>>", lambda _event: self.set_playback_fps(fps_box.get()))
        ttk.Checkbutton(toolbar, text="ArUco IDs", variable=self.show_aruco_var, command=self.render).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="SLEAP", variable=self.show_sleap_var, command=self.render).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(toolbar, text="Metrics", variable=self.show_metrics_var, command=self.render).pack(side=tk.LEFT, padx=(8, 0))

        self.canvas = tk.Canvas(self.root, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        panel = ttk.Frame(self.root, padding=(10, 6, 10, 6), width=330)
        panel.grid(row=1, column=1, sticky="ns")
        panel.grid_propagate(False)

        identity = ttk.LabelFrame(panel, text="Colony", padding=8)
        identity.pack(fill=tk.X)
        ttk.Label(identity, text="Side").grid(row=0, column=0, sticky="w")
        side_box = ttk.Combobox(identity, textvariable=self.side_var, values=("auto", "left", "right"), state="readonly", width=10)
        side_box.grid(row=0, column=1, sticky="e")
        identity.columnconfigure(1, weight=1)
        side_box.bind("<<ComboboxSelected>>", lambda _event: self.render())

        classifier = ttk.LabelFrame(panel, text="Sleep classifier", padding=8)
        classifier.pack(fill=tk.X, pady=(10, 0))
        fields = (
            ("Body threshold (mm/s)", "body_threshold_mm_s"),
            ("Antenna threshold (mm/s)", "antenna_threshold_mm_s"),
            ("History (s)", "window_seconds"),
            ("Low-motion fraction", "quiet_fraction"),
            ("Valid-frame fraction", "min_valid_frame_fraction"),
        )
        for row, (label, key) in enumerate(fields):
            ttk.Label(classifier, text=label).grid(row=row, column=0, sticky="w", pady=2)
            entry = ttk.Entry(classifier, textvariable=self.param_vars[key], width=9)
            entry.grid(row=row, column=1, sticky="e", pady=2)
            entry.bind("<Return>", lambda _event: self.apply_parameters())
        classifier.columnconfigure(0, weight=1)

        evidence = ttk.LabelFrame(panel, text="Motion evidence", padding=8)
        evidence.pack(fill=tk.X, pady=(10, 0))
        evidence_fields = (
            ("Body percentile", "body_percentile"),
            ("Antenna percentile", "antenna_percentile"),
            ("Valid bodypoint fraction", "min_valid_body_fraction"),
            ("Valid antenna fraction", "min_valid_antenna_fraction"),
            ("Maximum gap (frames)", "max_gap_frames"),
            ("Jump cutoff (mm/s)", "max_speed_mm_s"),
        )
        for row, (label, key) in enumerate(evidence_fields):
            ttk.Label(evidence, text=label).grid(row=row, column=0, sticky="w", pady=2)
            entry = ttk.Entry(evidence, textvariable=self.param_vars[key], width=9)
            entry.grid(row=row, column=1, sticky="e", pady=2)
            entry.bind("<Return>", lambda _event: self.apply_parameters())
        evidence.columnconfigure(0, weight=1)

        actions = ttk.Frame(panel)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="Apply", command=self.apply_parameters).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(actions, text="Reset", command=self.reset_parameters).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        ttk.Label(panel, textvariable=self.parameter_status_var, wraplength=305).pack(fill=tk.X, pady=(8, 0))

        slider_row = ttk.Frame(self.root, padding=(8, 5))
        slider_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        slider_row.columnconfigure(0, weight=1)
        self.slider = ttk.Scale(
            slider_row,
            from_=0,
            to=max(0, self.frame_count - 1),
            command=self.on_slider,
        )
        self.slider.grid(row=0, column=0, sticky="ew")
        self.slider.set(self.current_frame)
        ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(10, 3)).grid(
            row=3, column=0, columnspan=2, sticky="ew"
        )

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _event: self.toggle_playback())
        self.root.bind("<Left>", lambda _event: self.step(-1))
        self.root.bind("<Right>", lambda _event: self.step(1))
        self.root.bind("<Shift-Left>", lambda _event: self.step(-24))
        self.root.bind("<Shift-Right>", lambda _event: self.step(24))

    def parameter_values(self) -> SleepParameters:
        return SleepParameters(**{key: variable.get() for key, variable in self.param_vars.items()}).validate()

    def apply_parameters(self) -> None:
        try:
            params = self.parameter_values()
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid sleep parameters", str(exc), parent=self.root)
            return
        self.params = params
        self.motion_store.clear_blocks()
        self.parameter_status_var.set("Applied")
        self.render()

    def reset_parameters(self) -> None:
        defaults = SleepParameters()
        for key, variable in self.param_vars.items():
            variable.set(getattr(defaults, key))
        self.apply_parameters()

    def set_playback_fps(self, raw: str) -> None:
        try:
            self.playback_fps = max(0.25, float(raw))
        except ValueError:
            pass

    def on_slider(self, raw: str) -> None:
        if self.setting_slider:
            return
        target = min(max(0, int(round(float(raw)))), self.frame_count - 1)
        self.frame_var.set(str(target))
        if self.slider_job is not None:
            self.root.after_cancel(self.slider_job)
        self.slider_job = self.root.after(100, lambda: self.goto_frame(target, pause=True))

    def goto_from_entry(self) -> None:
        try:
            frame = int(self.frame_var.get())
        except ValueError:
            return
        self.goto_frame(frame, pause=True)

    def step(self, delta: int) -> None:
        self.goto_frame(self.current_frame + int(delta), pause=True)

    def goto_frame(self, frame: int, *, pause: bool) -> None:
        if pause:
            self.pause()
        self.current_frame = min(max(0, int(frame)), self.frame_count - 1)
        self.frame_var.set(str(self.current_frame))
        self.setting_slider = True
        self.slider.set(self.current_frame)
        self.setting_slider = False
        self.render()

    def toggle_playback(self) -> None:
        if self.is_playing:
            self.pause()
            return
        self.is_playing = True
        self.play_button.configure(text="Pause")
        self._schedule_play(1)

    def pause(self) -> None:
        self.is_playing = False
        self.play_button.configure(text="Play")
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def _schedule_play(self, delay_ms: int) -> None:
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
        self.play_job = self.root.after(max(1, int(delay_ms)), self.play_step)

    def play_step(self) -> None:
        self.play_job = None
        if not self.is_playing:
            return
        if self.current_frame >= self.frame_count - 1:
            self.pause()
            return
        started = time.perf_counter()
        self.current_frame += 1
        self.frame_var.set(str(self.current_frame))
        self.setting_slider = True
        self.slider.set(self.current_frame)
        self.setting_slider = False
        self.render()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._schedule_play(max(1, int(round(1000.0 / self.playback_fps - elapsed_ms))))

    def side_for_point(self, point: ArucoPoint) -> Optional[str]:
        selected = self.side_var.get().strip().lower()
        if selected in {"left", "right"}:
            return selected
        if self.panorama_h is not None:
            projected = self.panorama_h @ np.asarray([point.x, point.y, 1.0], dtype=np.float64)
            if np.isfinite(projected).all() and abs(float(projected[2])) > 1e-12:
                return "left" if float(projected[0] / projected[2]) < self.x_split_px else "right"
        return self.motion_store.only_side_for_track(point.track_id)

    def _draw_text(
        self,
        image: np.ndarray,
        text: str,
        point: tuple[int, int],
        color: tuple[int, int, int],
        *,
        scale: float = 0.65,
        thickness: int = 2,
    ) -> None:
        draw_scale = float(scale) * self.overlay_scale
        draw_thickness = max(1, int(round(float(thickness) * self.overlay_scale)))
        cv2.putText(
            image,
            text,
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            draw_scale,
            (0, 0, 0),
            draw_thickness + max(2, int(round(2 * self.overlay_scale))),
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            draw_scale,
            color,
            draw_thickness,
            cv2.LINE_AA,
        )

    def draw_sleap(self, image: np.ndarray, rows: np.ndarray, points: list[ArucoPoint]) -> int:
        if rows.size == 0:
            return 0
        instances: dict[int, dict[int, tuple[float, float, float]]] = {}
        for instance_f, bodypoint_f, x_f, y_f, score_f in rows:
            if not (math.isfinite(float(x_f)) and math.isfinite(float(y_f))):
                continue
            instances.setdefault(int(instance_f), {})[int(bodypoint_f)] = (
                float(x_f),
                float(y_f),
                float(score_f),
            )

        candidates: list[tuple[float, int, int]] = []
        max_distance_sq = self.match_distance_px**2
        for instance, nodes in instances.items():
            anchor = nodes.get(0)
            if anchor is None:
                continue
            for point in points:
                distance_sq = (anchor[0] - point.x) ** 2 + (anchor[1] - point.y) ** 2
                if distance_sq <= max_distance_sq:
                    candidates.append((distance_sq, instance, point.track_id))
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _distance, instance, track_id in sorted(candidates):
            if instance in assignments or track_id in used_tracks:
                continue
            assignments[instance] = track_id
            used_tracks.add(track_id)

        for instance, nodes in instances.items():
            track_id = assignments.get(instance)
            color = color_for_id(track_id) if track_id is not None else SLEAP_UNMATCHED_COLOR
            line_color = tuple(max(45, int(channel * 0.65)) for channel in color)
            for first, second in SLEAP_SKELETON_EDGES:
                if first not in nodes or second not in nodes:
                    continue
                p0 = (int(round(nodes[first][0])), int(round(nodes[first][1])))
                p1 = (int(round(nodes[second][0])), int(round(nodes[second][1])))
                cv2.line(image, p0, p1, line_color, max(2, int(round(2 * self.overlay_scale))), cv2.LINE_AA)
            for bodypoint, (x, y, _score) in nodes.items():
                radius = int(round((5 if bodypoint == 0 else 3) * self.overlay_scale))
                cv2.circle(image, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)
            anchor = nodes.get(0)
            if anchor is not None:
                label = f"S{instance}"
                self._draw_text(
                    image,
                    label,
                    (
                        int(round(anchor[0] - 36 * self.overlay_scale)),
                        int(round(anchor[1] - 10 * self.overlay_scale)),
                    ),
                    color,
                    scale=0.48,
                    thickness=1,
                )
        return len(instances)

    @staticmethod
    def state_style(label: str) -> tuple[tuple[int, int, int], str]:
        if label == "sleep":
            return SLEEP_COLOR, "SLEEP"
        if label == "wake":
            return WAKE_COLOR, "WAKE"
        if label == "warmup":
            return UNKNOWN_COLOR, "WARMUP"
        if label == "side?":
            return UNKNOWN_COLOR, "SIDE?"
        return UNKNOWN_COLOR, "NO DATA"

    def draw_aruco_and_sleep(
        self,
        image: np.ndarray,
        points: list[ArucoPoint],
        spec: ChunkSpec,
    ) -> dict[str, int]:
        counts = {"sleep": 0, "wake": 0, "unknown": 0}
        for point in points:
            center = (int(round(point.x)), int(round(point.y)))
            id_color = color_for_id(point.track_id)
            side = self.side_for_point(point)
            if side is None:
                state = {"label": "side?"}
            else:
                try:
                    state = self.motion_store.state_for(
                        side=side,
                        track_id=point.track_id,
                        frame=self.current_frame,
                        block_start=spec.start,
                        block_stop=spec.stop,
                        params=self.params,
                    )
                except Exception as exc:
                    state = {"label": "no data"}
                    self.parameter_status_var.set(str(exc))
            label = str(state["label"])
            state_color, state_text = self.state_style(label)
            count_key = label if label in {"sleep", "wake"} else "unknown"
            counts[count_key] += 1

            if self.show_aruco_var.get():
                cv2.circle(
                    image,
                    center,
                    int(round(11 * self.overlay_scale)),
                    id_color,
                    max(2, int(round(3 * self.overlay_scale))),
                    cv2.LINE_AA,
                )
                self._draw_text(
                    image,
                    f"T{point.track_id}",
                    (
                        int(round(center[0] + 13 * self.overlay_scale)),
                        int(round(center[1] - 11 * self.overlay_scale)),
                    ),
                    id_color,
                    scale=0.72,
                )
            cv2.circle(
                image,
                center,
                int(round(21 * self.overlay_scale)),
                state_color,
                max(2, int(round(3 * self.overlay_scale))),
                cv2.LINE_AA,
            )
            text = state_text
            if side is not None:
                text = f"{side[0].upper()} {text}"
            if self.show_metrics_var.get():
                body = float(state.get("body_speed_mm_s", np.nan))
                antenna = float(state.get("antenna_speed_mm_s", np.nan))
                quiet = float(state.get("quiet_fraction", np.nan))
                if math.isfinite(body):
                    text += f" B{body:.2f}"
                if math.isfinite(antenna):
                    text += f" A{antenna:.2f}"
                if math.isfinite(quiet):
                    text += f" q{quiet:.2f}"
            self._draw_text(
                image,
                text,
                (
                    int(round(center[0] + 13 * self.overlay_scale)),
                    int(round(center[1] + 22 * self.overlay_scale)),
                ),
                state_color,
                scale=0.62,
            )
        return counts

    def render(self) -> None:
        try:
            bgr = self.video.read_frame(self.current_frame)
            spec, points = self.labels.aruco_for_frame(self.current_frame)
            sleap_rows = self.labels.sleap_for_frame(self.current_frame) if self.show_sleap_var.get() else np.empty((0, 5))
            frame = bgr.copy()
            sleap_count = self.draw_sleap(frame, sleap_rows, points) if self.show_sleap_var.get() else 0
            counts = self.draw_aruco_and_sleep(frame, points, spec)
            self._draw_text(
                frame,
                f"Frame {self.current_frame}  chunk {spec.chunk:03d}",
                (12, 34),
                (255, 255, 255),
                scale=0.85,
            )
            self.current_annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_current_image()
            self.status_var.set(
                f"Frame {self.current_frame + 1:,}/{self.frame_count:,} | chunk {spec.chunk:03d} "
                f"local {spec.to_local(self.current_frame):,} | ArUco {len(points)} | SLEAP {sleap_count} | "
                f"sleep {counts['sleep']} | wake {counts['wake']} | unknown {counts['unknown']}"
            )
        except Exception as exc:
            self.status_var.set(str(exc))

    def display_current_image(self) -> None:
        if self.current_annotated_rgb is None:
            return
        canvas_width = max(10, self.canvas.winfo_width())
        canvas_height = max(10, self.canvas.winfo_height())
        source_height, source_width = self.current_annotated_rgb.shape[:2]
        scale = min(canvas_width / source_width, canvas_height / source_height)
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(self.current_annotated_rgb, (width, height), interpolation=interpolation)
        self.current_photo = ImageTk.PhotoImage(Image.fromarray(resized))
        x = max(0, (canvas_width - width) // 2)
        y = max(0, (canvas_height - height) // 2)
        if self.canvas_image is None:
            self.canvas_image = self.canvas.create_image(x, y, anchor=tk.NW, image=self.current_photo)
        else:
            self.canvas.coords(self.canvas_image, x, y)
            self.canvas.itemconfigure(self.canvas_image, image=self.current_photo)

    def on_canvas_resize(self, _event: tk.Event) -> None:
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(80, self.display_current_image)

    def close(self) -> None:
        self.pause()
        self.labels.close()
        self.video.close()
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Full camera video to review.")
    parser.add_argument("--block-dir", type=Path, default=None, help="Dataset block root. Defaults to the video directory.")
    parser.add_argument("--label-dir", type=Path, default=None, help="Chunked ArUco/SLEAP label directory.")
    parser.add_argument("--sleep-motion-root", type=Path, default=None, help="Root containing per-track sleep_motion caches.")
    parser.add_argument("--hmats", type=Path, default=None, help="Camera-to-panorama homography stack used by auto side selection.")
    parser.add_argument("--camera", type=int, default=None, help="One-based camera number. Defaults to the camNN video prefix.")
    parser.add_argument("--x-split-px", type=float, default=None, help="Panorama x coordinate separating left and right colonies.")
    parser.add_argument("--side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--playback-fps", type=float, default=10.0)
    parser.add_argument("--body-threshold", type=float, default=0.50)
    parser.add_argument("--antenna-threshold", type=float, default=0.70)
    parser.add_argument("--history-seconds", type=float, default=10.0)
    parser.add_argument("--quiet-fraction", type=float, default=0.90)
    parser.add_argument("--min-valid-frame-fraction", type=float, default=0.50)
    parser.add_argument("--body-percentile", type=float, default=75.0)
    parser.add_argument("--antenna-percentile", type=float, default=75.0)
    parser.add_argument("--min-valid-body-fraction", type=float, default=0.75)
    parser.add_argument("--min-valid-antenna-fraction", type=float, default=0.50)
    parser.add_argument("--max-gap-frames", type=int, default=5)
    parser.add_argument("--max-speed", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    block_dir = Path(args.block_dir) if args.block_dir is not None else video_path.parent
    label_dir = Path(args.label_dir) if args.label_dir is not None else block_dir / "data"
    sleep_motion_root = (
        Path(args.sleep_motion_root)
        if args.sleep_motion_root is not None
        else block_dir / "stitched" / "sleep_motion"
    )
    specs = resolve_chunk_specs(video_path, label_dir)
    motion_store = SleepMotionStore(sleep_motion_root)

    hmats_path = Path(args.hmats) if args.hmats is not None else infer_hmats_path(block_dir)
    camera_index = (
        int(args.camera) - 1 if args.camera is not None and int(args.camera) > 0 else args.camera
    )
    if camera_index is None:
        camera_index = parse_camera_index(video_path)
    panorama_h = None
    if hmats_path is not None and hmats_path.exists() and camera_index is not None:
        homographies = np.load(hmats_path)["H"]
        if 0 <= int(camera_index) < len(homographies):
            panorama_h = np.asarray(homographies[int(camera_index)], dtype=np.float64)
    x_split_px = float(args.x_split_px) if args.x_split_px is not None else infer_x_split(block_dir)

    params = SleepParameters(
        body_threshold_mm_s=float(args.body_threshold),
        antenna_threshold_mm_s=float(args.antenna_threshold),
        window_seconds=float(args.history_seconds),
        quiet_fraction=float(args.quiet_fraction),
        min_valid_frame_fraction=float(args.min_valid_frame_fraction),
        body_percentile=float(args.body_percentile),
        antenna_percentile=float(args.antenna_percentile),
        min_valid_body_fraction=float(args.min_valid_body_fraction),
        min_valid_antenna_fraction=float(args.min_valid_antenna_fraction),
        max_gap_frames=int(args.max_gap_frames),
        max_speed_mm_s=float(args.max_speed),
    ).validate()
    root = tk.Tk()
    ArucoSleepViewer(
        root,
        video_path=video_path,
        specs=specs,
        motion_store=motion_store,
        params=params,
        start_frame=int(args.start_frame),
        playback_fps=float(args.playback_fps),
        side=str(args.side),
        panorama_h=panorama_h,
        x_split_px=x_split_px,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
