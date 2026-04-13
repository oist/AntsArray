#!/usr/bin/env python3
"""
Interactive GUI for manual curation of ArUco detections before downstream tracking.

The app edits sparse `*_aruco_detections.csv` files alongside the source video.
It can then export:
  - curated CSV
  - dense H5 arrays compatible with the existing ArUco pipeline
  - JSON edit log

Core workflow
-------------
- Load one chunk video and its ArUco detections CSV.
- Navigate frames with keyboard, buttons, slider, or playback.
- Click detections to select them.
- Drag a selected detection to move it.
- Toggle add/update mode to place the tag ID shown in the tag box.
- Relabel or delete the selected detection.
- Save curated outputs without modifying the original files in place.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.font as tkfont
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


SUPPORTED_VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv")
SHORTCUTS_TEXT = (
    "Shortcuts:\n"
    "  Left / Right: previous or next frame\n"
    "  Shift+Left / Shift+Right: jump by 10 frames\n"
    "  Space: play or pause\n"
    "  A: toggle add/update mode\n"
    "  Delete: delete selected detection\n"
    "  Ctrl+Z / Ctrl+Y: undo or redo\n"
    "  Ctrl+S: save curated CSV + edit log\n"
)


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def color_for_id(instance: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(int(instance) + 19)
    vals = rng.integers(80, 255, size=3, dtype=np.int32)
    return int(vals[2]), int(vals[1]), int(vals[0])


@dataclass
class Detection:
    frame: int
    instance: int
    x: float
    y: float
    confidence: float = 1.0

    def clone(self) -> "Detection":
        return Detection(
            frame=int(self.frame),
            instance=int(self.instance),
            x=float(self.x),
            y=float(self.y),
            confidence=float(self.confidence),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "Frame": int(self.frame),
            "Instance": int(self.instance),
            "X": float(self.x),
            "Y": float(self.y),
            "Confidence": float(self.confidence),
        }


@dataclass
class FrameAction:
    kind: str
    frame: int
    before: List[Detection]
    after: List[Detection]
    details: Dict[str, object]
    created_at: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "frame": int(self.frame),
            "before": [d.to_dict() for d in self.before],
            "after": [d.to_dict() for d in self.after],
            "details": self.details,
            "created_at": float(self.created_at),
        }


class ArucoDetectionStore:
    REQUIRED_COLUMNS = {"Frame", "Instance", "X", "Y"}

    def __init__(self, source_csv: Path):
        self.source_csv = Path(source_csv)
        self.frame_map: Dict[int, Dict[int, Detection]] = {}
        self.id_to_frames: Dict[int, List[int]] = {}
        self.id_to_frame_sets: Dict[int, set[int]] = {}

    @classmethod
    def from_csv(cls, path: Path) -> "ArucoDetectionStore":
        path = Path(path)
        df = pd.read_csv(path)
        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"ArUco detections missing required columns: {sorted(missing)}")

        df = df.copy()
        if "Confidence" not in df.columns:
            df["Confidence"] = 1.0

        for col in ("Frame", "Instance", "X", "Y", "Confidence"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Frame", "Instance", "X", "Y", "Confidence"]).copy()
        df["Frame"] = df["Frame"].astype(int)
        df["Instance"] = df["Instance"].astype(int)

        store = cls(path)
        for row in df.itertuples(index=False):
            det = Detection(
                frame=int(row.Frame),
                instance=int(row.Instance),
                x=float(row.X),
                y=float(row.Y),
                confidence=float(row.Confidence),
            )
            store._set_detection(det)
        return store

    def _set_detection(self, det: Detection) -> None:
        frame_dets = self.frame_map.setdefault(det.frame, {})
        existed = det.instance in frame_dets
        frame_dets[det.instance] = det
        if not existed:
            self._add_frame_for_id(det.instance, det.frame)

    def _add_frame_for_id(self, instance: int, frame: int) -> None:
        frame_set = self.id_to_frame_sets.setdefault(instance, set())
        if frame in frame_set:
            return
        frame_set.add(frame)
        frame_list = self.id_to_frames.setdefault(instance, [])
        bisect.insort(frame_list, frame)

    def _remove_frame_for_id(self, instance: int, frame: int) -> None:
        frame_set = self.id_to_frame_sets.get(instance)
        if frame_set is None or frame not in frame_set:
            return
        frame_set.remove(frame)
        frame_list = self.id_to_frames.get(instance, [])
        idx = bisect.bisect_left(frame_list, frame)
        if 0 <= idx < len(frame_list) and frame_list[idx] == frame:
            frame_list.pop(idx)
        if not frame_list:
            self.id_to_frames.pop(instance, None)
            self.id_to_frame_sets.pop(instance, None)

    def snapshot_frame(self, frame: int) -> List[Detection]:
        frame_dets = self.frame_map.get(frame, {})
        return [frame_dets[key].clone() for key in sorted(frame_dets)]

    def apply_frame_snapshot(self, frame: int, snapshot: Iterable[Detection]) -> None:
        old_ids = set(self.frame_map.get(frame, {}))
        new_dets = [det.clone() for det in snapshot]
        new_ids = {det.instance for det in new_dets}

        for instance in old_ids - new_ids:
            self._remove_frame_for_id(instance, frame)
        for instance in new_ids - old_ids:
            self._add_frame_for_id(instance, frame)

        if new_dets:
            self.frame_map[frame] = {det.instance: det for det in new_dets}
        else:
            self.frame_map.pop(frame, None)

    def get_frame_detections(self, frame: int) -> List[Detection]:
        return self.snapshot_frame(frame)

    def get_detection(self, frame: int, instance: int) -> Optional[Detection]:
        frame_dets = self.frame_map.get(frame, {})
        det = frame_dets.get(instance)
        return det.clone() if det is not None else None

    def preview_move(self, frame: int, instance: int, x: float, y: float) -> None:
        det = self.frame_map.get(frame, {}).get(instance)
        if det is None:
            return
        det.x = float(x)
        det.y = float(y)

    def create_upsert_action(
        self,
        frame: int,
        instance: int,
        x: float,
        y: float,
        confidence: float = 1.0,
        *,
        note: str = "",
    ) -> FrameAction:
        before = self.snapshot_frame(frame)
        frame_dets = self.frame_map.setdefault(frame, {})
        frame_dets[instance] = Detection(
            frame=int(frame),
            instance=int(instance),
            x=float(x),
            y=float(y),
            confidence=float(confidence),
        )
        self._add_frame_for_id(int(instance), int(frame))
        after = self.snapshot_frame(frame)
        return FrameAction(
            kind="upsert",
            frame=int(frame),
            before=before,
            after=after,
            details={"instance": int(instance), "note": note},
            created_at=time.time(),
        )

    def create_delete_action(
        self,
        frame: int,
        instance: int,
        *,
        note: str = "",
    ) -> Optional[FrameAction]:
        if frame not in self.frame_map or instance not in self.frame_map[frame]:
            return None
        before = self.snapshot_frame(frame)
        self.frame_map[frame].pop(instance, None)
        self._remove_frame_for_id(int(instance), int(frame))
        if not self.frame_map.get(frame):
            self.frame_map.pop(frame, None)
        after = self.snapshot_frame(frame)
        return FrameAction(
            kind="delete",
            frame=int(frame),
            before=before,
            after=after,
            details={"instance": int(instance), "note": note},
            created_at=time.time(),
        )

    def create_relabel_action(
        self,
        frame: int,
        old_instance: int,
        new_instance: int,
        *,
        note: str = "",
    ) -> Optional[FrameAction]:
        if frame not in self.frame_map or old_instance not in self.frame_map[frame]:
            return None
        before = self.snapshot_frame(frame)
        det = self.frame_map[frame].pop(old_instance)
        self._remove_frame_for_id(int(old_instance), int(frame))
        det.instance = int(new_instance)
        self.frame_map[frame][int(new_instance)] = det
        self._add_frame_for_id(int(new_instance), int(frame))
        after = self.snapshot_frame(frame)
        return FrameAction(
            kind="relabel",
            frame=int(frame),
            before=before,
            after=after,
            details={
                "old_instance": int(old_instance),
                "new_instance": int(new_instance),
                "note": note,
            },
            created_at=time.time(),
        )

    def create_move_action(
        self,
        frame: int,
        before_snapshot: List[Detection],
        *,
        instance: int,
        note: str = "",
    ) -> Optional[FrameAction]:
        after = self.snapshot_frame(frame)
        if len(before_snapshot) != len(after):
            changed = True
        else:
            changed = any(
                (
                    int(a.instance) != int(b.instance)
                    or abs(float(a.x) - float(b.x)) > 1e-9
                    or abs(float(a.y) - float(b.y)) > 1e-9
                    or abs(float(a.confidence) - float(b.confidence)) > 1e-9
                )
                for a, b in zip(before_snapshot, after)
            )
        if not changed:
            return None
        return FrameAction(
            kind="move",
            frame=int(frame),
            before=[det.clone() for det in before_snapshot],
            after=after,
            details={"instance": int(instance), "note": note},
            created_at=time.time(),
        )

    def create_batch_upsert_action(
        self,
        frame: int,
        detections: Iterable[Detection],
        *,
        note: str = "",
    ) -> Optional[FrameAction]:
        to_apply = [det.clone() for det in detections]
        if not to_apply:
            return None
        before = self.snapshot_frame(frame)
        frame_dets = self.frame_map.setdefault(int(frame), {})
        for det in to_apply:
            det.frame = int(frame)
            frame_dets[int(det.instance)] = det
            self._add_frame_for_id(int(det.instance), int(frame))
        after = self.snapshot_frame(frame)
        return FrameAction(
            kind="batch_upsert",
            frame=int(frame),
            before=before,
            after=after,
            details={
                "instances": [int(det.instance) for det in to_apply],
                "count": int(len(to_apply)),
                "note": note,
            },
            created_at=time.time(),
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for frame in sorted(self.frame_map):
            for det in self.snapshot_frame(frame):
                rows.append(det.to_dict())
        if not rows:
            return pd.DataFrame(columns=["Frame", "Instance", "X", "Y", "Confidence"])
        return pd.DataFrame(rows, columns=["Frame", "Instance", "X", "Y", "Confidence"])

    def export_dense_h5(self, output_path: Path, frame_count: int, dictionary_size: int) -> None:
        tracks = np.zeros((int(frame_count), int(dictionary_size), 2), dtype=np.float32)
        confidences = np.zeros((int(frame_count), int(dictionary_size)), dtype=np.float32)

        for frame, frame_dets in self.frame_map.items():
            if frame < 0 or frame >= frame_count:
                continue
            for instance, det in frame_dets.items():
                if instance < 0 or instance >= dictionary_size:
                    continue
                tracks[frame, instance, 0] = float(det.x)
                tracks[frame, instance, 1] = float(det.y)
                confidences[frame, instance] = float(det.confidence)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as h5f:
            h5f.create_dataset("aruco_tracks", data=tracks)
            h5f.create_dataset("aruco_confidences", data=confidences)

    def present_frames_for_id(self, instance: int) -> List[int]:
        return list(self.id_to_frames.get(int(instance), []))

    def frame_has_instance(self, frame: int, instance: int) -> bool:
        return int(frame) in self.id_to_frame_sets.get(int(instance), set())

    def next_present_frame(self, instance: int, current: int) -> Optional[int]:
        frames = self.id_to_frames.get(int(instance), [])
        idx = bisect.bisect_right(frames, int(current))
        return int(frames[idx]) if idx < len(frames) else None

    def prev_present_frame(self, instance: int, current: int) -> Optional[int]:
        frames = self.id_to_frames.get(int(instance), [])
        idx = bisect.bisect_left(frames, int(current)) - 1
        return int(frames[idx]) if idx >= 0 else None

    def next_missing_frame(self, instance: int, current: int, frame_count: int) -> Optional[int]:
        seen = self.id_to_frame_sets.get(int(instance), set())
        for frame in range(int(current) + 1, int(frame_count)):
            if frame not in seen:
                return frame
        return None

    def prev_missing_frame(self, instance: int, current: int) -> Optional[int]:
        seen = self.id_to_frame_sets.get(int(instance), set())
        for frame in range(int(current) - 1, -1, -1):
            if frame not in seen:
                return frame
        return None


class SleapAnchorStore:
    def __init__(self, source_csv: Path, bodypoint: int):
        self.source_csv = Path(source_csv)
        self.bodypoint = int(bodypoint)
        self.frame_candidates: Dict[int, np.ndarray] = {}

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        bodypoint: int = 0,
        chunksize: int = 300_000,
    ) -> "SleapAnchorStore":
        path = Path(path)
        grouped: Dict[int, List[np.ndarray]] = {}
        wanted = {"Frame", "Bodypoint", "X", "Y"}

        reader = pd.read_csv(
            path,
            usecols=lambda c: c in wanted,
            chunksize=int(chunksize),
        )
        for chunk in reader:
            for col in ("Frame", "Bodypoint", "X", "Y"):
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunk = chunk.dropna(subset=["Frame", "Bodypoint", "X", "Y"])
            chunk = chunk[chunk["Bodypoint"].astype(int) == int(bodypoint)]
            if chunk.empty:
                continue
            chunk["Frame"] = chunk["Frame"].astype(int)
            for frame, g in chunk.groupby("Frame", sort=False):
                arr = g[["X", "Y"]].to_numpy(dtype=np.float32)
                if arr.size == 0:
                    continue
                grouped.setdefault(int(frame), []).append(arr)

        store = cls(path, bodypoint=int(bodypoint))
        store.frame_candidates = {
            int(frame): np.vstack(parts)
            for frame, parts in grouped.items()
            if parts
        }
        return store

    def candidates_for_frame(self, frame: int) -> np.ndarray:
        arr = self.frame_candidates.get(int(frame))
        if arr is None:
            return np.empty((0, 2), dtype=np.float32)
        return arr


def infer_detections_from_video(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_aruco_detections.csv")


def infer_sleap_from_video(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_sleap_data.csv")


def infer_video_from_detections(detections_path: Path) -> Optional[Path]:
    stem = detections_path.stem
    suffix = "_aruco_detections"
    if not stem.endswith(suffix):
        return None
    video_stem = stem[: -len(suffix)]
    for ext in SUPPORTED_VIDEO_SUFFIXES:
        cand = detections_path.with_name(video_stem + ext)
        if cand.exists():
            return cand
    return None


def infer_frame_counts_sidecar(video_path: Path) -> Optional[Path]:
    stem = video_path.stem
    if len(stem) >= 4 and stem[-4] == "_" and stem[-3:].isdigit():
        prefix = stem[:-4]
        cand = video_path.with_name(f"{prefix}_frame_counts.csv")
        if cand.exists():
            return cand
    return None


def lookup_frame_count_from_sidecar(sidecar: Path, video_name: str) -> Optional[int]:
    try:
        df = pd.read_csv(sidecar, header=None, names=["video", "frame_count"])
    except Exception:
        return None
    if df.empty:
        return None
    df["video"] = df["video"].astype(str)
    df["frame_count"] = pd.to_numeric(df["frame_count"], errors="coerce")
    row = df[df["video"] == video_name]
    if row.empty:
        return None
    val = row["frame_count"].iloc[0]
    if pd.isna(val):
        return None
    return int(val)


def resolve_frame_count(video_path: Path, cap: cv2.VideoCapture) -> int:
    sidecar = infer_frame_counts_sidecar(video_path)
    if sidecar is not None:
        sidecar_count = lookup_frame_count_from_sidecar(sidecar, video_path.name)
        if sidecar_count is not None and sidecar_count > 0:
            return int(sidecar_count)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        raise RuntimeError(f"Could not determine frame count for {video_path}")
    return count


def pick_path_with_dialogs(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.video is not None and args.detections is not None:
        return Path(args.video), Path(args.detections)

    chooser = tk.Tk()
    chooser.withdraw()

    video_path: Optional[Path] = Path(args.video) if args.video else None
    detections_path: Optional[Path] = Path(args.detections) if args.detections else None

    if video_path is None and detections_path is not None:
        inferred_video = infer_video_from_detections(detections_path)
        if inferred_video is not None:
            video_path = inferred_video

    if video_path is None:
        selected = filedialog.askopenfilename(
            title="Choose ArUco source video",
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")],
        )
        if not selected:
            chooser.destroy()
            raise SystemExit(1)
        video_path = Path(selected)

    if detections_path is None:
        inferred = infer_detections_from_video(video_path)
        if inferred.exists():
            detections_path = inferred
        else:
            selected = filedialog.askopenfilename(
                title="Choose ArUco detections CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not selected:
                chooser.destroy()
                raise SystemExit(1)
            detections_path = Path(selected)

    chooser.destroy()
    return video_path, detections_path


class ArucoCurationApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        video_path: Path,
        detections_path: Path,
        output_dir: Path,
        dictionary_size: int,
        start_frame: int,
        playback_fps: float,
    ):
        self.root = root
        self.video_path = Path(video_path)
        self.detections_path = Path(detections_path)
        self.output_dir = Path(output_dir)
        self.dictionary_size = int(dictionary_size)
        self.playback_delay_ms = max(10, int(round(1000.0 / max(playback_fps, 1.0))))
        self.sleap_path = infer_sleap_from_video(self.video_path)
        self.sleap_store: Optional[SleapAnchorStore] = None
        self.sleap_loading = False
        self.sleap_error: Optional[str] = None

        self.store = ArucoDetectionStore.from_csv(self.detections_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")

        self.frame_count = resolve_frame_count(self.video_path, self.cap)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1

        self.current_frame = clamp(int(start_frame), 0, max(0, self.frame_count - 1))
        self.current_bgr: Optional[np.ndarray] = None
        self.current_bgr_frame: Optional[int] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.trajectory_photo: Optional[ImageTk.PhotoImage] = None
        self.current_trajectory_render: Optional[Dict[str, object]] = None
        self.drag_before_snapshot: Optional[List[Detection]] = None
        self.drag_instance: Optional[int] = None
        self.dragging = False
        self.slider_job: Optional[str] = None
        self.play_job: Optional[str] = None
        self.canvas_resize_job: Optional[str] = None
        self.trajectory_canvas_resize_job: Optional[str] = None
        self.setting_slider = False
        self.updating_tree = False
        self.is_playing = False
        self.auto_bridge_active = False
        self.auto_bridge_scope: Optional[str] = None
        self.dirty = False
        self.status_note = ""
        self.decoded_frame_index: Optional[int] = None
        self.last_status_refresh_at = 0.0
        self.last_tree_refresh_at = 0.0
        self.last_controls_refresh_at = 0.0
        self.playback_status_interval_s = 0.15
        self.playback_tree_interval_s = 0.25
        self.playback_controls_interval_s = 0.15
        self.playback_trajectory_interval_s = 0.35
        self.last_trajectory_refresh_at = 0.0

        self.undo_stack: List[FrameAction] = []
        self.redo_stack: List[FrameAction] = []
        self.saved_actions: List[FrameAction] = []
        self.bridge_command_ids: List[int] = []
        self.next_bridge_command_id = 1
        self.trajectory_data_cache: Dict[int, Dict[str, object]] = {}
        self.trajectory_image_cache: Dict[Tuple[int, int, int], Dict[str, object]] = {}

        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.render_width = self.video_width
        self.render_height = self.video_height

        self.add_mode_var = tk.BooleanVar(value=False)
        self.frame_var = tk.StringVar(value=str(self.current_frame))
        self.tag_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="")
        self.selected_instance_var = tk.StringVar(value="None")
        self.trajectory_status_var = tk.StringVar(value="Trajectory: no tag selected")
        self.default_confidence_var = tk.DoubleVar(value=1.0)
        self.sleap_status_var = tk.StringVar(value="SLEAP: not loaded")
        self.bridge_max_distance_var = tk.DoubleVar(value=160.0)
        self.bridge_max_frames_var = tk.IntVar(value=250)
        self.bridge_preview_start_var = tk.IntVar(value=0)
        self.bridge_preview_end_var = tk.IntVar(value=max(0, self.frame_count - 1))
        self.trajectory_zoom_var = tk.DoubleVar(value=1.0)

        self.canvas_item_id: Optional[int] = None
        self.trajectory_canvas_item_id: Optional[int] = None
        self.selected_detection: Optional[Tuple[int, int]] = None

        self._build_ui()
        self.tag_var.trace_add("write", self.on_tag_var_changed)
        self.trajectory_zoom_var.trace_add("write", self.on_trajectory_zoom_changed)
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self.render_current_frame)
        self.root.after(100, self.start_sleap_load)

    def _configure_styles(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=15)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=15)
        heading_font = tkfont.nametofont("TkHeadingFont")
        heading_font.configure(size=16, weight="bold")
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(size=14)
        self.compact_font = default_font.copy()
        self.compact_font.configure(size=12)

        style = ttk.Style(self.root)
        style.configure("TButton", padding=(12, 8), font=default_font)
        style.configure("Compact.TButton", padding=(6, 4), font=self.compact_font)
        style.configure("TLabel", font=default_font)
        style.configure("TCheckbutton", font=default_font)
        style.configure("TLabelframe.Label", font=heading_font)
        style.configure("Treeview", font=default_font, rowheight=30)
        style.configure("Treeview.Heading", font=heading_font)
        style.configure("Transport.TButton", padding=(18, 12), font=heading_font)
        self.root.option_add("*Font", default_font)

    def _build_ui(self) -> None:
        self.root.title(f"ArUco Curation - {self.video_path.name}")
        self.root.geometry("1500x920")
        self._configure_styles()

        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        self.save_button = ttk.Button(top, text="Save CSV + Log", command=self.save_outputs)
        self.save_button.pack(side=tk.LEFT)
        self.export_h5_button = ttk.Button(top, text="Export Dense H5", command=self.export_dense_h5)
        self.export_h5_button.pack(side=tk.LEFT, padx=(8, 0))
        self.undo_button = ttk.Button(top, text="Undo", command=self.undo)
        self.undo_button.pack(side=tk.LEFT, padx=(12, 0))
        self.redo_button = ttk.Button(top, text="Redo", command=self.redo)
        self.redo_button.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text="Frame").pack(side=tk.LEFT)
        frame_entry = ttk.Entry(top, textvariable=self.frame_var, width=8)
        frame_entry.pack(side=tk.LEFT, padx=(4, 4))
        frame_entry.bind("<Return>", lambda _event: self.goto_frame_from_entry())
        ttk.Button(top, text="Go", command=self.goto_frame_from_entry).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Checkbutton(
            top,
            text="Add/Update Mode",
            variable=self.add_mode_var,
            command=self.refresh_status,
        ).pack(side=tk.LEFT)

        ttk.Label(top, text="Tag ID").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Spinbox(top, from_=0, to=max(999, self.dictionary_size - 1), textvariable=self.tag_var, width=8).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        ttk.Label(top, text="Confidence").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=self.default_confidence_var, width=6).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(top, text=f"Output: {self.output_dir}").pack(side=tk.RIGHT)

        slider_frame = ttk.Frame(self.root, padding=(10, 6, 10, 4))
        self.frame_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=max(0, self.frame_count - 1),
            orient=tk.HORIZONTAL,
            command=self.on_slider_changed,
        )
        self.frame_slider.pack(fill=tk.X)
        self.setting_slider = True
        self.frame_slider.set(self.current_frame)
        self.setting_slider = False

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)

        left = ttk.Frame(main, padding=(10, 4, 6, 10))
        main.add(left, weight=5)
        right = ttk.Frame(main, padding=(6, 4, 10, 10), width=360)
        main.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        compact_controls = ttk.Frame(right)
        compact_controls.pack(fill=tk.X, pady=(0, 8))
        compact_controls.columnconfigure(0, weight=1)
        compact_controls.columnconfigure(1, weight=1)

        info_box = ttk.LabelFrame(compact_controls, text="Selected")
        info_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(
            info_box,
            textvariable=self.selected_instance_var,
            justify=tk.LEFT,
            wraplength=150,
        ).pack(anchor=tk.W, padx=6, pady=(6, 4))
        info_actions = ttk.Frame(info_box)
        info_actions.pack(fill=tk.X, padx=6, pady=(0, 6))
        info_actions.columnconfigure(0, weight=1)
        info_actions.columnconfigure(1, weight=1)
        ttk.Button(
            info_actions,
            text="Relabel",
            style="Compact.TButton",
            command=self.relabel_selected,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            info_actions,
            text="Delete",
            style="Compact.TButton",
            command=self.delete_selected,
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0))

        nav_box = ttk.LabelFrame(compact_controls, text="Tag Navigation")
        nav_box.grid(row=0, column=1, sticky="nsew")
        nav_actions = ttk.Frame(nav_box)
        nav_actions.pack(fill=tk.X, padx=6, pady=6)
        nav_actions.columnconfigure(0, weight=1)
        nav_actions.columnconfigure(1, weight=1)
        ttk.Button(
            nav_actions,
            text="Prev Tag",
            style="Compact.TButton",
            command=lambda: self.jump_tag_presence(prev=True),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=(0, 4))
        ttk.Button(
            nav_actions,
            text="Next Tag",
            style="Compact.TButton",
            command=lambda: self.jump_tag_presence(prev=False),
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=(0, 4))
        ttk.Button(
            nav_actions,
            text="Prev Gap",
            style="Compact.TButton",
            command=lambda: self.jump_tag_missing(prev=True),
        ).grid(row=1, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            nav_actions,
            text="Next Gap",
            style="Compact.TButton",
            command=lambda: self.jump_tag_missing(prev=False),
        ).grid(row=1, column=1, sticky="ew", padx=(3, 0))

        sleap_box = ttk.LabelFrame(right, text="SLEAP Bridge")
        sleap_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(sleap_box, textvariable=self.sleap_status_var, wraplength=320, justify=tk.LEFT).pack(
            anchor=tk.W, padx=8, pady=(8, 6)
        )
        bridge_params = ttk.Frame(sleap_box)
        bridge_params.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(bridge_params, text="Max Dist").pack(side=tk.LEFT)
        ttk.Entry(bridge_params, textvariable=self.bridge_max_distance_var, width=7).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(bridge_params, text="Max Frames").pack(side=tk.LEFT)
        ttk.Entry(bridge_params, textvariable=self.bridge_max_frames_var, width=7).pack(side=tk.LEFT, padx=(4, 0))
        self.bridge_button = ttk.Button(
            sleap_box,
            text="Bridge Tag (SLEAP NN)",
            command=self.bridge_gap_with_sleap,
        )
        self.bridge_button.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.bridge_all_button = ttk.Button(
            sleap_box,
            text="Bridge All Tags (SLEAP NN)",
            command=self.bridge_all_gaps_with_sleap,
        )
        self.bridge_all_button.pack(fill=tk.X, padx=8, pady=(0, 6))
        preview_range = ttk.Frame(sleap_box)
        preview_range.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(preview_range, text="Range").pack(side=tk.LEFT)
        ttk.Entry(preview_range, textvariable=self.bridge_preview_start_var, width=7).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Label(preview_range, text="to").pack(side=tk.LEFT)
        ttk.Entry(preview_range, textvariable=self.bridge_preview_end_var, width=7).pack(side=tk.LEFT, padx=(4, 0))
        preview_actions = ttk.Frame(sleap_box)
        preview_actions.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.preview_all_bridge_button = ttk.Button(
            preview_actions,
            text="Preview All Frames",
            style="Compact.TButton",
            command=self.preview_bridge_all_frames,
        )
        self.preview_all_bridge_button.pack(side=tk.LEFT)
        self.preview_range_button = ttk.Button(
            preview_actions,
            text="Preview Range",
            style="Compact.TButton",
            command=self.preview_bridge_range_all,
        )
        self.preview_range_button.pack(side=tk.LEFT, padx=(6, 0))
        self.undo_bridge_button = ttk.Button(
            preview_actions,
            text="Back Bridge",
            style="Compact.TButton",
            command=self.undo_latest_bridge_command,
        )
        self.undo_bridge_button.pack(side=tk.RIGHT)

        self.right_notebook = ttk.Notebook(right)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)
        detections_tab = ttk.Frame(self.right_notebook)
        trajectory_tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(detections_tab, text="Current Frame Detections")
        self.right_notebook.add(trajectory_tab, text="Tag Trajectory")
        self.right_notebook.bind("<<NotebookTabChanged>>", self.on_right_tab_changed)
        self.detections_tab = detections_tab
        self.trajectory_tab = trajectory_tab

        tree_box = ttk.Frame(detections_tab)
        tree_box.pack(fill=tk.BOTH, expand=True)
        columns = ("instance", "x", "y", "confidence")
        self.tree = ttk.Treeview(tree_box, columns=columns, show="headings", height=18)
        self.tree.heading("instance", text="Tag")
        self.tree.heading("x", text="X")
        self.tree.heading("y", text="Y")
        self.tree.heading("confidence", text="Conf")
        self.tree.column("instance", width=60, anchor=tk.CENTER)
        self.tree.column("x", width=84, anchor=tk.E)
        self.tree.column("y", width=84, anchor=tk.E)
        self.tree.column("confidence", width=60, anchor=tk.E)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        tree_scroll = ttk.Scrollbar(tree_box, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=8, padx=(0, 8))
        self.tree.configure(yscrollcommand=tree_scroll.set)

        trajectory_info = ttk.Label(
            trajectory_tab,
            textvariable=self.trajectory_status_var,
            justify=tk.LEFT,
            wraplength=320,
        )
        trajectory_info.pack(fill=tk.X, padx=8, pady=(8, 4))
        trajectory_controls = ttk.Frame(trajectory_tab)
        trajectory_controls.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(trajectory_controls, text="Zoom").pack(side=tk.LEFT)
        zoom_spin = ttk.Spinbox(
            trajectory_controls,
            from_=1.0,
            to=12.0,
            increment=0.5,
            textvariable=self.trajectory_zoom_var,
            width=6,
        )
        zoom_spin.pack(side=tk.LEFT, padx=(4, 0))
        zoom_spin.bind("<Return>", lambda _event: self.on_trajectory_zoom_changed())
        ttk.Button(
            trajectory_controls,
            text="Reset",
            style="Compact.TButton",
            command=self.reset_trajectory_zoom,
        ).pack(side=tk.LEFT, padx=(8, 0))
        self.trajectory_canvas = tk.Canvas(trajectory_tab, bg="#111111", highlightthickness=0, height=420)
        self.trajectory_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.trajectory_canvas.bind("<Button-1>", self.on_trajectory_canvas_click)
        self.trajectory_canvas.bind("<Configure>", self.on_trajectory_canvas_configure)

        slider_frame.pack(side=tk.BOTTOM, fill=tk.X)

        transport = ttk.Frame(self.root, padding=(10, 2, 10, 8))
        transport.pack(side=tk.BOTTOM, fill=tk.X)
        self.rewind_button = ttk.Button(
            transport,
            text="<<",
            style="Transport.TButton",
            command=lambda: self.seek_relative(-100),
        )
        self.rewind_button.pack(side=tk.LEFT, padx=(0, 8))
        self.back_button = ttk.Button(
            transport,
            text="<",
            style="Transport.TButton",
            command=lambda: self.seek_relative(-10),
        )
        self.back_button.pack(side=tk.LEFT, padx=(0, 8))
        self.play_pause_button = ttk.Button(
            transport,
            text="Play",
            style="Transport.TButton",
            command=self.toggle_playback,
        )
        self.play_pause_button.pack(side=tk.LEFT, padx=(0, 8))
        self.forward_button = ttk.Button(
            transport,
            text=">",
            style="Transport.TButton",
            command=lambda: self.seek_relative(10),
        )
        self.forward_button.pack(side=tk.LEFT, padx=(0, 8))
        self.fast_forward_button = ttk.Button(
            transport,
            text=">>",
            style="Transport.TButton",
            command=lambda: self.seek_relative(100),
        )
        self.fast_forward_button.pack(side=tk.LEFT)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, padding=(10, 4))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        main.pack(fill=tk.BOTH, expand=True)

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _event: self.step_frame(-1))
        self.root.bind("<Right>", lambda _event: self.step_frame(1))
        self.root.bind("<Shift-Left>", lambda _event: self.step_frame(-10))
        self.root.bind("<Shift-Right>", lambda _event: self.step_frame(10))
        self.root.bind_all("<space>", self.on_space_hotkey)
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("<BackSpace>", lambda _event: self.delete_selected())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-y>", lambda _event: self.redo())
        self.root.bind("<Control-s>", lambda _event: self.save_outputs())
        self.root.bind("<KeyPress-a>", lambda _event: self.toggle_add_mode())

    def toggle_add_mode(self) -> None:
        self.add_mode_var.set(not self.add_mode_var.get())
        self.refresh_status()

    def on_space_hotkey(self, _event: tk.Event) -> str:
        self.toggle_playback()
        return "break"

    def start_sleap_load(self) -> None:
        if self.sleap_loading or self.sleap_store is not None:
            return
        if self.sleap_path is None or not self.sleap_path.exists():
            self.sleap_status_var.set("SLEAP: sidecar not found for this video")
            self.bridge_button.configure(state=tk.DISABLED)
            return

        self.sleap_loading = True
        self.sleap_error = None
        self.sleap_status_var.set(f"SLEAP: loading {self.sleap_path.name} (bodypoint 0)")
        self.bridge_button.configure(state=tk.DISABLED)
        worker = threading.Thread(target=self._load_sleap_worker, daemon=True)
        worker.start()

    def _load_sleap_worker(self) -> None:
        try:
            store = SleapAnchorStore.from_csv(self.sleap_path, bodypoint=0)
        except Exception as exc:
            self.root.after(0, lambda: self._finish_sleap_load(None, str(exc)))
            return
        self.root.after(0, lambda: self._finish_sleap_load(store, None))

    def _finish_sleap_load(
        self,
        store: Optional[SleapAnchorStore],
        error: Optional[str],
    ) -> None:
        self.sleap_loading = False
        self.sleap_error = error
        if error is not None:
            self.sleap_store = None
            self.sleap_status_var.set(f"SLEAP: load failed ({error})")
            self.update_bridge_button()
            return

        self.sleap_store = store
        loaded_frames = len(store.frame_candidates) if store is not None else 0
        self.sleap_status_var.set(f"SLEAP: loaded bodypoint 0 for {loaded_frames} frames")
        self.update_bridge_button()

    def update_bridge_button(self) -> None:
        if not hasattr(self, "bridge_button"):
            return
        single_text = "Bridge Tag (SLEAP NN): ON" if self.auto_bridge_scope == "single" else "Bridge Tag (SLEAP NN)"
        all_text = (
            "Bridge All Tags (SLEAP NN): ON" if self.auto_bridge_scope == "all" else "Bridge All Tags (SLEAP NN)"
        )
        state = tk.NORMAL
        if self.sleap_loading:
            single_text = "Bridge Tag (SLEAP NN): Loading"
            all_text = "Bridge All Tags (SLEAP NN): Loading"
            state = tk.DISABLED
        if self.sleap_store is None and not self.sleap_loading and self.sleap_path.exists() is False:
            state = tk.DISABLED
        self.bridge_button.configure(text=single_text, state=state)
        if hasattr(self, "bridge_all_button"):
            self.bridge_all_button.configure(text=all_text, state=state)
        if hasattr(self, "preview_all_bridge_button"):
            self.preview_all_bridge_button.configure(state=state)
        if hasattr(self, "preview_range_button"):
            self.preview_range_button.configure(state=state)
        if hasattr(self, "undo_bridge_button"):
            undo_state = tk.NORMAL if self.can_undo_latest_bridge_command() else tk.DISABLED
            self.undo_bridge_button.configure(state=undo_state)

    def current_right_tab(self) -> str:
        if not hasattr(self, "right_notebook"):
            return "detections"
        selected = self.right_notebook.select()
        if selected == str(self.trajectory_tab):
            return "trajectory"
        return "detections"

    def on_right_tab_changed(self, _event: tk.Event) -> None:
        self.refresh_right_panel(force=True)

    def on_tag_var_changed(self, *_args: object) -> None:
        if not hasattr(self, "root"):
            return
        if not self.is_playing and hasattr(self, "canvas"):
            self.render_current_frame()
        else:
            if self.current_right_tab() == "trajectory":
                self.refresh_trajectory_plot(force=True)
            self.refresh_status(force=True)

    def on_trajectory_zoom_changed(self, *_args: object) -> None:
        if not hasattr(self, "root"):
            return
        try:
            zoom = float(self.trajectory_zoom_var.get())
        except Exception:
            return
        if zoom <= 0:
            self.trajectory_zoom_var.set(1.0)
            return
        self.trajectory_image_cache.clear()
        self.current_trajectory_render = None
        if self.current_right_tab() == "trajectory":
            self.refresh_trajectory_plot(force=True)

    def reset_trajectory_zoom(self) -> None:
        self.trajectory_zoom_var.set(1.0)

    def refresh_right_panel(self, *, force: bool = True) -> None:
        if self.current_right_tab() == "trajectory":
            self.refresh_trajectory_plot(force=force)
        else:
            self.refresh_tree(force=force)

    def sync_frame_controls(self, *, force: bool = True) -> None:
        now = time.perf_counter()
        if (
            not force
            and self.is_playing
            and (now - self.last_controls_refresh_at) < self.playback_controls_interval_s
        ):
            return
        self.last_controls_refresh_at = now
        self.frame_var.set(str(self.current_frame))
        self.setting_slider = True
        self.frame_slider.set(self.current_frame)
        self.setting_slider = False

    def refresh_status(self, *, force: bool = True) -> None:
        now = time.perf_counter()
        if (
            not force
            and self.is_playing
            and (now - self.last_status_refresh_at) < self.playback_status_interval_s
        ):
            return
        self.last_status_refresh_at = now
        mode = "ADD/UPDATE" if self.add_mode_var.get() else "SELECT/MOVE"
        selected = "None"
        if self.selected_detection is not None:
            selected = f"ID {self.selected_detection[1]}"
        count = len(self.store.get_frame_detections(self.current_frame))
        if hasattr(self, "play_pause_button"):
            self.play_pause_button.configure(text="Pause" if self.is_playing else "Play", command=self.toggle_playback)
        self.update_bridge_button()
        if self.auto_bridge_active and self.auto_bridge_scope == "all":
            bridge_mode = "AUTO-BRIDGE ALL"
        elif self.auto_bridge_active:
            bridge_mode = "AUTO-BRIDGE TAG"
        else:
            bridge_mode = "AUTO-BRIDGE OFF"
        note = f" | {self.status_note}" if self.status_note else ""
        self.status_var.set(
            f"Frame {self.current_frame + 1}/{self.frame_count} | "
            f"Mode {mode} | Selected {selected} | "
            f"Frame detections {count} | {bridge_mode} | Dirty {self.dirty}{note}"
        )

    def on_slider_changed(self, value: str) -> None:
        if self.setting_slider:
            return
        target = clamp(int(float(value)), 0, max(0, self.frame_count - 1))
        self.frame_var.set(str(target))
        if self.slider_job is not None:
            self.root.after_cancel(self.slider_job)
        self.slider_job = self.root.after(80, lambda: self.goto_frame(target))

    def goto_frame_from_entry(self) -> None:
        try:
            frame = int(self.frame_var.get())
        except ValueError:
            return
        self.goto_frame(frame)

    def step_frame(self, delta: int) -> None:
        self.goto_frame(self.current_frame + int(delta))

    def seek_relative(self, delta: int) -> None:
        self.pause_playback()
        self.step_frame(int(delta))

    def goto_frame(
        self,
        frame: int,
        *,
        update_controls: bool = True,
        full_refresh: bool = True,
    ) -> None:
        frame = clamp(int(frame), 0, max(0, self.frame_count - 1))
        if self.slider_job is not None:
            self.root.after_cancel(self.slider_job)
            self.slider_job = None
        self.current_frame = frame
        self.sync_frame_controls(force=update_controls)
        self.sync_selected_for_current_frame()
        self.render_current_frame(full_refresh=full_refresh)

    def start_playback(self) -> None:
        if self.is_playing:
            self.refresh_status()
            return
        self.status_note = "Auto-bridge running" if self.auto_bridge_active else "Playing"
        self.is_playing = True
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
        self.play_job = self.root.after(self.playback_delay_ms, self.play_step)
        self.refresh_status()

    def pause_playback(self) -> None:
        self.is_playing = False
        self.auto_bridge_active = False
        self.auto_bridge_scope = None
        self.status_note = "Paused"
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        self.sync_frame_controls(force=True)
        self.refresh_right_panel(force=True)
        self.refresh_status()

    def toggle_playback(self) -> None:
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def play_step(self) -> None:
        if not self.is_playing:
            return
        self.play_job = None
        started_at = time.perf_counter()
        if self.current_frame >= self.frame_count - 1:
            self.pause_playback()
            return
        self.current_frame = min(self.frame_count - 1, self.current_frame + 1)
        self.sync_selected_for_current_frame()
        if self.auto_bridge_active:
            if self.auto_bridge_scope == "all":
                self.attempt_bridge_all_current_frame(refresh_ui=False)
            else:
                self.attempt_bridge_current_frame(refresh_ui=False)
        self.render_current_frame(full_refresh=False)
        elapsed_ms = int(round((time.perf_counter() - started_at) * 1000.0))
        next_delay = max(1, self.playback_delay_ms - elapsed_ms)
        self.play_job = self.root.after(next_delay, self.play_step)

    def read_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if self.current_bgr is not None and self.current_bgr_frame == frame_idx:
            return self.current_bgr

        need_seek = self.decoded_frame_index is None or frame_idx != (self.decoded_frame_index + 1)
        if need_seek:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ok, frame = self.cap.read()
        if (not ok or frame is None) and not need_seek:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from {self.video_path}")
        self.decoded_frame_index = frame_idx
        self.current_bgr = frame
        self.current_bgr_frame = frame_idx
        return frame

    def draw_text_with_outline(
        self,
        image: np.ndarray,
        text: str,
        org: Tuple[int, int],
        color: Tuple[int, int, int],
        *,
        scale: float,
        thickness: int,
    ) -> None:
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def render_current_frame(self, *, full_refresh: bool = True) -> None:
        try:
            bgr = self.read_frame(self.current_frame)
        except Exception as exc:
            self.status_var.set(str(exc))
            return

        frame = bgr.copy()
        current_dets = self.store.get_frame_detections(self.current_frame)
        selected_instance = self.selected_detection[1] if self.selected_detection is not None else None

        for det in current_dets:
            color = color_for_id(det.instance)
            center = (int(round(det.x)), int(round(det.y)))
            radius = 14 if det.instance == selected_instance else 11
            thickness = 4 if det.instance == selected_instance else 3
            cv2.circle(frame, center, radius, color, thickness)
            if det.instance == selected_instance:
                cv2.circle(frame, center, radius + 7, (0, 255, 255), 3)
            self.draw_text_with_outline(
                frame,
                str(det.instance),
                (center[0] + 12, center[1] - 12),
                color,
                scale=1.5 if det.instance == selected_instance else 1.25,
                thickness=4 if det.instance == selected_instance else 3,
            )

        mode_text = "ADD/UPDATE" if self.add_mode_var.get() else "SELECT/MOVE"
        self.draw_text_with_outline(
            frame,
            f"Frame {self.current_frame} | Mode {mode_text}",
            (10, 34),
            (255, 255, 255),
            scale=1.0,
            thickness=2,
        )
        self.draw_text_with_outline(
            frame,
            f"Tag box: {self.tag_var.get()}",
            (10, 70),
            (255, 255, 255),
            scale=1.0,
            thickness=2,
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas_w = max(10, self.canvas.winfo_width())
        canvas_h = max(10, self.canvas.winfo_height())
        scale = min(canvas_w / self.video_width, canvas_h / self.video_height)
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0

        render_w = max(1, int(round(self.video_width * scale)))
        render_h = max(1, int(round(self.video_height * scale)))
        resized = cv2.resize(rgb, (render_w, render_h), interpolation=cv2.INTER_AREA)

        self.display_scale = float(scale)
        self.render_width = render_w
        self.render_height = render_h
        self.display_offset_x = max(0, (canvas_w - render_w) // 2)
        self.display_offset_y = max(0, (canvas_h - render_h) // 2)

        image = Image.fromarray(resized)
        self.current_photo = ImageTk.PhotoImage(image=image)

        if self.canvas_item_id is None:
            self.canvas_item_id = self.canvas.create_image(
                self.display_offset_x,
                self.display_offset_y,
                anchor=tk.NW,
                image=self.current_photo,
            )
        else:
            self.canvas.coords(self.canvas_item_id, self.display_offset_x, self.display_offset_y)
            self.canvas.itemconfigure(self.canvas_item_id, image=self.current_photo)

        self.sync_frame_controls(force=full_refresh)
        self.refresh_right_panel(force=full_refresh)
        self.refresh_status(force=full_refresh)

    def refresh_tree(self, *, force: bool = True) -> None:
        now = time.perf_counter()
        if (
            not force
            and self.is_playing
            and (now - self.last_tree_refresh_at) < self.playback_tree_interval_s
        ):
            return
        self.last_tree_refresh_at = now
        self.updating_tree = True
        selected_instance = self.selected_detection[1] if self.selected_detection is not None else None
        current_dets = self.store.get_frame_detections(self.current_frame)
        self.tree.delete(*self.tree.get_children())

        selected_row = None
        for det in current_dets:
            iid = f"{det.instance}"
            self.tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(
                    int(det.instance),
                    f"{det.x:.1f}",
                    f"{det.y:.1f}",
                    f"{det.confidence:.2f}",
                ),
            )
            if selected_instance is not None and int(det.instance) == int(selected_instance):
                selected_row = iid

        if selected_row is not None:
            self.tree.selection_set(selected_row)
            self.tree.see(selected_row)
        self.updating_tree = False

    def on_canvas_configure(self, _event: tk.Event) -> None:
        if self.canvas_resize_job is not None:
            self.root.after_cancel(self.canvas_resize_job)
        self.canvas_resize_job = self.root.after(30, self._render_after_resize)

    def _render_after_resize(self) -> None:
        self.canvas_resize_job = None
        self.render_current_frame()

    def on_trajectory_canvas_configure(self, _event: tk.Event) -> None:
        if self.trajectory_canvas_resize_job is not None:
            self.root.after_cancel(self.trajectory_canvas_resize_job)
        self.trajectory_canvas_resize_job = self.root.after(40, self._render_trajectory_after_resize)

    def _render_trajectory_after_resize(self) -> None:
        self.trajectory_canvas_resize_job = None
        self.refresh_trajectory_plot(force=True)

    def on_trajectory_canvas_click(self, event: tk.Event) -> None:
        if self.current_trajectory_render is None:
            return
        frames = self.current_trajectory_render.get("frames")
        x_pixels = self.current_trajectory_render.get("x_pixels")
        y_pixels = self.current_trajectory_render.get("y_pixels")
        tag = int(self.current_trajectory_render.get("tag", int(self.tag_var.get())))
        if frames is None or x_pixels is None or y_pixels is None:
            return
        if len(frames) == 0:
            return

        click = np.array([float(event.x), float(event.y)], dtype=np.float32)
        points = np.column_stack((x_pixels.astype(np.float32), y_pixels.astype(np.float32)))
        d2 = np.sum((points - click[None, :]) * (points - click[None, :]), axis=1)
        idx = int(np.argmin(d2))
        target_frame = int(frames[idx])

        self.pause_playback()
        self.selected_detection = (target_frame, tag)
        self.status_note = f"Trajectory seek: tag {tag} frame {target_frame}"
        self.goto_frame(target_frame)

    def invalidate_trajectory_cache(self) -> None:
        self.trajectory_data_cache.clear()
        self.trajectory_image_cache.clear()
        self.current_trajectory_render = None

    def get_trajectory_data(self, tag: int) -> Dict[str, object]:
        tag = int(tag)
        cached = self.trajectory_data_cache.get(tag)
        if cached is not None:
            return cached

        frames_list = self.store.present_frames_for_id(tag)
        if not frames_list:
            data = {
                "tag": tag,
                "frames": np.empty((0,), dtype=np.int32),
                "xs": np.empty((0,), dtype=np.float32),
                "ys": np.empty((0,), dtype=np.float32),
                "frame_to_index": {},
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            }
            self.trajectory_data_cache[tag] = data
            return data

        frames = np.asarray(frames_list, dtype=np.int32)
        xs = np.empty(len(frames), dtype=np.float32)
        ys = np.empty(len(frames), dtype=np.float32)
        for idx, frame in enumerate(frames):
            det = self.store.frame_map[int(frame)][tag]
            xs[idx] = float(det.x)
            ys[idx] = float(det.y)

        x_min = float(xs.min())
        x_max = float(xs.max())
        y_min = float(ys.min())
        y_max = float(ys.max())
        if abs(x_max - x_min) < 1e-6:
            x_min -= 1.0
            x_max += 1.0
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0

        data = {
            "tag": tag,
            "frames": frames,
            "xs": xs,
            "ys": ys,
            "frame_to_index": {int(frame): idx for idx, frame in enumerate(frames.tolist())},
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }
        self.trajectory_data_cache[tag] = data
        return data

    def build_trajectory_image(self, tag: int, canvas_w: int, canvas_h: int) -> Dict[str, object]:
        cache_key = (int(tag), int(canvas_w), int(canvas_h), round(float(self.trajectory_zoom_var.get()), 3), int(self.current_frame))
        cached = self.trajectory_image_cache.get(cache_key)
        if cached is not None:
            return cached

        data = self.get_trajectory_data(tag)
        image = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)
        left = 42
        right = max(left + 10, canvas_w - 16)
        top = 24
        bottom = max(top + 10, canvas_h - 48)
        plot_w = max(1, right - left)
        plot_h = max(1, bottom - top)

        cv2.rectangle(image, (left, top), (right, bottom), (80, 80, 80), 1)
        self.draw_text_with_outline(image, "X", (left, canvas_h - 16), (230, 230, 230), scale=0.55, thickness=1)
        self.draw_text_with_outline(image, "Y", (10, top + 14), (230, 230, 230), scale=0.55, thickness=1)

        frames = data["frames"]
        x_pixels = np.empty((0,), dtype=np.int32)
        y_pixels = np.empty((0,), dtype=np.int32)
        visible_frames = np.empty((0,), dtype=np.int32)
        frame_to_index: Dict[int, int] = {}
        try:
            zoom = float(self.trajectory_zoom_var.get())
        except Exception:
            zoom = 1.0
        zoom = max(1.0, min(12.0, zoom))

        if len(frames) == 0:
            self.draw_text_with_outline(
                image,
                f"No detections for tag {int(tag)}",
                (left + 8, top + 24),
                (220, 220, 220),
                scale=0.65,
                thickness=1,
            )
        else:
            xs = data["xs"]
            ys = data["ys"]
            x_min = float(data["x_min"])
            x_max = float(data["x_max"])
            y_min = float(data["y_min"])
            y_max = float(data["y_max"])
            current_idx = data["frame_to_index"].get(int(self.current_frame))
            if current_idx is not None:
                focus_x = float(xs[int(current_idx)])
                focus_y = float(ys[int(current_idx)])
            else:
                focus_x = 0.5 * (x_min + x_max)
                focus_y = 0.5 * (y_min + y_max)

            def fit_window(lo_full: float, hi_full: float, focus: float) -> Tuple[float, float]:
                span = max(1e-6, hi_full - lo_full)
                view = max(span / zoom, 1e-6)
                if view >= span:
                    return lo_full, hi_full
                lo = focus - 0.5 * view
                hi = focus + 0.5 * view
                if lo < lo_full:
                    hi += lo_full - lo
                    lo = lo_full
                if hi > hi_full:
                    lo -= hi - hi_full
                    hi = hi_full
                lo = max(lo_full, lo)
                hi = min(hi_full, hi)
                if (hi - lo) < 1e-6:
                    hi = lo + 1e-6
                return lo, hi

            x_lo, x_hi = fit_window(x_min, x_max, focus_x)
            y_lo, y_hi = fit_window(y_min, y_max, focus_y)
            visible_mask = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)
            if not np.any(visible_mask):
                visible_mask = np.ones(len(frames), dtype=bool)
                x_lo, x_hi = x_min, x_max
                y_lo, y_hi = y_min, y_max

            if len(frames) == 1:
                colors_all = np.array([[[0, 255, 255]]], dtype=np.uint8).reshape(-1, 3)
            else:
                ramp = np.linspace(0, 255, len(frames), dtype=np.uint8).reshape(-1, 1)
                colors_all = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO).reshape(-1, 3)

            visible_frames = frames[visible_mask]
            visible_xs = xs[visible_mask]
            visible_ys = ys[visible_mask]
            visible_colors = colors_all[visible_mask]
            x_pixels = left + np.clip(
                np.round((visible_xs - x_lo) / max(1e-6, x_hi - x_lo) * plot_w).astype(np.int32),
                0,
                plot_w,
            )
            y_pixels = top + np.clip(
                np.round((visible_ys - y_lo) / max(1e-6, y_hi - y_lo) * plot_h).astype(np.int32),
                0,
                plot_h,
            )
            frame_to_index = {int(frame): idx for idx, frame in enumerate(visible_frames.tolist())}

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xp = np.clip(x_pixels + dx, 0, canvas_w - 1)
                    yp = np.clip(y_pixels + dy, 0, canvas_h - 1)
                    image[yp, xp] = visible_colors

            legend = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_TURBO)
            legend = cv2.resize(legend, (min(140, plot_w), 12), interpolation=cv2.INTER_LINEAR)
            legend_x = right - legend.shape[1]
            legend_y = bottom + 12
            image[legend_y : legend_y + legend.shape[0], legend_x : legend_x + legend.shape[1]] = legend
            self.draw_text_with_outline(
                image,
                f"{int(frames[0])}",
                (legend_x, legend_y + 28),
                (220, 220, 220),
                scale=0.45,
                thickness=1,
            )
            self.draw_text_with_outline(
                image,
                f"{int(frames[-1])}",
                (legend_x + legend.shape[1] - 28, legend_y + 28),
                (220, 220, 220),
                scale=0.45,
                thickness=1,
            )
            self.draw_text_with_outline(
                image,
                "early",
                (legend_x, legend_y - 4),
                (220, 220, 220),
                scale=0.42,
                thickness=1,
            )
            self.draw_text_with_outline(
                image,
                "late",
                (legend_x + legend.shape[1] - 26, legend_y - 4),
                (220, 220, 220),
                scale=0.42,
                thickness=1,
            )
            self.draw_text_with_outline(
                image,
                f"Zoom {zoom:.1f}x",
                (left, top - 6),
                (220, 220, 220),
                scale=0.45,
                thickness=1,
            )

        cached = {
            "image": image,
            "x_pixels": x_pixels,
            "y_pixels": y_pixels,
            "frame_to_index": frame_to_index,
            "frames": visible_frames,
            "zoom": zoom,
        }
        self.trajectory_image_cache[cache_key] = cached
        return cached

    def refresh_trajectory_plot(self, *, force: bool = True) -> None:
        if not hasattr(self, "trajectory_canvas"):
            return
        if self.current_right_tab() != "trajectory":
            return
        if self.is_playing and not force:
            return
        now = time.perf_counter()
        if (
            not force
            and self.is_playing
            and (now - self.last_trajectory_refresh_at) < self.playback_trajectory_interval_s
        ):
            return
        self.last_trajectory_refresh_at = now

        canvas_w = max(160, self.trajectory_canvas.winfo_width())
        canvas_h = max(160, self.trajectory_canvas.winfo_height())
        try:
            tag = int(self.tag_var.get())
        except Exception:
            tag = 0
        rendered = self.build_trajectory_image(tag, canvas_w, canvas_h)
        image = rendered["image"].copy()
        frame_to_index = rendered["frame_to_index"]
        x_pixels = rendered["x_pixels"]
        y_pixels = rendered["y_pixels"]
        frames = rendered["frames"]
        zoom = float(rendered.get("zoom", 1.0))
        self.current_trajectory_render = {
            "tag": int(tag),
            "frames": frames,
            "x_pixels": x_pixels,
            "y_pixels": y_pixels,
            "zoom": zoom,
        }

        current_idx = frame_to_index.get(int(self.current_frame))
        if current_idx is not None and len(x_pixels) > int(current_idx):
            x = int(x_pixels[current_idx])
            y = int(y_pixels[current_idx])
            cv2.circle(image, (x, y), 7, (255, 255, 255), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
            self.draw_text_with_outline(
                image,
                f"frame {self.current_frame}",
                (min(canvas_w - 90, x + 10), max(18, y - 10)),
                (255, 255, 255),
                scale=0.45,
                thickness=1,
            )

        self.trajectory_status_var.set(
            self.format_trajectory_status(tag=tag, frames=frames, current_present=current_idx is not None)
        )

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.trajectory_photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        if self.trajectory_canvas_item_id is None:
            self.trajectory_canvas_item_id = self.trajectory_canvas.create_image(
                0,
                0,
                anchor=tk.NW,
                image=self.trajectory_photo,
            )
        else:
            self.trajectory_canvas.coords(self.trajectory_canvas_item_id, 0, 0)
            self.trajectory_canvas.itemconfigure(self.trajectory_canvas_item_id, image=self.trajectory_photo)

    def format_trajectory_status(self, *, tag: int, frames: np.ndarray, current_present: bool) -> str:
        if len(frames) == 0:
            return f"Tag {int(tag)} has no detections in this chunk."
        current_text = "present on current frame" if current_present else "missing on current frame"
        try:
            zoom = float(self.trajectory_zoom_var.get())
        except Exception:
            zoom = 1.0
        return (
            f"Tag {int(tag)} | detections {len(frames)} | frame span {int(frames[0])} to {int(frames[-1])} | "
            f"current frame {self.current_frame}: {current_text} | zoom {zoom:.1f}x | click path to seek"
        )

    def sync_selected_for_current_frame(self) -> None:
        if self.selected_detection is None:
            return
        selected_id = int(self.selected_detection[1])
        det = self.store.get_detection(self.current_frame, selected_id)
        if det is None:
            self.selected_detection = None
            self.selected_instance_var.set("Selected: None")
            return
        self.selected_detection = (self.current_frame, selected_id)
        self.selected_instance_var.set(f"Selected: ID {selected_id} @ ({det.x:.1f}, {det.y:.1f})")

    def canvas_to_image_coords(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        ix = (float(x) - float(self.display_offset_x)) / float(self.display_scale)
        iy = (float(y) - float(self.display_offset_y)) / float(self.display_scale)
        if ix < 0 or iy < 0 or ix >= self.video_width or iy >= self.video_height:
            return None
        return ix, iy

    def nearest_detection(self, x: float, y: float, max_screen_radius: float = 16.0) -> Optional[Detection]:
        threshold = max_screen_radius / max(self.display_scale, 1e-6)
        best: Optional[Detection] = None
        best_d2 = threshold * threshold
        for det in self.store.get_frame_detections(self.current_frame):
            d2 = (float(det.x) - x) ** 2 + (float(det.y) - y) ** 2
            if d2 <= best_d2:
                best = det
                best_d2 = d2
        return best

    def on_canvas_click(self, event: tk.Event) -> None:
        coords = self.canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return
        x, y = coords

        if self.add_mode_var.get():
            self.upsert_at_current_frame(x, y)
            return

        nearest = self.nearest_detection(x, y)
        if nearest is None:
            self.selected_detection = None
            self.selected_instance_var.set("Selected: None")
            self.render_current_frame()
            return

        self.selected_detection = (self.current_frame, int(nearest.instance))
        self.tag_var.set(int(nearest.instance))
        self.selected_instance_var.set(
            f"Selected: ID {nearest.instance} @ ({nearest.x:.1f}, {nearest.y:.1f})"
        )
        self.drag_before_snapshot = self.store.snapshot_frame(self.current_frame)
        self.drag_instance = int(nearest.instance)
        self.dragging = False
        self.render_current_frame()

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.add_mode_var.get():
            return
        if self.drag_instance is None or self.selected_detection is None:
            return
        coords = self.canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return
        x, y = coords
        self.dragging = True
        self.store.preview_move(self.current_frame, self.drag_instance, x, y)
        self.selected_instance_var.set(f"Selected: ID {self.drag_instance} @ ({x:.1f}, {y:.1f})")
        self.render_current_frame()

    def on_canvas_release(self, _event: tk.Event) -> None:
        if self.drag_instance is None or self.drag_before_snapshot is None:
            return
        if self.dragging:
            action = self.store.create_move_action(
                self.current_frame,
                self.drag_before_snapshot,
                instance=self.drag_instance,
                note="drag move",
            )
            if action is not None:
                self.record_action(action)
        self.drag_instance = None
        self.drag_before_snapshot = None
        self.dragging = False
        self.render_current_frame()

    def on_tree_select(self, _event: tk.Event) -> None:
        if self.updating_tree:
            return
        selected = self.tree.selection()
        if not selected:
            return
        try:
            instance = int(selected[0])
        except ValueError:
            return
        det = self.store.get_detection(self.current_frame, instance)
        if det is None:
            return
        if self.selected_detection == (self.current_frame, instance):
            return
        self.selected_detection = (self.current_frame, instance)
        self.tag_var.set(instance)
        self.selected_instance_var.set(f"Selected: ID {instance} @ ({det.x:.1f}, {det.y:.1f})")
        self.render_current_frame()

    def upsert_at_current_frame(self, x: float, y: float) -> None:
        instance = int(self.tag_var.get())
        try:
            confidence = float(self.default_confidence_var.get())
        except Exception:
            confidence = 1.0
            self.default_confidence_var.set(confidence)
        action = self.store.create_upsert_action(
            self.current_frame,
            instance,
            x,
            y,
            confidence=confidence,
            note="canvas add/update",
        )
        self.record_action(action)
        self.selected_detection = (self.current_frame, instance)
        det = self.store.get_detection(self.current_frame, instance)
        if det is not None:
            self.selected_instance_var.set(f"Selected: ID {instance} @ ({det.x:.1f}, {det.y:.1f})")
        self.render_current_frame()

    def delete_selected(self) -> None:
        if self.selected_detection is None:
            return
        frame, instance = self.selected_detection
        action = self.store.create_delete_action(frame, instance, note="delete selected")
        if action is None:
            return
        self.record_action(action)
        self.selected_detection = None
        self.selected_instance_var.set("Selected: None")
        self.render_current_frame()

    def relabel_selected(self) -> None:
        if self.selected_detection is None:
            return
        frame, old_instance = self.selected_detection
        new_instance = int(self.tag_var.get())
        if new_instance == old_instance:
            return
        action = self.store.create_relabel_action(
            frame,
            old_instance,
            new_instance,
            note="relabel selected",
        )
        if action is None:
            return
        self.record_action(action)
        self.selected_detection = (frame, new_instance)
        det = self.store.get_detection(frame, new_instance)
        if det is not None:
            self.selected_instance_var.set(f"Selected: ID {new_instance} @ ({det.x:.1f}, {det.y:.1f})")
        self.render_current_frame()

    def jump_tag_presence(self, *, prev: bool) -> None:
        self.pause_playback()
        instance = int(self.tag_var.get())
        target = (
            self.store.prev_present_frame(instance, self.current_frame)
            if prev
            else self.store.next_present_frame(instance, self.current_frame)
        )
        if target is None:
            messagebox.showinfo("Tag navigation", f"No {'previous' if prev else 'next'} frame found for tag {instance}.")
            return
        self.goto_frame(target)
        det = self.store.get_detection(target, instance)
        if det is not None:
            self.selected_detection = (target, instance)
            self.selected_instance_var.set(f"Selected: ID {instance} @ ({det.x:.1f}, {det.y:.1f})")
        self.render_current_frame()

    def jump_tag_missing(self, *, prev: bool) -> None:
        self.pause_playback()
        instance = int(self.tag_var.get())
        target = (
            self.store.prev_missing_frame(instance, self.current_frame)
            if prev
            else self.store.next_missing_frame(instance, self.current_frame, self.frame_count)
        )
        if target is None:
            messagebox.showinfo(
                "Tag navigation",
                f"No {'previous' if prev else 'next'} missing frame found for tag {instance}.",
            )
            return
        self.goto_frame(target)

    def get_bridge_params(self) -> Tuple[float, int]:
        try:
            max_distance = float(self.bridge_max_distance_var.get())
        except Exception:
            max_distance = 160.0
            self.bridge_max_distance_var.set(max_distance)
        try:
            max_frames = int(self.bridge_max_frames_var.get())
        except Exception:
            max_frames = 250
            self.bridge_max_frames_var.set(max_frames)
        return max_distance, max_frames

    def get_bridge_preview_range(self) -> Tuple[int, int]:
        try:
            start = int(self.bridge_preview_start_var.get())
        except Exception:
            start = 0
        try:
            end = int(self.bridge_preview_end_var.get())
        except Exception:
            end = max(0, self.frame_count - 1)
        hi = max(0, self.frame_count - 1)
        start = clamp(start, 0, hi)
        end = clamp(end, 0, hi)
        if start > end:
            start, end = end, start
        self.bridge_preview_start_var.set(start)
        self.bridge_preview_end_var.set(end)
        return start, end

    def rebuild_bridge_command_ids(self) -> None:
        seen = set()
        ordered: List[int] = []
        for action in self.undo_stack:
            command_id = action.details.get("bridge_command_id")
            if isinstance(command_id, int) and command_id not in seen:
                seen.add(command_id)
                ordered.append(command_id)
        self.bridge_command_ids = ordered

    def can_undo_latest_bridge_command(self) -> bool:
        self.rebuild_bridge_command_ids()
        if not self.bridge_command_ids or not self.undo_stack:
            return False
        latest_id = int(self.bridge_command_ids[-1])
        return self.undo_stack[-1].details.get("bridge_command_id") == latest_id

    def record_bridge_actions(self, actions: List[FrameAction], *, label: str) -> int:
        valid_actions = [action for action in actions if action is not None]
        if not valid_actions:
            return 0
        command_id = int(self.next_bridge_command_id)
        self.next_bridge_command_id += 1
        for idx, action in enumerate(valid_actions):
            action.details["bridge_command_id"] = command_id
            action.details["bridge_command_label"] = label
            action.details["bridge_command_index"] = idx
            self.undo_stack.append(action)
        self.redo_stack.clear()
        self.dirty = True
        self.invalidate_trajectory_cache()
        self.rebuild_bridge_command_ids()
        return command_id

    def undo_latest_bridge_command(self) -> None:
        self.pause_playback()
        if not self.can_undo_latest_bridge_command():
            self.rebuild_bridge_command_ids()
            if not self.bridge_command_ids:
                self.status_note = "No bridge command to undo"
            else:
                self.status_note = "Undo newer edits before Back Bridge"
            self.refresh_status()
            return
        latest_id = int(self.bridge_command_ids[-1])

        reverted_actions = 0
        reverted_frames: List[int] = []
        while self.undo_stack and self.undo_stack[-1].details.get("bridge_command_id") == latest_id:
            action = self.undo_stack.pop()
            self.store.apply_frame_snapshot(action.frame, action.before)
            self.redo_stack.append(action)
            reverted_actions += 1
            reverted_frames.append(int(action.frame))

        self.dirty = True
        self.invalidate_trajectory_cache()
        self.rebuild_bridge_command_ids()
        if reverted_frames:
            self.current_frame = int(min(reverted_frames))
        self.sync_selected_for_current_frame()
        self.status_note = f"Back Bridge reverted {reverted_actions} frame edits"
        self.render_current_frame()

    def _bridge_tag_on_current_frame(
        self,
        tag: int,
        *,
        max_distance: float,
        max_frames: int,
    ) -> Optional[Detection]:
        if self.sleap_store is None:
            return None
        if self.store.get_detection(self.current_frame, tag) is not None:
            return None

        anchor_frame = self.store.prev_present_frame(tag, self.current_frame + 1)
        if anchor_frame is None or (self.current_frame - anchor_frame) > max_frames:
            return None

        anchor_det = self.store.get_detection(anchor_frame, tag)
        if anchor_det is None:
            return None

        candidates = self.sleap_store.candidates_for_frame(self.current_frame)
        if candidates.size == 0:
            return None

        prev_xy = np.array([float(anchor_det.x), float(anchor_det.y)], dtype=np.float32)
        deltas = candidates - prev_xy[None, :]
        d2 = np.sum(deltas * deltas, axis=1)
        idx = int(np.argmin(d2))
        if float(np.sqrt(d2[idx])) > max_distance:
            return None

        chosen = candidates[idx]
        return Detection(
            frame=self.current_frame,
            instance=int(tag),
            x=float(chosen[0]),
            y=float(chosen[1]),
            confidence=1.0,
        )

    def _create_bridge_all_action_for_frame(self, frame: int, *, note: str) -> Optional[FrameAction]:
        if self.sleap_store is None:
            return None
        max_distance, max_frames = self.get_bridge_params()
        if max_distance <= 0 or max_frames <= 0:
            return None

        candidates = self.sleap_store.candidates_for_frame(frame)
        if candidates.size == 0:
            return None

        max_distance_sq = float(max_distance) * float(max_distance)
        current_dets = self.store.get_frame_detections(frame)
        present_ids = {int(det.instance) for det in current_dets}

        reserve_pairs: List[Tuple[float, int]] = []
        for det in current_dets:
            det_xy = np.array([float(det.x), float(det.y)], dtype=np.float32)
            deltas = candidates - det_xy[None, :]
            d2 = np.sum(deltas * deltas, axis=1)
            if d2.size == 0:
                continue
            idx = int(np.argmin(d2))
            if float(d2[idx]) <= max_distance_sq:
                reserve_pairs.append((float(d2[idx]), idx))

        reserved_candidate_indices: set[int] = set()
        for _dist2, idx in sorted(reserve_pairs):
            if idx in reserved_candidate_indices:
                continue
            reserved_candidate_indices.add(idx)

        pairings: List[Tuple[float, int, int]] = []
        for tag in sorted(self.store.id_to_frames):
            tag = int(tag)
            if tag in present_ids:
                continue
            anchor_frame = self.store.prev_present_frame(tag, frame + 1)
            if anchor_frame is None or (frame - anchor_frame) > max_frames:
                continue
            anchor_det = self.store.get_detection(anchor_frame, tag)
            if anchor_det is None:
                continue
            prev_xy = np.array([float(anchor_det.x), float(anchor_det.y)], dtype=np.float32)
            deltas = candidates - prev_xy[None, :]
            d2 = np.sum(deltas * deltas, axis=1)
            valid_indices = np.flatnonzero(d2 <= max_distance_sq)
            for idx in valid_indices:
                idx_int = int(idx)
                if idx_int in reserved_candidate_indices:
                    continue
                pairings.append((float(d2[idx_int]), tag, idx_int))

        if not pairings:
            return None

        assigned_tags: set[int] = set()
        used_candidate_indices = set(reserved_candidate_indices)
        new_detections: List[Detection] = []
        for _dist2, tag, idx in sorted(pairings):
            if tag in assigned_tags or idx in used_candidate_indices:
                continue
            assigned_tags.add(tag)
            used_candidate_indices.add(idx)
            chosen = candidates[idx]
            new_detections.append(
                Detection(
                    frame=frame,
                    instance=tag,
                    x=float(chosen[0]),
                    y=float(chosen[1]),
                    confidence=1.0,
                )
            )

        if not new_detections:
            return None
        return self.store.create_batch_upsert_action(frame, new_detections, note=note)

    def attempt_bridge_current_frame(self, *, refresh_ui: bool = True) -> int:
        if self.sleap_loading or self.sleap_store is None:
            self.status_note = "Auto-bridge waiting for SLEAP load"
            if refresh_ui:
                self.refresh_status(force=False)
            return 0

        tag = int(self.tag_var.get())
        if self.store.get_detection(self.current_frame, tag) is not None:
            return 0

        max_distance, max_frames = self.get_bridge_params()
        if max_distance <= 0 or max_frames <= 0:
            self.status_note = "Auto-bridge parameters must be positive"
            if refresh_ui:
                self.refresh_status(force=False)
            return 0

        bridged = self._bridge_tag_on_current_frame(tag, max_distance=max_distance, max_frames=max_frames)
        if bridged is None:
            anchor_frame = self.store.prev_present_frame(tag, self.current_frame + 1)
            if anchor_frame is None:
                self.status_note = f"Auto-bridge: no anchor for tag {tag}"
            elif (self.current_frame - anchor_frame) > max_frames:
                self.status_note = f"Auto-bridge: gap too large for tag {tag}"
            else:
                self.status_note = f"Auto-bridge: no match for tag {tag}"
            if refresh_ui:
                self.refresh_status(force=False)
            return 0

        action = self.store.create_upsert_action(
            self.current_frame,
            tag,
            bridged.x,
            bridged.y,
            confidence=bridged.confidence,
            note="sleap_nn_bridge",
        )
        self.record_bridge_actions([action], label=f"bridge tag frame {self.current_frame}")
        self.selected_detection = (self.current_frame, tag)
        det = self.store.get_detection(self.current_frame, tag)
        if det is not None:
            self.selected_instance_var.set(f"Selected: ID {tag} @ ({det.x:.1f}, {det.y:.1f})")
        self.status_note = f"Auto-bridged frame {self.current_frame} for tag {tag}"
        if refresh_ui:
            self.refresh_status(force=False)
        return 1

    def attempt_bridge_all_current_frame(self, *, refresh_ui: bool = True) -> int:
        if self.sleap_loading or self.sleap_store is None:
            self.status_note = "Auto-bridge waiting for SLEAP load"
            if refresh_ui:
                self.refresh_status(force=False)
            return 0

        action = self._create_bridge_all_action_for_frame(self.current_frame, note="sleap_nn_bridge_all")
        if action is None:
            self.status_note = f"Auto-bridge all: no bridgeable tags on frame {self.current_frame}"
            if refresh_ui:
                self.refresh_status(force=False)
            return 0

        self.record_bridge_actions([action], label=f"bridge all tags frame {self.current_frame}")
        self.sync_selected_for_current_frame()
        count = int(action.details.get("count", max(0, len(action.after) - len(action.before))))
        self.status_note = f"Auto-bridged frame {self.current_frame} for {count} tags"
        if refresh_ui:
            self.refresh_status(force=False)
        return count

    def preview_bridge_all_frames(self) -> None:
        self.bridge_preview_start_var.set(0)
        self.bridge_preview_end_var.set(max(0, self.frame_count - 1))
        self.preview_bridge_range_all()

    def preview_bridge_range_all(self) -> None:
        self.pause_playback()
        if self.sleap_loading:
            self.status_note = "Wait for SLEAP load to finish before preview bridging"
            self.refresh_status()
            return
        if self.sleap_store is None:
            self.start_sleap_load()
            self.status_note = "Started SLEAP load; rerun Preview Bridge Range when ready"
            self.refresh_status()
            return

        start, end = self.get_bridge_preview_range()
        self.status_var.set(f"Preview bridging all tags from frame {start} to {end} ...")
        self.root.update_idletasks()

        actions: List[FrameAction] = []
        total_tags = 0
        first_changed: Optional[int] = None
        current_frame_before = int(self.current_frame)
        for offset, frame in enumerate(range(start, end + 1)):
            action = self._create_bridge_all_action_for_frame(frame, note="sleap_nn_bridge_range_preview")
            if action is not None:
                actions.append(action)
                total_tags += int(action.details.get("count", max(0, len(action.after) - len(action.before))))
                if first_changed is None:
                    first_changed = int(frame)
            if offset % 250 == 0:
                self.status_var.set(
                    f"Preview bridging frames {frame}/{end} | changed frames {len(actions)} | tags added {total_tags}"
                )
                self.root.update_idletasks()

        if not actions:
            self.status_note = f"Preview bridge range found no bridgeable tags from {start} to {end}"
            self.refresh_status()
            return

        self.record_bridge_actions(actions, label=f"preview bridge range {start}-{end}")
        if first_changed is not None and not any(action.frame == current_frame_before for action in actions):
            self.current_frame = int(first_changed)
        self.sync_selected_for_current_frame()
        self.status_note = (
            f"Preview bridge added {total_tags} tags across {len(actions)} frames from {start} to {end}"
        )
        self.render_current_frame()

    def arm_bridge_mode(self, scope: str) -> None:
        scope_label = "all tags" if scope == "all" else f"tag {int(self.tag_var.get())}"
        self.auto_bridge_active = True
        self.auto_bridge_scope = scope
        if self.sleap_loading:
            self.status_note = f"SLEAP loading; auto-bridge for {scope_label} will start when ready"
            self.start_playback()
            return
        if self.sleap_store is None:
            self.start_sleap_load()
            self.status_note = f"Started SLEAP load; auto-bridge playback armed for {scope_label}"
            self.start_playback()
            return

        bridged = (
            self.attempt_bridge_all_current_frame(refresh_ui=False)
            if scope == "all"
            else self.attempt_bridge_current_frame(refresh_ui=False)
        )
        self.render_current_frame(full_refresh=False)
        if bridged > 0:
            if scope == "all":
                self.status_note = f"Bridge armed; bridged {bridged} tags on frame {self.current_frame}"
            else:
                self.status_note = f"Bridge armed; bridged current frame for tag {int(self.tag_var.get())}"
        else:
            self.status_note = f"Bridge armed for {scope_label}; waiting for bridgeable frame"
        self.start_playback()
        self.refresh_status()

    def bridge_gap_with_sleap(self) -> None:
        self.arm_bridge_mode("single")

    def bridge_all_gaps_with_sleap(self) -> None:
        self.arm_bridge_mode("all")

    def record_action(self, action: FrameAction) -> None:
        if action is None:
            return
        self.undo_stack.append(action)
        self.redo_stack.clear()
        self.dirty = True
        self.invalidate_trajectory_cache()
        self.rebuild_bridge_command_ids()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        self.store.apply_frame_snapshot(action.frame, action.before)
        self.redo_stack.append(action)
        self.dirty = True
        self.invalidate_trajectory_cache()
        self.rebuild_bridge_command_ids()
        if self.selected_detection is not None and self.store.get_detection(*self.selected_detection) is None:
            self.selected_detection = None
            self.selected_instance_var.set("Selected: None")
        self.render_current_frame()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        action = self.redo_stack.pop()
        self.store.apply_frame_snapshot(action.frame, action.after)
        self.undo_stack.append(action)
        self.dirty = True
        self.invalidate_trajectory_cache()
        self.rebuild_bridge_command_ids()
        self.render_current_frame()

    def output_paths(self) -> Tuple[Path, Path, Path]:
        stem = self.detections_path.stem
        if stem.endswith("_aruco_detections"):
            base = stem[: -len("_aruco_detections")]
        else:
            base = stem
        csv_path = self.output_dir / f"{base}_aruco_detections_curated.csv"
        h5_path = self.output_dir / f"{base}_aruco_tracks_curated.h5"
        json_path = self.output_dir / f"{base}_aruco_edits.json"
        return csv_path, h5_path, json_path

    def save_outputs(self) -> None:
        self.pause_playback()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path, h5_path, json_path = self.output_paths()

        df = self.store.to_dataframe()
        df.to_csv(csv_path, index=False, float_format="%.3f")

        save_payload = {
            "video_path": str(self.video_path),
            "source_csv": str(self.detections_path),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame_count": int(self.frame_count),
            "dictionary_size": int(self.dictionary_size),
            "current_frame": int(self.current_frame),
            "num_detections": int(len(df)),
            "actions": [action.to_dict() for action in self.undo_stack],
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(save_payload, fh, indent=2)

        self.saved_actions = list(self.undo_stack)
        self.dirty = False
        self.refresh_status()
        messagebox.showinfo(
            "Saved",
            f"Curated outputs written:\n\n{csv_path}\n{json_path}\n\n"
            f"Dense H5 is exported separately with the 'Export Dense H5' button.",
        )

    def export_dense_h5(self) -> None:
        self.pause_playback()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _, h5_path, _ = self.output_paths()
        self.status_var.set(
            f"Exporting dense H5 to {h5_path.name} ... this can take a while for large chunks."
        )
        self.root.update_idletasks()
        self.store.export_dense_h5(h5_path, frame_count=self.frame_count, dictionary_size=self.dictionary_size)
        self.refresh_status()
        messagebox.showinfo("Export complete", f"Dense H5 written:\n\n{h5_path}")

    def on_close(self) -> None:
        if self.is_playing and self.play_job is not None:
            self.root.after_cancel(self.play_job)
        if self.dirty:
            ok = messagebox.askyesno(
                "Unsaved changes",
                "There are unsaved changes. Close without saving?",
            )
            if not ok:
                return
        self.cap.release()
        self.root.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive ArUco detections curation GUI.")
    parser.add_argument("--video", type=Path, default=None, help="Path to source video.")
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="Path to *_aruco_detections.csv. If omitted, try to infer from the video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for curated CSV/H5/JSON outputs. Default: <detections_dir>/curation",
    )
    parser.add_argument("--dictionary-size", type=int, default=1000, help="Dense H5 marker dimension.")
    parser.add_argument("--start-frame", type=int, default=0, help="Initial frame.")
    parser.add_argument("--fps", type=float, default=20.0, help="Playback FPS inside the GUI.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    video_path, detections_path = pick_path_with_dialogs(args)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not detections_path.exists():
        inferred = infer_detections_from_video(video_path)
        raise FileNotFoundError(
            f"Detections file not found: {detections_path}\n"
            f"Expected inferred path: {inferred}"
        )

    output_dir = Path(args.output_dir) if args.output_dir is not None else detections_path.parent / "curation"

    root = tk.Tk()
    app = ArucoCurationApp(
        root,
        video_path=video_path,
        detections_path=detections_path,
        output_dir=output_dir,
        dictionary_size=int(args.dictionary_size),
        start_frame=int(args.start_frame),
        playback_fps=float(args.fps),
    )
    print(SHORTCUTS_TEXT, flush=True)
    app.refresh_status()
    root.mainloop()


if __name__ == "__main__":
    main()
