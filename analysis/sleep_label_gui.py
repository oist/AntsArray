#!/usr/bin/env python3
"""Minimal crop-video GUI for labeling per-frame crop-video states."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}
LABEL_TO_VALUE = {"wake": 0, "sleep": 1}
VALUE_TO_LABEL = {-1: "unlabeled", 0: "wake", 1: "sleep"}
UNLABELED_LABEL = "unlabeled"
BINARY_LABELS = tuple(LABEL_TO_VALUE)
LABEL_BAR_COLORS = {
    UNLABELED_LABEL: (70, 70, 70),
    "wake": (55, 150, 255),
    "sleep": (255, 170, 70),
}
FAST_PLAYBACK_RATES = (2.0, 4.0, 8.0, 16.0, 30.0)
PLAYBACK_TIMER_MS = 15
HOTKEY_HELP = "\n".join(
    [
        "space: play/pause",
        "left/right: previous/next frame",
        "a/d: jump backward/forward 1 second",
        "A/D: play backward/forward at 2x, then faster up to 30x on repeated presses",
        "s: start/switch/end sleep",
        "w or n: start/switch/end wake",
        "enter in Label box or Apply: start/switch/end the text label",
        "c: start/switch to clearing as unlabeled, or clear the selected range",
        "e: end the active label at the current frame",
        "[: set range start for a one-shot interval label",
        ",/.: previous/next crop video",
        "ctrl+s: save",
        "q: save and close",
    ]
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required to label crop videos; install the cv2 package.") from exc
    return cv2


def clamp(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(value)))


def format_rate(rate: float) -> str:
    return str(int(rate)) if float(rate).is_integer() else f"{rate:.1f}"


def list_crop_videos(video_dir: Path) -> list[Path]:
    video_dir = Path(video_dir)
    videos = [p for p in sorted(video_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if not videos:
        raise FileNotFoundError(f"No crop videos found in {video_dir}")
    return videos


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def discover_crop_videos(
    video: Path | None,
    video_dir: Path | None,
    *,
    single_video: bool = False,
) -> tuple[list[Path], int]:
    if video is not None:
        video_path = Path(video)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        if single_video:
            return [video_path], 0

        search_dir = Path(video_dir) if video_dir is not None else video_path.parent
        videos = list_crop_videos(search_dir)
        for index, path in enumerate(videos):
            if same_path(path, video_path):
                return videos, index

        if video_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video suffix for {video_path}")
        return [video_path, *videos], 0
    if video_dir is None:
        raise ValueError("Provide --video or --video_dir")
    return list_crop_videos(video_dir), 0


def default_labels_dir(video_paths: list[Path], video_index: int = 0) -> Path:
    return Path(video_paths[video_index]).resolve().parent / "label_vectors"


def label_paths(labels_dir: Path, video_path: Path) -> tuple[Path, Path, Path]:
    stem = Path(video_path).stem
    labels_dir = Path(labels_dir)
    return (
        labels_dir / f"{stem}_labels.npy",
        labels_dir / f"{stem}_labels.parquet",
        labels_dir / f"{stem}_metadata.json",
    )


def label_text_path(labels_dir: Path, video_path: Path) -> Path:
    return Path(labels_dir) / f"{Path(video_path).stem}_label_text.npy"


def canonical_label_text(label: object) -> str:
    try:
        if pd.isna(label):
            return UNLABELED_LABEL
    except (TypeError, ValueError):
        pass
    text = str(label).strip()
    if not text:
        return UNLABELED_LABEL
    if text.lower() == UNLABELED_LABEL:
        return UNLABELED_LABEL
    return text


def labels_to_legacy_values(labels: np.ndarray) -> np.ndarray:
    values = np.full(len(labels), -1, dtype=np.int8)
    for label, value in LABEL_TO_VALUE.items():
        values[np.char.lower(labels.astype(str)) == label] = int(value)
    return values


def _custom_color_for_label(label: object, *, collision_offset: int = 0) -> tuple[int, int, int]:
    key = f"{canonical_label_text(label).lower()}|{int(collision_offset)}"
    digest = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), byteorder="big")
    hue = digest / float(1 << 64)
    saturation = 0.62 + (((digest >> 8) & 0xFF) / 255.0) * 0.23
    value = 0.82 + (((digest >> 16) & 0xFF) / 255.0) * 0.14
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(round(channel * 255.0)) for channel in (red, green, blue))


def color_for_label(label: object, *, used_colors: set[tuple[int, int, int]] | None = None) -> tuple[int, int, int]:
    text = canonical_label_text(label)
    key = text.lower()
    if key in LABEL_BAR_COLORS:
        return LABEL_BAR_COLORS[key]

    used_colors = set() if used_colors is None else used_colors
    for offset in range(256):
        color = _custom_color_for_label(text, collision_offset=offset)
        if color not in used_colors:
            return color
    return _custom_color_for_label(text, collision_offset=256)


def label_color_map(labels: np.ndarray) -> dict[str, tuple[int, int, int]]:
    unique_labels = sorted({canonical_label_text(label) for label in labels}, key=lambda label: label.lower())
    colors: dict[str, tuple[int, int, int]] = {}
    used_colors = set(LABEL_BAR_COLORS.values())
    for label in unique_labels:
        key = label.lower()
        if key in LABEL_BAR_COLORS:
            color = LABEL_BAR_COLORS[key]
        else:
            color = color_for_label(label, used_colors=used_colors)
        colors[label] = color
        used_colors.add(color)
    return colors


def vector_to_frame_table(label_vector: np.ndarray, video_path: Path) -> pd.DataFrame:
    labels = np.asarray([canonical_label_text(label) for label in label_vector], dtype=str)
    values = labels_to_legacy_values(labels)
    return pd.DataFrame(
        {
            "Frame": np.arange(len(values), dtype=np.int64),
            "label_value": values.astype(np.int8),
            "label": labels,
            "is_sleep_wake_label": np.isin(np.char.lower(labels.astype(str)), list(BINARY_LABELS)),
            "video_path": str(video_path),
        }
    )


def load_label_vector(labels_dir: Path, video_path: Path, frame_count: int) -> np.ndarray:
    npy_path, table_path, _metadata_path = label_paths(labels_dir, video_path)
    text_path = label_text_path(labels_dir, video_path)
    labels = np.full(int(frame_count), UNLABELED_LABEL, dtype=object)
    if text_path.exists():
        loaded = np.load(text_path, allow_pickle=False).astype(str, copy=False).reshape(-1)
        n = min(len(labels), len(loaded))
        labels[:n] = [canonical_label_text(label) for label in loaded[:n]]
        return labels
    if npy_path.exists():
        loaded = np.load(npy_path, allow_pickle=False).reshape(-1)
        n = min(len(labels), len(loaded))
        if loaded.dtype.kind in {"i", "u", "f", "b"}:
            values = loaded[:n].astype(np.int8, copy=False)
            labels[:n] = [VALUE_TO_LABEL.get(int(value), UNLABELED_LABEL) for value in values]
        else:
            labels[:n] = [canonical_label_text(label) for label in loaded[:n].astype(str, copy=False)]
        return labels
    if table_path.exists():
        table = pd.read_parquet(table_path)
        if "Frame" in table.columns and "label" in table.columns:
            frames = pd.to_numeric(table["Frame"], errors="coerce").to_numpy()
            values = table["label"].astype(str).to_numpy()
            valid = np.isfinite(frames)
            frames = frames[valid].astype(np.int64)
            values = values[valid]
            in_range = (frames >= 0) & (frames < len(labels))
            labels[frames[in_range]] = [canonical_label_text(label) for label in values[in_range]]
        elif "Frame" in table.columns and "label_value" in table.columns:
            frames = pd.to_numeric(table["Frame"], errors="coerce").to_numpy()
            values = pd.to_numeric(table["label_value"], errors="coerce").to_numpy()
            valid = np.isfinite(frames) & np.isfinite(values)
            frames = frames[valid].astype(np.int64)
            values = values[valid].astype(np.int8)
            in_range = (frames >= 0) & (frames < len(labels))
            labels[frames[in_range]] = [VALUE_TO_LABEL.get(int(value), UNLABELED_LABEL) for value in values[in_range]]
    return labels


def save_label_vector(
    *,
    labels_dir: Path,
    video_path: Path,
    label_vector: np.ndarray,
    fps: float,
    frame_count: int,
) -> None:
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    npy_path, table_path, metadata_path = label_paths(labels_dir, video_path)
    text_path = label_text_path(labels_dir, video_path)
    labels = np.asarray([canonical_label_text(label) for label in label_vector], dtype=str)
    values = labels_to_legacy_values(labels)
    np.save(npy_path, values)
    np.save(text_path, labels)
    table = vector_to_frame_table(labels, video_path)
    try:
        table.to_parquet(table_path, index=False)
    except Exception:
        table_path = table_path.with_suffix(".csv")
        table.to_csv(table_path, index=False)
    label_counts = table["label"].value_counts(dropna=False).sort_index().to_dict()
    metadata = {
        "video_path": str(video_path),
        "label_vector": str(text_path),
        "legacy_sleep_wake_label_vector": str(npy_path),
        "label_table": str(table_path),
        "frame_count": int(frame_count),
        "fps": float(fps),
        "label_map": LABEL_TO_VALUE,
        "unlabeled_label": UNLABELED_LABEL,
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "n_sleep": int(np.sum(values == LABEL_TO_VALUE["sleep"])),
        "n_wake": int(np.sum(values == LABEL_TO_VALUE["wake"])),
        "n_unlabeled": int(np.sum(labels == UNLABELED_LABEL)),
        "n_custom_labeled": int(np.sum((labels != UNLABELED_LABEL) & (values < 0))),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


class OpenCvSequentialReader:
    def __init__(self, video_path: Path):
        self.cv2 = require_cv2()
        self.video_path = Path(video_path)
        self.cap = self.cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")
        self.width = int(self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH)) or 1
        self.height = int(self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        self.fps = float(self.cap.get(self.cv2.CAP_PROP_FPS)) or 24.0
        self.frame_count = int(self.cap.get(self.cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.decoded_frame_index: int | None = None

    def read_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        need_seek = self.decoded_frame_index is None or frame_idx != self.decoded_frame_index + 1
        max_grab_skip = max(2, int(round(self.fps * 2)))
        can_grab_forward = (
            self.decoded_frame_index is not None
            and frame_idx > self.decoded_frame_index + 1
            and frame_idx - self.decoded_frame_index <= max_grab_skip
        )
        if can_grab_forward:
            ok = True
            while self.decoded_frame_index < frame_idx - 1:
                ok = self.cap.grab()
                if not ok:
                    break
                self.decoded_frame_index += 1
            if ok:
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self.decoded_frame_index = frame_idx
                    return frame
            need_seek = True
        if need_seek:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if (not ok or frame is None) and not need_seek:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from {self.video_path}")
        self.decoded_frame_index = frame_idx
        return frame

    def close(self) -> None:
        self.cap.release()


class CropLabelWindow:
    def __init__(
        self,
        *,
        video_paths: list[Path],
        labels_dir: Path,
        start_frame: int,
        start_video_index: int = 0,
    ) -> None:
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import messagebox, ttk
        from PIL import Image, ImageTk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.Image = Image
        self.ImageTk = ImageTk
        self.tkfont = tkfont
        self.cv2 = require_cv2()
        self.video_paths = [Path(path) for path in video_paths]
        self.labels_dir = Path(labels_dir)
        self.initial_start_frame = int(start_frame)

        self.video_index = clamp(start_video_index, 0, len(self.video_paths) - 1)
        self.reader: OpenCvSequentialReader | None = None
        self.video_path = self.video_paths[self.video_index]
        self.frame_count = 0
        self.video_width = 1
        self.video_height = 1
        self.fps = 24.0
        self.playback_delay_ms = 42
        self.current_frame = 0
        self.current_bgr: np.ndarray | None = None
        self.current_bgr_frame: int | None = None
        self.label_vector = np.empty((0,), dtype=object)

        self.active_label_text: str | None = None
        self.active_start_frame: int | None = None
        self.active_last_painted_frame: int | None = None
        self.range_start_frame: int | None = None
        self.status_note = ""
        self.is_playing = False
        self.playback_direction = 1
        self.playback_rate = 1.0
        self.playback_accumulator = 0.0
        self.last_playback_tick_at = 0.0
        self.play_job = None
        self.slider_job = None
        self.resize_job = None
        self.setting_slider = False
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.render_width = 1
        self.render_height = 1
        self.current_photo = None
        self.canvas_item_id = None
        self.label_bar_photo = None
        self.label_bar_image_id = None
        self.label_bar_cursor_id = None
        self.label_bar_dirty = True
        self.label_bar_last_size = (0, 0)
        self.last_status_refresh_at = 0.0
        self.last_controls_refresh_at = 0.0

        self.root = tk.Tk()
        self.root.title("Sleep/wake crop labels")
        self.frame_var = tk.StringVar(value="0")
        self.label_text_var = tk.StringVar(value="sleep")
        self.status_var = tk.StringVar(value="")
        self.play_pause_button = None

        self.open_video(self.video_index, start_frame=self.initial_start_frame, save_previous=False)
        self.build_ui()
        self.bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update_idletasks()
        self.redraw(full_refresh=True)

    def configure_styles(self) -> None:
        default_font = self.tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=15)
        heading_font = self.tkfont.nametofont("TkHeadingFont")
        heading_font.configure(size=16, weight="bold")
        style = self.ttk.Style(self.root)
        style.configure("TButton", padding=(12, 8), font=default_font)
        style.configure("Transport.TButton", padding=(16, 10), font=heading_font)
        style.configure("TLabel", font=default_font)
        style.configure("TEntry", font=default_font)
        self.root.option_add("*Font", default_font)

    def build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk
        self.configure_styles()
        self.root.geometry("1080x760")
        self.root.minsize(720, 520)

        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)
        for text, callback in [
            ("Prev video", lambda: self.change_video(-1)),
            ("Next video", lambda: self.change_video(1)),
            ("Play", self.toggle_playback),
            ("Prev", lambda: self.seek_relative(-1)),
            ("Next", lambda: self.seek_relative(1)),
            ("-1 s", lambda: self.seek_relative(-round(self.fps))),
            ("+1 s", lambda: self.seek_relative(round(self.fps))),
            ("Rewind", lambda: self.start_fast_playback(-1)),
            ("Fast forward", lambda: self.start_fast_playback(1)),
            ("Sleep", lambda: self.apply_named_label("sleep")),
            ("Wake", lambda: self.apply_named_label("wake")),
            ("Clear", self.clear_label_selection),
            ("End", self.end_active_label),
            ("Save", self.save_labels),
        ]:
            button = ttk.Button(top, text=text, command=callback, style="Transport.TButton")
            button.pack(side=tk.LEFT, padx=3)
            if text == "Play":
                self.play_pause_button = button

        row = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(row, text="Frame").pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=self.frame_var, width=11)
        entry.pack(side=tk.LEFT, padx=(5, 10))
        entry.bind("<Return>", lambda _event: self.goto_frame_from_entry())
        ttk.Button(row, text="Go", command=self.goto_frame_from_entry).pack(side=tk.LEFT)
        ttk.Button(row, text="Hotkeys", command=self.show_hotkeys).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(row, text="Label").pack(side=tk.LEFT, padx=(16, 0))
        self.label_entry = ttk.Entry(row, textvariable=self.label_text_var, width=18)
        self.label_entry.pack(side=tk.LEFT, padx=(5, 4))
        self.label_entry.bind("<Return>", lambda _event: self.apply_label_from_entry())
        ttk.Button(row, text="Apply", command=self.apply_label_from_entry).pack(side=tk.LEFT)

        self.status_label = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side=tk.TOP, fill=tk.X, padx=8)

        self.slider = tk.Scale(
            self.root,
            from_=0,
            to=max(0, self.frame_count - 1),
            orient=tk.HORIZONTAL,
            showvalue=False,
            command=self.on_slider,
        )
        self.slider.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 5))

        self.label_bar_canvas = tk.Canvas(self.root, height=22, bg="#111111", highlightthickness=0)
        self.label_bar_canvas.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
        self.label_bar_canvas.bind("<Configure>", self.on_label_bar_configure)
        self.label_bar_canvas.bind("<Button-1>", self.on_label_bar_click)

        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.video_canvas = tk.Canvas(main, width=720, height=560, bg="#111111", highlightthickness=0)
        self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Configure>", self.on_canvas_configure)

    def bind_keys(self) -> None:
        self.root.bind("<space>", lambda event: self.run_hotkey(event, self.toggle_playback))
        self.root.bind("<Left>", lambda event: self.run_hotkey(event, lambda: self.seek_relative(-1)))
        self.root.bind("<Right>", lambda event: self.run_hotkey(event, lambda: self.seek_relative(1)))
        self.root.bind("<KeyPress-a>", lambda event: self.run_hotkey(event, lambda: self.seek_relative(-round(self.fps))))
        self.root.bind("<KeyPress-d>", lambda event: self.run_hotkey(event, lambda: self.seek_relative(round(self.fps))))
        self.root.bind("<KeyPress-A>", lambda event: self.run_hotkey(event, lambda: self.start_fast_playback(-1)))
        self.root.bind("<KeyPress-D>", lambda event: self.run_hotkey(event, lambda: self.start_fast_playback(1)))
        self.root.bind("<KeyPress-s>", lambda event: self.run_hotkey(event, lambda: self.apply_named_label("sleep")))
        self.root.bind("<KeyPress-w>", lambda event: self.run_hotkey(event, lambda: self.apply_named_label("wake")))
        self.root.bind("<KeyPress-n>", lambda event: self.run_hotkey(event, lambda: self.apply_named_label("wake")))
        self.root.bind("<KeyPress-c>", lambda event: self.run_hotkey(event, self.clear_label_selection))
        self.root.bind("<KeyPress-e>", lambda event: self.run_hotkey(event, self.end_active_label))
        self.root.bind("<KeyPress-bracketleft>", lambda event: self.run_hotkey(event, self.set_range_start))
        self.root.bind("<KeyPress-comma>", lambda event: self.run_hotkey(event, lambda: self.change_video(-1)))
        self.root.bind("<KeyPress-period>", lambda event: self.run_hotkey(event, lambda: self.change_video(1)))
        self.root.bind("<Control-s>", lambda _event: self.save_labels())
        self.root.bind("<KeyPress-q>", lambda event: self.run_hotkey(event, self.close))

    def run_hotkey(self, event, callback) -> str | None:
        widget = getattr(event, "widget", None)
        try:
            widget_class = widget.winfo_class() if widget is not None else ""
        except Exception:
            widget_class = ""
        if widget_class in {"Entry", "TEntry"}:
            return None
        callback()
        return "break"

    def show_hotkeys(self) -> None:
        self.messagebox.showinfo("Hotkeys", HOTKEY_HELP, parent=self.root)

    def show(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.pause_playback()
        self.paint_active_label_to_frame(self.current_frame)
        self.save_labels()
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
            self.resize_job = None
        if self.slider_job is not None:
            self.root.after_cancel(self.slider_job)
            self.slider_job = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.root.destroy()

    def open_video(self, index: int, *, start_frame: int = 0, save_previous: bool = True) -> None:
        if save_previous and self.reader is not None:
            self.paint_active_label_to_frame(self.current_frame)
            self.save_labels()
        if self.reader is not None:
            self.reader.close()

        self.video_index = clamp(index, 0, len(self.video_paths) - 1)
        self.video_path = self.video_paths[self.video_index]
        self.reader = OpenCvSequentialReader(self.video_path)
        self.frame_count = max(1, int(self.reader.frame_count))
        self.video_width = int(self.reader.width)
        self.video_height = int(self.reader.height)
        self.fps = float(self.reader.fps)
        self.playback_delay_ms = max(1, int(round(1000.0 / max(self.fps, 1.0))))
        self.current_frame = clamp(int(start_frame), 0, self.frame_count - 1)
        self.current_bgr = None
        self.current_bgr_frame = None
        self.label_vector = load_label_vector(self.labels_dir, self.video_path, self.frame_count)
        self.label_bar_dirty = True
        self.active_label_text = None
        self.active_start_frame = None
        self.active_last_painted_frame = None
        self.range_start_frame = None
        self.status_note = f"Opened {self.video_path.name}"
        if hasattr(self, "slider"):
            self.slider.configure(to=max(0, self.frame_count - 1))
        log(f"opened crop video {self.video_index + 1}/{len(self.video_paths)}: {self.video_path}")

    def change_video(self, delta: int) -> None:
        if len(self.video_paths) <= 1:
            self.status_note = "Only one crop video loaded"
            self.redraw_status(force=True)
            return
        self.pause_playback()
        new_index = clamp(self.video_index + int(delta), 0, len(self.video_paths) - 1)
        if new_index == self.video_index:
            edge = "last" if int(delta) > 0 else "first"
            self.status_note = f"Already at {edge} crop video"
            self.redraw_status(force=True)
            return
        self.open_video(new_index, start_frame=0, save_previous=True)
        self.redraw(full_refresh=True)

    def read_video_frame(self, frame: int) -> np.ndarray:
        frame = int(frame)
        if self.current_bgr is not None and self.current_bgr_frame == frame:
            return self.current_bgr
        assert self.reader is not None
        img = self.reader.read_frame(frame)
        self.current_bgr = img
        self.current_bgr_frame = frame
        return img

    def on_slider(self, value: str) -> None:
        if self.setting_slider:
            return
        target = clamp(int(float(value)), 0, max(0, self.frame_count - 1))
        if self.slider_job is not None:
            self.root.after_cancel(self.slider_job)
        self.slider_job = self.root.after(80, lambda: self.goto_frame(target))

    def sync_controls(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and self.is_playing and (now - self.last_controls_refresh_at) < 0.5:
            return
        self.last_controls_refresh_at = now
        self.frame_var.set(str(self.current_frame))
        self.setting_slider = True
        self.slider.set(self.current_frame)
        self.setting_slider = False
        self.update_label_bar_cursor()

    def goto_frame_from_entry(self) -> None:
        try:
            frame = int(self.frame_var.get())
        except ValueError:
            return
        self.pause_playback()
        self.goto_frame(frame)

    def goto_frame(self, frame: int, *, full_refresh: bool = True) -> None:
        if self.slider_job is not None:
            self.root.after_cancel(self.slider_job)
            self.slider_job = None
        self.current_frame = clamp(int(frame), 0, max(0, self.frame_count - 1))
        self.reset_active_label_anchor(self.current_frame)
        self.sync_controls(force=True)
        self.redraw(full_refresh=full_refresh)

    def seek_relative(self, delta: int) -> None:
        self.pause_playback()
        self.goto_frame(self.current_frame + int(delta))

    def toggle_playback(self) -> None:
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback(direction=1, rate=1.0)

    def next_fast_rate(self) -> float:
        for rate in FAST_PLAYBACK_RATES:
            if self.playback_rate < rate:
                return rate
        return FAST_PLAYBACK_RATES[-1]

    def start_fast_playback(self, direction: int) -> None:
        direction = 1 if int(direction) >= 0 else -1
        if self.is_playing and self.playback_direction == direction and self.playback_rate > 1.0:
            rate = self.next_fast_rate()
        else:
            rate = FAST_PLAYBACK_RATES[0]
        self.start_playback(direction=direction, rate=rate)

    def start_playback(self, *, direction: int = 1, rate: float = 1.0) -> None:
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        self.playback_direction = 1 if int(direction) >= 0 else -1
        self.playback_rate = max(1.0, min(float(rate), FAST_PLAYBACK_RATES[-1]))
        self.playback_accumulator = 0.0
        self.last_playback_tick_at = time.perf_counter()
        self.is_playing = True
        self.status_note = f"Playing {self.playback_mode_text()}"
        self.play_job = self.root.after(0, self.play_step)
        self.redraw_status(force=True)

    def pause_playback(self, *, status_note: str = "Paused") -> None:
        self.is_playing = False
        self.status_note = status_note
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        self.sync_controls(force=True)
        self.redraw_status(force=True)

    def play_step(self) -> None:
        if not self.is_playing:
            return
        self.play_job = None
        started_at = time.perf_counter()
        if self.playback_direction > 0 and self.current_frame >= self.frame_count - 1:
            self.pause_playback(status_note="Reached end")
            return
        if self.playback_direction < 0 and self.current_frame <= 0:
            self.pause_playback(status_note="Reached start")
            return

        now = time.perf_counter()
        elapsed_s = max(0.0, now - self.last_playback_tick_at)
        self.last_playback_tick_at = now
        self.playback_accumulator += elapsed_s * max(self.fps, 1.0) * self.playback_rate
        frames_to_move = int(self.playback_accumulator)
        if frames_to_move <= 0:
            self.play_job = self.root.after(PLAYBACK_TIMER_MS, self.play_step)
            return

        self.playback_accumulator -= frames_to_move
        self.current_frame = clamp(
            self.current_frame + self.playback_direction * frames_to_move,
            0,
            max(0, self.frame_count - 1),
        )
        self.paint_active_label_to_frame(self.current_frame)
        self.redraw(full_refresh=False)
        elapsed_ms = int(round((time.perf_counter() - started_at) * 1000.0))
        self.play_job = self.root.after(max(1, PLAYBACK_TIMER_MS - elapsed_ms), self.play_step)

    def paint_label_interval(self, start: int, end: int, label: str) -> tuple[int, int, bool]:
        start = clamp(int(start), 0, self.frame_count - 1)
        end = clamp(int(end), 0, self.frame_count - 1)
        if end < start:
            start, end = end, start
        label = canonical_label_text(label)
        segment = self.label_vector[start : end + 1]
        changed = bool(np.any(segment != label))
        if changed:
            segment[:] = label
            self.label_bar_dirty = True
        return start, end, changed

    def paint_active_label_to_frame(self, frame: int, *, redraw_bar: bool = False) -> bool:
        if self.active_label_text is None:
            return False
        frame = clamp(int(frame), 0, self.frame_count - 1)
        start = self.active_last_painted_frame
        if start is None:
            start = self.active_start_frame if self.active_start_frame is not None else frame
        _start, _end, changed = self.paint_label_interval(start, frame, self.active_label_text)
        self.active_last_painted_frame = frame
        if redraw_bar and changed:
            self.draw_label_bar()
        return changed

    def reset_active_label_anchor(self, frame: int) -> None:
        if self.active_label_text is None:
            return
        frame = clamp(int(frame), 0, self.frame_count - 1)
        self.active_start_frame = frame
        self.active_last_painted_frame = None

    def fill_interval(self, start: int, end: int, label: str) -> None:
        label = canonical_label_text(label)
        start, end, _changed = self.paint_label_interval(start, end, label)
        self.status_note = f"Labeled {label}: frames {start}-{end}"
        self.save_labels()
        self.draw_label_bar()

    def set_range_start(self) -> None:
        self.range_start_frame = int(self.current_frame)
        self.status_note = f"Range start {self.range_start_frame}"
        self.redraw_status(force=True)

    def apply_named_label(self, label: str) -> None:
        label = canonical_label_text(label)
        self.label_text_var.set(label)
        self.apply_or_switch_label(label)

    def apply_label_from_entry(self) -> None:
        label = canonical_label_text(self.label_text_var.get())
        if label == UNLABELED_LABEL:
            self.status_note = "Type a label other than unlabeled, or use Clear"
            self.redraw_status(force=True)
            return
        self.label_text_var.set(label)
        self.apply_or_switch_label(label)

    def apply_or_switch_label(self, label: str) -> None:
        label = canonical_label_text(label)
        if label == UNLABELED_LABEL:
            self.clear_label_selection()
            return
        if self.range_start_frame is not None:
            self.fill_interval(self.range_start_frame, self.current_frame, label)
            self.range_start_frame = None
            self.redraw(full_refresh=False)
            return
        if self.active_label_text is None:
            self.active_label_text = label
            self.active_start_frame = int(self.current_frame)
            self.active_last_painted_frame = None
            self.paint_active_label_to_frame(self.current_frame, redraw_bar=True)
            self.status_note = f"Started {label} at frame {self.active_start_frame}"
        elif self.active_label_text == label:
            self.end_active_label()
            return
        else:
            start = int(self.active_start_frame if self.active_start_frame is not None else self.current_frame)
            current = int(self.current_frame)
            if current > start:
                self.fill_interval(start, current - 1, self.active_label_text)
            elif current < start:
                self.fill_interval(current + 1, start, self.active_label_text)
            self.active_label_text = label
            self.active_start_frame = current
            self.active_last_painted_frame = None
            self.paint_active_label_to_frame(current, redraw_bar=True)
            self.status_note = f"Switched to {label} at frame {self.active_start_frame}"
        self.redraw_status(force=True)

    def end_active_label(self) -> None:
        if self.active_label_text is None or self.active_start_frame is None:
            self.status_note = "No active label"
            self.redraw_status(force=True)
            return
        self.fill_interval(self.active_start_frame, self.current_frame, self.active_label_text)
        self.active_label_text = None
        self.active_start_frame = None
        self.active_last_painted_frame = None
        self.redraw(full_refresh=False)

    def clear_label_selection(self) -> None:
        if self.range_start_frame is not None:
            start = self.range_start_frame
            end = self.current_frame
            self.range_start_frame = None
            self.active_label_text = None
            self.active_start_frame = None
            self.active_last_painted_frame = None
            start = clamp(int(start), 0, self.frame_count - 1)
            end = clamp(int(end), 0, self.frame_count - 1)
            if end < start:
                start, end = end, start
            self.label_vector[start : end + 1] = UNLABELED_LABEL
            self.label_bar_dirty = True
            self.save_labels()
            self.status_note = f"Cleared frames {start}-{end}"
            self.redraw(full_refresh=False)
            return

        current = int(self.current_frame)
        if self.active_label_text == UNLABELED_LABEL:
            self.paint_active_label_to_frame(current, redraw_bar=True)
            self.active_label_text = None
            self.active_start_frame = None
            self.active_last_painted_frame = None
            self.save_labels()
            self.status_note = f"Stopped clearing at frame {current}"
            self.redraw(full_refresh=False)
            return

        if self.active_label_text is not None:
            start = int(self.active_start_frame if self.active_start_frame is not None else current)
            if current > start:
                self.fill_interval(start, current - 1, self.active_label_text)
            elif current < start:
                self.fill_interval(current + 1, start, self.active_label_text)

        self.active_label_text = UNLABELED_LABEL
        self.active_start_frame = current
        self.active_last_painted_frame = None
        self.paint_active_label_to_frame(current, redraw_bar=True)
        self.save_labels()
        if self.current_label() == UNLABELED_LABEL:
            self.status_note = f"Started clearing at frame {current}"
        else:
            self.status_note = f"Started clearing at frame {current}; current frame not cleared"
        self.redraw(full_refresh=False)

    def save_labels(self) -> None:
        save_label_vector(
            labels_dir=self.labels_dir,
            video_path=self.video_path,
            label_vector=self.label_vector,
            fps=self.fps,
            frame_count=self.frame_count,
        )
        self.status_note = f"Saved labels for {self.video_path.name}"
        log(self.status_note)

    def current_label(self) -> str:
        if len(self.label_vector) == 0:
            return UNLABELED_LABEL
        return canonical_label_text(self.label_vector[self.current_frame])

    def resize_frame_to_canvas(self, img: np.ndarray) -> np.ndarray:
        canvas_w = max(10, int(self.video_canvas.winfo_width()))
        canvas_h = max(10, int(self.video_canvas.winfo_height()))
        h, w = img.shape[:2]
        scale = min(canvas_w / max(1, w), canvas_h / max(1, h))
        self.display_scale = float(scale)
        self.render_width = max(1, int(round(w * scale)))
        self.render_height = max(1, int(round(h * scale)))
        self.display_offset_x = max(0, (canvas_w - self.render_width) // 2)
        self.display_offset_y = max(0, (canvas_h - self.render_height) // 2)
        if self.render_width == w and self.render_height == h:
            return img
        interpolation = self.cv2.INTER_AREA if scale < 1.0 else self.cv2.INTER_LINEAR
        return self.cv2.resize(img, (self.render_width, self.render_height), interpolation=interpolation)

    def redraw_video(self) -> None:
        try:
            img = self.read_video_frame(self.current_frame)
        except Exception as exc:
            img = np.zeros((max(1, self.video_height), max(1, self.video_width), 3), dtype=np.uint8)
            self.cv2.putText(
                img,
                str(exc),
                (12, 30),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                self.cv2.LINE_AA,
            )
        rgb = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        shown = self.resize_frame_to_canvas(rgb)
        photo = self.ImageTk.PhotoImage(image=self.Image.fromarray(shown))
        self.current_photo = photo
        if self.canvas_item_id is None:
            self.canvas_item_id = self.video_canvas.create_image(
                self.display_offset_x,
                self.display_offset_y,
                image=photo,
                anchor=self.tk.NW,
            )
        else:
            self.video_canvas.coords(self.canvas_item_id, self.display_offset_x, self.display_offset_y)
            self.video_canvas.itemconfigure(self.canvas_item_id, image=photo)

    def draw_label_bar(self, *, force: bool = False) -> None:
        if not hasattr(self, "label_bar_canvas"):
            return
        width = max(2, int(self.label_bar_canvas.winfo_width()))
        height = max(8, int(self.label_bar_canvas.winfo_height()))
        size = (width, height)
        if force or self.label_bar_dirty or self.label_bar_last_size != size:
            frame_indices = np.floor(np.arange(width, dtype=np.float64) * max(1, self.frame_count) / width).astype(
                np.int64
            )
            frame_indices = np.clip(frame_indices, 0, max(0, len(self.label_vector) - 1))
            values = (
                self.label_vector[frame_indices].astype(str)
                if len(self.label_vector)
                else np.full(width, UNLABELED_LABEL, dtype=object)
            )
            values = np.asarray([canonical_label_text(value) for value in values], dtype=str)
            row = np.zeros((width, 3), dtype=np.uint8)
            colors = label_color_map(values)
            for value, color in colors.items():
                row[values == value] = np.asarray(color, dtype=np.uint8)
            image = np.repeat(row.reshape(1, width, 3), height, axis=0)
            image[0, :, :] = 25
            image[-1, :, :] = 25
            self.label_bar_photo = self.ImageTk.PhotoImage(image=self.Image.fromarray(image))
            if self.label_bar_image_id is None:
                self.label_bar_image_id = self.label_bar_canvas.create_image(
                    0,
                    0,
                    image=self.label_bar_photo,
                    anchor=self.tk.NW,
                )
            else:
                self.label_bar_canvas.itemconfigure(self.label_bar_image_id, image=self.label_bar_photo)
                self.label_bar_canvas.coords(self.label_bar_image_id, 0, 0)
            self.label_bar_dirty = False
            self.label_bar_last_size = size
        self.update_label_bar_cursor()

    def update_label_bar_cursor(self) -> None:
        if not hasattr(self, "label_bar_canvas"):
            return
        width = max(2, int(self.label_bar_canvas.winfo_width()))
        height = max(8, int(self.label_bar_canvas.winfo_height()))
        x = int(round(self.current_frame / max(1, self.frame_count - 1) * (width - 1)))
        if self.label_bar_cursor_id is None:
            self.label_bar_cursor_id = self.label_bar_canvas.create_line(x, 0, x, height, fill="#ffffff", width=2)
        else:
            self.label_bar_canvas.coords(self.label_bar_cursor_id, x, 0, x, height)
        self.label_bar_canvas.tag_raise(self.label_bar_cursor_id)

    def on_label_bar_click(self, event) -> None:
        width = max(2, int(self.label_bar_canvas.winfo_width()))
        x = clamp(int(event.x), 0, width - 1)
        frame = int(round(float(x) / max(1, width - 1) * max(0, self.frame_count - 1)))
        self.pause_playback()
        self.goto_frame(frame)

    def on_label_bar_configure(self, _event) -> None:
        self.label_bar_dirty = True
        self.draw_label_bar(force=True)

    def playback_mode_text(self) -> str:
        if not self.is_playing:
            return "paused"
        direction = "reverse" if self.playback_direction < 0 else "forward"
        return f"{direction} {format_rate(self.playback_rate)}x"

    def redraw_status(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and self.is_playing and (now - self.last_status_refresh_at) < 0.35:
            return
        self.last_status_refresh_at = now
        if self.play_pause_button is not None:
            self.play_pause_button.configure(text="Pause" if self.is_playing else "Play")
        active = (
            f"{self.active_label_text} from {self.active_start_frame}"
            if self.active_label_text is not None
            else "-"
        )
        range_text = str(self.range_start_frame) if self.range_start_frame is not None else "-"
        lower_labels = np.char.lower(self.label_vector.astype(str)) if len(self.label_vector) else np.asarray([])
        n_sleep = int(np.sum(lower_labels == "sleep"))
        n_wake = int(np.sum(lower_labels == "wake"))
        n_labeled = int(np.sum(lower_labels != UNLABELED_LABEL))
        n_custom = max(0, n_labeled - n_sleep - n_wake)
        labeled_pct = 100.0 * float(n_labeled) / max(1, len(self.label_vector))
        note = f" | {self.status_note}" if self.status_note else ""
        self.status_var.set(
            f"Video {self.video_index + 1}/{len(self.video_paths)}: {self.video_path.name} | "
            f"Frame {self.current_frame + 1}/{self.frame_count} | mode {self.playback_mode_text()} | "
            f"current {self.current_label()} | "
            f"active {active} | range {range_text} | labeled {labeled_pct:.1f}% "
            f"(sleep {n_sleep:,}, wake {n_wake:,}, other {n_custom:,}){note}"
        )

    def redraw(self, *, full_refresh: bool = True) -> None:
        self.redraw_video()
        self.draw_label_bar(force=full_refresh)
        self.redraw_status(force=full_refresh or not self.is_playing)
        self.sync_controls(force=full_refresh or not self.is_playing)

    def on_canvas_configure(self, _event) -> None:
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        delay = 30 if self.is_playing else 80
        self.resize_job = self.root.after(delay, self.finish_canvas_resize)

    def finish_canvas_resize(self) -> None:
        self.resize_job = None
        self.redraw(full_refresh=not self.is_playing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None, help="Crop video to label.")
    parser.add_argument("--video_dir", type=Path, default=None, help="Directory of crop videos to label.")
    parser.add_argument("--labels_dir", type=Path, default=None, help="Output label-vector directory. Default: <video_dir>/label_vectors.")
    parser.add_argument("--start_frame", type=int, default=0, help="Initial frame for the first video.")
    parser.add_argument(
        "--single_video",
        action="store_true",
        help="With --video, load only that file instead of starting within its sibling crop-video folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_paths, video_index = discover_crop_videos(args.video, args.video_dir, single_video=args.single_video)
    labels_dir = Path(args.labels_dir) if args.labels_dir is not None else default_labels_dir(video_paths, video_index)
    window = CropLabelWindow(
        video_paths=video_paths,
        labels_dir=labels_dir,
        start_frame=args.start_frame,
        start_video_index=video_index,
    )
    window.show()


if __name__ == "__main__":
    main()
