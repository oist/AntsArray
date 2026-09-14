#!/usr/bin/env python3
"""Generate a calibrated panorama and annotate semantic regions on it.

The saved JSON contains geometry in both displayed-image coordinates and the
raw panorama coordinates produced by the homographies (the coordinate system
used by colony tracking). A flat CSV and a full-resolution annotated panorama
PNG are written alongside it for analysis and visual verification.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from PIL import Image, ImageDraw, ImageFont, ImageTk
from tkinter import messagebox, ttk


# Calibrations in the older tracking coordinate system can produce trusted
# local mosaics above Pillow's generic 178-megapixel safety threshold.
Image.MAX_IMAGE_PIXELS = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera_cal.panorama_from_hmats import (  # noqa: E402
    load_frames_from_videos_dir_aligned,
    load_h_mats,
    make_panorama_from_hmats,
    panorama_geometry,
)
from camera_cal.region_paths import (  # noqa: E402
    TRACKING_ARENA_LABELS, arena_label_side, panorama_dataset_dir,
    panorama_regions_path, tracking_split_from_region_rows,
)


DEFAULT_BLOCK_DIR = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260723/block02"
)
DEFAULT_HMATS = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/cameraArray_calib/"
    "20260623_calib_elevated_by_2mm_from_arenafloor/frame0/aruco_stitch/"
    "aruco_H_mats.npz"
)
DEFAULT_SEMANTIC_LABELS = ("colony", "colony_entrance", "food", "water")
LABEL_COLORS = {
    "arena": "#53b66b",
    "arena_left": "#53b66b",
    "arena_right": "#53b66b",
    "colony": "#00b7c7",
    "colony_entrance": "#f4b942",
    "food": "#ef6f6c",
}
FALLBACK_COLORS = ("#78c679", "#c994c7", "#80b1d3", "#fdb462", "#b3de69")


def resolve_homographies(block_dir: Path, requested: Path | None, metadata_path: Path) -> Path:
    """Prefer this recording date's calibration over the pilot default."""
    if requested is not None:
        return requested.resolve()
    dataset_dir = panorama_dataset_dir(block_dir)
    calibration_root = dataset_dir.parent / "cameraArray_calib"
    candidates = sorted(calibration_root.glob(
        f"{dataset_dir.name}_calib*/frame0/aruco_stitch/aruco_H_mats.npz"
    ))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        raise ValueError("Multiple dataset calibrations; specify --hmats:\n"
                         + "\n".join(map(str, candidates)))
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return Path(metadata["homographies"]).resolve()
    return DEFAULT_HMATS.resolve()


def region_color(semantic_label: str) -> str:
    """Return the deterministic color used by both the GUI and saved PNG."""
    if semantic_label in LABEL_COLORS:
        return LABEL_COLORS[semantic_label]
    return FALLBACK_COLORS[sum(map(ord, semantic_label)) % len(FALLBACK_COLORS)]


def default_annotated_panorama_path(panorama_path: Path) -> Path:
    panorama_path = Path(panorama_path)
    return panorama_path.with_name(f"{panorama_path.stem}_annotated.png")


def arena_split_in_image(regions: list[dict[str, Any]]) -> float:
    """Use the tracker's exact region rule to preview the split on the image."""
    return tracking_split_from_region_rows([
        {"semantic_label": r["semantic_label"], "name": r["name"], "shape": r["shape"],
         "tracking_x_min_px": r["geometry"].get("x_min", float("nan")),
         "tracking_x_max_px": r["geometry"].get("x_max", float("nan"))}
        for r in regions
    ])


def _annotation_font(image: Image.Image) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_size = max(20, int(round(min(image.size) * 0.009)))
    for font_name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_annotated_panorama(
    panorama_path: Path,
    annotated_path: Path,
    regions: list[dict[str, Any]],
) -> Path:
    """Render image-coordinate region outlines, semantic labels, and names."""
    panorama_path = Path(panorama_path)
    annotated_path = Path(annotated_path)
    if not panorama_path.is_file():
        raise FileNotFoundError(f"Panorama image does not exist: {panorama_path}")

    with Image.open(panorama_path) as source:
        annotated = source.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _annotation_font(annotated)
    line_width = max(3, int(round(min(annotated.size) * 0.0012)))
    text_padding = max(4, line_width)
    label_gap = max(4, line_width)

    for region in regions:
        shape = str(region.get("shape", ""))
        if shape not in {"rectangle", "circle"}:
            raise ValueError(f"Unsupported region shape: {shape!r}")
        geometry = region.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"Region {region.get('region_id', '<unknown>')} has no image geometry")

        semantic_label = str(region.get("semantic_label", "")).strip()
        name = str(region.get("name", "")).strip()
        if not semantic_label:
            raise ValueError(f"Region {region.get('region_id', '<unknown>')} has no semantic label")
        if not name:
            name = str(region.get("region_id", semantic_label))
        color = region_color(semantic_label)

        if shape == "rectangle":
            x0 = float(geometry["x_min"])
            y0 = float(geometry["y_min"])
            x1 = float(geometry["x_max"])
            y1 = float(geometry["y_max"])
            draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
            label_anchor_x = x0 + line_width
            label_anchor_y = y0 + line_width
        else:
            center_x = float(geometry["center_x"])
            center_y = float(geometry["center_y"])
            radius = float(geometry["radius"])
            x0, y0 = center_x - radius, center_y - radius
            x1, y1 = center_x + radius, center_y + radius
            draw.ellipse((x0, y0, x1, y1), outline=color, width=line_width)
            label_anchor_x = x0
            label_anchor_y = y0 - label_gap

        label_text = semantic_label if name == semantic_label else f"{semantic_label}\n{name}"
        text_box = draw.multiline_textbbox(
            (0, 0),
            label_text,
            font=font,
            spacing=2,
            stroke_width=1,
        )
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        if shape == "circle":
            label_anchor_y -= text_height + 2 * text_padding
        label_x = min(
            max(text_padding, label_anchor_x),
            max(text_padding, annotated.width - text_width - 3 * text_padding),
        )
        label_y = min(
            max(text_padding, label_anchor_y),
            max(text_padding, annotated.height - text_height - 3 * text_padding),
        )
        background = (
            label_x - text_padding,
            label_y - text_padding,
            label_x + text_width + text_padding,
            label_y + text_height + text_padding,
        )
        draw.rounded_rectangle(background, radius=text_padding, fill=(0, 0, 0, 180), outline=color, width=1)
        draw.multiline_text(
            (label_x, label_y),
            label_text,
            font=font,
            fill=color,
            spacing=2,
            stroke_width=1,
            stroke_fill="#000000",
        )

    try:
        split_x = arena_split_in_image(regions)
    except ValueError:
        pass  # Regions can be saved for analyses before both arenas are drawn.
    else:
        draw.line((split_x, 0, split_x, annotated.height), fill="#ff3344", width=line_width)
        draw.text((split_x + 2 * line_width, 10), "Tracking split", font=font,
                  fill="#ff3344", stroke_width=2, stroke_fill="#000000")
    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = annotated_path.with_name(f".{annotated_path.stem}.{os.getpid()}.tmp.png")
    try:
        # Moderate compression keeps GUI saves quick for large panoramas; the
        # source image is already the dominant information in the output.
        annotated.save(temporary, format="PNG", compress_level=4)
        os.replace(temporary, annotated_path)
    finally:
        temporary.unlink(missing_ok=True)
    return annotated_path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "region_id",
        "semantic_label",
        "name",
        "shape",
        "image_x_min_px",
        "image_y_min_px",
        "image_x_max_px",
        "image_y_max_px",
        "image_center_x_px",
        "image_center_y_px",
        "tracking_x_min_px",
        "tracking_y_min_px",
        "tracking_x_max_px",
        "tracking_y_max_px",
        "tracking_center_x_px",
        "tracking_center_y_px",
        "radius_px",
        "radius_mm",
        "area_px2",
        "area_mm2",
        "mm_per_pixel",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def generate_panorama(
    block_dir: Path,
    hmats_path: Path,
    panorama_path: Path,
    metadata_path: Path,
    frame_index: int,
    mm_per_pixel: float,
) -> dict[str, Any]:
    """Build and save a panorama using the pipeline homographies."""
    homographies = load_h_mats(hmats_path)
    source_files, frames = load_frames_from_videos_dir_aligned(
        block_dir,
        n_expected=len(homographies),
        frame_index=frame_index,
    )
    translation, (width, height), raw_bounds = panorama_geometry(
        frames,
        [homographies[index] for index in range(len(homographies))],
    )
    panorama = make_panorama_from_hmats(homographies, frames)
    if panorama.shape[:2] != (height, width):
        raise RuntimeError(
            f"Panorama shape {panorama.shape[:2]} disagrees with geometry {(height, width)}"
        )

    panorama_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(panorama_path), panorama):
        raise IOError(f"Failed to write panorama: {panorama_path}")

    raw_x_min, raw_y_min, raw_x_max, raw_y_max = raw_bounds
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "block_dir": str(block_dir.resolve()),
        "panorama_image": str(panorama_path.resolve()),
        "homographies": str(hmats_path.resolve()),
        "sampled_frame_index": int(frame_index),
        "source_videos": [str(path.resolve()) if path is not None else None for path in source_files],
        "image_width_px": int(width),
        "image_height_px": int(height),
        "mm_per_pixel": float(mm_per_pixel),
        "raw_tracking_bounds_px": {
            "x_min": raw_x_min,
            "y_min": raw_y_min,
            "x_max": raw_x_max,
            "y_max": raw_y_max,
        },
        "raw_tracking_to_image_matrix": translation.tolist(),
        "image_to_raw_tracking_matrix": np.linalg.inv(translation).tolist(),
        "image_to_raw_tracking_offset_px": [raw_x_min, raw_y_min],
    }
    _write_json_atomic(metadata_path, metadata)
    return metadata


def load_panorama_metadata(metadata_path: Path, panorama_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing panorama metadata {metadata_path}; regenerate the panorama so annotation "
            "coordinates can be mapped to tracking coordinates."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with Image.open(panorama_path) as image:
        actual_size = image.size
    expected_size = (
        int(metadata.get("image_width_px", -1)),
        int(metadata.get("image_height_px", -1)),
    )
    if actual_size != expected_size:
        raise ValueError(
            f"Panorama is {actual_size}, but metadata records {expected_size}: {metadata_path}"
        )
    return metadata


def _translated_geometry(
    region: dict[str, Any],
    offset_x: float,
    offset_y: float,
) -> dict[str, float]:
    geometry = region["geometry"]
    if region["shape"] == "rectangle":
        return {
            "x_min": float(geometry["x_min"]) + offset_x,
            "y_min": float(geometry["y_min"]) + offset_y,
            "x_max": float(geometry["x_max"]) + offset_x,
            "y_max": float(geometry["y_max"]) + offset_y,
        }
    return {
        "center_x": float(geometry["center_x"]) + offset_x,
        "center_y": float(geometry["center_y"]) + offset_y,
        "radius": float(geometry["radius"]),
    }


def _metric_geometry(geometry: dict[str, float], mm_per_pixel: float) -> dict[str, float]:
    return {f"{key}_mm": float(value) * mm_per_pixel for key, value in geometry.items()}


def _area_px2(region: dict[str, Any]) -> float:
    geometry = region["geometry"]
    if region["shape"] == "rectangle":
        return (float(geometry["x_max"]) - float(geometry["x_min"])) * (
            float(geometry["y_max"]) - float(geometry["y_min"])
        )
    return math.pi * float(geometry["radius"]) ** 2


def build_annotation_payload(
    regions: list[dict[str, Any]],
    panorama_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    offset_x, offset_y = [
        float(value) for value in metadata["image_to_raw_tracking_offset_px"]
    ]
    mm_per_pixel = float(metadata["mm_per_pixel"])
    output_regions = []
    csv_rows = []

    for region in regions:
        image_geometry = {key: float(value) for key, value in region["geometry"].items()}
        tracking_geometry = _translated_geometry(region, offset_x, offset_y)
        area_px2 = _area_px2(region)
        output_regions.append(
            {
                "region_id": region["region_id"],
                "semantic_label": region["semantic_label"],
                "name": region["name"],
                "shape": region["shape"],
                "image_geometry_px": image_geometry,
                "tracking_geometry_px": tracking_geometry,
                "tracking_geometry_mm": _metric_geometry(tracking_geometry, mm_per_pixel),
                "area_px2": area_px2,
                "area_mm2": area_px2 * mm_per_pixel**2,
            }
        )

        row: dict[str, Any] = {
            "region_id": region["region_id"],
            "semantic_label": region["semantic_label"],
            "name": region["name"],
            "shape": region["shape"],
            "radius_px": "",
            "radius_mm": "",
            "area_px2": area_px2,
            "area_mm2": area_px2 * mm_per_pixel**2,
            "mm_per_pixel": mm_per_pixel,
        }
        if region["shape"] == "rectangle":
            for axis_key in ("x_min", "y_min", "x_max", "y_max"):
                row[f"image_{axis_key}_px"] = image_geometry[axis_key]
                row[f"tracking_{axis_key}_px"] = tracking_geometry[axis_key]
            row["image_center_x_px"] = (image_geometry["x_min"] + image_geometry["x_max"]) / 2
            row["image_center_y_px"] = (image_geometry["y_min"] + image_geometry["y_max"]) / 2
            row["tracking_center_x_px"] = (
                tracking_geometry["x_min"] + tracking_geometry["x_max"]
            ) / 2
            row["tracking_center_y_px"] = (
                tracking_geometry["y_min"] + tracking_geometry["y_max"]
            ) / 2
        else:
            row["image_center_x_px"] = image_geometry["center_x"]
            row["image_center_y_px"] = image_geometry["center_y"]
            row["tracking_center_x_px"] = tracking_geometry["center_x"]
            row["tracking_center_y_px"] = tracking_geometry["center_y"]
            row["radius_px"] = image_geometry["radius"]
            row["radius_mm"] = image_geometry["radius"] * mm_per_pixel
        csv_rows.append(row)

    payload = {
        "schema_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "panorama_image": str(panorama_path.resolve()),
        "annotated_panorama_image": str(
            default_annotated_panorama_path(panorama_path).resolve()
        ),
        "panorama_metadata": str(metadata_path.resolve()),
        "homographies": metadata["homographies"],
        "mm_per_pixel": mm_per_pixel,
        "coordinate_systems": {
            "image_geometry_px": "Displayed panorama pixels; origin at image upper left; y increases down.",
            "tracking_geometry_px": "Raw homography output used by tracking; y increases down.",
            "tracking_geometry_mm": "Raw tracking coordinates multiplied by mm_per_pixel.",
        },
        "image_to_raw_tracking_matrix": metadata["image_to_raw_tracking_matrix"],
        "regions": output_regions,
    }
    return payload, csv_rows


def save_annotations(
    regions: list[dict[str, Any]],
    annotation_path: Path,
    csv_path: Path,
    panorama_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    *,
    copy_regions_to: list[Path] | None = None,
) -> Path:
    payload, csv_rows = build_annotation_payload(
        regions,
        panorama_path,
        metadata_path,
        metadata,
    )
    _write_json_atomic(annotation_path, payload)
    _write_csv_atomic(csv_path, csv_rows)
    for block_dir in copy_regions_to or []:
        # Copies retain the source panorama/calibration as provenance; only
        # tracking-space geometry is used to analyze the other blocks.
        _write_json_atomic(Path(block_dir) / "panorama_regions.json", payload)
        _write_csv_atomic(Path(block_dir) / "panorama_regions.csv", csv_rows)
    return save_annotated_panorama(
        panorama_path,
        default_annotated_panorama_path(panorama_path),
        regions,
    )


def load_regions(annotation_path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if not annotation_path.exists():
        return []
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    expected_hmats = str(Path(metadata["homographies"]).resolve())
    saved_hmats = str(Path(payload.get("homographies", "")).resolve())
    if saved_hmats != expected_hmats:
        raise ValueError(
            "Existing annotations were made with different homographies:\n"
            f"{saved_hmats}\nCurrent: {expected_hmats}"
        )

    regions = []
    for saved in payload.get("regions", []):
        if saved.get("shape") not in {"rectangle", "circle"}:
            continue
        geometry = saved.get("image_geometry_px")
        tracking_geometry = saved.get("tracking_geometry_px")
        if isinstance(tracking_geometry, dict):
            # Shared regions remain fixed in tracking space even if another
            # block's panorama has a different displayed-image origin.
            offset_x, offset_y = metadata["image_to_raw_tracking_offset_px"]
            geometry = _translated_geometry(
                {"shape": saved["shape"], "geometry": tracking_geometry},
                -float(offset_x), -float(offset_y),
            )
        elif payload.get("image_to_raw_tracking_matrix") != metadata["image_to_raw_tracking_matrix"]:
            raise ValueError("Image-only annotations use a different panorama origin")
        if saved.get("shape") not in {"rectangle", "circle"} or not isinstance(geometry, dict):
            continue
        regions.append(
            {
                "region_id": str(saved["region_id"]),
                "semantic_label": str(saved["semantic_label"]),
                "name": str(saved.get("name", saved["region_id"])),
                "shape": str(saved["shape"]),
                "geometry": {key: float(value) for key, value in geometry.items()},
            }
        )
    return regions


def points_in_region(
    xy: np.ndarray,
    region: dict[str, Any],
    coordinate_system: str = "tracking",
) -> np.ndarray:
    """Return a boolean mask for Nx2 points and one saved JSON region."""
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {points.shape}")
    geometry_key = f"{coordinate_system}_geometry_px"
    if geometry_key not in region:
        raise KeyError(f"Region has no {geometry_key!r}")
    geometry = region[geometry_key]
    finite = np.isfinite(points).all(axis=1)
    inside = np.zeros(len(points), dtype=bool)
    if region["shape"] == "rectangle":
        inside[finite] = (
            (points[finite, 0] >= float(geometry["x_min"]))
            & (points[finite, 0] <= float(geometry["x_max"]))
            & (points[finite, 1] >= float(geometry["y_min"]))
            & (points[finite, 1] <= float(geometry["y_max"]))
        )
    else:
        dx = points[finite, 0] - float(geometry["center_x"])
        dy = points[finite, 1] - float(geometry["center_y"])
        inside[finite] = dx * dx + dy * dy <= float(geometry["radius"]) ** 2
    return inside


class PanoramaRegionAnnotator:
    def __init__(
        self,
        root: tk.Tk,
        panorama_path: Path,
        metadata_path: Path,
        annotation_path: Path,
        csv_path: Path,
        metadata: dict[str, Any],
        regions: list[dict[str, Any]],
        *,
        copy_regions_to: list[Path] | None = None,
    ) -> None:
        self.root = root
        self.panorama_path = panorama_path
        self.metadata_path = metadata_path
        self.annotation_path = annotation_path
        self.csv_path = csv_path
        self.metadata = metadata
        self.regions = regions
        self.copy_regions_to = copy_regions_to or []
        self.undo_stack: list[list[dict[str, Any]]] = []
        self.redo_stack: list[list[dict[str, Any]]] = []
        self.selected_index: int | None = None
        self.drag_start: tuple[float, float] | None = None
        self.select_press_window: tuple[int, int] | None = None
        self.select_press_image: tuple[float, float] | None = None
        self.select_dragging = False
        self.preview_item: int | None = None
        self.dirty = False
        self.zoom = 1.0
        self.fit_to_window = True
        self.render_after_id: str | None = None
        self.resize_after_id: str | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.original_image = Image.open(panorama_path).convert("RGB")
        self.offset_x, self.offset_y = [
            float(value) for value in metadata["image_to_raw_tracking_offset_px"]
        ]

        self.mode_var = tk.StringVar(value="select")
        self.label_var = tk.StringVar(value=DEFAULT_SEMANTIC_LABELS[0])
        self.name_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.zoom_text_var = tk.StringVar(value="100%")
        self.tracking_split_var = tk.StringVar(value="")

        self._configure_fonts()
        self._build_ui()
        self._bind_events()
        self.root.update_idletasks()
        self._fit_image()
        self._refresh_region_list()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_fonts(self) -> None:
        font_sizes = {
            "TkDefaultFont": 10,
            "TkTextFont": 10,
            "TkFixedFont": 10,
            "TkMenuFont": 10,
            "TkHeadingFont": 10,
            "TkCaptionFont": 11,
            "TkSmallCaptionFont": 9,
        }
        for name, size in font_sizes.items():
            tkfont.nametofont(name).configure(size=size)

        default_family = tkfont.nametofont("TkDefaultFont").actual("family")
        self.region_label_font = tkfont.Font(
            root=self.root,
            family=default_family,
            size=10,
            weight="bold",
        )
        style = ttk.Style(self.root)
        style.configure("Treeview", rowheight=34)
        style.configure("TButton", padding=(8, 5))
        style.configure("TRadiobutton", padding=(3, 3))

    def _build_ui(self) -> None:
        self.root.title(f"Panorama region annotator - {self.panorama_path.parent.name}")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.root.resizable(width=True, height=True)
        self.root.minsize(1000, 650)
        self.root.geometry(f"{min(1800, screen_w - 80)}x{min(1100, screen_h - 100)}+30+30")

        toolbar = ttk.Frame(self.root, padding=(8, 5))
        toolbar.pack(fill="x")
        edit_bar = ttk.Frame(toolbar)
        edit_bar.pack(fill="x")
        ttk.Label(edit_bar, text="Mode").pack(side="left", padx=(0, 4))
        for text, value in (("Select", "select"), ("Rectangle", "rectangle"), ("Circle", "circle")):
            ttk.Radiobutton(edit_bar, text=text, value=value, variable=self.mode_var).pack(side="left")

        ttk.Separator(edit_bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(edit_bar, text="Semantic label").pack(side="left", padx=(0, 4))
        self.label_box = ttk.Combobox(
            edit_bar,
            textvariable=self.label_var,
            values=DEFAULT_SEMANTIC_LABELS,
            width=18,
        )
        self.label_box.pack(side="left")
        ttk.Label(edit_bar, text="Name").pack(side="left", padx=(10, 4))
        ttk.Entry(edit_bar, textvariable=self.name_var, width=22).pack(side="left", fill="x", expand=True)
        ttk.Button(edit_bar, text="Apply", command=self._apply_fields).pack(side="left", padx=(5, 0))
        ttk.Button(edit_bar, text="Delete", command=self._delete_selected).pack(side="left", padx=(5, 0))

        arena_bar = ttk.Frame(toolbar)
        arena_bar.pack(fill="x", pady=(5, 0))
        ttk.Label(arena_bar, text="Full arena boundary").pack(side="left", padx=(0, 6))
        for side, label in TRACKING_ARENA_LABELS.items():
            ttk.Button(arena_bar, text=f"{side.title()} arena",
                       command=lambda value=label: self.mode_var.set(value)).pack(side="left", padx=(0, 4))
        ttk.Label(arena_bar, textvariable=self.tracking_split_var, wraplength=750).pack(
            side="left", padx=(12, 0))

        view_bar = ttk.Frame(toolbar)
        view_bar.pack(fill="x", pady=(5, 0))
        ttk.Button(view_bar, text="Undo", command=self._undo).pack(side="left")
        ttk.Button(view_bar, text="Redo", command=self._redo).pack(side="left", padx=(4, 0))
        ttk.Separator(view_bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(view_bar, text="Zoom").pack(side="left", padx=(0, 4))
        ttk.Button(
            view_bar,
            text="Zoom out",
            command=lambda: self._set_zoom(self.zoom / 1.25),
        ).pack(side="left")
        ttk.Label(view_bar, textvariable=self.zoom_text_var, width=7, anchor="center").pack(side="left")
        ttk.Button(
            view_bar,
            text="Zoom in",
            command=lambda: self._set_zoom(self.zoom * 1.25),
        ).pack(side="left")
        ttk.Button(view_bar, text="Fit window", command=self._fit_image).pack(side="left", padx=(4, 0))
        ttk.Button(view_bar, text="Save", command=self._save).pack(side="right")

        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        canvas_frame = ttk.Frame(paned)
        self.canvas = tk.Canvas(canvas_frame, bg="#202326", highlightthickness=0)
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self._on_xscroll)
        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._on_yscroll)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        paned.add(canvas_frame, weight=5)

        sidebar = ttk.Frame(paned, padding=(8, 4))
        ttk.Label(sidebar, text="Regions").pack(anchor="w")
        self.region_tree = ttk.Treeview(
            sidebar,
            columns=("label", "name", "shape"),
            show="headings",
            selectmode="browse",
            height=24,
        )
        self.region_tree.heading("label", text="Label")
        self.region_tree.heading("name", text="Name")
        self.region_tree.heading("shape", text="Shape")
        self.region_tree.column("label", width=120, stretch=True)
        self.region_tree.column("name", width=130, stretch=True)
        self.region_tree.column("shape", width=70, stretch=False)
        self.region_tree.pack(fill="both", expand=True, pady=(4, 0))
        paned.add(sidebar, weight=1)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(8, 4))
        status.pack(fill="x")

    def _bind_events(self) -> None:
        self.mode_var.trace_add("write", self._on_mode_changed)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan_canvas)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._pan_canvas)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._set_zoom(self.zoom * 1.15, event))
        self.canvas.bind("<Button-5>", lambda event: self._set_zoom(self.zoom / 1.15, event))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.region_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.root.bind("<Control-s>", lambda _event: self._save())
        self.root.bind("<Control-z>", lambda _event: self._undo())
        self.root.bind("<Control-y>", lambda _event: self._redo())
        self.root.bind("<Control-plus>", lambda _event: self._set_zoom(self.zoom * 1.25))
        self.root.bind("<Control-equal>", lambda _event: self._set_zoom(self.zoom * 1.25))
        self.root.bind("<Control-minus>", lambda _event: self._set_zoom(self.zoom / 1.25))
        self.root.bind("<Control-Key-0>", lambda _event: self._fit_image())
        self.root.bind("<Delete>", lambda _event: self._delete_selected())
        self.root.bind("<Escape>", lambda _event: self.mode_var.set("select"))

    def _on_mode_changed(self, *_args: str) -> None:
        mode = self.mode_var.get()
        if self.preview_item is not None:
            self.canvas.delete(self.preview_item)
            self.preview_item = None
        self.drag_start = None
        if mode != "select":
            self.selected_index = None
            self.region_tree.selection_remove(self.region_tree.selection())
            self.name_var.set("")
            if mode in TRACKING_ARENA_LABELS.values():
                self.label_var.set(mode)
                self.status_var.set("Drag a rectangle around the full arena, including the foraging area.")
            elif arena_label_side(self.label_var.get()) is not None:
                self.label_var.set(DEFAULT_SEMANTIC_LABELS[0])
        self._update_label_state()

    def _update_label_state(self) -> None:
        fixed = self.mode_var.get() in TRACKING_ARENA_LABELS.values()
        if self.selected_index is not None:
            fixed |= arena_label_side(self.regions[self.selected_index]["semantic_label"]) is not None
        self.label_box.configure(state="disabled" if fixed else "normal")

    def _fit_image(self) -> None:
        self.root.update_idletasks()
        available_w = max(100, self.canvas.winfo_width() - 20)
        available_h = max(100, self.canvas.winfo_height() - 20)
        fit = min(available_w / self.original_image.width, available_h / self.original_image.height)
        self.fit_to_window = True
        self.zoom = max(0.03, min(8.0, fit))
        self._update_scrollregion()
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)
        self._render()
        self._update_zoom_text()

    def _set_zoom(self, requested: float, event: tk.Event | None = None) -> None:
        new_zoom = max(0.03, min(8.0, float(requested)))
        old_zoom = self.zoom
        if event is None:
            focus_x = self.canvas.canvasx(self.canvas.winfo_width() / 2) / old_zoom
            focus_y = self.canvas.canvasy(self.canvas.winfo_height() / 2) / old_zoom
            window_x = self.canvas.winfo_width() / 2
            window_y = self.canvas.winfo_height() / 2
        else:
            focus_x = self.canvas.canvasx(event.x) / old_zoom
            focus_y = self.canvas.canvasy(event.y) / old_zoom
            window_x = event.x
            window_y = event.y
        self.zoom = new_zoom
        self.fit_to_window = False
        self._update_scrollregion()
        scaled_w = self.original_image.width * self.zoom
        scaled_h = self.original_image.height * self.zoom
        if scaled_w > 0:
            self.canvas.xview_moveto(max(0.0, (focus_x * self.zoom - window_x) / scaled_w))
        if scaled_h > 0:
            self.canvas.yview_moveto(max(0.0, (focus_y * self.zoom - window_y) / scaled_h))
        self._render()
        self._update_zoom_text()
        self.status_var.set(f"Zoom {self.zoom * 100:.0f}%")

    def _update_scrollregion(self) -> None:
        width = max(1, int(round(self.original_image.width * self.zoom)))
        height = max(1, int(round(self.original_image.height * self.zoom)))
        self.canvas.configure(scrollregion=(0, 0, width, height))

    def _update_zoom_text(self) -> None:
        self.zoom_text_var.set(f"{self.zoom * 100:.0f}%")

    def _render(self) -> None:
        if self.render_after_id is not None:
            self.root.after_cancel(self.render_after_id)
            self.render_after_id = None
        self._update_scrollregion()

        view_left = max(0.0, self.canvas.canvasx(0))
        view_top = max(0.0, self.canvas.canvasy(0))
        view_right = min(
            self.original_image.width * self.zoom,
            self.canvas.canvasx(self.canvas.winfo_width()),
        )
        view_bottom = min(
            self.original_image.height * self.zoom,
            self.canvas.canvasy(self.canvas.winfo_height()),
        )
        image_x0 = max(0, int(math.floor(view_left / self.zoom)) - 2)
        image_y0 = max(0, int(math.floor(view_top / self.zoom)) - 2)
        image_x1 = min(
            self.original_image.width,
            int(math.ceil(view_right / self.zoom)) + 2,
        )
        image_y1 = min(
            self.original_image.height,
            int(math.ceil(view_bottom / self.zoom)) + 2,
        )

        self.canvas.delete("all")
        if image_x1 <= image_x0 or image_y1 <= image_y0:
            return
        visible = self.original_image.crop((image_x0, image_y0, image_x1, image_y1))
        render_width = max(1, int(round((image_x1 - image_x0) * self.zoom)))
        render_height = max(1, int(round((image_y1 - image_y0) * self.zoom)))
        resized = visible.resize((render_width, render_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(
            image_x0 * self.zoom,
            image_y0 * self.zoom,
            image=self.photo,
            anchor="nw",
            tags=("panorama",),
        )
        for index, region in enumerate(self.regions):
            self._draw_region(index, region, selected=index == self.selected_index)
        try:
            split_x = arena_split_in_image(self.regions)
        except ValueError as error:
            self.tracking_split_var.set(f"Tracking split unavailable: {str(error).split(': ', 1)[-1]}")
        else:
            self.tracking_split_var.set(f"Tracking split X = {split_x + self.offset_x:.2f} px (red line)")
            self.canvas.create_line(split_x * self.zoom, 0, split_x * self.zoom,
                                    self.original_image.height * self.zoom,
                                    fill="#ff3344", width=3, dash=(8, 5), tags=("tracking_split",))

    def _schedule_render(self) -> None:
        if self.render_after_id is not None:
            self.root.after_cancel(self.render_after_id)
        self.render_after_id = self.root.after_idle(self._render)

    def _on_xscroll(self, *args: str) -> None:
        self.canvas.xview(*args)
        self._schedule_render()

    def _on_yscroll(self, *args: str) -> None:
        self.canvas.yview(*args)
        self._schedule_render()

    def _start_pan(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _pan_canvas(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self._schedule_render()

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self.resize_after_id is not None:
            self.root.after_cancel(self.resize_after_id)
        self.resize_after_id = self.root.after(80, self._finish_canvas_resize)

    def _finish_canvas_resize(self) -> None:
        self.resize_after_id = None
        if self.fit_to_window:
            self._fit_image()
        else:
            self._render()

    def _on_double_click(self, event: tk.Event) -> None:
        if self.mode_var.get() == "select":
            self._set_zoom(self.zoom * 1.5, event)

    def _region_color(self, semantic_label: str) -> str:
        return region_color(semantic_label)

    def _draw_region(self, index: int, region: dict[str, Any], selected: bool) -> None:
        geometry = region["geometry"]
        color = self._region_color(region["semantic_label"])
        width = 4 if selected else 2
        tags = ("region", f"region_{index}")
        if region["shape"] == "rectangle":
            x0, y0 = geometry["x_min"] * self.zoom, geometry["y_min"] * self.zoom
            x1, y1 = geometry["x_max"] * self.zoom, geometry["y_max"] * self.zoom
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width, tags=tags)
            label_x, label_y = x0 + 4, y0 + 4
        else:
            cx, cy = geometry["center_x"] * self.zoom, geometry["center_y"] * self.zoom
            radius = geometry["radius"] * self.zoom
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                outline=color,
                width=width,
                tags=tags,
            )
            label_x, label_y = cx - radius + 4, cy - radius + 4
        label_text = region["name"] or region["semantic_label"]
        self.canvas.create_text(
            label_x + 1,
            label_y + 1,
            text=label_text,
            anchor="nw",
            fill="#000000",
            font=self.region_label_font,
            tags=tags,
        )
        self.canvas.create_text(
            label_x,
            label_y,
            text=label_text,
            anchor="nw",
            fill=color,
            font=self.region_label_font,
            tags=tags,
        )

    def _image_xy(self, event: tk.Event) -> tuple[float, float]:
        x = self.canvas.canvasx(event.x) / self.zoom
        y = self.canvas.canvasy(event.y) / self.zoom
        return (
            max(0.0, min(float(self.original_image.width), x)),
            max(0.0, min(float(self.original_image.height), y)),
        )

    def _on_motion(self, event: tk.Event) -> None:
        x, y = self._image_xy(event)
        self.status_var.set(
            f"Image ({x:.1f}, {y:.1f}) px   Tracking ({x + self.offset_x:.1f}, "
            f"{y + self.offset_y:.1f}) px   Zoom {self.zoom:.2f}x"
        )

    def _on_mousewheel(self, event: tk.Event) -> None:
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self._set_zoom(self.zoom * factor, event)

    def _on_left_press(self, event: tk.Event) -> None:
        point = self._image_xy(event)
        if self.mode_var.get() == "select":
            self.select_press_window = (event.x, event.y)
            self.select_press_image = point
            self.select_dragging = False
            self.canvas.scan_mark(event.x, event.y)
            return
        self.drag_start = point
        self.preview_item = None

    def _on_left_drag(self, event: tk.Event) -> None:
        if self.mode_var.get() == "select":
            if self.select_press_window is None:
                return
            dx = event.x - self.select_press_window[0]
            dy = event.y - self.select_press_window[1]
            if math.hypot(dx, dy) >= 4:
                self.select_dragging = True
            if self.select_dragging:
                self.canvas.scan_dragto(event.x, event.y, gain=1)
                self._schedule_render()
            return
        if self.drag_start is None:
            return
        if self.preview_item is not None:
            self.canvas.delete(self.preview_item)
        x0, y0 = self.drag_start
        x1, y1 = self._image_xy(event)
        color = self._region_color(self.label_var.get().strip())
        if self.mode_var.get() in ("rectangle", *TRACKING_ARENA_LABELS.values()):
            coordinates = (x0 * self.zoom, y0 * self.zoom, x1 * self.zoom, y1 * self.zoom)
            self.preview_item = self.canvas.create_rectangle(*coordinates, outline=color, width=3, dash=(5, 3))
        else:
            radius = math.hypot(x1 - x0, y1 - y0)
            coordinates = (
                (x0 - radius) * self.zoom,
                (y0 - radius) * self.zoom,
                (x0 + radius) * self.zoom,
                (y0 + radius) * self.zoom,
            )
            self.preview_item = self.canvas.create_oval(*coordinates, outline=color, width=3, dash=(5, 3))

    def _on_left_release(self, event: tk.Event) -> None:
        if self.mode_var.get() == "select":
            if self.select_press_window is not None and not self.select_dragging:
                self._select_at(self.select_press_image or self._image_xy(event))
            self.select_press_window = None
            self.select_press_image = None
            self.select_dragging = False
            return
        if self.drag_start is None:
            return
        if self.preview_item is not None:
            self.canvas.delete(self.preview_item)
            self.preview_item = None
        x0, y0 = self.drag_start
        x1, y1 = self._image_xy(event)
        self.drag_start = None
        mode = self.mode_var.get()
        is_arena_tool = mode in TRACKING_ARENA_LABELS.values()
        semantic_label = mode if is_arena_tool else self.label_var.get().strip()
        if not semantic_label:
            messagebox.showerror("Missing label", "Enter a semantic label before drawing a region.")
            return

        shape = "rectangle" if is_arena_tool else mode
        if shape == "rectangle":
            geometry = {
                "x_min": min(x0, x1),
                "y_min": min(y0, y1),
                "x_max": max(x0, x1),
                "y_max": max(y0, y1),
            }
            if geometry["x_max"] - geometry["x_min"] < 4 or geometry["y_max"] - geometry["y_min"] < 4:
                return
        else:
            radius = math.hypot(x1 - x0, y1 - y0)
            if radius < 4:
                return
            geometry = {"center_x": x0, "center_y": y0, "radius": radius}

        self._push_undo()
        region_id = self._next_region_id()
        name = self.name_var.get().strip() or self._next_default_name(semantic_label)
        new_region = {
            "region_id": region_id,
            "semantic_label": semantic_label,
            "name": name,
            "shape": shape,
            "geometry": geometry,
        }
        # Redrawing an arena replaces that side; Undo restores its previous bounds.
        existing = next((i for i, region in enumerate(self.regions)
                         if is_arena_tool and arena_label_side(region["semantic_label"])
                         == arena_label_side(semantic_label)), None)
        if existing is None:
            self.regions.append(new_region)
            self.selected_index = len(self.regions) - 1
        else:
            new_region["region_id"] = self.regions[existing]["region_id"]
            self.regions[existing] = new_region
            self.selected_index = existing
        self.name_var.set("")
        if is_arena_tool:
            self.label_var.set(semantic_label)
            self.name_var.set(new_region["name"])
            self.mode_var.set("select")
        self.dirty = True
        self._refresh_region_list()
        self._render()

    def _next_region_id(self) -> str:
        used = {region["region_id"] for region in self.regions}
        index = 1
        while f"region_{index:03d}" in used:
            index += 1
        return f"region_{index:03d}"

    def _next_default_name(self, semantic_label: str) -> str:
        matching = sum(region["semantic_label"] == semantic_label for region in self.regions)
        return f"{semantic_label}_{matching + 1}"

    def _select_at(self, point: tuple[float, float]) -> None:
        x, y = point
        selected = None
        for index in range(len(self.regions) - 1, -1, -1):
            region = self.regions[index]
            geometry = region["geometry"]
            if region["shape"] == "rectangle":
                inside = (
                    geometry["x_min"] <= x <= geometry["x_max"]
                    and geometry["y_min"] <= y <= geometry["y_max"]
                )
            else:
                inside = (
                    (x - geometry["center_x"]) ** 2 + (y - geometry["center_y"]) ** 2
                    <= geometry["radius"] ** 2
                )
            if inside:
                selected = index
                break
        self._select_index(selected)

    def _select_index(self, index: int | None) -> None:
        self.selected_index = index
        if index is not None:
            region = self.regions[index]
            self.label_var.set(region["semantic_label"])
            self.name_var.set(region["name"])
            item = str(index)
            if self.region_tree.exists(item):
                self.region_tree.selection_set(item)
                self.region_tree.see(item)
        else:
            self.region_tree.selection_remove(self.region_tree.selection())
        self._update_label_state()
        self._render()

    def _on_tree_select(self, _event: tk.Event) -> None:
        selection = self.region_tree.selection()
        if selection:
            index = int(selection[0])
            if index != self.selected_index:
                self._select_index(index)

    def _refresh_region_list(self) -> None:
        self.region_tree.delete(*self.region_tree.get_children())
        for index, region in enumerate(self.regions):
            self.region_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(region["semantic_label"], region["name"], region["shape"]),
            )
        if self.selected_index is not None and self.selected_index < len(self.regions):
            self.region_tree.selection_set(str(self.selected_index))
        self._update_label_state()

    def _apply_fields(self) -> None:
        if self.selected_index is None:
            return
        semantic_label = self.label_var.get().strip()
        original_label = self.regions[self.selected_index]["semantic_label"]
        if arena_label_side(original_label) is not None:
            semantic_label = original_label
        if not semantic_label:
            messagebox.showerror("Missing label", "Semantic label cannot be empty.")
            return
        self._push_undo()
        region = self.regions[self.selected_index]
        region["semantic_label"] = semantic_label
        region["name"] = self.name_var.get().strip() or region["region_id"]
        self.dirty = True
        self._refresh_region_list()
        self._render()

    def _delete_selected(self) -> None:
        if self.selected_index is None:
            return
        self._push_undo()
        del self.regions[self.selected_index]
        self.selected_index = None
        self.dirty = True
        self._refresh_region_list()
        self._render()

    def _push_undo(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.regions))
        self.undo_stack = self.undo_stack[-100:]
        self.redo_stack.clear()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.regions))
        self.regions = self.undo_stack.pop()
        self.selected_index = None
        self.dirty = True
        self._refresh_region_list()
        self._render()

    def _redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(copy.deepcopy(self.regions))
        self.regions = self.redo_stack.pop()
        self.selected_index = None
        self.dirty = True
        self._refresh_region_list()
        self._render()

    def _save(self) -> None:
        try:
            annotated_path = save_annotations(
                self.regions,
                self.annotation_path,
                self.csv_path,
                self.panorama_path,
                self.metadata_path,
                self.metadata,
                copy_regions_to=self.copy_regions_to,
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.dirty = False
        self.status_var.set(
            f"Saved {len(self.regions)} regions to {self.annotation_path}, "
            f"{self.csv_path.name}, and {annotated_path.name}"
            + (f"; copied to {len(self.copy_regions_to)} other blocks" if self.copy_regions_to else "")
        )

    def _on_close(self) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel("Unsaved regions", "Save region annotations before closing?")
            if answer is None:
                return
            if answer:
                self._save()
                if self.dirty:
                    return
        self.original_image.close()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a calibrated panorama and annotate arena, colony, entrance, food, and water regions."
    )
    parser.add_argument("--block-dir", type=Path, default=DEFAULT_BLOCK_DIR)
    parser.add_argument("--hmats", type=Path, default=None,
                        help="Homographies; default: matching date calibration, then saved panorama calibration, then legacy default")
    parser.add_argument("--panorama", type=Path, default=None, help="Default: <block-dir>/panorama_from_hmats.png")
    parser.add_argument("--metadata", type=Path, default=None, help="Default: <block-dir>/panorama_from_hmats_metadata.json")
    parser.add_argument("--annotations", type=Path, default=None, help="Default: <block-dir>/panorama_regions.json")
    parser.add_argument("--csv", type=Path, default=None, help="Default: panorama_regions.csv beside the annotation JSON")
    parser.add_argument("--copy-regions-to", type=Path, action="append", default=[], metavar="BLOCK_DIR",
                        help="Also save JSON/CSV copies in this block on every Save; repeat for multiple blocks")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=720,
        help="Video frame sampled for the panorama. Default: 720 (30 seconds at 24 fps).",
    )
    parser.add_argument("--mm-per-pixel", type=float, default=0.016)
    parser.add_argument("--regenerate", action="store_true", help="Overwrite an existing panorama and metadata.")
    parser.add_argument("--no-gui", action="store_true", help="Generate/validate files without opening the annotator.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    block_dir = args.block_dir.resolve()
    panorama_path = (args.panorama or block_dir / "panorama_from_hmats.png").resolve()
    metadata_path = (args.metadata or block_dir / "panorama_from_hmats_metadata.json").resolve()
    hmats_path = resolve_homographies(block_dir, args.hmats, metadata_path)
    annotation_path = (args.annotations or panorama_regions_path(block_dir, ".json", for_write=True)).resolve()
    csv_path = (args.csv or annotation_path.with_suffix(".csv")).resolve()

    if not block_dir.is_dir():
        raise FileNotFoundError(f"Block directory not found: {block_dir}")
    if not hmats_path.is_file():
        raise FileNotFoundError(f"Homography file not found: {hmats_path}")

    if args.regenerate or not panorama_path.exists() or not metadata_path.exists():
        print(f"Generating panorama from {block_dir}")
        metadata = generate_panorama(
            block_dir,
            hmats_path,
            panorama_path,
            metadata_path,
            frame_index=max(0, int(args.frame_index)),
            mm_per_pixel=float(args.mm_per_pixel),
        )
        print(f"Saved panorama: {panorama_path}")
        print(f"Saved panorama metadata: {metadata_path}")
    else:
        metadata = load_panorama_metadata(metadata_path, panorama_path)
        if Path(metadata["homographies"]).resolve() != hmats_path:
            raise ValueError(
                f"Existing panorama uses {metadata['homographies']}, but selected calibration is {hmats_path}. "
                "Use --regenerate to rebuild the panorama before annotating."
            )
        print(f"Using existing panorama: {panorama_path}")

    load_path = annotation_path if args.annotations else panorama_regions_path(block_dir, ".json")
    regions = load_regions(load_path, metadata)
    print(f"Region save paths: {annotation_path} and {csv_path}", flush=True)
    if load_path != annotation_path and load_path.is_file():
        print(f"Loaded shared annotations from {load_path}; Save will write a block-local copy.")
    copy_regions_to = [path.resolve() for path in args.copy_regions_to]
    for path in copy_regions_to:
        if not path.is_dir():
            raise FileNotFoundError(f"Region copy destination does not exist: {path}")
        print(f"Also saving region copies to: {path}")
    if args.no_gui:
        print(f"Panorama size: {metadata['image_width_px']} x {metadata['image_height_px']} px")
        print(f"Existing regions: {len(regions)}")
        return

    root = tk.Tk()
    PanoramaRegionAnnotator(
        root,
        panorama_path,
        metadata_path,
        annotation_path,
        csv_path,
        metadata,
        regions,
        copy_regions_to=copy_regions_to,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
