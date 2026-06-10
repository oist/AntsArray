#!/usr/bin/env python3
"""Estimate camera pixel resolution from ArUco tag detections in PNG images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import cv2.aruco as aruco
import numpy as np


@dataclass(frozen=True)
class MarkerMeasurement:
    image_path: Path
    marker_id: int
    side_px: float
    px_per_um: float


def _normalize_window_param(value: int) -> int:
    out = max(3, int(value))
    if out % 2 == 0:
        out += 1
    return out


def _set_if_attr(obj: object, name: str, value: object) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _dictionary_from_name(name: str) -> aruco.Dictionary:
    attr = name if name.startswith("DICT_") else f"DICT_{name}"
    if not hasattr(aruco, attr):
        known = sorted(k for k in dir(aruco) if k.startswith("DICT_"))
        raise SystemExit(f"Unknown ArUco dictionary {name!r}. Known examples: {', '.join(known[:8])}, ...")
    return aruco.getPredefinedDictionary(getattr(aruco, attr))


def _load_custom_dictionary(path: Path) -> aruco.Dictionary:
    data = np.load(str(path), allow_pickle=True)
    if "bytesList" not in data:
        raise SystemExit(f"Custom dictionary missing bytesList: {path}")
    if "max_correction_bits" not in data:
        raise SystemExit(f"Custom dictionary missing max_correction_bits: {path}")

    custom = aruco.Dictionary()
    custom.bytesList = data["bytesList"]
    custom.markerSize = int(data["marker_size"]) if "marker_size" in data.files else 4
    custom.maxCorrectionBits = int(data["max_correction_bits"])
    return custom


def build_detector(args: argparse.Namespace) -> aruco.ArucoDetector:
    dictionary = _load_custom_dictionary(args.custom_dict) if args.custom_dict else _dictionary_from_name(args.dict)
    params = aruco.DetectorParameters()
    _set_if_attr(params, "adaptiveThreshConstant", float(args.adaptive_thresh_constant))
    _set_if_attr(params, "adaptiveThreshWinSizeMin", _normalize_window_param(args.adaptive_thresh_win_min))
    _set_if_attr(params, "adaptiveThreshWinSizeMax", _normalize_window_param(args.adaptive_thresh_win_max))
    _set_if_attr(params, "adaptiveThreshWinSizeStep", max(1, int(args.adaptive_thresh_win_step)))
    _set_if_attr(params, "errorCorrectionRate", float(args.error_correction_rate))
    _set_if_attr(params, "minMarkerPerimeterRate", float(args.min_marker_perimeter_rate))
    _set_if_attr(params, "maxMarkerPerimeterRate", float(args.max_marker_perimeter_rate))
    _set_if_attr(params, "polygonalApproxAccuracyRate", float(args.polygonal_approx_accuracy_rate))
    if hasattr(aruco, "CORNER_REFINE_CONTOUR"):
        _set_if_attr(params, "cornerRefinementMethod", int(aruco.CORNER_REFINE_CONTOUR))
    return aruco.ArucoDetector(dictionary, params)


def image_paths(input_dir: Path, *, recursive: bool) -> list[Path]:
    pattern = "**/*.png" if recursive else "*.png"
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def marker_side_px(corners: np.ndarray) -> float:
    pts = np.asarray(corners, dtype=float).reshape(4, 2)
    edge_lengths = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    return float(np.mean(edge_lengths))


def detect_measurements(
    paths: Iterable[Path],
    *,
    detector: aruco.ArucoDetector,
    tag_size_um: float,
) -> list[MarkerMeasurement]:
    measurements: list[MarkerMeasurement] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: could not read {path}")
            continue

        corners, ids, _rejected = detector.detectMarkers(image)
        if ids is None or len(ids) == 0:
            print(f"{path.name}: 0 markers")
            continue

        ids_flat = ids.reshape(-1)
        image_sides: list[float] = []
        for marker_corners, marker_id in zip(corners, ids_flat, strict=False):
            side_px = marker_side_px(marker_corners)
            if not np.isfinite(side_px) or side_px <= 0:
                continue
            measurements.append(
                MarkerMeasurement(
                    image_path=path,
                    marker_id=int(marker_id),
                    side_px=side_px,
                    px_per_um=side_px / tag_size_um,
                )
            )
            image_sides.append(side_px)

        if image_sides:
            image_px_per_um = np.asarray(image_sides, dtype=float) / tag_size_um
            print(
                f"{path.name}: {len(image_sides)} markers, "
                f"median {np.median(image_px_per_um):.6g} px/um "
                f"({1.0 / np.median(image_px_per_um):.6g} um/px)"
            )
        else:
            print(f"{path.name}: 0 usable markers")
    return measurements


def print_summary(measurements: list[MarkerMeasurement], *, tag_size_um: float) -> None:
    if not measurements:
        raise SystemExit("No ArUco markers detected; no resolution estimate available.")

    px_per_um = np.asarray([m.px_per_um for m in measurements], dtype=float)
    side_px = np.asarray([m.side_px for m in measurements], dtype=float)
    image_count = len({m.image_path for m in measurements})
    marker_ids = sorted({m.marker_id for m in measurements})

    print()
    print("Resolution estimate")
    print(f"Images with usable detections: {image_count}")
    print(f"Markers measured: {len(measurements)}")
    print(f"Unique marker IDs: {len(marker_ids)}")
    print(f"Physical tag side: {tag_size_um / 1000.0:g} mm ({tag_size_um:g} um), excluding white margin")
    print(f"Median marker side: {np.median(side_px):.6g} px")
    print(f"Mean marker side: {np.mean(side_px):.6g} px")
    print(f"Median resolution: {np.median(px_per_um):.9g} px/um")
    print(f"Mean resolution: {np.mean(px_per_um):.9g} px/um")
    print(f"Std resolution: {np.std(px_per_um, ddof=1) if len(px_per_um) > 1 else 0.0:.9g} px/um")
    print(f"Median inverse: {1.0 / np.median(px_per_um):.9g} um/px")
    print(f"Mean inverse: {1.0 / np.mean(px_per_um):.9g} um/px")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect ArUco tags in PNG images and estimate pixels per micron from "
            "the detected marker square. The physical tag size excludes any white margin."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing PNG images.")
    parser.add_argument("--recursive", action="store_true", help="Search for PNG files recursively.")
    parser.add_argument("--tag_size_mm", type=float, default=1.5, help="ArUco marker side length in mm. Default: 1.5.")
    parser.add_argument("--dict", default="DICT_4X4_1000", help="OpenCV predefined dictionary. Default: DICT_4X4_1000.")
    parser.add_argument("--custom_dict", type=Path, default=None, help="Optional custom ArUco dictionary .npz.")
    parser.add_argument("--adaptive-thresh-constant", type=float, default=3.0)
    parser.add_argument("--adaptive-thresh-win-min", type=int, default=10)
    parser.add_argument("--adaptive-thresh-win-max", type=int, default=40)
    parser.add_argument("--adaptive-thresh-win-step", type=int, default=10)
    parser.add_argument("--error-correction-rate", type=float, default=1.0)
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.03)
    parser.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
    parser.add_argument("--polygonal-approx-accuracy-rate", type=float, default=0.03)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {args.input_dir}")
    if args.tag_size_mm <= 0:
        raise SystemExit("--tag_size_mm must be > 0.")

    paths = image_paths(args.input_dir, recursive=args.recursive)
    if not paths:
        raise SystemExit(f"No PNG files found in {args.input_dir}")

    tag_size_um = float(args.tag_size_mm) * 1000.0
    detector = build_detector(args)
    measurements = detect_measurements(paths, detector=detector, tag_size_um=tag_size_um)
    print_summary(measurements, tag_size_um=tag_size_um)


if __name__ == "__main__":
    main()
