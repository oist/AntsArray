#!/usr/bin/env python3
"""
ArUco marker detection for video files.

Outputs:
- Raw numpy arrays (H5): aruco_tracks, aruco_confidences
- DataFrame (CSV and/or H5): Frame, Instance, X, Y, Confidence

Debug features:
- On-screen visualization and/or saved debug video.
- Optional rejected-candidate overlay.

Parameter optimization mode:
- Sweeps selected OpenCV ArUco detector parameters.
- Scores each config for "exactly one marker per frame" videos.
- Saves ranked CSV and prints the best parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import os
import sys

import cv2
import cv2.aruco as aruco
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Detector config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectorConfig:
    corner_refinement: str = "contour"
    adaptive_thresh_constant: float = 3.0
    adaptive_thresh_win_min: int = 10
    adaptive_thresh_win_max: int = 40
    adaptive_thresh_win_step: int = 10
    error_correction_rate: float = 1.0
    min_marker_perimeter_rate: float = 0.03
    max_marker_perimeter_rate: float = 4.0
    polygonal_approx_accuracy_rate: float = 0.03


def _set_if_attr(obj: object, name: str, value: object) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _corner_refine_enum(name: str) -> int:
    key = name.strip().lower()
    mapping = {
        "none": "CORNER_REFINE_NONE",
        "subpix": "CORNER_REFINE_SUBPIX",
        "contour": "CORNER_REFINE_CONTOUR",
        "apriltag": "CORNER_REFINE_APRILTAG",
    }
    attr = mapping.get(key)
    if attr is None:
        raise ValueError(f"Unknown corner refinement mode: {name}")
    if not hasattr(aruco, attr):
        # Fall back conservatively if OpenCV build lacks this enum.
        return int(getattr(aruco, "CORNER_REFINE_CONTOUR"))
    return int(getattr(aruco, attr))


def _normalize_window_param(v: int) -> int:
    # OpenCV expects positive odd windows.
    vv = max(3, int(v))
    if vv % 2 == 0:
        vv += 1
    return vv


def load_custom_aruco_dict(npz_path: str | Path) -> aruco.Dictionary:
    data = np.load(str(npz_path), allow_pickle=True)
    if "bytesList" not in data:
        raise ValueError(f"Custom ArUco dictionary missing bytesList: {npz_path}")
    if "max_correction_bits" not in data:
        raise ValueError(f"Custom ArUco dictionary missing max_correction_bits: {npz_path}")

    custom = aruco.Dictionary()
    custom.bytesList = data["bytesList"]
    custom.markerSize = int(data["marker_size"]) if "marker_size" in data.files else 4
    custom.maxCorrectionBits = int(data["max_correction_bits"])
    return custom


def build_aruco_detector(
    config: DetectorConfig,
    *,
    aruco_dict: Optional[aruco.Dictionary] = None,
) -> aruco.ArucoDetector:
    params = aruco.DetectorParameters()
    _set_if_attr(params, "cornerRefinementMethod", _corner_refine_enum(config.corner_refinement))
    _set_if_attr(params, "adaptiveThreshConstant", float(config.adaptive_thresh_constant))
    _set_if_attr(params, "adaptiveThreshWinSizeMin", _normalize_window_param(config.adaptive_thresh_win_min))
    _set_if_attr(params, "adaptiveThreshWinSizeMax", _normalize_window_param(config.adaptive_thresh_win_max))
    _set_if_attr(params, "adaptiveThreshWinSizeStep", max(1, int(config.adaptive_thresh_win_step)))
    _set_if_attr(params, "errorCorrectionRate", float(config.error_correction_rate))
    _set_if_attr(params, "minMarkerPerimeterRate", float(config.min_marker_perimeter_rate))
    _set_if_attr(params, "maxMarkerPerimeterRate", float(config.max_marker_perimeter_rate))
    _set_if_attr(params, "polygonalApproxAccuracyRate", float(config.polygonal_approx_accuracy_rate))

    if aruco_dict is None:
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    return aruco.ArucoDetector(aruco_dict, params)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def _annotate_debug_frame(
    frame_bgr: np.ndarray,
    corners: Sequence[np.ndarray],
    ids: Optional[np.ndarray],
    rejected: Sequence[np.ndarray],
    frame_idx: int,
    *,
    show_rejected: bool,
    scale_text: float = 0.6,
) -> np.ndarray:
    """
    Draw detected markers and optional rejected candidates.
    """
    vis = frame_bgr.copy()

    det_count = 0
    id_list: List[int] = []
    if ids is not None and len(ids) > 0:
        det_count = int(len(ids))
        # Draw borders only (no built-in IDs) to avoid duplicate labels.
        aruco.drawDetectedMarkers(vis, corners, None)
        ids_flat = ids.flatten().astype(int)
        id_list = ids_flat.tolist()

        for i, marker_id in enumerate(ids_flat):
            corner = corners[i][0]  # (4, 2)
            center = np.mean(corner, axis=0)
            cx, cy = int(round(center[0])), int(round(center[1]))
            cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(
                vis,
                f"ID {int(marker_id)}",
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale_text,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    rej_count = 0
    if show_rejected and rejected is not None and len(rejected) > 0:
        rej_count = int(len(rejected))
        for r in rejected:
            pts = r.reshape(-1, 2).astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 1, cv2.LINE_AA)

    id_txt = ",".join(map(str, id_list[:12]))
    if len(id_list) > 12:
        id_txt += "..."

    cv2.putText(
        vis,
        f"Frame {frame_idx}  Det:{det_count}  Rej:{rej_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale_text,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if id_txt:
        cv2.putText(
            vis,
            f"IDs: {id_txt}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale_text,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return vis


# -----------------------------------------------------------------------------
# Core detection
# -----------------------------------------------------------------------------
def tracks_to_dataframe(tracks: np.ndarray, confidences: np.ndarray) -> pd.DataFrame:
    """
    Convert tracks/confidences arrays to a DataFrame.
    """
    frame_idx, inst_idx = np.where(confidences > 0)
    if len(frame_idx) == 0:
        return pd.DataFrame(columns=["Frame", "Instance", "X", "Y", "Confidence"])

    coords = tracks[frame_idx, inst_idx, :]
    confs = confidences[frame_idx, inst_idx]
    return pd.DataFrame(
        {
            "Frame": frame_idx.astype(np.int32),
            "Instance": inst_idx.astype(np.int32),
            "X": coords[:, 0].astype(np.float32),
            "Y": coords[:, 1].astype(np.float32),
            "Confidence": confs.astype(np.float32),
        }
    )


def detect_aruco_in_video(
    video_file: str | Path,
    detector_config: DetectorConfig,
    *,
    dictionary_size: int = 300,
    aruco_dict: Optional[aruco.Dictionary] = None,
    debug_vis: bool = False,
    debug_every: int = 1,
    debug_save_path: Optional[Path] = None,
    debug_max_frames: int = 0,
    debug_pause: bool = False,
    debug_show_rejected: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect ArUco markers in a video and return tracks/confidences arrays.

    Returns:
      tracks: (num_frames, dictionary_size, 2) float32
      confidences: (num_frames, dictionary_size) float32
    """
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = build_aruco_detector(detector_config, aruco_dict=aruco_dict)

    writer = None
    if debug_save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(debug_save_path),
            fourcc,
            fps if fps > 0 else 30.0,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Could not open debug video writer at: {debug_save_path}")

    debug_every = max(1, int(debug_every))
    if debug_vis:
        cv2.namedWindow("ArUco debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ArUco debug", width, height)

    detections_per_frame: List[List[tuple[int, float, float]]] = []
    frame_idx = 0
    total_for_pbar = reported_frames if reported_frames > 0 else None
    pbar = tqdm(total=total_for_pbar, desc="Processing frames")

    while True:
        if debug_max_frames > 0 and frame_idx >= debug_max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        frame_dets: List[tuple[int, float, float]] = []
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            for i, marker_id in enumerate(ids_flat):
                mid = int(marker_id)
                if 0 <= mid < dictionary_size:
                    center = np.mean(corners[i][0], axis=0)
                    frame_dets.append((mid, float(center[0]), float(center[1])))
        detections_per_frame.append(frame_dets)

        do_debug = (debug_vis or writer is not None) and (frame_idx % debug_every == 0)
        if do_debug:
            vis = _annotate_debug_frame(
                frame,
                corners,
                ids,
                rejected,
                frame_idx,
                show_rejected=debug_show_rejected,
            )
            if writer is not None:
                writer.write(vis)
            if debug_vis:
                cv2.imshow("ArUco debug", vis)
                key = (cv2.waitKey(0) if debug_pause else cv2.waitKey(1)) & 0xFF
                if key == ord("q"):
                    break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    if debug_vis:
        cv2.destroyAllWindows()

    num_frames = len(detections_per_frame)
    tracks = np.zeros((num_frames, dictionary_size, 2), dtype=np.float32)
    confidences = np.zeros((num_frames, dictionary_size), dtype=np.float32)

    for f, dets in enumerate(detections_per_frame):
        for mid, x, y in dets:
            tracks[f, mid, 0] = x
            tracks[f, mid, 1] = y
            confidences[f, mid] = 1.0

    return tracks, confidences


# -----------------------------------------------------------------------------
# Parameter optimization
# -----------------------------------------------------------------------------
def _parse_num_list(text: str, cast) -> List:
    vals = []
    for t in text.split(","):
        tt = t.strip()
        if not tt:
            continue
        vals.append(cast(tt))
    if not vals:
        raise ValueError(f"Could not parse values from: '{text}'")
    return vals


def evaluate_detector_on_video(
    video_file: str | Path,
    detector_config: DetectorConfig,
    *,
    dictionary_size: int,
    aruco_dict: Optional[aruco.Dictionary] = None,
    frame_stride: int,
    max_frames: int,
) -> Dict[str, object]:
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    detector = build_aruco_detector(detector_config, aruco_dict=aruco_dict)
    frame_idx = 0

    eval_frames = 0
    zero = 0
    one = 0
    multi = 0
    total_detected_markers = 0
    one_ids: List[int] = []

    while True:
        if max_frames > 0 and eval_frames >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_stride > 1 and (frame_idx % frame_stride) != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        n = 0
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            valid = [int(mid) for mid in ids_flat if 0 <= int(mid) < dictionary_size]
            n = len(valid)
            if n == 1:
                one_ids.append(valid[0])

        if n == 0:
            zero += 1
        elif n == 1:
            one += 1
        else:
            multi += 1
        total_detected_markers += n

        eval_frames += 1
        frame_idx += 1

    cap.release()

    if eval_frames == 0:
        raise RuntimeError("No frames were evaluated during optimization.")

    zero_frac = zero / eval_frames
    one_frac = one / eval_frames
    multi_frac = multi / eval_frames
    mean_count = total_detected_markers / eval_frames

    mode_id = None
    mode_id_frac = 0.0
    if one_ids:
        s = pd.Series(one_ids)
        mode_id = int(s.mode().iloc[0])
        mode_id_frac = float((s == mode_id).mean())

    # Rank for "single marker per frame" videos.
    score = one_frac - 0.5 * multi_frac - 0.1 * abs(mean_count - 1.0) + 0.05 * mode_id_frac

    return {
        "score": float(score),
        "frames_eval": int(eval_frames),
        "exactly_one_frac": float(one_frac),
        "zero_frac": float(zero_frac),
        "multi_frac": float(multi_frac),
        "mean_detected_markers": float(mean_count),
        "mode_id": mode_id,
        "mode_id_frac": float(mode_id_frac),
        "corner_refinement": detector_config.corner_refinement,
        "adaptive_thresh_constant": float(detector_config.adaptive_thresh_constant),
        "adaptive_thresh_win_min": int(detector_config.adaptive_thresh_win_min),
        "adaptive_thresh_win_max": int(detector_config.adaptive_thresh_win_max),
        "adaptive_thresh_win_step": int(detector_config.adaptive_thresh_win_step),
        "error_correction_rate": float(detector_config.error_correction_rate),
        "min_marker_perimeter_rate": float(detector_config.min_marker_perimeter_rate),
        "max_marker_perimeter_rate": float(detector_config.max_marker_perimeter_rate),
        "polygonal_approx_accuracy_rate": float(detector_config.polygonal_approx_accuracy_rate),
    }


def run_parameter_search(
    *,
    video_file: str | Path,
    out_dir: Path,
    name_no_ext: str,
    dictionary_size: int,
    aruco_dict: Optional[aruco.Dictionary] = None,
    base_config: DetectorConfig,
    frame_stride: int,
    max_frames: int,
    sweep_adaptive_const: str,
    sweep_win_min: str,
    sweep_win_max: str,
    sweep_win_step: str,
    sweep_err_rate: str,
    sweep_min_perim: str,
) -> tuple[pd.DataFrame, DetectorConfig]:
    const_vals = _parse_num_list(sweep_adaptive_const, float)
    min_vals = _parse_num_list(sweep_win_min, int)
    max_vals = _parse_num_list(sweep_win_max, int)
    step_vals = _parse_num_list(sweep_win_step, int)
    err_vals = _parse_num_list(sweep_err_rate, float)
    min_perim_vals = _parse_num_list(sweep_min_perim, float)

    configs: List[DetectorConfig] = []
    for c, wmin, wmax, wstep, err, min_perim in product(
        const_vals, min_vals, max_vals, step_vals, err_vals, min_perim_vals
    ):
        if wmax < wmin:
            continue
        configs.append(
            DetectorConfig(
                corner_refinement=base_config.corner_refinement,
                adaptive_thresh_constant=float(c),
                adaptive_thresh_win_min=int(wmin),
                adaptive_thresh_win_max=int(wmax),
                adaptive_thresh_win_step=int(wstep),
                error_correction_rate=float(err),
                min_marker_perimeter_rate=float(min_perim),
                max_marker_perimeter_rate=base_config.max_marker_perimeter_rate,
                polygonal_approx_accuracy_rate=base_config.polygonal_approx_accuracy_rate,
            )
        )

    if not configs:
        raise RuntimeError("Parameter sweep produced zero candidate configurations.")

    print(f"[INFO] Evaluating {len(configs)} parameter configurations...")
    rows: List[Dict[str, object]] = []
    for cfg in tqdm(configs, desc="Param search"):
        rows.append(
            evaluate_detector_on_video(
                video_file,
                cfg,
                dictionary_size=dictionary_size,
                aruco_dict=aruco_dict,
                frame_stride=frame_stride,
                max_frames=max_frames,
            )
        )

    df = pd.DataFrame(rows).sort_values(
        by=["score", "exactly_one_frac", "multi_frac"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    csv_path = out_dir / f"{name_no_ext}_aruco_param_search.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Wrote parameter search results: {csv_path}")

    best = df.iloc[0]
    best_config = DetectorConfig(
        corner_refinement=str(best["corner_refinement"]),
        adaptive_thresh_constant=float(best["adaptive_thresh_constant"]),
        adaptive_thresh_win_min=int(best["adaptive_thresh_win_min"]),
        adaptive_thresh_win_max=int(best["adaptive_thresh_win_max"]),
        adaptive_thresh_win_step=int(best["adaptive_thresh_win_step"]),
        error_correction_rate=float(best["error_correction_rate"]),
        min_marker_perimeter_rate=float(best["min_marker_perimeter_rate"]),
        max_marker_perimeter_rate=float(best["max_marker_perimeter_rate"]),
        polygonal_approx_accuracy_rate=float(best["polygonal_approx_accuracy_rate"]),
    )
    return df, best_config


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    try:
        p = argparse.ArgumentParser("aruco-track")
        p.add_argument("--video-file", required=True, help="Path to input video file")
        p.add_argument("--output-path", required=True, help="Directory for output files")
        p.add_argument(
            "--dictionary-size",
            type=int,
            default=300,
            help="Maximum marker ID to track (default: 300)",
        )
        p.add_argument(
            "--custom-dict",
            type=Path,
            default=None,
            help="Path to custom ArUco dictionary .npz with bytesList and max_correction_bits.",
        )
        p.add_argument(
            "--output-format",
            choices=["csv", "h5", "both"],
            default="h5",
            help="Output format for DataFrame export: csv, h5, or both (default: h5)",
        )

        # Detector params (single-run mode and base for optimization mode)
        p.add_argument(
            "--corner-refinement",
            choices=["none", "subpix", "contour", "apriltag"],
            default="contour",
            help="OpenCV corner refinement method.",
        )
        p.add_argument("--adaptive-thresh-constant", type=float, default=3.0)
        p.add_argument("--adaptive-thresh-win-min", type=int, default=10)
        p.add_argument("--adaptive-thresh-win-max", type=int, default=40)
        p.add_argument("--adaptive-thresh-win-step", type=int, default=10)
        p.add_argument("--error-correction-rate", type=float, default=1.0)
        p.add_argument("--min-marker-perimeter-rate", type=float, default=0.03)
        p.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
        p.add_argument("--polygonal-approx-accuracy-rate", type=float, default=0.03)

        # Debug options
        p.add_argument("--debug-vis", action="store_true", help="Show debug visualization window.")
        p.add_argument("--debug-every", type=int, default=1, help="Debug every N frames.")
        p.add_argument("--debug-pause", action="store_true", help="Pause each debug frame.")
        p.add_argument("--debug-max-frames", type=int, default=0, help="Stop after N frames (0 = no limit).")
        p.add_argument("--debug-save-video", action="store_true", help="Save annotated debug video.")
        p.add_argument(
            "--debug-show-rejected",
            action="store_true",
            help="Overlay rejected marker candidates in red (for tuning).",
        )

        # Optimization mode
        p.add_argument(
            "--optimize-params",
            action="store_true",
            help="Run parameter search for one-marker-per-frame debug videos.",
        )
        p.add_argument(
            "--optimize-apply-best",
            action="store_true",
            help="After optimization, run full detection using best found parameters.",
        )
        p.add_argument(
            "--optimize-frame-stride",
            type=int,
            default=5,
            help="Evaluate every Nth frame during parameter search.",
        )
        p.add_argument(
            "--optimize-max-frames",
            type=int,
            default=600,
            help="Max evaluated frames in parameter search (0 = all).",
        )
        p.add_argument(
            "--optimize-adaptive-thresh-constant",
            type=str,
            default="1,3,5,7",
            help="Comma-separated sweep values.",
        )
        p.add_argument(
            "--optimize-adaptive-thresh-win-min",
            type=str,
            default="5,10",
            help="Comma-separated sweep values.",
        )
        p.add_argument(
            "--optimize-adaptive-thresh-win-max",
            type=str,
            default="31,41",
            help="Comma-separated sweep values.",
        )
        p.add_argument(
            "--optimize-adaptive-thresh-win-step",
            type=str,
            default="4,8",
            help="Comma-separated sweep values.",
        )
        p.add_argument(
            "--optimize-error-correction-rate",
            type=str,
            default="0.6,1.0",
            help="Comma-separated sweep values.",
        )
        p.add_argument(
            "--optimize-min-marker-perimeter-rate",
            type=str,
            default="0.01,0.02,0.03",
            help="Comma-separated sweep values.",
        )

        args = p.parse_args()

        if not os.path.isfile(args.video_file):
            raise FileNotFoundError(f"Video file not found: {args.video_file}")
        os.makedirs(args.output_path, exist_ok=True)

        basename = os.path.basename(args.video_file)
        name_no_ext = os.path.splitext(basename)[0]
        out_dir = Path(args.output_path)

        base_config = DetectorConfig(
            corner_refinement=args.corner_refinement,
            adaptive_thresh_constant=args.adaptive_thresh_constant,
            adaptive_thresh_win_min=args.adaptive_thresh_win_min,
            adaptive_thresh_win_max=args.adaptive_thresh_win_max,
            adaptive_thresh_win_step=args.adaptive_thresh_win_step,
            error_correction_rate=args.error_correction_rate,
            min_marker_perimeter_rate=args.min_marker_perimeter_rate,
            max_marker_perimeter_rate=args.max_marker_perimeter_rate,
            polygonal_approx_accuracy_rate=args.polygonal_approx_accuracy_rate,
        )
        aruco_dict = load_custom_aruco_dict(args.custom_dict) if args.custom_dict is not None else None

        detector_config = base_config
        if args.optimize_params:
            print(f"[INFO] Running parameter search on: {basename}", flush=True)
            search_df, best_config = run_parameter_search(
                video_file=args.video_file,
                out_dir=out_dir,
                name_no_ext=name_no_ext,
                dictionary_size=args.dictionary_size,
                aruco_dict=aruco_dict,
                base_config=base_config,
                frame_stride=max(1, int(args.optimize_frame_stride)),
                max_frames=max(0, int(args.optimize_max_frames)),
                sweep_adaptive_const=args.optimize_adaptive_thresh_constant,
                sweep_win_min=args.optimize_adaptive_thresh_win_min,
                sweep_win_max=args.optimize_adaptive_thresh_win_max,
                sweep_win_step=args.optimize_adaptive_thresh_win_step,
                sweep_err_rate=args.optimize_error_correction_rate,
                sweep_min_perim=args.optimize_min_marker_perimeter_rate,
            )

            print("\n[INFO] Top 10 parameter sets:")
            show_cols = [
                "score",
                "exactly_one_frac",
                "zero_frac",
                "multi_frac",
                "mean_detected_markers",
                "mode_id",
                "mode_id_frac",
                "adaptive_thresh_constant",
                "adaptive_thresh_win_min",
                "adaptive_thresh_win_max",
                "adaptive_thresh_win_step",
                "error_correction_rate",
                "min_marker_perimeter_rate",
            ]
            print(search_df[show_cols].head(10).to_string(index=False))

            print("\n[INFO] Best config selected:")
            print(best_config)

            if not args.optimize_apply_best:
                print("[INFO] Parameter search complete. Use --optimize-apply-best to run detection with best config.")
                return
            detector_config = best_config

        debug_video_path: Optional[Path] = None
        if args.debug_save_video:
            debug_video_path = out_dir / f"{name_no_ext}_aruco_debug.mp4"

        print(f"[INFO] Processing {basename}...", flush=True)
        print(f"[INFO] Detector config: {detector_config}", flush=True)

        tracks, confidences = detect_aruco_in_video(
            args.video_file,
            detector_config=detector_config,
            dictionary_size=args.dictionary_size,
            aruco_dict=aruco_dict,
            debug_vis=args.debug_vis,
            debug_every=args.debug_every,
            debug_save_path=debug_video_path,
            debug_max_frames=args.debug_max_frames,
            debug_pause=args.debug_pause,
            debug_show_rejected=args.debug_show_rejected,
        )

        raw_h5_path = out_dir / f"{name_no_ext}_aruco_tracks.h5"
        with h5py.File(raw_h5_path, "w") as hdf:
            hdf.create_dataset("aruco_tracks", data=tracks)
            hdf.create_dataset("aruco_confidences", data=confidences)
        print(f"[INFO] Saved raw arrays to: {raw_h5_path}", flush=True)

        print("[INFO] Converting to DataFrame...", flush=True)
        df = tracks_to_dataframe(tracks, confidences)
        print(f"[INFO] Created DataFrame with {len(df)} detections", flush=True)

        if args.output_format in ("csv", "both"):
            csv_path = out_dir / f"{name_no_ext}_aruco_detections.csv"
            df.to_csv(csv_path, index=False, float_format="%.1f")
            print(f"[INFO] Saved CSV to: {csv_path}", flush=True)

        if args.output_format in ("h5", "both"):
            df_h5_path = out_dir / f"{name_no_ext}_aruco_detections.h5"
            try:
                import tables  # noqa: F401

                df.to_hdf(df_h5_path, key="detections", mode="w", format="table")
                print(f"[INFO] Saved DataFrame H5 to: {df_h5_path}", flush=True)
            except ImportError:
                print("[WARN] 'tables' module not found. Skipping HDF5 DataFrame export.", flush=True)

        if debug_video_path is not None:
            print(f"[INFO] Saved debug video to: {debug_video_path}", flush=True)

    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
