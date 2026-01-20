#!/usr/bin/env python3
"""
ArUco marker detection for video files.

Outputs:
- Raw numpy arrays (H5): aruco_tracks, aruco_confidences
- DataFrame (CSV and/or H5): Frame, Instance, X, Y, Confidence

Optional debug visualization:
- On-screen window (resizable) and/or saved annotated debug video.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import h5py
import sys


def _annotate_debug_frame(frame_bgr, corners, ids, frame_idx, scale_text=0.6):
    """
    Draw detected marker borders + center dots + IDs on a copy of the input frame.
    Avoids duplicate ID text by drawing borders only via OpenCV's helper.
    """
    vis = frame_bgr.copy()

    if ids is not None and len(ids) > 0:
        # Draw borders only (no built-in IDs) to avoid double text.
        aruco.drawDetectedMarkers(vis, corners, None)

        ids_flat = ids.flatten()
        for i, marker_id in enumerate(ids_flat):
            corner = corners[i][0]  # (4,2)
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

    cv2.putText(
        vis,
        f"Frame {frame_idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale_text,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def tracks_to_dataframe(tracks, confidences):
    """
    Convert tracks and confidences arrays to a DataFrame using vectorization.
    """
    frames_indices, instance_indices = np.where(confidences > 0)
    if len(frames_indices) == 0:
        return pd.DataFrame(columns=["Frame", "Instance", "X", "Y", "Confidence"])

    coords = tracks[frames_indices, instance_indices, :]
    confs = confidences[frames_indices, instance_indices]

    return pd.DataFrame(
        {
            "Frame": frames_indices.astype(np.int32),
            "Instance": instance_indices.astype(np.int32),
            "X": coords[:, 0].astype(np.float32),
            "Y": coords[:, 1].astype(np.float32),
            "Confidence": confs.astype(np.float32),
        }
    )


def detect_aruco_in_video(
    video_file,
    dictionary_size=300,
    debug_vis=False,
    debug_every=1,
    debug_save_path=None,
    debug_max_frames=0,
    debug_pause=False,
):
    """
    Detect ArUco markers in a video file and return tracks/confidences arrays.

    Returns:
        tracks: (num_frames, dictionary_size, 2) float32
        confidences: (num_frames, dictionary_size) float32
    """
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    # Video properties
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ArUco detector
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    detector_params = aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    detector_params.adaptiveThreshConstant = 3
    detector_params.adaptiveThreshWinSizeMin = 10
    detector_params.adaptiveThreshWinSizeMax = 40
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.errorCorrectionRate = 1
    detector = aruco.ArucoDetector(aruco_dict, detector_params)

    # Debug video writer
    writer = None
    if debug_save_path is not None:
        debug_save_path = str(debug_save_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            debug_save_path, fourcc, fps if fps > 0 else 30.0, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Could not open debug video writer at: {debug_save_path}")

    # Debug window
    debug_every = max(1, int(debug_every))
    if debug_vis:
        cv2.namedWindow("ArUco debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ArUco debug", width, height)

    # We cannot trust CAP_PROP_FRAME_COUNT for all codecs.
    # So accumulate detections per frame, then build arrays once we know actual frame count.
    detections_per_frame = []  # list of list of (id, x, y)
    frame_idx = 0

    # Progress bar: if reported_frames looks valid, show it; otherwise unknown.
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

        frame_dets = []
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            for i, marker_id in enumerate(ids_flat):
                mid = int(marker_id)
                if 0 <= mid < dictionary_size:
                    corner = corners[i][0]
                    center = np.mean(corner, axis=0)
                    frame_dets.append((mid, float(center[0]), float(center[1])))

        detections_per_frame.append(frame_dets)

        # Debug visualization / saving
        do_debug = (debug_vis or writer is not None) and (frame_idx % debug_every == 0)
        if do_debug:
            vis = _annotate_debug_frame(frame, corners, ids, frame_idx)
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

    # Build arrays with actual number of processed frames
    num_frames = len(detections_per_frame)
    tracks = np.zeros((num_frames, dictionary_size, 2), dtype=np.float32)
    confidences = np.zeros((num_frames, dictionary_size), dtype=np.float32)

    for f, dets in enumerate(detections_per_frame):
        for mid, x, y in dets:
            tracks[f, mid, 0] = x
            tracks[f, mid, 1] = y
            confidences[f, mid] = 1.0

    return tracks, confidences


def main():
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
            "--output-format",
            choices=["csv", "h5", "both"],
            default="both",
            help="Output format for DataFrame: csv, h5, or both (default: both)",
        )

        # Debug options
        p.add_argument("--debug-vis", action="store_true", help="Show debug visualization window.")
        p.add_argument("--debug-every", type=int, default=1, help="Debug every N frames (default 1).")
        p.add_argument("--debug-pause", action="store_true", help="Pause each debug frame; any key advances; q quits.")
        p.add_argument("--debug-max-frames", type=int, default=0, help="Stop after N frames (0 = no limit).")
        p.add_argument("--debug-save-video", action="store_true", help="Save annotated debug video alongside outputs.")

        args = p.parse_args()

        if not os.path.isfile(args.video_file):
            raise FileNotFoundError(f"Video file not found: {args.video_file}")
        os.makedirs(args.output_path, exist_ok=True)

        basename = os.path.basename(args.video_file)
        name_no_ext = os.path.splitext(basename)[0]
        out_dir = Path(args.output_path)

        debug_video_path = None
        if args.debug_save_video:
            debug_video_path = out_dir / f"{name_no_ext}_aruco_debug.mp4"

        print(f"[INFO] Processing {basename}...", flush=True)

        # Run detection (returns raw arrays)
        tracks, confidences = detect_aruco_in_video(
            args.video_file,
            dictionary_size=args.dictionary_size,
            debug_vis=args.debug_vis,
            debug_every=args.debug_every,
            debug_save_path=debug_video_path,
            debug_max_frames=args.debug_max_frames,
            debug_pause=args.debug_pause,
        )

        # Save raw arrays (HDF5)
        raw_h5_path = out_dir / f"{name_no_ext}_aruco_tracks.h5"
        with h5py.File(raw_h5_path, "w") as hdf:
            hdf.create_dataset("aruco_tracks", data=tracks)
            hdf.create_dataset("aruco_confidences", data=confidences)
        print(f"[INFO] Saved raw arrays to: {raw_h5_path}", flush=True)

        # Convert to DataFrame
        print("[INFO] Converting to DataFrame...", flush=True)
        df = tracks_to_dataframe(tracks, confidences)
        print(f"[INFO] Created DataFrame with {len(df)} detections", flush=True)

        # Save DataFrame
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
