#!/usr/bin/env python3
"""
Simple ArUco marker detection from video files
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm


def _annotate_debug_frame(frame_bgr, corners, ids, frame_idx, scale_text=0.6):
    """
    Draw detected markers + centers + IDs on a copy of the input frame.
    """
    vis = frame_bgr.copy()

    if ids is not None and len(ids) > 0:
        # OpenCV expects ids as Nx1 or Nx? int array for drawing
        aruco.drawDetectedMarkers(vis, corners, None) 

        ids_flat = ids.flatten()
        for i, marker_id in enumerate(ids_flat):
            corner = corners[i][0]  # shape (4,2)
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

    # Frame index overlay (useful when stepping)
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


def detect_aruco_in_video(
    video_file,
    dictionary_size=1000,
    max_gap=100,
    min_fraction=0.125,
    debug_vis=False,
    debug_every=1,
    debug_save_path=None,
    debug_max_frames=0,
    debug_pause=False,
):
    """
    Detect ArUco markers in a video file and return tracking data.

    Args:
        video_file: Path to video file
        dictionary_size: Size of ArUco dictionary
        max_gap: Maximum gap between detections for a marker (currently unused)
        min_fraction: Minimum fraction of frames a marker must be detected in (currently unused)

        debug_vis: If True, show an OpenCV window with detections overlaid.
        debug_every: Visualize/save debug output every N frames (>=1).
        debug_save_path: If provided, write an annotated debug video to this path.
        debug_max_frames: If >0, stop after this many frames (handy for quick debugging).
        debug_pause: If True and debug_vis=True, pause each shown frame; press any key to advance,
                     or 'q' to quit early.

    Returns:
        DataFrame with columns: Frame, Instance, X, Y, Confidence
    """
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        # Some codecs don't report frame count reliably; fall back to streaming mode
        total_frames = None

    # Initialize ArUco detector
    # NOTE: dictionary_size is kept for output array sizing; the actual dict is fixed at DICT_4X4_1000.
    # If you later want a true "dictionary selection", add a CLI option for the dict constant.
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    detector_params = aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    detector_params.adaptiveThreshConstant = 3
    detector_params.adaptiveThreshWinSizeMin = 10
    detector_params.adaptiveThreshWinSizeMax = 40
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.errorCorrectionRate = 1
    detector = aruco.ArucoDetector(aruco_dict, detector_params)

    # Prepare debug video writer if requested
    writer = None
    if debug_save_path is not None:
        debug_save_path = str(debug_save_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(debug_save_path, fourcc, fps if fps > 0 else 30.0, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Could not open debug video writer at: {debug_save_path}")

    # Initialize tracking data
    # If total_frames is unknown, we will store detections in a list and build dataframe directly.
    use_fixed_arrays = total_frames is not None
    if use_fixed_arrays:
        tracks = np.zeros((total_frames, dictionary_size, 2), dtype=np.float32)
        confidences = np.zeros((total_frames, dictionary_size), dtype=np.float32)
        frame_iter = range(total_frames)
        pbar = tqdm(frame_iter, total=total_frames, desc="Processing frames")
    else:
        tracks = None
        confidences = None
        pbar = tqdm(desc="Processing frames")

    frame_idx = 0
    rows_streaming = []  # used only if total_frames is unknown

    # Validate debug_every
    debug_every = max(1, int(debug_every))
    
    if debug_vis:
        cv2.namedWindow("ArUco debug", cv2.WINDOW_NORMAL)
        # Optional: set an initial size so it doesn't jump around
        cv2.resizeWindow("ArUco debug", width, height)
        
    while True:
        if debug_max_frames > 0 and frame_idx >= debug_max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(gray)

        # Store detections
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            for i, marker_id in enumerate(ids_flat):
                if 0 <= int(marker_id) < dictionary_size:
                    corner = corners[i][0]
                    center = np.mean(corner, axis=0)

                    if use_fixed_arrays:
                        tracks[frame_idx, int(marker_id), :] = center
                        confidences[frame_idx, int(marker_id)] = 1.0
                    else:
                        rows_streaming.append(
                            {
                                "Frame": frame_idx,
                                "Instance": int(marker_id),
                                "X": float(center[0]),
                                "Y": float(center[1]),
                                "Confidence": 1.0,
                            }
                        )

        # Debug visualization / saving
        do_debug = (debug_vis or writer is not None) and (frame_idx % debug_every == 0)
        if do_debug:
            vis = _annotate_debug_frame(frame, corners, ids, frame_idx)

            if writer is not None:
                writer.write(vis)

            if debug_vis:
                cv2.imshow("ArUco debug", vis)
                if debug_pause:
                    key = cv2.waitKey(0) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

        frame_idx += 1
        if use_fixed_arrays:
            pbar.update(1)
        else:
            pbar.update(1)

    pbar.close()
    cap.release()

    if writer is not None:
        writer.release()
    if debug_vis:
        cv2.destroyAllWindows()

    # Convert to DataFrame
    if not use_fixed_arrays:
        return pd.DataFrame(rows_streaming)

    rows = []
    for f in range(frame_idx):  # frame_idx may be < total_frames if early break
        for marker_id in range(dictionary_size):
            if confidences[f, marker_id] > 0:
                x, y = tracks[f, marker_id, :]
                rows.append(
                    {
                        "Frame": f,
                        "Instance": marker_id,
                        "X": float(x),
                        "Y": float(y),
                        "Confidence": float(confidences[f, marker_id]),
                    }
                )

    return pd.DataFrame(rows)


def main():
    try:
        p = argparse.ArgumentParser('aruco-track')
        p.add_argument('--video-file', required=True, help='Path to input video file')
        p.add_argument('--output-path', required=True, help='Directory for output files')
        p.add_argument('--dictionary-size', type=int, default=300, 
                       help='Maximum marker ID to track (default: 300)')
        p.add_argument('--output-format', choices=['csv', 'h5', 'both'], default='both',
                       help='Output format for DataFrame: csv, h5, or both (default: both)')
        
        args = p.parse_args()
        
        # Validate inputs
        if not os.path.isfile(args.video_file):
            raise FileNotFoundError(f"Video file not found: {args.video_file}")
        os.makedirs(args.output_path, exist_ok=True)

        basename = os.path.basename(args.video_file)
        name_no_ext = os.path.splitext(basename)[0]
        
        print(f"[INFO] Processing {basename}...", flush=True)
        
        # Load ArUco detector
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        detect_params = aruco.DetectorParameters()
        detect_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        detect_params.adaptiveThreshConstant = 3
        detect_params.adaptiveThreshWinSizeMin = 10
        detect_params.adaptiveThreshWinSizeMax = 40
        detect_params.adaptiveThreshWinSizeStep = 10
        detect_params.errorCorrectionRate = 1
        detector = aruco.ArucoDetector(aruco_dict, detect_params)

        # Run detection
        tracks, confidences = get_aruco_tracks(args.video_file, detector, args.dictionary_size)
        
        # Save raw arrays in HDF5 format (legacy format)
        hdf5_path = os.path.join(args.output_path, name_no_ext + '_aruco_tracks_.h5')
        try:
            with h5py.File(hdf5_path, 'w') as hdf:
                hdf.create_dataset('aruco_tracks', data=tracks)
                hdf.create_dataset('aruco_confidences', data=confidences)
            print(f"[INFO] Saved raw arrays to: {hdf5_path}", flush=True)
        except Exception as e:
            print(f"[ERR] Failed to save raw HDF5: {e}", flush=True)
            raise

        # Convert to DataFrame
        print("[INFO] Converting to DataFrame...", flush=True)
        df = tracks_to_dataframe(tracks, confidences)
        print(f"[INFO] Created DataFrame with {len(df)} detections", flush=True)
        
        # Save DataFrame in requested format(s)
        if args.output_format in ('csv', 'both'):
            csv_path = os.path.join(args.output_path, f"{name_no_ext}_aruco_detections.csv")
            df.to_csv(csv_path, index=False, float_format="%.1f")
            print(f"[INFO] Saved CSV to: {csv_path}", flush=True)
        
        if args.output_format in ('h5', 'both'):
            try:
                import tables  # Check availability
                df_h5_path = os.path.join(args.output_path, f"{name_no_ext}_aruco_detections.h5")
                df.to_hdf(df_h5_path, key='detections', mode='w', format='table')
                print(f"[INFO] Saved DataFrame H5 to: {df_h5_path}", flush=True)
            except ImportError:
                print("[WARN] 'tables' module not found. Skipping HDF5 DataFrame export.", flush=True)
            except Exception as e:
                print(f"[ERR] Failed to save DataFrame H5: {e}", flush=True)
                raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    import sys
    main()  


