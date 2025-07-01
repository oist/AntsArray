#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import h5py



def main():
    parser = argparse.ArgumentParser(
        description="Combine SLEAP and ArUco detections into complete tracks."
    )
    parser.add_argument(
        "--sleap_file_name",
        type=str,
        required=True,
        help="Path to SLEAP file (e.g., .h5)."
    )
    parser.add_argument(
        "--video_name",
        type=str,
        required=True,
        help="Path to the video file (e.g., .avi)."
    )
    parser.add_argument(
        "--aruco_name",
        type=str,
        required=True,
        help="Path to the ArUco .npy detection file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output folder."
    )
    args = parser.parse_args()

    # --- Load SLEAP detections ---
    with h5py.File(args.sleap_file_name, 'r') as sleap_File:
        sleap_detection = pd.DataFrame({
            'X':     np.squeeze(sleap_File['X'][:]),
            'Y':     np.squeeze(sleap_File['Y'][:]),
            'Frame': np.squeeze(sleap_File['Frame'][:])
        })

    # --- Load ArUco detections ---
    aruco_opencv = np.load(args.aruco_name)
    nFrames, nIDs, _ = aruco_opencv.shape

    # Create the ArUco DataFrame
    df = pd.DataFrame({
        'X':            aruco_opencv[:, :, 0].flatten(),
        'Y':            aruco_opencv[:, :, 1].flatten(),
        'Frame':        np.repeat(np.arange(nFrames), nIDs),
        'ARUCO_number': np.tile(np.arange(nIDs), nFrames),
    })

    # Filter out zero-coordinates (assumed invalid)
    aruco_detection = df[(df['X'] != 0) | (df['Y'] != 0)]

    # --- Combine the data using your custom function ---
    # Ensure get_complete_tracks2 is correctly imported or defined.
    all_pos = get_complete_tracks2(
        args.output_path,
        aruco_detection,
        sleap_detection,
        args.video_name,
        False,           # set to True/False as needed
        harvest_crops=True,
        harvest_interval=25,
        crops_output_dir=args.output_path 
    )


if __name__ == "__main__":
    main()
