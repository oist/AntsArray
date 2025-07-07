#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import h5py
import sys

# Add the parent directory to the path to import from tracking module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.tracking_utils import get_complete_tracks



def main():
    parser = argparse.ArgumentParser(
        description="Combine SLEAP and ArUco detections into complete tracks."
    )
    parser.add_argument(
        "--video_name",
        type=str,
        required=True,
        help="Path to the video file (e.g., .avi)."
    )
    parser.add_argument(
        "--output_path",
        type=str,

        required=True,
        help="Path to the output folder."
    )
    args = parser.parse_args()

    # --- Infer SLEAP and ArUco file paths ---
    base_name = os.path.splitext(os.path.basename(args.video_name))[0]
    data_folder = os.path.dirname(args.video_name)
    sleap_file_name = os.path.join(data_folder, base_name + "_sleap_data.h5")
    aruco_name = os.path.join(data_folder, base_name + "_aruco.csv")

    # --- Load SLEAP detections ---
    with h5py.File(sleap_file_name, 'r') as sleap_File:
        sleap_detection = pd.DataFrame({
            'X':     np.squeeze(sleap_File['X'][:]),
            'Y':     np.squeeze(sleap_File['Y'][:]),
            'Frame': np.squeeze(sleap_File['Frame'][:]),
            'Instance': np.squeeze(sleap_File['Instance'][:]),
            'Bodypoint': np.squeeze(sleap_File['Bodypoint'][:]),
            'Score_node': np.squeeze(sleap_File['Score_node'][:])
                    })
        sleap_detection = sleap_detection.dropna()
     #   sleap_detection = sleap_detection[sleap_detection['Bodypoint'] == 0] #just the aruco tag for now

    # --- Load ArUco detections ---
    aruco_detection = pd.read_csv(aruco_name)

    tracks = get_complete_tracks(
        output_path=args.output_path,   # pickled dataframe
        aruco_detection=aruco_detection,
        sleap_detection=sleap_detection,
        video_file=args.video_name,
        #harvest_crops=True,
        #crops_output_dir=args.output_path+'aruco_crops'
        visualize=True,
        video_out_path=args.output_path+'/tracking_video.avi'
    )


if __name__ == "__main__":
    main()
