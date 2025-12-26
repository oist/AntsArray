#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:17:05 2023
Modified on Mon Dec 22 16:50:00 2025

@author: sam
@modifier: Makoto

ArUco marker detection for video files.
Outputs both raw numpy arrays (H5) and a DataFrame (CSV/H5) for downstream analysis.
"""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from cv2 import aruco
import argparse
import os
import h5py


def get_aruco_tracks(video_file, detector, dictionary_size=300):
    """
    Detect ArUco markers in every frame of a video.
    
    Args:
        video_file: Path to video file
        detector: ArUco detector instance
        dictionary_size: Maximum marker ID to track (default 300)
    
    Returns:
        tracks: (num_frames, dictionary_size, 2) array of x,y positions
        confidences: (num_frames, dictionary_size) array of detection confidence (1.0 if detected)
    """
    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create arrays for tracks and confidences
    tracks = np.zeros((length, dictionary_size, 2))
    confidences = np.zeros((length, dictionary_size))

    for frame_count in tqdm(range(length), total=length):
        ret, img = cap.read()
        if not ret:
            break
        corners, ids, rejected = detector.detectMarkers(img)

        if ids is not None:
            # Process each detection - iterate through ids and corners together
            # This fixes the bug where np.unique() broke the id-to-com mapping
            for i, (corner, marker_id) in enumerate(zip(corners, ids)):
                curr_id = marker_id[0]
                # Only store if curr_id < dictionary_size
                if curr_id < dictionary_size:
                    # Calculate center of mass from corners
                    com = np.mean(corner[0], axis=0)
                    tracks[frame_count, curr_id, :] = com
                    confidences[frame_count, curr_id] = 1.0  # Binary confidence for now
    
    cap.release()
    return tracks, confidences


# ... (imports remain similar)

def tracks_to_dataframe(tracks, confidences):
    """
    Convert tracks and confidences arrays to a DataFrame using vectorization.
    
    Args:
        tracks: (num_frames, dictionary_size, 2) array of x,y positions
        confidences: (num_frames, dictionary_size) array of detection confidence
    
    Returns:
        DataFrame with columns: Frame, Instance, X, Y, Confidence
    """
    # Find indices where confidence > 0
    # frames_indices: (N,) array of frame numbers
    # instance_indices: (N,) array of instance IDs
    frames_indices, instance_indices = np.where(confidences > 0)
    
    if len(frames_indices) == 0:
        return pd.DataFrame(columns=['Frame', 'Instance', 'X', 'Y', 'Confidence'])
    
    # Extract coordinates using the indices
    # tracks[frames, instances, :] gives (N, 2) array of X,Y
    coords = tracks[frames_indices, instance_indices, :]
    
    # Extract confidences
    confs = confidences[frames_indices, instance_indices]
    
    # Create DataFrame dictionary
    data = {
        'Frame': frames_indices,
        'Instance': instance_indices,
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Confidence': confs
    }
    
    return pd.DataFrame(data)


def main():
    try:
        p = argparse.ArgumentParser('aruco-track')
        p.add_argument('--video-file', required=True, help='Path to input video file')
        p.add_argument('--output-path', required=True, help='Directory for output files')
        p.add_argument('--dictionary-size', type=int, default=300, 
                       help='Maximum marker ID to track (default: 300)')
        p.add_argument('--output-format', choices=['csv', 'h5', 'both', 'none'], default='none',
                       help='Output format for DataFrame: csv, h5, both, or none (default: none)')
        
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

            print(f"[INFO] Saved raw arrays to: {hdf5_path}", flush=True)
        except Exception as e:
            print(f"[ERR] Failed to save raw HDF5: {e}", flush=True)
            raise

        if args.output_format != 'none':
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


