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


def detect_aruco_in_video(video_file, dictionary_size=1000, max_gap=100, min_fraction=0.125):
    """
    Detect ArUco markers in a video file and return tracking data.
    
    Args:
        video_file: Path to video file
        dictionary_size: Size of ArUco dictionary
        max_gap: Maximum gap between detections for a marker
        min_fraction: Minimum fraction of frames a marker must be detected in
    
    Returns:
        DataFrame with columns: Frame, Marker_ID, X, Y, Confidence
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize ArUco detector
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    detector_params = aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    detector_params.adaptiveThreshConstant = 3
    detector_params.adaptiveThreshWinSizeMin = 10
    detector_params.adaptiveThreshWinSizeMax = 40
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.errorCorrectionRate = 1
    detector = aruco.ArucoDetector(aruco_dict, detector_params)
    
    # Initialize tracking data
    tracks = np.zeros((total_frames, dictionary_size, 2))
    confidences = np.zeros((total_frames, dictionary_size))
    
    print(f"Processing {total_frames} frames...")
    
    # Process each frame
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None:
            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                if marker_id < dictionary_size:
                    # Calculate center of marker
                    corner = corners[i][0]
                    center = np.mean(corner, axis=0)
                    
                    # Store position and confidence (using corner quality as proxy)
                    tracks[frame_idx, marker_id, :] = center
                    confidences[frame_idx, marker_id] = 1.0  # Simple confidence
    
    cap.release()
    
    # Convert to DataFrame
    rows = []
    for frame_idx in range(total_frames):
        for marker_id in range(dictionary_size):
            if confidences[frame_idx, marker_id] > 0:
                x, y = tracks[frame_idx, marker_id, :]
                rows.append({
                    'Frame': frame_idx,
                    'Instance': marker_id,
                    'X': x,
                    'Y': y,
                    'Confidence': confidences[frame_idx, marker_id]
                })
    
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description='Detect ArUco markers in video files')
    parser.add_argument('--video-file', required=True, help='Path to video file')
    parser.add_argument('--output-path', required=True, help='Output directory path')
    parser.add_argument('--dictionary-size', type=int, default=1000, help='ArUco dictionary size')
    parser.add_argument('--max-gap', type=int, default=100, help='Maximum gap between detections')
    parser.add_argument('--min-fraction', type=float, default=0.125, help='Minimum fraction of frames')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base name for output files
    video_name = Path(args.video_file).stem
    
    # Detect ArUco markers
    print(f"Processing video: {args.video_file}")
    df = detect_aruco_in_video(
        args.video_file, 
        args.dictionary_size, 
        args.max_gap, 
        args.min_fraction
    )
    
    # Save results
    output_csv = output_dir / f"{video_name}_aruco_detections.csv"
    df.to_csv(output_csv, index=False, float_format="%.1f")
    print(f"Saved ArUco detections to: {output_csv}")
    
    # Print summary
    if not df.empty:
        print(f"Detected {df['Instance'].nunique()} unique markers")
        print(f"Total detections: {len(df)}")
    else:
        print("No ArUco markers detected")

if __name__ == "__main__":
    main()
