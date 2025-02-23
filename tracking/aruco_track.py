#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:59:17 2024

@author: sam
"""

import numpy as np
import glob
import cv2
import re
from tqdm import tqdm
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import h5py
from tqdm import tqdm
import pickle
import os
import uuid

def interpolate_missing_positions(merged_all_pos, max_gap):
    """
    Linearly interpolates missing positions (-1) in the merged_all_pos array over gaps less than max_gap frames,
    but does not interpolate beyond the first or last valid positions.

    Parameters:
    - merged_all_pos (numpy.ndarray): Array of shape (tracks, time, 2) with missing positions as -1.
    - max_gap (int): Maximum number of consecutive frames to interpolate.

    Returns:
    - numpy.ndarray: Interpolated merged_all_pos array.
    """
    interpolated_pos = merged_all_pos.copy()
    num_tracks, num_frames, _ = merged_all_pos.shape

    for t_idx in range(num_tracks):
        for dim in range(2):  # Interpolate separately for X and Y
            track_values = merged_all_pos[t_idx, :, dim]
            valid_indices = np.where(track_values != -1)[0]

            if len(valid_indices) > 1:  # At least two points to interpolate
                first_valid = valid_indices[0]
                last_valid = valid_indices[-1]

                for start_idx, end_idx in zip(valid_indices[:-1], valid_indices[1:]):
                    if (end_idx - start_idx - 1) <= max_gap:
                        # Perform interpolation within this gap
                        gap_range = np.arange(start_idx + 1, end_idx)
                        interpolated_values = np.interp(
                            gap_range,
                            [start_idx, end_idx],
                            [track_values[start_idx], track_values[end_idx]]
                        )
                        interpolated_pos[t_idx, gap_range, dim] = interpolated_values

    return interpolated_pos



def read_pickle_file_with_load(pickle_path, pickle_name):
    """
    Read a DataFrame from a pickle file using `pickle.load`, ensuring the file exists.

    Parameters:
    - pickle_path (str): Path to the folder where the pickle file is located.
    - pickle_name (str): Name of the pickle file (including '.pkl').

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    # Full path to the pickle file
    pickle_full_path = os.path.join(pickle_path, pickle_name)
    
    try:
        print(f"Reading DataFrame from {pickle_full_path} using `pickle.load`...")
        with open(pickle_full_path, 'rb') as f:
            data = pickle.load(f)
        print(f"File '{pickle_name}' read successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File '{pickle_full_path}' not found.")
    except Exception as e:
        print(f"Error while reading file: {e}")      


def convert_to_pickle_with_dump(folder_path, pickle_name, df_to_convert): 
    """
    Save a DataFrame to a pickle file using `pickle.dump`, ensuring the folder exists.

    Parameters:
    - folder_path (str): Path to the folder where the pickle file will be saved.
    - pickle_name (str): Name of the pickle file (including '.pkl').
    - df_to_convert (pd.DataFrame): DataFrame to save.

    Returns:
    None
    """
    # Ensure the folder exists, create it if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
    
    # Full path to the pickle file
    pickle_full_path = os.path.join(folder_path, pickle_name)
    
    try:
        # Save the DataFrame to a pickle file using `pickle.dump`
        print(f"Saving DataFrame to {pickle_full_path} using `pickle.dump`...")
        with open(pickle_full_path, 'wb') as f:
            pickle.dump(df_to_convert, f)
        print(f"File '{pickle_name}' saved successfully.")
    except Exception as e:
        print(f"Error while saving file: {e}")



def get_complete_tracks2(
    output_path,
    aruco_detection,
    sleap_detection,
    video_file,
    visualize=True,
    harvest_crops=False,
    crops_output_dir=None,
    crop_size=128,
    max_distance=50,
    startFrame=0,
    harvest_interval=5
):
    """
    Track objects across frames and optionally harvest crops of each track.
    
    :param aruco_detection: pandas DataFrame with columns:
                            ['Frame', 'ARUCO_number', 'X', 'Y', ...]
    :param sleap_detection: pandas DataFrame with columns:
                            ['Frame', 'X', 'Y', ...] (no ID)
    :param video_file:      path to video
    :param visualize:       whether to display a live tracking preview
    :param harvest_crops:   if True, save a crop of each track at each frame.
    :param crops_output_dir: directory to save the track crops (used if harvest_crops=True)
    :param crop_size:       width/height of each cropped image around the track position
    :param max_distance:    threshold distance for "nearest-neighbor" matches
    :param startFrame:      which frame in the video to start reading from
    :return: dict of {frame_idx: { track_id: (x, y), ... }}
    """

    # --------------------------
    # 1) Group detections by frame
    # --------------------------
    grouped_aruco = dict(iter(aruco_detection.groupby('Frame')))
    grouped_sleap = dict(iter(sleap_detection.groupby('Frame')))

    # Decide how many frames to iterate
    cap = None
    if visualize or harvest_crops:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {video_file}")
            return {}
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    else:
        # No video reading if not visualizing or cropping
        max_aruco_frame = aruco_detection['Frame'].max() if not aruco_detection.empty else 0
        max_sleap_frame = sleap_detection['Frame'].max() if not sleap_detection.empty else 0
        num_frames = int(max(max_aruco_frame, max_sleap_frame)) + 1

    # --------------------------
    # 2) Tracking data structures
    # --------------------------
    # all_pos[f] = { track_id: (x, y), ... }
    all_pos = [{} for _ in range(num_frames)]
    # tracked_positions[track_id] = [(frame_idx, (x, y)), ... ] (optional history)
    tracked_positions = {}

    # If we are harvesting crops, ensure the output directory exists
    if harvest_crops and crops_output_dir:
        os.makedirs(crops_output_dir, exist_ok=True)

    # --------------------------
    # 3) Main tracking loop
    # --------------------------
    for frame_idx in tqdm(range(num_frames)):
        # Read frame if we are visualizing or harvesting
        img = None
        if cap is not None:
            success, frame = cap.read()
            if not success:
                print(f"Warning: Unable to read frame {frame_idx}. Stopping.")
                break
            img = frame.copy()

        # --- 3A) Gather ArUco detections for this frame ---
        if frame_idx in grouped_aruco:
            frame_data = grouped_aruco[frame_idx]
        else:
            frame_data = pd.DataFrame(columns=['ARUCO_number','X','Y'])  # empty

        # Build array of (aruco_id, x, y)
        aruco_detections = []
        for aruco_id, row in frame_data.groupby('ARUCO_number'):
            for _, r in row.iterrows():
                aruco_detections.append((aruco_id, r['X'], r['Y']))
        aruco_detections = np.array(aruco_detections)  # shape (N, 3)

        # --- 3B) Gather SLEAP detections for this frame ---
        if frame_idx in grouped_sleap:
            sleap_xy = grouped_sleap[frame_idx][['X', 'Y']].to_numpy()  # shape (M, 2)
        else:
            sleap_xy = np.empty((0,2))

        # We'll track which detections are used
        used_aruco = set()
        used_sleap = set()

        # Tracks from the previous frame:
        if frame_idx > 0:
            prev_tracks = list(all_pos[frame_idx - 1].keys())
        else:
            prev_tracks = []

        # ===============
        # 3C) Update existing tracks from previous frame
        # ===============
        for tid in prev_tracks:
            prev_pos = all_pos[frame_idx - 1][tid]  # (x, y) from the last frame
            assigned_pos = None

            # --- Try ArUco first ---
            if aruco_detections.shape[0] > 0:
                det_xy = aruco_detections[:, 1:3]  # shape (N, 2)
                dists = np.linalg.norm(det_xy - prev_pos, axis=1)
                within_idxs = np.where(dists <= max_distance)[0]
                within_idxs = [i for i in within_idxs if i not in used_aruco]

                if len(within_idxs) == 1:
                    idx_det = within_idxs[0]
                    assigned_pos = (det_xy[idx_det, 0], det_xy[idx_det, 1])
                    used_aruco.add(idx_det)
                # If multiple or none, remain None => fallback to SLEAP

            # --- If still not assigned, try SLEAP ---
            if assigned_pos is None and sleap_xy.shape[0] > 0:
                dists_sleap = np.linalg.norm(sleap_xy - prev_pos, axis=1)
                within_idxs_sleap = np.where(dists_sleap <= max_distance)[0]
                within_idxs_sleap = [i for i in within_idxs_sleap if i not in used_sleap]

                if len(within_idxs_sleap) == 1:
                    idx_sl = within_idxs_sleap[0]
                    assigned_pos = (sleap_xy[idx_sl, 0], sleap_xy[idx_sl, 1])
                    used_sleap.add(idx_sl)

            # Store it if found
            if assigned_pos is not None:
                all_pos[frame_idx][tid] = assigned_pos
                if tid not in tracked_positions:
                    tracked_positions[tid] = []
                tracked_positions[tid].append((frame_idx, assigned_pos))

        # ===============
        # 3D) Create new tracks from leftover ArUco
        # ===============
        if aruco_detections.shape[0] > 0:
            for i_det, (aruco_id, x, y) in enumerate(aruco_detections):
                if i_det in used_aruco:
                    continue
                # If not yet used, we treat this ArUco ID as a new track
                if aruco_id not in all_pos[frame_idx]:
                    all_pos[frame_idx][aruco_id] = (x, y)
                    used_aruco.add(i_det)
                    if aruco_id not in tracked_positions:
                        tracked_positions[aruco_id] = []
                    tracked_positions[aruco_id].append((frame_idx, (x, y)))

        # ===============
        # 3F) If harvest_crops==True, save a crop for each track
        # ===============
        if harvest_crops and img is not None and crops_output_dir is not None:
            if frame_idx % harvest_interval ==0:
                for tid, (x, y) in all_pos[frame_idx].items():
                    # Convert to int for indexing
                    x_center = int(x)
                    y_center = int(y)
                    half = crop_size // 2
    
                    x_min = max(0, x_center - half)
                    x_max = x_min + crop_size
                    y_min = max(0, y_center - half)
                    y_max = y_min + crop_size
    
                    # Clamp to image bounds:
                    x_max = min(x_max, img.shape[1])
                    y_max = min(y_max, img.shape[0])
    
                    crop = img[y_min:y_max, x_min:x_max]
                    # Optionally resize to ensure exact crop_size x crop_size
                    if (crop.shape[0] != crop_size) or (crop.shape[1] != crop_size):
                        crop = cv2.resize(crop, (crop_size, crop_size))
    
                    # Save to folder named after tid
                    tid_folder = os.path.join(crops_output_dir, str(tid))
                    os.makedirs(tid_folder, exist_ok=True)
    
                    # Create a unique filename
                    crop_filename = f"{frame_idx}_{uuid.uuid4().hex}.png"
                    crop_path = os.path.join(tid_folder, crop_filename)
                    cv2.imwrite(crop_path, crop)

        # ===============
        # 3G) Visualization (optional)
        # ===============
        if visualize and img is not None:
            disp_img = img.copy()
            
            # Draw SLEAP detections as blue
            for (sx, sy) in sleap_xy:
                cv2.circle(disp_img, (int(sx), int(sy)), 8, (255, 0, 0), -1)

            # Draw final track assignments in red
            for tid, (x, y) in all_pos[frame_idx].items():
                px, py = int(x), int(y)
                cv2.circle(disp_img, (px, py), 10, (0, 0, 255), -1)
                cv2.putText(disp_img, str(tid), (px+10, py+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

            cv2.putText(disp_img, f"Frame {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            show_img = cv2.resize(disp_img, (1080, 720))
            cv2.imshow("Tracking", show_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    if cap is not None:
        cap.release()
    if visualize:
        cv2.destroyAllWindows()

    return all_pos
