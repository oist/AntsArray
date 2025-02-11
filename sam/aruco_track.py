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


def is_well_isolated(track_id, positions_dict, max_dist):
    """
    Returns True if EVERY position for track_id in positions_dict
    is > max_dist away from every other (valid) track's positions.
    
    positions_dict[track_id] should be a list of positions (x,y).
    We say 'well isolated' if each (x,y) for 'track_id' is far
    from all other IDs' positions.
    """
    my_positions = positions_dict.get(track_id, [])
    if len(my_positions) == 0:
        return False  # No valid positions
    # Flatten all other IDs' positions
    for other_id, other_positions in positions_dict.items():
        if other_id == track_id:
            continue
        for my_pos in my_positions:
            for other_pos in other_positions:
                if np.any(other_pos < 0):
                    continue
                dist = np.linalg.norm(my_pos - other_pos)
                if dist < max_dist:
                    return False
    return True


# video_file='/home/sam/bucket/Ants/trials/20241108_1/20241108_1_first_hour/cam0_2024-11-07-23-40-20_cam18_first_hour.avi'
# aruco_file='/home/sam/saionWork/sam/aruco_files/cam0_2024-11-07-23-40-20_cam18_first_hour_000.pkl'
# sleap_file='/home/sam/bucket/sam/ant_tracking/20241108_1/slp_files/cam0_2024-11-07-23-40-20_cam18_first_hour_000.csv'


# aruco_file='/home/sam/bucket/sam/ant_tracking/20241108_1/20241108_1_aruco_realigned_col1.pkl'
# sleap_file='/home/sam/bucket/sam/ant_tracking/20241108_1/20241108_1_sleap_realigned_col1.pkl'

video_file='/home/sam/bucket/Ants/basler/20250123_1/data/cam5_2025-01-23-11-15-11_cam06_000.avi'
aruco_file='/home/sam/bucket/Ants/basler/20250123_1/data/tracks_aruco/cam5_2025-01-23-11-15-11_cam06_000.pkl'
sleap_file='/home/sam/bucket/Ants/basler/20250123_1/data/cam5_2025-01-23-11-15-11_cam06_000.csv'

# aruco
with open(aruco_file, "rb") as f:
    aruco_detection = pickle.load(f)
    
# sleap
# with open(sleap_file, "rb") as f:
#     sleap_detection = pickle.load(f)

#sleap
sleap_detection = pd.read_csv(sleap_file)
sleap_detection['Frame']=sleap_detection['Frame']-1 #0 base frames
sleap_detection = sleap_detection.drop(['Score_node'], axis=1)
# Reset instance counter for each new frame
sleap_detection['Instance'] = sleap_detection.groupby('Frame').cumcount() - 1


#%% simple track

# Parameters
max_distance = 60
min_sep_distance = 10.0
min_speed = 50.0
brightness_factor = 1.2
visualize = True

startFrame = 0

#############################
# Initialization
#############################

grouped_aruco = dict(iter(aruco_detection.groupby('Frame')))
grouped_sleap = dict(iter(sleap_detection.groupby('Frame')))

length = len(grouped_aruco)

if visualize:
    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

# We’ll track positions as: tracked_positions[aruco_id] = [ (x,y), (x2,y2), ... ]
tracked_positions = {}  # last known positions from the previous frame
all_pos = {}            # store final results (by frame) -> { id: [ (x,y), ... ], ... }

# Pre-fill all_pos if desired
for frame_idx in range(length):
    # each frame -> dictionary of ID -> list of positions
    all_pos[frame_idx] = {}

#############################
# Main Loop
#############################

for tally in tqdm(range(length)):
    if visualize:
        success, img = cap.read()
        if not success:
            print(f"Failed to read frame {tally}.")
            break

    # -- 1) Get SLEAP centroids for this frame --
    if tally in grouped_sleap:
        curr_sleap = grouped_sleap[tally][['X', 'Y']].to_numpy()
    else:
        curr_sleap = np.empty((0, 2))

    if tally in grouped_aruco:
        frame_data = grouped_aruco[tally]  # DataFrame with all detections at this frame
    else:
        frame_data = pd.DataFrame(columns=['ARUCO_number', 'X', 'Y'])  # empty

    aruco_frame_data = {}
    for aruco_id, group in frame_data.groupby('ARUCO_number'):
        # Convert to a list/array of (x, y)
        aruco_positions = group[['X', 'Y']].to_numpy()
        aruco_frame_data[aruco_id] = aruco_positions

    # Initialize new_tracked_positions as a dict of ID->list-of-positions
    new_tracked_positions = {
        aruco_id: [] for aruco_id in aruco_frame_data.keys()
    }

    # ==========================================================
    # 2A) Assign ArUco detections to their IDs
    # ==========================================================
    for aruco_id, aruco_positions in aruco_frame_data.items():
        # old_positions for this aruco_id (if any)
        old_positions = tracked_positions.get(aruco_id, [])
        # Convert to np.array
        old_positions_array = np.array(old_positions)

        for aruco_pos in aruco_positions:
            # Distance to old positions of the same aruco_id
            if old_positions_array.size > 0:
                dists_to_old = np.linalg.norm(old_positions_array - aruco_pos, axis=1)
                if dists_to_old.min() <= max_distance:
                    new_tracked_positions[aruco_id].append(aruco_pos)
                    # Optionally continue if each new detection can only match once
                    # continue

            # Otherwise, store under this aruco_id
            else:
                new_tracked_positions[aruco_id].append(aruco_pos)

    # -------------------------------------------------------------------
    # 2B) Missing detection logic: match old IDs to SLEAP centroids if
    #     no current ArUco detection for them
    # -------------------------------------------------------------------
    # Gather assigned (aruco_id, x, y) from new_tracked_positions
    assigned_list = []
    for aruco_id, assigned_positions in new_tracked_positions.items():
        for pos in assigned_positions:
            assigned_list.append((aruco_id, pos[0], pos[1]))
    assigned_array = np.array(assigned_list, dtype=float)  # shape (M, 3)

    used_centroids = set()
    if len(assigned_array) > 0 and len(curr_sleap) > 0:
        # cdist between the new_arUco_positions (M,2) and curr_sleap (N,2)
        aruco_positions_2d = assigned_array[:, 1:3]  # just (x,y)
        dist_matrix = cdist(aruco_positions_2d, curr_sleap)  # shape (M,N)

        # For each assigned position, find the nearest centroid
        nearest_centroid_idx = dist_matrix.argmin(axis=1)  # shape (M,)
        nearest_centroid_dist = dist_matrix.min(axis=1)    # shape (M,)

        # Mark that centroid as used if distance <= max_distance
        for i, dist_val in enumerate(nearest_centroid_dist):
            if dist_val <= max_distance:
                used_centroids.add(nearest_centroid_idx[i])

    unused_centroids = set(range(len(curr_sleap))) - used_centroids

    # For each old ID not updated by ArUco
    for old_id in tracked_positions.keys():
        if (old_id not in new_tracked_positions) or (len(new_tracked_positions[old_id]) == 0):
            old_pos_list = tracked_positions[old_id]
            if len(old_pos_list) == 0:
                continue

            # Last known position
            old_pos = old_pos_list[-1]

            # Try matching to a single SLEAP centroid
            if len(curr_sleap) > 0:
                dists = np.linalg.norm(curr_sleap - old_pos, axis=1)
                valid_idx = [i for i in unused_centroids if dists[i] <= max_distance]

                if len(valid_idx) == 1:
                    matched_idx = valid_idx[0]
                    matched_pos = curr_sleap[matched_idx]
                    if old_id not in new_tracked_positions:
                        new_tracked_positions[old_id] = []
                    new_tracked_positions[old_id].append(matched_pos)
                    unused_centroids.remove(matched_idx)
                    continue
                # else => none or multiple => treat as missing

            # Check speed if no match found
            if len(old_pos_list) >= 2:
                prev_pos = old_pos_list[-2]
                speed = np.linalg.norm(old_pos - prev_pos)
                if speed < min_speed:
                    # Speed is below threshold => assume same location
                    if old_id not in new_tracked_positions:
                        new_tracked_positions[old_id] = []
                    new_tracked_positions[old_id].append(old_pos)
                    continue
            # If speed >= min_speed or not enough history => no update

    # ===============================================================
    # 3) Conflict Resolution: prevent double counting for same/diff IDs
    #    if positions are within min_sep_distance
    # ===============================================================
    flat_list = []
    for aruco_id, pos_list in new_tracked_positions.items():
        for pos in pos_list:
            flat_list.append((aruco_id, pos[0], pos[1]))

    keep = [True] * len(flat_list)
    for i in range(len(flat_list)):
        if not keep[i]:
            continue
        id_i, x_i, y_i = flat_list[i]
        pos_i = np.array([x_i, y_i], dtype=float)
        for j in range(i+1, len(flat_list)):
            if not keep[j]:
                continue
            id_j, x_j, y_j = flat_list[j]
            pos_j = np.array([x_j, y_j], dtype=float)
            dist_ij = np.linalg.norm(pos_i - pos_j)
            if dist_ij < min_sep_distance:
                # Remove the second detection (j)
                keep[j] = False

    filtered_new_positions = {}
    for (flag, (aruco_id, x, y)) in zip(keep, flat_list):
        if flag:
            if aruco_id not in filtered_new_positions:
                filtered_new_positions[aruco_id] = []
            filtered_new_positions[aruco_id].append(np.array([x, y], dtype=float))

    # ---------------------------------------------------------
    # 4) Update our global tracking: tracked_positions & all_pos
    # ---------------------------------------------------------
    tracked_positions = filtered_new_positions
    all_pos[tally] = filtered_new_positions

    # -----------------------------------
    # 5) Visualization (optional)
    # -----------------------------------
    if visualize:
        disp_img = np.clip(img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        for aruco_id, pos_list in filtered_new_positions.items():
            for pos in pos_list:
                px, py = int(pos[0]), int(pos[1])
                cv2.circle(disp_img, (px, py), 15, (0, 0, 255), -1)
                cv2.putText(disp_img, str(aruco_id), (px + 15, py + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)
        cv2.putText(disp_img, f"Frame {tally}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        show_img = cv2.resize(disp_img, (1080, 720))
        cv2.imshow("Centroid Tracking", show_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Safely release cap if visualize was True
if visualize:
    cap.release()
    cv2.destroyAllWindows()

# ------------------------------------
# 6) Save all_pos to a pickle file
# ------------------------------------
with open('/home/sam/bucket/sam/ant_tracking/sleap_aruco_tracks.pkl', "wb") as f:
    pickle.dump(all_pos, f)

if visualize:
    cap.release()
    cv2.destroyAllWindows()

    
#%% Optimized Visualization Settings
brightness_factor = 1.2
visualize = 1  # Set to 1 to enable visualization
skip_frames = 50  # Visualize every 5th frame

tally = 0
cap = cv2.VideoCapture(video_file)

if visualize:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        for _ in range(skip_frames):  # Skip frames for faster playback
            success, img = cap.read()
            if not success:
                break
        tally += skip_frames

        if not success or img is None:
            print(f"End of video or failed to read frame {tally}.")
            break

        # Make the image brighter
        disp_img = np.clip(img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

        # Get valid positions for the current frame from all_pos
        if tally in all_pos:
            frame_positions = all_pos[tally]
            for aruco_id, positions in frame_positions.items():
                for pos in positions:
                    if np.all(pos >= 0):  # Only draw valid positions
                        px, py = int(pos[0]), int(pos[1])
                        cv2.circle(disp_img, (px, py), 20, (0, 0, 255), -1)
                        cv2.putText(disp_img, str(aruco_id),
                                    (px + 15, py + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        # Add frame information
        cv2.putText(disp_img, f"Frame {tally}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        show_img = cv2.resize(disp_img, (1080, 720))
        cv2.imshow("Centroid Tracking with ArUco IDs", show_img)

        # Quit visualization with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



#%% Visualize no video
import cv2
import numpy as np

min_posX=int(np.min(aruco_detection['X'].values))
max_posX=int(np.max(aruco_detection['X'].values))
min_posY=int(np.min(aruco_detection['Y'].values))
max_posY=int(np.max(aruco_detection['Y'].values))


# Optimized Visualization Settings
brightness_factor = 1.2
visualize = 1  # Set to 1 to enable visualization
skip_frames = 1  # Visualize every 50th frame
image_size = (max_posX-min_posX,max_posY-min_posY, 3)  # Height, Width, Channels


if visualize:
    frame_indices = sorted(all_pos.keys())  # Get all valid frame indices

    for frame_idx in frame_indices[::skip_frames]:  # Skip frames for faster visualization
        if frame_idx not in all_pos:
            continue  # Skip empty frames

        # Create a blank image
        disp_img = np.zeros(image_size, dtype=np.uint8)

        # Make the image brighter
        disp_img[:, :] = (disp_img[:, :] * brightness_factor).clip(0, 255).astype(np.uint8)

        # Get positions and IDs for the current frame
        frame_positions = all_pos[frame_idx] 
        for aruco_id, positions in frame_positions.items():
            
            for pos in positions:
                curr_pos=pos-[min_posX, min_posY]
                if np.all(curr_pos >= 0):  # Only draw valid positions
                    px, py = int(curr_pos[0]), int(curr_pos[1])
                    # Draw a circle for each position
                    cv2.circle(disp_img, (px, py), 40, (0, 0, 255), -1)
                    # Add the ArUco ID as text
                    cv2.putText(disp_img, str(aruco_id),
                                (px + 15, py + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2)

        # Add frame information
        cv2.putText(disp_img, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        show_img = cv2.resize(disp_img, (1080, 720))
        cv2.imshow("Centroid Tracking with ArUco IDs", show_img)

        # Quit visualization with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




#%%

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def post_process_all_pos_with_unknown(all_pos, n_frames, max_distance):
    """
    Converts `all_pos` dictionary into a NumPy array for post-processing,
    retaining unmatched detections in an 'unknown' track.

    Parameters:
        all_pos: dict
            Dictionary of frame-indexed ArUco ID positions:
            {frame_index: {aruco_id: [(x, y), ...], ...}, ...}
        n_frames: int
            Total number of frames in the video.
        max_distance: float
            Maximum allowed distance for nearest-neighbor matching.

    Returns:
        positions: np.ndarray
            Array of shape (n_ids + 1, n_frames, 2), where each ID maps to
            a time series of positions (x, y). The last row corresponds to
            the 'unknown' track.
        id_index: np.ndarray
            Array of unique IDs corresponding to rows in `positions`.
            The last element is 'unknown'.
    """
    # Collect all unique IDs and ensure they are strings
    unique_ids = sorted(list(set(str(aruco_id) for frame in all_pos.values() for aruco_id in frame.keys())))
    n_ids = len(unique_ids)

    # Map IDs to array indices
    id_index = np.array(unique_ids + ['unknown'], dtype=str)  # Add 'unknown' as the last ID
    positions = np.full((n_ids + 1, n_frames, 2), np.nan, dtype=float)  # Initialize with invalid (-1, -1)

    # Process each frame with tqdm for progress tracking
    for frame_idx in tqdm(range(n_frames), desc="Processing Frames"):
        if frame_idx not in all_pos:
            continue  # Skip empty frames
        frame_data = all_pos[frame_idx]

        for aruco_id, pos_list in frame_data.items():
            aruco_id = str(aruco_id)  # Ensure the ID is a string
            if len(pos_list) == 0:
                continue  # No valid positions

            # Find index of the ID or assign to 'unknown'
            if aruco_id in id_index:
                id_idx = np.where(id_index == aruco_id)[0][0]
            else:
                id_idx = np.where(id_index == 'unknown')[0][0]

            if len(pos_list) == 1:
                # Assign the single detected position
                positions[id_idx, frame_idx] = pos_list[0]
            else:
                # Handle multiple detections using nearest neighbor logic
                prev_pos = positions[id_idx, frame_idx - 1] if frame_idx > 0 else None
                if prev_pos is not None and np.all(prev_pos >= 0):
                    # Find the nearest neighbor to the last known position
                    dists = cdist([prev_pos], pos_list)
                    nearest_idx = np.argmin(dists)
                  #  if dists[0, nearest_idx] <= max_distance:
                    positions[id_idx, frame_idx] = pos_list[nearest_idx]
                    # Assign remaining detections to 'unknown'
                    unknown_idx = np.where(id_index == 'unknown')[0][0]
                    for i, pos in enumerate(pos_list):
                        if i != nearest_idx:
                            positions[unknown_idx, frame_idx] = pos
                    # else:
                    #     # If no valid match, assign all to 'unknown'
                    #     unknown_idx = np.where(id_index == 'unknown')[0][0]
                    #     for pos in pos_list:
                    #         positions[unknown_idx, frame_idx] = pos
                else:
                    # If no previous position, assign the first instance to ID and others to 'unknown'
                    positions[id_idx, frame_idx] = pos_list[0]
                    unknown_idx = np.where(id_index == 'unknown')[0][0]
                    for i, pos in enumerate(pos_list):
                        if i != 0:
                            positions[unknown_idx, frame_idx] = pos

    return positions, id_index


tracks_file='/home/sam/bucket/sam/ant_tracking/20241108_1/sleap_aruco_tracks.pkl'
with open(tracks_file, "rb") as f:
    all_pos = pickle.load(f)

length=len(all_pos)
max_distance = 60
positions, instance_ids = post_process_all_pos_with_unknown(all_pos, n_frames=length, max_distance=max_distance)



all_tags=np.load('/home/sam/bucket/sam/ant_tracking/aruco_imgs/classnames.npy')

tag_id=[]
for tag in all_tags:
    tag_id.append(int(tag))

# matching_indices = np.array([np.where(instance_ids == tag)[0][0] if tag in instance_ids else -1 for tag in all_tags])
# matching_indices=matching_indices[matching_indices>-1]
# sorted_tags=positions[matching_indices,:,:]

sorted_tags=positions[tag_id,:,:]


non_nan_count = np.sum(~np.isnan(sorted_tags[:,:,0]),axis=1) 
good_tracks=np.where(non_nan_count>20000)[0]

np.save('/home/sam/bucket/sam/ant_tracking/20241108_1/final_pos.npy',positions)

#%%
import os
output_folder='/home/sam/bucket/sam/ant_tracking/track_figs'
os.makedirs(output_folder, exist_ok=True)

min_posX=int(np.min(aruco_detection['X'].values))
max_posX=int(np.max(aruco_detection['X'].values))
min_posY=int(np.min(aruco_detection['Y'].values))
max_posY=int(np.max(aruco_detection['Y'].values))

# Ensure the data for the selected ID exists
for idx in range( positions.shape[0]-1):
    if idx in good_tracks:
        curr_instance=int(instance_ids[idx])
        curr_id=all_tags[curr_instance]
        x_positions = positions[idx, :, 0]
        y_positions = positions[idx, :, 1]
    
        # Create a color map based on the frame indices
        time_indices = np.arange(positions.shape[1])  # Frame indices
        valid_indices = x_positions != -1  # Only include valid positions
        colors = time_indices[valid_indices]
    
        # Plot positions with color corresponding to time
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x_positions[valid_indices], 
                               y_positions[valid_indices], 
                               c=colors, cmap='viridis', s=50)
        plt.colorbar(scatter, label='Frame Index')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Trajectory of ID {curr_id} Over Time')
        plt.xlim(min_posX, max_posX)  # Set consistent X axis range
        plt.ylim(min_posY, max_posY)  # Set consistent Y axis range
        plt.grid(True)
        plt.show()
    
        output_path = os.path.join(output_folder, f"track_{curr_id}.png")
        plt.savefig(output_path)
        plt.close()

        
        
