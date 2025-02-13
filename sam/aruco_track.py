#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:59:17 2024

Modified by ChatGPT

Now using an ArUco detection file that is a numpy array of shape
   (n_frames, n_ids, 2)
with missing detections indicated by [-1, -1].
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pickle
import os

def interpolate_missing_positions(merged_all_pos, max_gap):
    """
    Linearly interpolates missing positions (-1) in the merged_all_pos array over gaps less than max_gap frames,
    but does not interpolate beyond the first or last valid positions.
    """
    interpolated_pos = merged_all_pos.copy()
    num_tracks, num_frames, _ = merged_all_pos.shape

    for t_idx in range(num_tracks):
        for dim in range(2):  # Interpolate X and Y separately
            track_values = merged_all_pos[t_idx, :, dim]
            valid_indices = np.where(track_values != -1)[0]

            if len(valid_indices) > 1:
                first_valid = valid_indices[0]
                last_valid = valid_indices[-1]
                for start_idx, end_idx in zip(valid_indices[:-1], valid_indices[1:]):
                    if (end_idx - start_idx - 1) <= max_gap:
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
    Returns True if every position for track_id in positions_dict is
    farther than max_dist from every other valid track's position.
    """
    my_positions = positions_dict.get(track_id, [])
    if len(my_positions) == 0:
        return False  # No valid positions
    for other_id, other_positions in positions_dict.items():
        if other_id == track_id:
            continue
        for my_pos in my_positions:
            for other_pos in other_positions:
                if np.any(other_pos < 0):
                    continue
                if np.linalg.norm(my_pos - other_pos) < max_dist:
                    return False
    return True

#%% File paths
video_file = '/home/sam/bucket/Ants/basler/20250123_1/data/cam5_2025-01-23-11-15-11_cam06_000.avi'
aruco_file = '/home/sam/bucket/Ants/basler/20250123_1/data/cam5_2025-01-23-11-15-11_cam06_000.aviaruco_tracks_.npy'
sleap_file = '/home/sam/bucket/Ants/basler/20250123_1/data/cam5_2025-01-23-11-15-11_cam06_000.csv'

#%% Load detections
# ArUco detections are now stored in a numpy array with shape (n_frames, n_ids, 2)
aruco_detection = np.load(aruco_file)
n_frames_aruco, n_ids = aruco_detection.shape[0], aruco_detection.shape[1]

# Load SLEAP detections as before
sleap_detection = pd.read_csv(sleap_file)
sleap_detection['Frame'] = sleap_detection['Frame'] - 1  # zero-based frames
sleap_detection = sleap_detection.drop(['Score_node'], axis=1)
sleap_detection['Instance'] = sleap_detection.groupby('Frame').cumcount() - 1

#%% Tracking parameters
max_distance = 80
min_sep_distance = 10.0
min_speed = 60.0
brightness_factor = 1.2
visualize = True
startFrame = 0

# Group SLEAP detections by frame for easier lookup
grouped_sleap = dict(iter(sleap_detection.groupby('Frame')))
# Use the number of frames from the ArUco array for the main loop.
length = n_frames_aruco

if visualize:
    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

# We track positions per ArUco ID. Both "tracked_positions" (last frame)
# and "all_pos" (by frame) are dictionaries mapping ID -> list-of-positions.
tracked_positions = {}
all_pos = {}
for frame_idx in range(length):
    all_pos[frame_idx] = {}

#%% Main Tracking Loop
for tally in tqdm(range(length), desc="Tracking frames"):
    if visualize:
        success, img = cap.read()
        if not success:
            print(f"Failed to read frame {tally}.")
            break

    # -- SLEAP centroids for current frame --
    if tally in grouped_sleap:
        curr_sleap = grouped_sleap[tally][['X', 'Y']].to_numpy()
    else:
        curr_sleap = np.empty((0, 2))

    # -- Get ArUco detections for this frame from the numpy array --
    if tally < n_frames_aruco:
        frame_data = aruco_detection[tally]  # shape: (n_ids, 2)
    else:
        frame_data = np.full((n_ids, 2), -1)  # no detections if out-of-bounds

    # Instead of grouping by a column, use the index as the ArUco ID.
    aruco_frame_data = {}
    for aruco_id in range(frame_data.shape[0]):
        pos = frame_data[aruco_id]
        # Skip if detection is missing (assumed marked as [-1, -1])
        if np.all(pos == -1):
            continue
        # Wrap in an array so that it resembles the original “list of detections”
        aruco_frame_data[aruco_id] = np.array([pos])

    # Initialize new tracked positions for this frame
    new_tracked_positions = {aruco_id: [] for aruco_id in aruco_frame_data.keys()}

    # ==========================================================
    # 2A) Assign ArUco detections to their IDs
    # ==========================================================
    for aruco_id, aruco_positions in aruco_frame_data.items():
        old_positions = tracked_positions.get(aruco_id, [])
        old_positions_array = np.array(old_positions)
        for aruco_pos in aruco_positions:
            if old_positions_array.size > 0:
                dists_to_old = np.linalg.norm(old_positions_array - aruco_pos, axis=1)
                if dists_to_old.min() <= max_distance:
                    new_tracked_positions[aruco_id].append(aruco_pos)
                    continue  # detection matched; if you want one match per detection
            else:
                new_tracked_positions[aruco_id].append(aruco_pos)

    # -----------------------------------------------------------
    # 2B) Missing detection logic: try to update missing IDs using SLEAP centroids
    # -----------------------------------------------------------
    # Build a flat list of assigned (aruco_id, x, y)
    assigned_list = []
    for aruco_id, assigned_positions in new_tracked_positions.items():
        for pos in assigned_positions:
            assigned_list.append((aruco_id, pos[0], pos[1]))
    assigned_array = np.array(assigned_list, dtype=float)  # shape (M, 3)
    used_centroids = set()
    if len(assigned_array) > 0 and len(curr_sleap) > 0:
        aruco_positions_2d = assigned_array[:, 1:3]
        dist_matrix = cdist(aruco_positions_2d, curr_sleap)
        nearest_centroid_idx = dist_matrix.argmin(axis=1)
        nearest_centroid_dist = dist_matrix.min(axis=1)
        for i, dist_val in enumerate(nearest_centroid_dist):
            if dist_val <= max_distance:
                used_centroids.add(nearest_centroid_idx[i])
    unused_centroids = set(range(len(curr_sleap))) - used_centroids

    # For each previously tracked ID with no new detection, try to match a SLEAP centroid.
    for old_id in tracked_positions.keys():
        if (old_id not in new_tracked_positions) or (len(new_tracked_positions[old_id]) == 0):
            old_pos_list = tracked_positions[old_id]
            if len(old_pos_list) == 0:
                continue
            old_pos = old_pos_list[-1]
            if len(curr_sleap) > 0:
                dists = np.linalg.norm(curr_sleap - old_pos, axis=1)
                valid_idx = [i for i in unused_centroids if dists[i] <= max_distance]
                if len(valid_idx) == 1:
                    matched_idx = valid_idx[0]
                    matched_pos = curr_sleap[matched_idx]
                    new_tracked_positions.setdefault(old_id, []).append(matched_pos)
                    unused_centroids.remove(matched_idx)
                    continue
            # If no match and the previous movement was small, assume the object stayed in place.
            if len(old_pos_list) >= 2:
                prev_pos = old_pos_list[-2]
                speed = np.linalg.norm(old_pos - prev_pos)
                if speed < min_speed:
                    new_tracked_positions.setdefault(old_id, []).append(old_pos)
                    continue

    # =======================================================
    # 3) Conflict Resolution: remove duplicate detections too close together
    # =======================================================
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
        for j in range(i + 1, len(flat_list)):
            if not keep[j]:
                continue
            id_j, x_j, y_j = flat_list[j]
            pos_j = np.array([x_j, y_j], dtype=float)
            if np.linalg.norm(pos_i - pos_j) < min_sep_distance:
                keep[j] = False
    filtered_new_positions = {}
    for (flag, (aruco_id, x, y)) in zip(keep, flat_list):
        if flag:
            filtered_new_positions.setdefault(aruco_id, []).append(np.array([x, y], dtype=float))

    # ---------------------------------------------------------
    # 4) Update global tracking: tracked_positions & all_pos
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

# Clean up video capture if using visualization
if visualize:
    cap.release()
    cv2.destroyAllWindows()

# ------------------------------------
# 6) Save all_pos to a pickle file
# ------------------------------------
with open('/home/sam/bucket/sam/ant_tracking/sleap_aruco_tracks.pkl', "wb") as f:
    pickle.dump(all_pos, f)

#%% (Optional) Further Visualization and Post-processing remain unchanged.
# For example, you can use the post_process_all_pos_with_unknown function below
# to convert the dictionary "all_pos" into a NumPy array (frame x ID x [X,Y])
# for further analysis or plotting.

def post_process_all_pos_with_unknown(all_pos, n_frames, max_distance):
    """
    Converts the `all_pos` dictionary into a NumPy array with shape
    (n_ids + 1, n_frames, 2) where the last row corresponds to unmatched detections ('unknown').
    """
    unique_ids = sorted(list(set(str(aruco_id) for frame in all_pos.values() for aruco_id in frame.keys())))
    n_ids = len(unique_ids)
    id_index = np.array(unique_ids + ['unknown'], dtype=str)
    positions = np.full((n_ids + 1, n_frames, 2), np.nan, dtype=float)

    for frame_idx in tqdm(range(n_frames), desc="Processing Frames"):
        if frame_idx not in all_pos:
            continue
        frame_data = all_pos[frame_idx]
        for aruco_id, pos_list in frame_data.items():
            aruco_id = str(aruco_id)
            if len(pos_list) == 0:
                continue
            if aruco_id in id_index:
                id_idx = np.where(id_index == aruco_id)[0][0]
            else:
                id_idx = np.where(id_index == 'unknown')[0][0]
            if len(pos_list) == 1:
                positions[id_idx, frame_idx] = pos_list[0]
            else:
                prev_pos = positions[id_idx, frame_idx - 1] if frame_idx > 0 else None
                if prev_pos is not None and np.all(prev_pos >= 0):
                    dists = cdist([prev_pos], pos_list)
                    nearest_idx = np.argmin(dists)
                    positions[id_idx, frame_idx] = pos_list[nearest_idx]
                    unknown_idx = np.where(id_index == 'unknown')[0][0]
                    for i, pos in enumerate(pos_list):
                        if i != nearest_idx:
                            positions[unknown_idx, frame_idx] = pos
                else:
                    positions[id_idx, frame_idx] = pos_list[0]
                    unknown_idx = np.where(id_index == 'unknown')[0][0]
                    for i, pos in enumerate(pos_list):
                        if i != 0:
                            positions[unknown_idx, frame_idx] = pos
    return positions, id_index

tracks_file = '/home/sam/bucket/sam/ant_tracking/sleap_aruco_tracks.pkl'
with open(tracks_file, "rb") as f:
    all_pos = pickle.load(f)

length = len(all_pos)
positions, instance_ids = post_process_all_pos_with_unknown(all_pos, n_frames=length, max_distance=max_distance)

# (Further plotting code below remains similar.)
all_tags = np.load('/home/sam/bucket/sam/ant_tracking/aruco_imgs/classnames.npy')
tag_id = [int(tag) for tag in all_tags]
sorted_tags = positions[tag_id, :, :]

non_nan_count = np.sum(~np.isnan(sorted_tags[:, :, 0]), axis=1)
good_tracks = np.where(non_nan_count > 20000)[0]

np.save('/home/sam/bucket/sam/ant_tracking/20241108_1/final_pos.npy', positions)

#%% Plotting individual tracks
output_folder = '/home/sam/bucket/sam/ant_tracking/track_figs'
os.makedirs(output_folder, exist_ok=True)

min_posX = int(np.min(aruco_detection[:, :, 0]))
max_posX = int(np.max(aruco_detection[:, :, 0]))
min_posY = int(np.min(aruco_detection[:, :, 1]))
max_posY = int(np.max(aruco_detection[:, :, 1]))

for idx in range(positions.shape[0] - 1):
    if idx in good_tracks:
        curr_instance = int(instance_ids[idx])
        curr_id = all_tags[curr_instance]
        x_positions = positions[idx, :, 0]
        y_positions = positions[idx, :, 1]

        time_indices = np.arange(positions.shape[1])
        valid_indices = x_positions != -1
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x_positions[valid_indices],
                              y_positions[valid_indices],
                              c=time_indices[valid_indices], cmap='viridis', s=50)
        plt.colorbar(scatter, label='Frame Index')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Trajectory of ID {curr_id} Over Time')
        plt.xlim(min_posX, max_posX)
        plt.ylim(min_posY, max_posY)
        plt.grid(True)
        output_path = os.path.join(output_folder, f"track_{curr_id}.png")
        plt.savefig(output_path)
        plt.show()
        plt.close()
