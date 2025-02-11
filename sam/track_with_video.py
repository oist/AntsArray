#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 09:42:05 2024

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
import pickle


def interpolate_missing_positions(merged_all_pos):
    """
    Linearly interpolates missing positions (-1) in the merged_all_pos array.

    Parameters:
    - merged_all_pos (numpy.ndarray): Array of shape (tracks, time, 2) with missing positions as -1.

    Returns:
    - numpy.ndarray: Interpolated merged_all_pos array.
    """
    interpolated_pos = merged_all_pos.copy()
    num_tracks, num_frames, _ = merged_all_pos.shape

    for t_idx in range(num_tracks):
        for dim in range(2):  # Interpolate separately for X and Y
            track_values = merged_all_pos[t_idx, :, dim]
            valid_indices = np.where(track_values != -1)[0]
            invalid_indices = np.where(track_values == -1)[0]

            if len(valid_indices) > 1:  # At least two points to interpolate
                # Interpolation function
                interp_func = np.interp(
                    np.arange(num_frames),  # Full range of frames
                    valid_indices,          # Frames with valid data
                    track_values[valid_indices]  # Corresponding valid values
                )
                # Apply interpolation for the invalid indices
                interpolated_pos[t_idx, invalid_indices, dim] = interp_func[invalid_indices]

    return interpolated_pos



video_file='/home/sam/bucket/Ants/basler/20250123_1/data/cam5_2025-01-23-11-15-11_cam06_000.avi'
aruco_file='/home/sam/bucket/Ants/basler/20250123_1/data/tracks_aruco/cam5_2025-01-23-11-15-11_cam06_000.pkl'
sleap_file='/home/sam/bucket/Ants/basler/20250123_1/data/cam5_2025-01-23-11-15-11_cam06_000.csv'

#aruco
with open(aruco_file, "rb") as f:
    aruco_detection = pickle.load(f)
    
num_frames, num_arucos, num_positions = aruco_tracks.shape
reshaped_array = aruco_tracks.reshape((num_arucos * num_frames, num_positions))

#sleap
df = pd.read_csv(sleap_file)
df['Frame']=df['Frame']-1 #0 base frames
df = df.drop(['Score_node'], axis=1)
# Reset instance counter for each new frame
df['Instance'] = df.groupby('Frame').cumcount() - 1



#%% 
#------------------- PARAMETERS -------------------
brightness_factor = 1.2
visualize = 1

max_sleap = 1000
max_distance = 60        # detection->track distance threshold
lost_frames = 200        # how long we try to recover a lost track
live_thresh = 25         # track must exist >= this many frames to be kept
min_sep_distance = 10.0  # min separation to spawn or maintain separate tracks

#--------------------------------------------------

cap = cv2.VideoCapture(video_file)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
all_pos = np.full((max_sleap, length, 2), -1, dtype=float)

ret, img = cap.read()
if not ret:
    print("Failed to read the first frame.")
    cap.release()
    exit()

tally = 0
curr_sleap = df[df['Frame'] == tally][['X', 'Y']].to_numpy()

# Track data
tracked_centroids = np.full((max_sleap, 2), -1, dtype=float)  # current (x,y) for each track
lost_positions = np.full((max_sleap, 2), -1, dtype=float)     # last known position if lost
lost_count = np.zeros(max_sleap, dtype=int)                   # how many frames track is lost
track_exists_count = np.zeros(max_sleap, dtype=int)           # how many frames track has been active

# IDs available to spawn new tracks
track_id_pool = list(range(max_sleap))

#------------------- INIT FRAME (0) -------------------
n_init = min(len(curr_sleap), max_sleap)
for i in range(n_init):
    t_idx = track_id_pool.pop(0)
    tracked_centroids[t_idx] = curr_sleap[i]
    all_pos[t_idx, 0, :] = curr_sleap[i]
    track_exists_count[t_idx] = 1

tally = 1

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    print(tally)
    curr_sleap = df[df['Frame'] == tally][['X', 'Y']].to_numpy()

    # Container for the next frame's positions
    new_tracked_centroids = np.full((max_sleap, 2), -1, dtype=float)

    matched_detection_indices = set()  # which detections are claimed

    #================= A) ACTIVE TRACKS =================
    active_tracks = np.where(tracked_centroids[:, 0] != -1)[0]

    for t_idx in active_tracks:
        old_pt = tracked_centroids[t_idx].reshape(1, 2)

        if len(curr_sleap) == 0:
            # No new detections => track goes lost
            lost_positions[t_idx] = old_pt[0]
            lost_count[t_idx] = 1
            continue

        # Distances from old_pt to each detection
        dists = cdist(old_pt, curr_sleap)[0]
        valid_detections = np.where(dists <= max_distance)[0]

        if len(valid_detections) == 0:
            # No detection => lost
            lost_positions[t_idx] = old_pt[0]
            lost_count[t_idx] = 1

        elif len(valid_detections) == 1:
            # Exactly one detection => track continues
            det_idx = valid_detections[0]
            # If detection is already matched, skip or mark this track lost
            if det_idx in matched_detection_indices:
                # We'll just lose this track to keep it simple
                lost_positions[t_idx] = old_pt[0]
                lost_count[t_idx] = 1
            else:
                # Assign detection
                new_tracked_centroids[t_idx] = curr_sleap[det_idx]
                matched_detection_indices.add(det_idx)
                # reset lost
                lost_positions[t_idx] = [-1, -1]
                lost_count[t_idx] = 0
                track_exists_count[t_idx] += 1

        else:
            # Multiple => pick the single closest detection
            closest_det_idx = valid_detections[np.argmin(dists[valid_detections])]
            if closest_det_idx in matched_detection_indices:
                # If it's claimed, we lose this track to keep code simple
                lost_positions[t_idx] = old_pt[0]
                lost_count[t_idx] = 1
            else:
                new_tracked_centroids[t_idx] = curr_sleap[closest_det_idx]
                matched_detection_indices.add(closest_det_idx)
                lost_positions[t_idx] = [-1, -1]
                lost_count[t_idx] = 0
                track_exists_count[t_idx] += 1

            # Then we spawn new tracks for the other detections
            # only if they're at least min_sep_distance from the chosen detection 
            chosen_coord = curr_sleap[closest_det_idx]

            for det_idx in valid_detections:
                if det_idx == closest_det_idx:
                    continue
                if det_idx in matched_detection_indices:
                    continue

                new_coord = curr_sleap[det_idx]
                dist = np.linalg.norm(new_coord - chosen_coord)

                if dist < min_sep_distance:
                    # Too close => skip
                    continue

                if track_id_pool:
                    new_id = track_id_pool.pop(0)
                    new_tracked_centroids[new_id] = new_coord
                    track_exists_count[new_id] = 1
                    matched_detection_indices.add(det_idx)

    #================= B) LOST TRACK RECOVERY =================
    unmatched = set(range(len(curr_sleap))) - matched_detection_indices
    lost_candidates = np.where(
        (tracked_centroids[:, 0] == -1)
        & (lost_positions[:, 0] != -1)
        & (lost_count < lost_frames)
    )[0]

    for t_idx in lost_candidates:
        old_pt = lost_positions[t_idx].reshape(1, 2)
        if not unmatched:
            lost_count[t_idx] += 1
            continue

        # Distances from the lost position to unmatched detections
        det_list = np.array(list(unmatched))
        possible_pts = curr_sleap[det_list, :]
        dists = cdist(old_pt, possible_pts)[0]
        valid_det_indices = np.where(dists <= max_distance)[0]

        if len(valid_det_indices) == 0:
            lost_count[t_idx] += 1
        else:
            # Pick the single closest
            closest_det_idx = valid_det_indices[np.argmin(dists[valid_det_indices])]
            actual_det_idx = det_list[closest_det_idx]

            if actual_det_idx in matched_detection_indices:
                # detection is claimed => skip
                lost_count[t_idx] += 1
            else:
                # recover this lost track
                new_tracked_centroids[t_idx] = curr_sleap[actual_det_idx]
                track_exists_count[t_idx] += 1
                matched_detection_indices.add(actual_det_idx)
                unmatched.remove(actual_det_idx)
                lost_positions[t_idx] = [-1, -1]
                lost_count[t_idx] = 0

    #================= C) NEW TRACKS FOR REMAINING UNMATCHED =================
    # We also require new tracks to be at least min_sep_distance from each other
    # and from any newly assigned track in new_tracked_centroids
    existing_new_coords = []  # coords for newly assigned or created tracks

    # Collect coords from newly assigned/continued tracks
    assigned_indices = np.where(new_tracked_centroids[:, 0] != -1)[0]
    for idx in assigned_indices:
        assigned_coord = new_tracked_centroids[idx]
        existing_new_coords.append(assigned_coord)

    # Now spawn new tracks
    for det_idx in (unmatched - matched_detection_indices):
        if not track_id_pool:
            break

        coord = curr_sleap[det_idx]

        # Check distance from all existing_new_coords
        if any(np.linalg.norm(coord - existing) < min_sep_distance for existing in existing_new_coords):
            continue  # skip too-close detection

        # Otherwise create a new track
        new_id = track_id_pool.pop(0)
        new_tracked_centroids[new_id] = coord
        track_exists_count[new_id] = 1
        existing_new_coords.append(coord)

    #================= D) UPDATE TRACKED CENTROIDS =================
    tracked_centroids = new_tracked_centroids.copy()
    all_pos[:, tally, :] = tracked_centroids

    #================= E) END OR LOST TRACKS CLEANUP =================
    # If track is lost > lost_frames, decide if ephemeral
    for t_idx in range(max_sleap):
        if tracked_centroids[t_idx, 0] == -1:
            # track is not active this frame
            lost_count[t_idx] += 1

            if lost_count[t_idx] >= lost_frames:
                # ephemeral check
                if track_exists_count[t_idx] < live_thresh:
                    all_pos[t_idx, :, :] = -1

                    # free ID
                    if t_idx not in track_id_pool:
                        track_id_pool.append(t_idx)
    
                    # reset
                    track_exists_count[t_idx] = 0
                    lost_count[t_idx] = 0
                    lost_positions[t_idx] = [-1, -1]

    #================= VISUALIZATION (OPTIONAL) =================
    if visualize:
        disp_img = np.clip(img.astype(np.float32)*brightness_factor, 0, 255).astype(np.uint8)
        for t_idx, pos in enumerate(tracked_centroids):
            if pos[0] != -1:
                cv2.circle(disp_img, (int(pos[0]), int(pos[1])), 20, (0, 0, 255), -1)
                cv2.putText(disp_img, str(t_idx), (int(pos[0]) + 15, int(pos[1]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(disp_img, f"Frame {tally}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        show_img = cv2.resize(disp_img, (1080, 720))
        cv2.imshow("Centroid Tracking", show_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tally += 1

cap.release()
cv2.destroyAllWindows()

print("Tracking completed. Final shape of all_pos:", all_pos.shape)

#np.save('/home/sam/bucket/sam/ant_tracking/sleapTrack.npy', all_pos)

#%%
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict

# Example shapes:
# all_pos: tracks x time x 2
# aruco_tracks: time x arucoID x 2

# Masks for valid positions
valid_tracks = (all_pos[..., 0] != -1)  # Shape: (tracks, time)
valid_aruco = (aruco_tracks[..., 0] != 0)  # Shape: (time, arucoID)

# Initialize a storage array for closest ArUco IDs
closest_aruco_ids = np.full_like(all_pos[..., 0], fill_value=-1, dtype=int)  # Shape: (tracks, time)

# Iterate over all valid time frames
for t in range(all_pos.shape[1]):  # Loop over time
    valid_track_indices = np.where(valid_tracks[:, t])[0]  # Valid track indices for this frame
    valid_aruco_indices = np.where(valid_aruco[t])[0]  # Valid ArUco indices for this frame

    if len(valid_track_indices) == 0 or len(valid_aruco_indices) == 0:
        continue  # Skip if no valid data in this frame

    # Get valid positions for tracks and ArUco markers
    track_positions = all_pos[valid_track_indices, t]  # Shape: (valid_tracks, 2)
    aruco_positions = aruco_tracks[t, valid_aruco_indices]  # Shape: (valid_aruco, 2)

    # Compute distances
    distances = cdist(track_positions, aruco_positions)  # Shape: (valid_tracks, valid_aruco)

    # Find the closest ArUco for each track
    closest_indices = np.argmin(distances, axis=1)  # Shape: (valid_tracks,)
    closest_aruco_ids[valid_track_indices, t] = valid_aruco_indices[closest_indices]  # Map to actual ArUco IDs

# Determine the most frequent ArUco ID for each track
track_to_aruco_id = {}
for track in range(all_pos.shape[0]):
    track_aruco_matches = closest_aruco_ids[track, valid_tracks[track]]
    if track_aruco_matches.size > 0:
        most_frequent_aruco = Counter(track_aruco_matches).most_common(1)[0][0]
        track_to_aruco_id[track] = most_frequent_aruco
    else:
        track_to_aruco_id[track] = None  # No valid ArUco ID

# Create merged tracks array
unique_aruco_ids = list(set(track_to_aruco_id.values()) - {None})  # Get unique ArUco IDs
merged_all_pos = np.full((len(unique_aruco_ids), all_pos.shape[1], 2), -1, dtype=float)
merged_aruco_ids = np.full(len(unique_aruco_ids), -1, dtype=int)  # Array to store ArUco IDs for merged tracks

aruco_to_track_mapping = defaultdict(list)
for track, aruco_id in track_to_aruco_id.items():
    if aruco_id is not None:
        aruco_to_track_mapping[aruco_id].append(track)

for idx, aruco_id in enumerate(unique_aruco_ids):
    # Initialize merged track for the current ArUco ID
    merged_track = np.full((all_pos.shape[1], 2), -1, dtype=float)
    
    for track in aruco_to_track_mapping[aruco_id]:
        valid_positions = (all_pos[track, :, 0] != -1)  # Identify valid positions in the current track
        
        # Add valid positions to the merged track only if the corresponding positions in merged_track are invalid
        overwrite_positions = valid_positions & (merged_track[:, 0] == -1)
        merged_track[overwrite_positions] = all_pos[track, overwrite_positions]
    
    # Assign the merged track to the result array
    merged_all_pos[idx] = merged_track
    merged_aruco_ids[idx] = aruco_id  # Save the corresponding ArUco ID

print("Merged all_pos Shape:", merged_all_pos.shape)
print("Merged ArUco IDs:", merged_aruco_ids)



# Apply interpolation to merged_all_pos
merged_all_pos_interpolated = interpolate_missing_positions(merged_all_pos)


#%% Visualization settings
brightness_factor = 1.2
visualize = 1  # Set to 1 to enable visualization

tally = 0
cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Make the image brighter
    disp_img = np.clip(img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    # Plot each track's position for the current frame
    for t_idx, pos in enumerate(merged_all_pos_interpolated[:, tally, :]):
        if pos[0] != -1:  # Only consider valid positions
            # Draw the track position
            cv2.circle(disp_img, (int(pos[0]), int(pos[1])), 20, (0, 0, 255), -1)

            # Overlay the mapped ArUco ID
            mapped_aruco_id = merged_aruco_ids[t_idx]
            cv2.putText(disp_img, str(mapped_aruco_id), 
                        (int(pos[0]) + 15, int(pos[1]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    # Add frame information
    cv2.putText(disp_img, f"Frame {tally}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Resize and display the frame
    show_img = cv2.resize(disp_img, (1080, 720))
    cv2.imshow("Centroid Tracking with ArUco IDs", show_img)

    # Quit visualization with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    tally += 1

cap.release()
cv2.destroyAllWindows()
