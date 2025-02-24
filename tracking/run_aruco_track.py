#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:03:05 2025

@author: sam
"""
                
from aruco_track import *

def remove_small_folders(output_dir, min_images):
    """
    Remove folders with fewer than `min_images` in the output directory.
    Args:
        output_dir: Path to the root directory containing subfolders.
        min_images: Minimum number of images required to keep a folder.
    """
    for subfolder in os.listdir(output_dir):
        subfolder_path = os.path.join(output_dir, subfolder)
        if os.path.isdir(subfolder_path):
            num_images = len([file for file in os.listdir(subfolder_path) if file.endswith(('.png', '.jpg', '.jpeg'))])
            if num_images < min_images:
                shutil.rmtree(subfolder_path)
                print(f"Removed folder: {subfolder_path} (contained {num_images} images)")


exp_name = '20250123_1'
input_file_path = '/home/sam/bucket/anouk/ant_tracking/'+ exp_name + '/' 
output_path = '/home/sam/bucket/sam/ant_tracking/'+ exp_name + '/' 



aruco_file_name_col1 = exp_name +'_aruco_realigned_col1.pkl'
sleap_file_name_col1 = exp_name + '_sleap_realigned_col1.pkl'

aruco_file_name_col2 = exp_name +'_aruco_realigned_col2.pkl'
sleap_file_name_col2 = exp_name + '_sleap_realigned_col2.pkl'

# load dataframes, do tracking
# aruco_detection_col1 = read_pickle_file_with_load(input_file_path, aruco_file_name_col1)
# sleap_detection_col1 = read_pickle_file_with_load(input_file_path, sleap_file_name_col1)
# pos_col1, idx_col1 = get_complete_tracks(output_path, aruco_detection_col1, sleap_detection_col1, '', False, 'col1')

aruco_detection_col2 = read_pickle_file_with_load(input_file_path, aruco_file_name_col2)
sleap_detection_col2 = read_pickle_file_with_load(input_file_path, sleap_file_name_col2) 
pos_col2, idx_col2 = get_complete_tracks(output_path, aruco_detection_col2, sleap_detection_col2, '', False, 'col2', startFrame =21000)


#it should work on single videos as well

aruco_file_name_col2='/home/sam/saionWork/sam/cam5_2025-01-23-11-15-11_cam06_000.pkl'
sleap_file_name='/home/sam/bucket/Ants/basler/20250123_1/data/cam5_2025-01-23-11-15-11_cam06_000.h5'
video_name='/home/sam/bucket/Ants/basler/20250123_1/data/cam5_2025-01-23-11-15-11_cam06_000.avi'

with open(aruco_file_name_col2, 'rb') as f:
    aruco_detection_col2 = pickle.load(f)
    
    
# single cam sleap is csv and needs some modification. Put this in the conversion code upstream
sleap_detection = pd.read_csv(sleap_file_name_col2)
sleap_detection['Frame'] = sleap_detection['Frame'] - 1  # zero-based frames
sleap_detection = sleap_detection.drop(['Score_node'], axis=1)
sleap_detection['Instance'] = sleap_detection.groupby('Frame').cumcount() - 1


sleap_File = h5py.File(sleap_file_name,'r')
sleap_detection=pd.DataFrame({
    'X': np.squeeze(sleap_File['X'][:]),
    'Y': np.squeeze(sleap_File['Y'][:]),
    'Frame': np.squeeze(sleap_File['Frame'][:]) })



#trying one more time aruco opencv
# aruco_opencv=np.load('/home/sam/bucket/Ants/basler/20250123_1/data/data_tmp/tracks_opencv_sleap_old/cam5_2025-01-23-11-15-11_cam06_000.aviaruco_tracks_.npy')

# #reformat to dataframe

# # Get the number of frames and IDs
# nFrames, nIDs, _ = aruco_opencv.shape

# # Create the DataFrame
# df = pd.DataFrame({
#     'X': aruco_opencv[:,:,0].flatten(),
#     'Y': aruco_opencv[:,:,1].flatten(),
#     'Frame': np.repeat(np.arange(nFrames), nIDs),
#     'ARUCO_number': np.tile(np.arange(nIDs), nFrames),
# })

# aruco_detection_col2=df[(df['X'] != 0) | (df['Y'] != 0)]

#so aruco opencv has good specificity but misses a lot. I can try using it +NN to get better tracking, take this result and use to train a neural net

#%%

from aruco_track import *
all_pos = get_complete_tracks2(output_path, aruco_detection_col2,sleap_detection, video_name, True, harvest_crops=False,
crops_output_dir='/home/sam/Pictures/aruco')


remove_small_folders('/home/sam/Pictures/aruco',5)

#%%


# Get all unique track IDs across all frames
track_ids = sorted({tid for frame in all_pos for tid in frame.keys()})

# Create a mapping from track ID to index
tid_to_index = {tid: i for i, tid in enumerate(track_ids)}

num_tracks = len(track_ids)
num_frames = len(all_pos)

# Initialize the array with NaNs (or another placeholder for missing values)
positions_array = np.full((num_tracks, num_frames, 2), np.nan)

# Fill in the positions
for frame_idx, frame in enumerate(all_pos):
    for tid, (x, y) in frame.items():
        track_idx = tid_to_index[tid]  # Get the row index for this track
        positions_array[track_idx, frame_idx] = [x, y]




#%%
import cv2
import numpy as np

# Define display settings
frame_width = 4000  # Adjust based on expected range of x-coordinates
frame_height = 3000  # Adjust based on expected range of y-coordinates

# Define colors
sleap_color = (255, 0, 0)  # Blue
text_color = (255, 255, 255)  # White

# Iterate over frames
for frame_idx in range(positions_array.shape[1]):
    # Create a blank image (black background)
    img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Draw all positions
    for track_idx, (x, y) in enumerate(positions_array[:, frame_idx]):
        if not np.isnan(x) and not np.isnan(y):  # Skip missing data
            px, py = int(x), int(y)
            cv2.circle(img, (px, py), 8, sleap_color, -1)  # Blue circles
            cv2.putText(img, str(track_ids[track_idx]), (px+10, py+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Display frame number
    cv2.putText(img, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Resize for display
    show_img = cv2.resize(img, (1080, 720))

    # Show frame
    cv2.imshow("Tracking", show_img)

    # Exit on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

#%% Tracking down where multiple instances on a single position come from

import pandas as pd
import numpy as np

# Let's assume your DataFrame is called `df` with columns:
# ['Frame', 'Instance', 'Bodypoint', 'X', 'Y']
df=aruco_detection_col2


# Define your distance threshold
Z = 10  # distance in pixels

# Step 1: Self-merge on 'Frame' so we only compare detections in the same frame
merged = df.merge(df, on='Frame', suffixes=('_1', '_2'))

# Step 2: Exclude comparisons of the same detection.
# We assume that "same detection" means same ARUCO_number and same Cam.
# You can adjust logic here depending on your exact criteria.
merged = merged[
    ~(
        (merged['ARUCO_number_1'] == merged['ARUCO_number_2']) & 
        (merged['Cam_1'] == merged['Cam_2'])
    )
]

# Alternatively, if you only want to ensure the ARUCO_number is different,
# comment out the above step and use the below one instead:
# merged = merged[merged['ARUCO_number_1'] != merged['ARUCO_number_2']]

# Step 3: Compute the Euclidean distance between (X_1, Y_1) and (X_2, Y_2)
merged['distance'] = np.sqrt(
    (merged['X_1'] - merged['X_2'])**2 +
    (merged['Y_1'] - merged['Y_2'])**2
)

# Step 4: Filter for distances <= Z
close_detections = merged[merged['distance'] <= Z]

# close_detections now contains rows where pairs of detections 
# (from different ARUCO markers or different cameras) 
# are in the same frame and within Z pixels.



    #%% Optimized Visualization Settings
    brightness_factor = 1.2
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




    min_posX=int(np.min(aruco_detection['X'].values))
    max_posX=int(np.max(aruco_detection['X'].values))
    min_posY=int(np.min(aruco_detection['Y'].values))
    max_posY=int(np.max(aruco_detection['Y'].values))


    # Optimized Visualization Settings
    brightness_factor = 1.2
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
        print('Converting to array ...\n')
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



    length=len(all_pos)
    max_distance = 60
    positions, instance_ids = post_process_all_pos_with_unknown(all_pos, n_frames=length, max_distance=max_distance)
    return positions, instance_ids



    # all_tags=np.load('/home/sam/bucket/sam/ant_tracking/aruco_imgs/classnames.npy')

    # tag_id=[]
    # for tag in all_tags:
    #     tag_id.append(int(tag))

    # matching_indices = np.array([np.where(instance_ids == tag)[0][0] if tag in instance_ids else -1 for tag in all_tags])
    # matching_indices=matching_indices[matching_indices>-1]
    # sorted_tags=positions[matching_indices,:,:]

    #sorted_tags=positions[tag_id,:,:]


    # non_nan_count = np.sum(~np.isnan(sorted_tags[:,:,0]),axis=1) 
    # good_tracks=np.where(non_nan_count>20000)[0]


