#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:13:48 2024

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
import matplotlib
import os
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
        
        
# Get active aruco IDs using arbitrary cutoff 
def get_aruco_ids(df, max_aruco_ind, aruco_occurance_thresh):

    filtered_df = df[df['ARUCO_number'] < max_aruco_ind]

    
    # Group by 'Aruco_id' and count non-zero values in 'X'
    counts = filtered_df.groupby('ARUCO_number')['X'].apply(lambda x: (x != 0).sum())

    # Find the maximum count
    max_count = counts.max()
    # Identify Aruco_ids with counts greater than one-eighth of the maximum count
    #aruco_ids_to_keep = counts[counts > (max_count / 200)].index
    aruco_ids_to_keep = counts[counts > aruco_occurance_thresh].index

    print(aruco_ids_to_keep)

    # Filter the DataFrame to keep only the rows with Aruco_ids to keep
    filtered_df = filtered_df[filtered_df['ARUCO_number'].isin(aruco_ids_to_keep)]

    print('Number of valid arucos', aruco_ids_to_keep.shape[0])

    return filtered_df
        

def plot_tracks(original_df, corrected_df, aruco_id):
    """
    Plots the original tracks and the corrected (optimal) track for a single Aruco marker.
    
    Parameters:
    - original_df (pd.DataFrame): DataFrame containing the original track data for the Aruco marker.
    - corrected_df (pd.DataFrame): DataFrame containing the corrected (optimal) track for the Aruco marker.
    - aruco_id (int or str): The ID of the Aruco marker being processed (used for labeling).
    """
    

    plt.figure(figsize=(10, 8))
    
    # Plot the original track (all detected positions)
    plt.plot(original_df['X'], original_df['Y'], 'o', label='Original Track (All Detections)', color='lightgray', markersize=8)
    
    # Plot the corrected track (chosen optimal path)
    plt.plot(corrected_df['X'], corrected_df['Y'], 'o', color='red', markersize=5, label='Corrected Track')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Track Debugging for Aruco Marker {aruco_id}')
    
    # Show the legend and grid for clarity
    plt.legend()
    plt.grid(True)
    plt.show()


def get_optimal_track_single_aruco(df, batch_size=100):
    """
    Processes an Aruco marker's tracking data to select optimal track points 
    while minimizing speed for each frame in batches.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing tracking data for a single Aruco marker.
    - batch_size (int): Number of frames to process in each batch.
    
    Returns:
    - final_df (pd.DataFrame): DataFrame containing the optimized track for the Aruco marker.
    """
    final_df = pd.DataFrame(columns=df.columns)  # Initialize an empty DataFrame for the final track
    
    # Get sorted unique frames for this Aruco marker
    frames = np.sort(df['Frame_number'].unique())
    batch_results = []  # Temporary storage for batching

    # Initialize progress bar
    with tqdm(total=len(frames), desc="Frames", leave=False) as frame_pbar:
        for i in range(0, len(frames), batch_size):
            # Process frames in batches
            batch_frames = frames[i:i + batch_size]
            
            for frame in batch_frames:
                frame_df = df[df['Frame_number'] == frame]
                
                # If there are multiple detections in the frame, find the one that minimizes speed
                if frame_df.shape[0] > 1:
                    min_mean_speed = float('inf')
                    min_mean_idx = None

                    if frame == 0:
                        for _, row in frame_df.iterrows():
                            temp_df = df[df['Frame_number'] != frame]
                            temp_df = pd.concat([temp_df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)
                            temp_df = temp_df.sort_values('Frame_number').reset_index(drop=True)
                            temp_df['Speed'] = np.sqrt(temp_df['X'].diff() ** 2 + temp_df['Y'].diff() ** 2)
                            mean_speed = temp_df['Speed'].mean()
                            if mean_speed < min_mean_speed:
                                min_mean_speed = mean_speed
                                min_mean_idx = row.name
                        # Check if min_mean_idx was updated; if not, use the first row as a fallback
                        if min_mean_idx is not None:
                            min_row = df.loc[min_mean_idx]
                        else:
                            min_row = frame_df.iloc[0]
                        batch_results.append(min_row)
                        frame_pbar.update(1)
                        continue
                    else:
                        for _, row in frame_df.iterrows():
                            temp_df = final_df[final_df['Frame_number'] != frame]
                            temp_df = pd.concat([temp_df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)
                            temp_df['Speed'] = np.sqrt(temp_df['X'].diff() ** 2 + temp_df['Y'].diff() ** 2)
                            mean_speed = temp_df['Speed'].mean()
                            if mean_speed < min_mean_speed:
                                min_mean_speed = mean_speed
                                min_mean_idx = row.name
                        # Check if min_mean_idx was updated; if not, use the first row as a fallback
                        if min_mean_idx is not None:
                            batch_results.append(df.loc[min_mean_idx])
                        else:
                            batch_results.append(frame_df.iloc[0])
                else:
                    batch_results.append(frame_df.iloc[0])
                
                # Update progress bar
                frame_pbar.update(1)

            # Concatenate the current batch of results to `final_df`
            final_df = pd.concat([final_df, pd.DataFrame(batch_results, columns=df.columns)], ignore_index=True)
            batch_results.clear()  # Clear batch_results for the next batch

    return final_df

        
exp_name = '20241108_1'
curr_dir='/home/sam/bucket/Ants/trials/' + exp_name + '/data/'
output_folder_path = '/home/sam/bucket/sam/ant_tracking/' + exp_name + '/'
hmats_dir = '/home/sam/bucket/sam/ant_tracking//bundle_adjustment_paras.mat'
pickle_name_aruco_1 = '_aruco_realigned_col1.pkl'
pickle_name_aruco_2 = '_aruco_realigned_col2.pkl'
pickle_name_sleap_1 = '_sleap_realigned_col1.pkl'
pickle_name_sleap_2 = '_sleap_realigned_col2.pkl'

max_aruco_ind=301
aruco_occurance_thresh=500

#load aruco data
df_colony_1 = read_pickle_file_with_load(output_folder_path, exp_name + pickle_name_aruco_1)
df_colony_2 = read_pickle_file_with_load(output_folder_path, exp_name + pickle_name_aruco_2)

#filter for active tags
arucos_col1 = get_aruco_ids(df_colony_1,max_aruco_ind, aruco_occurance_thresh)
arucos_col2 = get_aruco_ids(df_colony_2,max_aruco_ind, aruco_occurance_thresh)

#load sleap data
sleap_col1 = read_pickle_file_with_load(output_folder_path, exp_name + pickle_name_sleap_1)
sleap_col2 = read_pickle_file_with_load(output_folder_path, exp_name + pickle_name_sleap_2)



# Initialize an empty DataFrame to store the cleaned tracks for all Aruco markers
cleaned_aruco_col_1 = pd.DataFrame(columns=arucos_col1.columns)
cleaned_aruco_col_2 = pd.DataFrame(columns=arucos_col2.columns)

grouped_by_aruco_1 = arucos_col1.groupby('ARUCO_number')
grouped_by_aruco_2 = arucos_col2.groupby('ARUCO_number')

#this is too slow!
#Iterate over each Aruco marker group in `grouped_aruco`
for group_name, group_data in grouped_by_aruco_1:
    print(f"Processing Aruco marker: {group_name}")
    
    # Get the optimal track for the current Aruco marker
    aruco_df = get_optimal_track_single_aruco(group_data)
    plot_tracks(group_data, aruco_df, group_name)
    
    # Concatenate the result to `cleaned_aruco` DataFrame
    cleaned_aruco_col_1 = pd.concat([cleaned_aruco_col_1, aruco_df], ignore_index=True)
    
    

