#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:42:47 2024

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
import matplotlib.cm as cm
from collections import defaultdict
import tqdm

def get_Hmats(curr_dir, im_n=25):

    #how to load up parameters of the mapping
    mat = scipy.io.loadmat(curr_dir)  
    paras = np.squeeze(mat['paras'])

    H_pair = [[np.eye(3) if i == j else None for j in range(im_n)] for i in range(im_n)]
    for ii in range(1, im_n):
        currParams = paras[(4*(ii-1)):(4*ii)]
    
        S = np.array([[currParams[0], currParams[1], currParams[2]],
                      [currParams[1], currParams[0], currParams[3]]])
        H_pair[0][ii] = np.vstack([S, [0, 0, 1]])
        H_pair[ii][0] = np.linalg.inv(H_pair[0][ii])
    
    for i in range(1, im_n-1):
        for j in range(i+1, im_n):
            H_pair[i][j] = np.dot(H_pair[0][j], H_pair[i][0])
            H_pair[j][i] = np.linalg.inv(H_pair[i][j])
    
    H_mats = H_pair[12]
    H_mats_flipped = np.flip(H_mats, axis=0)
    return H_mats_flipped
    
def map_points(points, H):
    
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points_homogeneous = homogeneous_points @ H.T  # Matrix multiplication
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, [2]]

    return transformed_points

def map_points_inv(points, H):
    
    
    transformed_points = np.hstack([points, np.ones((points.shape[0], 1))])
    original_points_homogeneous = transformed_points @ np.linalg.inv(H).T  # Matrix multiplication
    original_points = original_points_homogeneous[:, :2] / original_points_homogeneous[:, [2]]

    return original_points

def convert_to_pickle(folder_path, pickle_name, df_to_convert): 
    #Create a new folder with the combined files in the desired directory 
    folder_name = folder_path.split('/')[-2]
    if not os.path.exists(folder_path):
        # If the folder does not exist, create it
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created in '{folder_path}'.")
    else:
        print(f"Folder '{folder_name}' already exists in '{folder_path}'.")

    df_to_convert.to_pickle(folder_path +  pickle_name )
    
    return

def check_calibration(xy_arra, H_mats):
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    cmap = plt.colormaps['viridis']
  
    for curr_cam in range(0,25):
        curr_H=H_mats[curr_cam]
        mapped_points=map_points(xy_array,curr_H)
        plt.plot(mapped_points[0:10000,0],mapped_points[0:10000,1])  
        plt.text(mapped_points[-1,0],mapped_points[-1,1], str(curr_cam))



def combine_aruco(H_mats, curr_aruco_dir, output_folder_path, output_file_name):
    """
    Combines and processes ArUco detection data from multiple pickle files, maps coordinates using
    homography matrices, and saves the final DataFrame.

    Parameters:
        H_mats (list): List of homography matrices, one for each camera.
        aruco_files (list): List of paths to pickle files containing `aruco_detection` dictionaries.
        output_folder_path (str): Path to save the combined output.
        output_file_name (str): Base name for the output files.
    """
    
    pkl_files = sorted(glob.glob(os.path.join(curr_aruco_dir, '*.pkl')))
    aruco_files = [file for file in pkl_files if 'global' not in file]

    grouped_files = defaultdict(lambda: defaultdict(list))

    for file in aruco_files:
        index = file.split('_')[-1].split('.')[0]
        # Extract the camera index: find the 2 numbers after the second 'cam'
        parts = os.path.basename(file).split('cam')
        if len(parts) > 2:
            cam_number = parts[2][:2]  # Take the first two characters after the second 'cam'
        else:
            raise ValueError(f"Filename {file} does not have the required format with two 'cam' occurrences.")
        
        grouped_files[index][cam_number].append(file)

    
    total_aruco_data = []  # Use a list to store DataFrames for concatenation later
    
    for ind, (index, chunk_files) in enumerate(grouped_files.items()):
        for cam_number, file_list in chunk_files.items():
            curr_cam = int(cam_number) - 1  # Adjust cam_number for 0-based indexing
            pkl_file = file_list[0]  # Assuming the first file in the list for this cam is the relevant one
    
            print('Processing file:', pkl_file)
    
            # Load the ArUco detection dictionary
            with open(pkl_file, "rb") as f:
                df_aruco = pickle.load(f)
           
            if df_aruco.empty:
                print(f"No valid data in {pkl_file}, skipping...")
                continue  # Skip empty data
            print(df_aruco['Frame'].iloc[-1])
            # Map points using the homography matrix
            curr_H = H_mats[curr_cam]
            xy_array = df_aruco[['X', 'Y']].to_numpy()
            mapped_points = map_points(xy_array, curr_H)
    
            # Update DataFrame with mapped points
            df_aruco[['X', 'Y']] = mapped_points
    
            # Append to the list for final concatenation
            total_aruco_data.append(df_aruco)
    
    # Concatenate all DataFrames at once
    if total_aruco_data:
        total_df_aruco = pd.concat(total_aruco_data, ignore_index=True)
        output_file = f"{output_file_name[:-4]}_{ind:03d}.pkl"
        convert_to_pickle(output_folder_path, output_file, total_df_aruco)
        print(f"Data saved to {output_file}")
    else:
        print("No valid data found across all groups.")



def combine_sleap(H_mats, curr_dir,output_folder_path, output_file_name):
    # Retrieve all CSV files in the current_directory if they are not frame counts 
    csv_files = sorted(glob.glob(os.path.join(curr_dir, '*.csv')))
    filtered_files = [file for file in csv_files if 'frame_counts' not in file and 'global' not in file]
    

    grouped_files = defaultdict(list)
    for file in filtered_files:
        # Extract the index (e.g., '003') from the filename
        index = file.split('_')[-1].split('.')[0]
        grouped_files[index].append(file)

    for ind, chunk_files in enumerate(grouped_files.items()):
        # Extract sleap files and convert to DataFrame 
        df = pd.read_csv(filtered_files[0])
        df = df.drop(['Score_node'], axis=1)
        total_df = pd.DataFrame(columns=df.columns)
        
        for csv in chunk_files[1]:
            print("File:", csv)
            df = pd.read_csv(csv)
            df = df.drop(['Score_node'], axis=1)
    
            # Reset instance counter for each new frame
            df['Instance'] = df.groupby('Frame').cumcount() - 1
    
            # Add the camera number column
            curr_cam = int(os.path.basename(csv).split('_')[2][-2:]) - 1  # Adjust camera index
            df['Cam'] = curr_cam
            df['Frame'] = df['Frame'] - 1  # Convert to 0-based indexing
            # Map points to the panorama
            curr_H = H_mats[curr_cam]
            xy_array = df[['X', 'Y']].to_numpy()
            mapped_points = map_points(xy_array, curr_H)
            df['X'] = mapped_points[:, 0]
            df['Y'] = mapped_points[:, 1]
    
            total_df = pd.concat([total_df, df], ignore_index=True)
      
        output_file = f"{output_file_name[:-4]}_{ind:03d}.pkl"
        convert_to_pickle(output_folder_path, output_file,total_df)


def combine_aruco_old(H_mats, curr_dir, output_folder_path, output_file_name):
    # Retrieve all .npy files in the current directory if they are not frame counts
    npy_files = glob.glob(os.path.join(curr_dir, '*.npy'))
    npy_files = [file for file in npy_files if 'global' not in file]
    
    grouped_files = defaultdict(list)
    for file in npy_files:
        # Extract the index (e.g., '003') from the filename
        index = file.split('_')[-1].split('.')[0]
        grouped_files[index].append(file)

    for ind, chunk_files in enumerate(grouped_files.items()):
        
        total_aruco_data = []  # Use a list to store DataFrames for concatenation later
    
        # For each camera, map the points to the panorama using H_mats and load new coordinates into a DataFrame
        for npy_file in chunk_files[1]:
            # Extract camera index from the filename
            print('Processing file:', npy_file)
            curr_cam = int(os.path.basename(npy_file).split('_')[2][-2:]) - 1  # Adjust camera index
    
            # Load the tracks
            aruco_tracks = np.load(npy_file)
    
            # Reshape array and create DataFrame
            num_frames, num_arucos, num_positions = aruco_tracks.shape
            reshaped_array = aruco_tracks.reshape((num_arucos * num_frames, num_positions))
            df_aruco = pd.DataFrame(reshaped_array, columns=['X', 'Y'])
            
            # Add frame number, ARUCO number, and camera columns directly
            df_aruco['Frame_number'] = np.repeat(np.arange(num_frames), num_arucos)
            df_aruco['ARUCO_number'] = np.tile(np.arange(num_arucos), num_frames)
            df_aruco['Cam'] = curr_cam
    
            # Filter out rows where both X and Y are zero
            df_aruco = df_aruco[(df_aruco['X'] != 0) | (df_aruco['Y'] != 0)]
    
            # Map points using homography matrix
            curr_H = H_mats[curr_cam]
            xy_array = df_aruco[['X', 'Y']].to_numpy()
            mapped_points = map_points(xy_array, curr_H)
            
            # Update DataFrame with mapped points
            df_aruco[['X', 'Y']] = mapped_points
    
            # Append to the list for final concatenation
            total_aruco_data.append(df_aruco)
    
        # Concatenate all DataFrames at once
        total_df_aruco = pd.concat(total_aruco_data, ignore_index=True)
        output_file = f"{output_file_name[:-4]}_{ind:03d}.pkl"
        convert_to_pickle(output_folder_path, output_file,total_df_aruco)
    

#directories 
exp_name = '20241108_1'
curr_slp_dir='/home/sam/bucket/sam/ant_tracking/20241108_1/slp_files/'
curr_aruco_dir='/home/sam/saionWork/sam/aruco_files/'
#curr_aruco_dir='/home/sam/bucket/Ants/trials/20241108_1/20241108_1_first_hour/data/'
output_folder_path = '/home/sam/bucket/sam/ant_tracking/' + exp_name + '/'
hmats_dir = '/home/sam/bucket/sam/ant_tracking//bundle_adjustment_paras.mat'
aruco_file_name = exp_name + '_aruco_panorama_frame.pkl'
sleap_file_name = exp_name + '_sleap_panorama_frame.pkl'



#check calibration
H_mats=get_Hmats(hmats_dir)
x = np.linspace(0,1000,100)
y = 1000*np.sin(x)
xy_array = np.array([x, y]).T
check_calibration(xy_array.T, H_mats)

#step 1 and 2 can be done automatically, parallelize over chunks

#%%
#step1
combine_aruco(H_mats, curr_aruco_dir,output_folder_path, aruco_file_name)

#step2
combine_sleap(H_mats, curr_slp_dir,output_folder_path, sleap_file_name)




