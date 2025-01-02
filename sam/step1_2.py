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

    grouped_files = defaultdict(list)
    for file in aruco_files:
        # Extract the camera index: find the 2 numbers after the second 'cam'
        parts = os.path.basename(file).split('cam')
        if len(parts) > 2:
            cam_number = parts[2][:2]  # Take the first two characters after the second 'cam'
        else:
            raise ValueError(f"Filename {file} does not have the required format with two 'cam' occurrences.")
        
        grouped_files[cam_number].append(file)

    #there is a naming bug here!!!

    # Process files for each group    

    total_aruco_data = []  # Use a list to store DataFrames for concatenation later
      
    for ind, chunk_files in enumerate(grouped_files.items()):

        curr_cam = int(chunk_files[0]) -1  # First entry: the group key (e.g., camera index)
        pkl_file = chunk_files[1][0] 
      
        print('Processing file:', pkl_file)
       
      
        #Load the ArUco detection dictionary
        with open(pkl_file, "rb") as f:
            aruco_detection = pickle.load(f)
       
        # Create a DataFrame for all frames in this file
        rows = []
        for frame_number, detections in tqdm.tqdm(aruco_detection.items(), total=len(aruco_detection)):
            for aruco_id, centroids in detections.items():
                if isinstance(centroids, list):  # Multiple centroids for a single ArUco ID
                    for centroid in centroids:
                        if len(centroid) == 2 and np.any(centroid):  # Valid (non-zero) centroid
                            rows.append({
                                'X': centroid[0],
                                'Y': centroid[1],
                                'Frame_number': frame_number,
                                'ARUCO_number': aruco_id,
                                'Cam': curr_cam
                            })
                elif len(centroids) == 2 and np.any(centroids):  # Single centroid
                    rows.append({
                        'X': centroids[0],
                        'Y': centroids[1],
                        'Frame_number': frame_number,
                        'ARUCO_number': aruco_id,
                        'Cam': curr_cam
                    })
            
            # Convert to DataFrame
            df_aruco = pd.DataFrame(rows)
        
            if df_aruco.empty:
                continue  # Skip empty data
    
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
    else:
        print(f"No valid data found for group {ind}")


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
    
            # Map points to the panorama
            curr_H = H_mats[curr_cam]
            xy_array = df[['X', 'Y']].to_numpy()
            mapped_points = map_points(xy_array, curr_H)
            df['X'] = mapped_points[:, 0]
            df['Y'] = mapped_points[:, 1]
    
            total_df = pd.concat([total_df, df], ignore_index=True)
      
        output_file = f"{output_file_name[:-4]}_{ind:03d}.pkl"
        convert_to_pickle(output_folder_path, output_file,total_df)


#directories 
exp_name = '20241108_1'
curr_dir='/home/sam/bucket/Ants/trials/' + exp_name + '/20241108_1_first_hour/data/'
curr_aruco_dir='/home/sam/saionWork/sam/aruco_files/'
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
combine_sleap(H_mats, curr_dir,output_folder_path, sleap_file_name)




