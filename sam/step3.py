#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 05:02:55 2024

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
import matplotlib as mpl
import os
import pickle
import seaborn as sns
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
# Increase chunksize
mpl.rcParams['agg.path.chunksize'] = 10000  # Adjust as needed
# Optionally enable path simplification
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.1

def show_original_alignment(df, col):
    
    grouped_by_cam = df.groupby('Cam')
    
    plt.figure(figsize=(15, 15))
    
    for cam, data in grouped_by_cam: 
        plt.scatter(data['X'], data['Y'], s = 1, label = cam)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.title('Original aligment for Colony' + str(col))
    plt.show()

def transform_points_with_similarity(points, transform_matrix):
    """
    Transform points using a 2x3 similarity transform matrix.
    
    Args:
        points (ndarray): Points to transform, shape (n, 2).
        transform_matrix (ndarray): 2x3 similarity transform matrix.
    
    Returns:
        ndarray: Transformed points, shape (n, 2).
    """
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply the transform matrix
    transformed_points = points_homogeneous @ transform_matrix.T

    return transformed_points
    
def align_cams(df, camera_order, col):
        
    def find_overlapping_points(ref_data, target_data):
        """
        Find overlapping points between reference and target cameras based on
        matching ARUCO_number and Frame_number.
        
        Args:
            ref_data (DataFrame): Data for the reference camera.
            target_data (DataFrame): Data for the target camera.
        
        Returns:
            DataFrame, DataFrame: Filtered data for reference and target cameras.
        """
             # Filter out rows where ARUCO_number appears more than once per Frame_number
        ref_data_filtered = ref_data[
            ref_data.groupby(['Frame_number', 'ARUCO_number'])['ARUCO_number']
            .transform('size') == 1
        ]
        target_data_filtered = target_data[
            target_data.groupby(['Frame_number', 'ARUCO_number'])['ARUCO_number']
            .transform('size') == 1
        ]
   
        # Merge on ARUCO_number and Frame_number to find matches
        merged_data = pd.merge(
            ref_data_filtered,
            target_data_filtered,
            on=['ARUCO_number', 'Frame_number'],
            suffixes=('_ref', '_target')
        )
        
        # Extract matched points
        ref_points = merged_data[['X_ref', 'Y_ref']]
        target_points = merged_data[['X_target', 'Y_target']]
        return ref_points, target_points

    def align_camera_with_overlap(ref_data, target_data):
        """
        Align target_data to ref_data by minimizing distances for overlapping points.
        
        Args:
            ref_data (DataFrame): Reference camera data with X, Y, ARUCO_number, and Frame_number.
            target_data (DataFrame): Target camera data to align.
        
        Returns:
            tuple: Optimal translation (dx, dy).
        """
        # Find overlapping points
        ref_points, target_points = find_overlapping_points(ref_data, target_data)
        ref_points = ref_points.values
        target_points = target_points.values
        
        def objective(translation):
            dx, dy = translation
            transformed_target = target_points + np.array([dx, dy])
            distances = np.linalg.norm(ref_points - transformed_target, axis=1)
            return np.sum(distances)
    
        # Initial guess for translation (dx=0, dy=0)
        initial_translation = [0, 0]
        result = minimize(objective, initial_translation, method='Powell')
        return result.x  # Optimized dx, dy
    
    def align_camera_with_similarity(ref_points, target_points):
        """
        Align target_points to ref_points by fitting a similarity transformation.
        
        Args:
            ref_points (ndarray): Reference points, shape (n, 2).
            target_points (ndarray): Target points, shape (n, 2).
        
        Returns:
            ndarray: 2x3 similarity transform matrix.
        """
        # Ensure at least 2 points for similarity transform
        if len(ref_points) < 2 or len(target_points) < 2:
            print("Not enough points for similarity transform. Returning identity transform.")
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # Return 2x3 identity matrix
    
        # Compute the similarity transform
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
    target_points,  # Source points
    ref_points,     # Destination points
    method=cv2.RANSAC,  # Optional: Use RANSAC for robust estimation
    ransacReprojThreshold=3.0  # Optional: Set threshold for RANSAC
)
        
        # If no valid transform could be computed, return identity
        if transform_matrix is None:
            print("Failed to compute similarity transform. Returning identity transform.")
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
       # transform_matrix=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return transform_matrix


    # Full alignment process
    realigned_df = df.copy()
    transformation_mats = []

    for i in range(1, len(camera_order)):
        ref_cam = camera_order[i - 1]
        target_cam = camera_order[i]
        
        ref_data = realigned_df[realigned_df['Cam'] == ref_cam]
        target_data = realigned_df[realigned_df['Cam'] == target_cam]
        
        # Find overlapping points
        ref_points, target_points = find_overlapping_points(ref_data, target_data)
        ref_points = ref_points.values
        target_points = target_points.values
        #print(str(ref_cam) + '_' + str(target_cam) + ' ' + str(len(target_points)))
        transform_matrix = align_camera_with_similarity(ref_points, target_points)
        all_points=np.array([realigned_df.loc[realigned_df['Cam'] == target_cam, 'X'].values,realigned_df.loc[realigned_df['Cam'] == target_cam, 'Y'].values]).T

        aligned_points = transform_points_with_similarity(all_points, transform_matrix)
        
        realigned_df.loc[realigned_df['Cam'] == target_cam, 'X'] = aligned_points[:,0]
        realigned_df.loc[realigned_df['Cam'] == target_cam, 'Y'] = aligned_points[:,1]
        transformation_mats.append(transform_matrix)

    # Plotting the aligned data
    plt.figure(figsize=(15, 15))
    for cam, data in realigned_df.groupby('Cam'):
        plt.scatter(data['X'], data['Y'], s=1, label=cam)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Camera ID")
    plt.title('New alignement for Colony ' + str(col))
    plt.tight_layout()
    plt.show()

    return realigned_df, transformation_mats


def apply_transform(df, transformations, camera_order, col,plot=0):
  

    # Copy the DataFrame to avoid modifying the original
    realigned_df = df.copy()

    for i in range(1, len(camera_order)):
        target_cam = camera_order[i]
        
        all_points=np.array([realigned_df.loc[realigned_df['Cam'] == target_cam, 'X'].values,realigned_df.loc[realigned_df['Cam'] == target_cam, 'Y'].values]).T

        aligned_points = transform_points_with_similarity(all_points, transformations[i-1])
        
        realigned_df.loc[realigned_df['Cam'] == target_cam, 'X'] = aligned_points[:,0]
        realigned_df.loc[realigned_df['Cam'] == target_cam, 'Y'] = aligned_points[:,1]

    # Plot the aligned data
    if plot !=0:
        plt.figure(figsize=(15, 15))
        for cam, data in realigned_df.groupby('Cam'):
            plt.scatter(data['X'], data['Y'], s=1, label=cam)
        
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Camera ID")
        plt.title(f'Aligned Data for {col}')
        plt.tight_layout()
        plt.show()

    return realigned_df


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
        
        
        
exp_name = '20241108_1'
curr_dir='/home/sam/bucket/Ants/trials/' + exp_name + '/20241108_1_first_hour/data/'
output_folder_path = '/home/sam/bucket/sam/ant_tracking/' + exp_name + '/'
hmats_dir = '/home/sam/bucket/sam/ant_tracking//bundle_adjustment_paras.mat'
aruco_file_name = exp_name + '_aruco_panorama_frame.pkl'
sleap_file_name = exp_name + '_sleap_panorama_frame.pkl'
colony_split_pixel=2000
col_1, col_2 = 9, 11
camera_order_1 = [20,21,22, 17,16,15, 10,11,12, 7,6,5,0,1,2]
camera_order_2 = [22,23,24, 19,18,17, 12, 13,14, 9,8,7,2,3,4]
camera_order_2 = [4,3,2,7,8,9,14,13,12,17,18,19,24,23,22]

aruco_files=sorted(glob.glob(f"{output_folder_path}{aruco_file_name[:-4]}" +'*'))
sleap_files=sorted(glob.glob(f"{output_folder_path}{sleap_file_name[:-4]}" +'*'))


for aruco_file, sleap_file in zip(aruco_files, sleap_files):
   
    with open(aruco_file, 'rb') as f:
        aruco_df = pickle.load(f)
        
    with open(sleap_file, 'rb') as f:
        sleap_df = pickle.load(f)
    
    df_1_aruco = aruco_df[aruco_df['X'] < colony_split_pixel]
    df_2_aruco = aruco_df[aruco_df['X'] >= colony_split_pixel]

    df_1_sleap = sleap_df[sleap_df['X'] < colony_split_pixel]
    df_2_sleap = sleap_df[sleap_df['X'] >= colony_split_pixel]
    
    if  aruco_file == aruco_files[0]: #find similarity transforms, realign, combine across chunks
       
        aligned_df_1_aruco, transforms_1 = align_cams(df_1_aruco, camera_order_1, col_1)
        aligned_df_2_aruco, transforms_2 = align_cams(df_2_aruco, camera_order_2, col_2)
        aligned_df_1_sleap = apply_transform(df_1_sleap, transforms_1, camera_order_1, col_1)
        aligned_df_2_sleap = apply_transform(df_2_sleap, transforms_2, camera_order_2, col_2)
        
        combined_aruco_1 = aligned_df_1_aruco.copy()
        combined_aruco_2 = aligned_df_2_aruco.copy()
        combined_sleap_1 = aligned_df_1_sleap.copy()
        combined_sleap_2 = aligned_df_2_sleap.copy()

    else: #use the already computed transforms
        aligned_df_1_aruco = apply_transform(df_1_aruco, transforms_1, camera_order_1, col_1) #try after missing camera fix
        aligned_df_2_aruco = apply_transform(df_2_aruco, transforms_2, camera_order_2, col_2)
        aligned_df_1_sleap = apply_transform(df_1_sleap, transforms_1, camera_order_1, col_1)
        aligned_df_2_sleap = apply_transform(df_2_sleap, transforms_2, camera_order_2, col_2)
        
        combined_aruco_1 = pd.concat([combined_aruco_1, aligned_df_1_aruco], ignore_index=True)
        combined_aruco_2 = pd.concat([combined_aruco_2, aligned_df_2_aruco], ignore_index=True)
        combined_sleap_1 = pd.concat([combined_sleap_1, aligned_df_1_sleap], ignore_index=True)
        combined_sleap_2 = pd.concat([combined_sleap_2, aligned_df_2_sleap], ignore_index=True)
   
    convert_to_pickle_with_dump(output_folder_path, exp_name + '_aruco_realigned_col1.pkl', combined_aruco_1)
    convert_to_pickle_with_dump(output_folder_path, exp_name + '_aruco_realigned_col2.pkl', combined_aruco_2)
    convert_to_pickle_with_dump(output_folder_path, exp_name + '_sleap_realigned_col1.pkl', combined_sleap_1)
    convert_to_pickle_with_dump(output_folder_path, exp_name + '_sleap_realigned_col2.pkl', combined_sleap_2)

  