#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:17:05 2023

@author: sam
"""
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cv2 import aruco
from matplotlib import cm
import argparse
import pandas as pd

def selective_interpolate(series, max_gap):
    
    # Indices of all elements and NaN elements
    x = np.arange(len(series))
    nan_indices = np.isnan(series)
    
    # Indices where the series is NOT NaN (real values)
    not_nan_indices = np.where(~nan_indices)[0]
    
    # Only interpolate if there are enough real values to interpolate between
    if len(not_nan_indices) == 0:
        return series  # No real values to interpolate from
    
    for i in range(len(not_nan_indices) - 1):
        # Check gap size between consecutive real values
        gap_size = not_nan_indices[i+1] - not_nan_indices[i] - 1
        if 0 < gap_size <= max_gap:
            # Indices to interpolate
            interp_indices = np.arange(not_nan_indices[i] + 1, not_nan_indices[i+1])
            # Perform interpolation
            series[interp_indices] = np.interp(interp_indices, x[~nan_indices], series[~nan_indices])
    
    return series


def csv_to_tracks(sleap_file):
    
    df = pd.read_csv(sleap_file)
    df=df.drop(['Score_node'], axis=1)
    
    num_instances = df['Instance'].max()
    num_bodypoints = df['Bodypoint'].max()
    num_frames = df['Frame'].max()
    
    # Pivot separately for X and Y
    pivot_x = df.pivot_table(index=['Instance', 'Bodypoint'], columns='Frame', values='X', fill_value=np.nan)
    pivot_y = df.pivot_table(index=['Instance', 'Bodypoint'], columns='Frame', values='Y', fill_value=np.nan)
    array_x = pivot_x.values.reshape(num_instances, num_bodypoints, num_frames)
    array_y = pivot_y.values.reshape(num_instances, num_bodypoints, num_frames)
    tracks_matrix = np.stack([array_x, array_y], axis=1)
    
    return tracks_matrix



def csv_to_tracks_float16(sleap_file, chunksize=10000):
    # Initial pass to find maxima
    max_instance, max_bodypoint, max_frame = -1, -1, -1
    for chunk in pd.read_csv(sleap_file, chunksize=chunksize, usecols=['Instance', 'Bodypoint', 'Frame']):
        max_instance = max(max_instance, chunk['Instance'].max())
        max_bodypoint = max(max_bodypoint, chunk['Bodypoint'].max())
        max_frame = max(max_frame, chunk['Frame'].max())

    # Initialize arrays with dimensions +1 to account for zero-based indexing and dtype float16 to save memory
    num_instances = int(max_instance) + 1
    num_bodypoints = int(max_bodypoint) + 1
    num_frames = int(max_frame) + 1
    array_x = np.full((num_instances, num_bodypoints, num_frames), np.nan, dtype=np.float16)
    array_y = np.full((num_instances, num_bodypoints, num_frames), np.nan, dtype=np.float16)

    # Main data processing pass
    for chunk in pd.read_csv(sleap_file, chunksize=chunksize):
        chunk.dropna(subset=['X', 'Y'], inplace=True)  # Ensure rows with NaN in 'X' or 'Y' are not processed
        for _, row in chunk.iterrows():
            instance, bodypoint, frame = int(row['Instance']), int(row['Bodypoint']), int(row['Frame'])
            # Directly update the arrays, converting values to float16
            array_x[instance, bodypoint, frame] = np.float16(row['X'])
            array_y[instance, bodypoint, frame] = np.float16(row['Y'])

    # Stack arrays to form the tracks matrix
    tracks_matrix = np.stack([array_x, array_y], axis=1)
    
    return tracks_matrix



def interpolate_nans(time, positions):
    nan_mask = np.isnan(positions)
    not_nan_indices = np.arange(len(positions))[~nan_mask]
    nan_indices = np.arange(len(positions))[nan_mask]
    positions[nan_indices] = np.interp(nan_indices, not_nan_indices, positions[not_nan_indices])
    return positions

def get_aruco_tracks(video_file,dictionary_size=100):
    
    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracks=np.zeros((length,dictionary_size,2))

    for frame_count in tqdm(range(length),total=length):
      
        ret, img = cap.read() 
        corners,ids,rejected=detector.detectMarkers(img)
        
        if ids is not None:
        
            #get com from corners
            com=[]
            for corner in corners:
                com.append(np.mean(corner[0], axis=0))
                        
            extract_ids=[]
            for i, curr_id in enumerate(ids):
                extract_ids.append(curr_id[0])
            extract_ids=np.array(extract_ids)
            extract_ids=np.unique(extract_ids)
            
            for i, curr_id in enumerate(extract_ids):
                 tracks[frame_count,ids[i][0],:] = [com[i][0],com[i][1]]  
    
    return tracks
             
             
def filter_arcuo_tracks(tracks):
    
    counts=np.sum(tracks[:,:,0]>0,axis=0)
    max_count=np.max(counts)
    aruco_ids=np.where(counts>(max_count/8))[0]  #may need to adjust exclusion criterion    
    id_tracks=tracks[:,aruco_ids,:]
    id_tracks=np.swapaxes(id_tracks,1,0) 
    id_tracks[id_tracks==0]=np.nan

   
    
#filter trajectory, play with different things. Can probably implement a velocity filter
    filtered_tracks=np.zeros_like(id_tracks)

    for i, track in enumerate(id_tracks):
        filtered_tracks[i,:,0]=selective_interpolate(track[:,0],100)
        filtered_tracks[i,:,1]=selective_interpolate(track[:,1],100)       
        
    return filtered_tracks 

             
def link_to_sleap(filtered_tracks, sleap_file, d_thresh):

    
    #get all the info from the df. Filter for frame, match to aruco
    
    df = pd.read_csv(sleap_file)
    sleap_instance_num = df['Instance'].max()
    posture_num = df['Bodypoint'].max()
    frame_num = df['Frame'].max()

    aruco_instance_num, frame_num, _ = filtered_tracks.shape

    refined_tracks=np.nan*np.ones((aruco_instance_num,2,posture_num,frame_num))
    for frame in range(0,frame_num):
        for aruco_ind in range(0, aruco_instance_num):
           
           curr_aruco=filtered_tracks[aruco_ind, frame,:]
           
           
           
           curr_sleaps=tracks_matrix[:,:,8,frame]  #make it automatic to look for the one called aruco!
           distances = np.sqrt(np.sum((curr_sleaps - curr_aruco)**2, axis=1))
           distances[np.isnan(distances)]=d_thresh
           curr_id=np.argmin(distances)
           if np.min(distances)<d_thresh:
              # sleap_id_mat[aruco_ind, frame]=curr_id
               refined_tracks[aruco_ind,:,:,frame]=tracks_matrix[curr_id,:,:,frame]      
               
    return refined_tracks
               


p = argparse.ArgumentParser('arcuo-sleap')
p.add_argument('--video-file')  
p.add_argument('--sleap-file')  
p.add_argument('--d-thresh', default=50, type=int)  
p.add_argument('--plot-output', action='store_true')
p.add_argument('--output-path')  

args = p.parse_args()

#load aruco detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
detectParams=aruco.DetectorParameters()
detectParams.adaptiveThreshConstant=3
detectParams.adaptiveThreshWinSizeMin=10
detectParams.adaptiveThreshWinSizeMax=40
detectParams.adaptiveThreshWinSizeStep=10
detectParams.errorCorrectionRate=1
detector=aruco.ArucoDetector(aruco_dict,detectParams)


# video_file='/home/sam/Videos/testAnt_out.mp4'
# sleap_file='/home/sam/saionWork/ant_tmp/20240117_1/cam1_2024-01-17-10-12-17_cam19_000.csv'
# output_path='/home/sam/Videos/'

tracks=get_aruco_tracks(args.video_file,100) #size of aruco dict

np.save(args.output_path + 'sleap_tracks_1.npy', tracks)  

filtered_tracks=filter_arcuo_tracks(tracks)
refined_tracks=link_to_sleap(filtered_tracks, args.sleap_file, args.d_thresh)
           
np.save(args.output_path + 'sleap_tracks.npy', tracks, filtered_tracks, refined_tracks)   


if args.plot_output:

    frame_num=refined_tracks.shape[3]
    instance_num=refined_tracks.shape[0]
    # Number of rows and columns in the subplot grid
    num_rows = int(np.ceil(np.sqrt(instance_num)))
    num_cols = int(np.ceil(np.sqrt(instance_num)))
    
    # Create a 3x2 subplot
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 12))
    tally=0
    # Example plots to fill the subplots
    for i in range(num_rows):
        for j in range(num_cols):
     
          #  axes[i, j].plot(id_tracks[tally,0:11000,0],id_tracks[tally,0:11000,1])
            axes[i, j].scatter(refined_tracks[tally,0,8,0:frame_num],refined_tracks[tally,1,8,0:frame_num], c=range(0,frame_num), cmap=cm.viridis, marker='o', alpha=0.8)
            axes[i, j].invert_yaxis()
            axes[i, j].set_title('instance' + str(tally))
          
            tally+=1
        


