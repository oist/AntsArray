#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:41:47 2023

@author: Sam

This script performs camera array calibration by detecting and matching keypoints
in videos from different cameras and calculating homography matrices.
"""

import numpy as np
import glob
import cv2
import re
from tqdm import tqdm
import pandas as pd
import scipy.io

# Function to detect and describe keypoints in a grayscale image
def detectAndDescribe(gray_image, descriptor):
    kps, features = descriptor.detectAndCompute(gray_image, None)
    return kps, features

# Function to match keypoints between two images using FLANN-based matcher
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, ransac_threshold, MIN_MATCH_COUNT):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = [m for m, n in rawMatches if m.distance < ratio * n.distance]

    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kpsA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpsB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, np.ones(len(src_pts)), cv2.RANSAC, ransac_threshold)
        goodMatches = mask.ravel().tolist()
        goodSrc = np.squeeze(src_pts[np.array(goodMatches) == 1, :, :])
        goodDst = np.squeeze(dst_pts[np.array(goodMatches) == 1, :, :])

        goodSrc = np.append(goodSrc, np.ones([goodSrc.shape[0], 1]), axis=1)
        goodDst = np.append(goodDst, np.ones([goodDst.shape[0], 1]), axis=1)
    else:
        goodSrc, goodDst, M = None, None, None

    return goodSrc, goodDst, M

# Calibration settings
ransac_threshold = 5.0
ratio = 0.5
MIN_MATCH_COUNT = 10
array_cal_folder = '/home/sam/bucket/Ants/basler/cameraArray_calib/2023-12-26-22-42_AruCo_DICT_6X6_1000_glass/'

# Collecting all camera video files
cam_list = glob.glob(array_cal_folder + "*.avi")

# Extracting camera numbers from file paths
camera_numbers = [int(re.search(r'_cam(\d+)', file_path).group(1)) for file_path in cam_list if re.search(r'_cam(\d+)', file_path)]
sort_inds = np.argsort(camera_numbers)
sorted_cams = [cam_list[i] for i in sort_inds]

# Generating a layout of the camera array
array_size = [5, 5]
cam_layout = np.reshape(np.array(range(25)), (array_size[0], array_size[1]))
cam_neighbors = [cam_layout[max(row - 1, 0):min(row + 2, array_size[0]), max(column - 1, 0):min(column + 2, array_size[1])].ravel() for row in range(array_size[0]) for column in range(array_size[1])]


#how to load up parameters of the mapping
mat = scipy.io.loadmat('/home/sam/bucket/Ants/basler/cameraArray_calib/2023-12-26-22-42_AruCo_DICT_6X6_1000_glass/bundle_adjustment_paras.mat')  
paras = np.squeeze(mat['paras'])

im_n = 25# set this to the number of images or the appropriate value
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




#practice mapping
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cv2 import aruco
from matplotlib import cm
import argparse

def map_points(points, H):
    
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points_homogeneous = homogeneous_points @ H.T  # Matrix multiplication
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, [2]]

    return transformed_points

sleap_file='/home/sam/Videos/predictions/ant_reencode_test.slp.231221_111428.predictions.000_testAnt_out.analysis.h5'
with h5py.File(sleap_file, 'r') as f:
    tracks_matrix = f['tracks'][:] 
    
sleap_instance_num,_,posture_num,frame_num=tracks_matrix.shape
 
 
curr_track=tracks_matrix[0,:,3,0:1000].T
 
cmap = cm.viridis
custom_cmap = cm.get_cmap(cmap, 25)

fig, ax = plt.subplots()
ax.invert_yaxis()
for curr_cam in range(0,25):
    curr_H=H_mats[curr_cam]
    mapped_points=map_points(curr_track,curr_H)
    color = custom_cmap(curr_cam / (25 - 1))
    plt.plot(mapped_points[:,0],mapped_points[:,1],color=color)
             



# # Initialize SIFT descriptor
# descriptor = cv2.SIFT_create()


# #save images from each camera for testing matlab bundle adjustment script
# for cam_num, cam_file in tqdm(enumerate(sorted_cams), total=len(sorted_cams)):
#     cap = cv2.VideoCapture(cam_file)
#     ret, img = cap.read() 
#     cv2.imwrite(array_cal_folder + 'cam_' + f"{cam_num:02}" + '.png', img)
    

# # Detect keypoints and compute features for each camera
# all_kps, all_f = {}, {}
# for cam_num, cam_file in tqdm(enumerate(sorted_cams), total=len(sorted_cams)):
#     cap = cv2.VideoCapture(cam_file)
#     ret, img = cap.read() 
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     k, f = detectAndDescribe(img_gray, descriptor)
#     all_kps[str(cam_num)] = [k]
#     all_f[str(cam_num)] = [f]

# # Compute homography matrices between cameras
# Hmat = np.empty((len(sorted_cams), len(sorted_cams)), dtype=object)
# for cam_num in tqdm(range(len(sorted_cams)), total=len(sorted_cams)):
#     neighbors = cam_neighbors[cam_num]
#     cam_keypts = all_kps[str(cam_num)][0]
#     cam_features = all_f[str(cam_num)][0]

#     for n in neighbors:
#         if n == cam_num:
#             Hmat[cam_num, n] = np.eye(3)
#         elif Hmat[cam_num, n] is None:
#             n_keypts = all_kps[str(n)][0]
#             n_features = all_f[str(n)][0]
#             src_pts, dst_pts, M = matchKeypoints(cam_keypts, n_keypts, cam_features, n_features, ratio, ransac_threshold, MIN_MATCH_COUNT)

#             if M is not None:
#                 Hmat[cam_num, n] = np.vstack([M, [0, 0, 1]])
#                 Hmat[n, cam_num] = np.linalg.inv(Hmat[cam_num, n])

# # Refining the homography matrices
# for _ in range(10):
#     for cam_num in range(len(sorted_cams)):
#         neighbors = [b for b in range(len(sorted_cams)) if Hmat[cam_num, b] is not None]

#         for n1 in neighbors:
#             for n2 in neighbors:
#                 if Hmat[n1, n2] is None or Hmat[n2, n1] is None:
#                     Hmat[n2, n1] = np.dot(Hmat[n2, cam_num], Hmat[cam_num, n1])
#                     Hmat[n1, n2] = np.dot(Hmat[n1, cam_num], Hmat[cam_num, n2])

# # Saving the homography matrices
# df = pd.DataFrame({str(img): Hmat[img] for img in range(len(sorted_cams))})
# df.to_pickle(array_cal_folder + 'cam_homographies.pkl')