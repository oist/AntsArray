#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:17:05 2023

@author: sam
"""
import cv2
import numpy as np
from tqdm import tqdm
from cv2 import aruco
import argparse
import os

def get_aruco_tracks(video_file,dictionary_size=1000):
    
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




p = argparse.ArgumentParser('arcuo-track')
p.add_argument('--video-file')  
p.add_argument('--output-path')  

args = p.parse_args()
p.add_argument('--video-file')  
p.add_argument('--output-path')  

args = p.parse_args()

basename=os.path.basename(args.video_file)
#load aruco detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
detectParams=aruco.DetectorParameters()
detectParams.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
detectParams.adaptiveThreshConstant=3
detectParams.adaptiveThreshWinSizeMin=10
detectParams.adaptiveThreshWinSizeMax=40
detectParams.adaptiveThreshWinSizeStep=10
detectParams.errorCorrectionRate=1
detector=aruco.ArucoDetector(aruco_dict,detectParams)


tracks=get_aruco_tracks(args.video_file,1000) #size of aruco dict
np.save(args.output_path + basename + 'aruco_tracks_.npy', tracks)  


