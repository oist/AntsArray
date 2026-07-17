#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:13:48 2023

@author: sam
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.signal import welch,convolve
import pandas as pd
import glob
from pathlib import Path
from DemoReadSGLXData.readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI
import os
from scipy.interpolate import interp1d
try:
    import nibabel as nb
except ModuleNotFoundError:
    nb = None
import h5py
from scipy.ndimage import gaussian_filter,distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label, find_objects, gaussian_filter,zoom
from scipy.signal import medfilt2d
import numpy as np
from skimage.morphology import binary_closing, binary_erosion,binary_opening,ball
from skimage.measure import label, regionprops
import random
#import pywt
#from rfest.utils import build_design_matrix
import cv2

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist, squareform




def resize_mat(mat,target_size):
            
        # Define the grid for the original data
    x = np.linspace(0, 1, mat.shape[1])
    y = np.linspace(0, 1, mat.shape[0])
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Create the interpolation function
    interp_func = RegularGridInterpolator((y, x), mat, method='linear')
    
    # Create the new grid for interpolation
    x_new = np.linspace(0, 1, target_size[1])
    y_new = np.linspace(0, 1, target_size[0])
    grid_x_new, grid_y_new = np.meshgrid(x_new, y_new)
    points_new = np.array([grid_y_new.ravel(), grid_x_new.ravel()]).T
    
    # Interpolate the data
    mat_resized = interp_func(points_new).reshape(target_size)
    
    return mat_resized
        
        
def load_nifti(src: str):
    """
    Load nifti image. Array is rearranged so that it aligns with TIFF or HDF5 format.
    Parameters
    ----------
    src : str
    """
    if nb is None:
        raise ImportError("nibabel is required for load_nifti()")
    stack = nb.load(src).get_fdata()
    # the below line is needed to make it consistent with TIFF array order
    stack = np.swapaxes(stack, 0, 2)
    return stack



def add_path_prefix(paths, on_deigo):
    """
    Add path prefixes based on the specified platform.

    Parameters:
    - paths (list): List of paths to be prefixed.
    - on_deigo (bool): Flag indicating whether the platform is on Deigo.

    Returns:
    - refined_paths (list): List of refined paths with added prefixes.
    """
    refined_paths = []
    for path in paths:
        if path:
            if on_deigo:
                path_prefix = '/bucket/ReiterU'
            else:
                path_prefix = '/home/sam/bucket'
                
            path_split = path.split('/')
            for s in path_split[3:]:
                path_prefix = path_prefix + '/' + s
                
            refined_paths.append(path_prefix) 
        
    return refined_paths


def load_data(exp, probe, data_type, chanList, tStart, tEnd):
    """
    Load experimental data.

    Parameters:
    - exp (dict): Experiment information dictionary.
    - data_type (str): Type of data to load ('lfp' or 'ap').
    - chanList (list): List of channel indices.
    - tStart (float): Start time for data extraction.
    - tEnd (float): End time for data extraction.

    Returns:
    - convData (numpy.ndarray): Converted data matrix.
    - d_times (numpy.ndarray): Time vector.
    """
    if data_type == 'lfp':
        binFullPath = Path(sorted(glob.glob(exp['curr_experiment_path'] + '/raw_data/**/**/*lf.bin'))[probe])
    elif data_type == 'ap':
        binFullPath = Path(sorted(glob.glob(exp['curr_experiment_path'] + '/raw_data/**/**/*ap.bin'))[probe])
    elif data_type == 'ni':
        binFullPath = Path(sorted(glob.glob(exp['curr_experiment_path'] + '/*nidq.bin'))[0])    
        
    meta = readMeta(binFullPath)    
    sRate = SampRate(meta)
    firstSamp = int(sRate*tStart)
    lastSamp = int(sRate*tEnd)
    rawData = makeMemMapRaw(binFullPath, meta)
    selectData = rawData[chanList, firstSamp:lastSamp+1]
    
    if meta['typeThis'] == 'imec':
        # apply gain correction and convert to uV
        convData = 1e6 * GainCorrectIM(selectData, chanList, meta)
    else:
        # apply gain correction and convert to mV
        convData = 1e3 * GainCorrectNI(selectData, chanList, meta)
     
    d_times = np.array(range(0, convData.shape[1])) / sRate   
    return convData, d_times, sRate, meta


def lfp_preprocessing(convData,d_times,srate,out_of_brain_s,out_of_brain_e,spatialDS,tempDS,resamp=True):
    
    out_of_brain_s=int(out_of_brain_s/spatialDS)
    out_of_brain_e=int(out_of_brain_e/spatialDS)
    new_data=convData[0::spatialDS,0::tempDS]
    filt_data=butter_highpass(new_data, 0.1, srate/tempDS, order=4)
    if resamp is True:
        oob_sig=np.mean(filt_data[out_of_brain_s:out_of_brain_e],axis=0)
        reref_data=filt_data-oob_sig[np.newaxis,:]
    else:
        reref_data=filt_data
    new_times=d_times[0::tempDS]
    
    return reref_data, new_times


def resample_mat(mat, curr_t, desired_fs, interp_kind='linear'):
    """
    Resample a data matrix to the desired sampling rate.

    Parameters:
    - mat (numpy.ndarray): Input data matrix.
    - curr_t (numpy.ndarray): Time vector of the input data.
    - desired_fs (float): Desired sampling rate.

    Returns:
    - new_mat (numpy.ndarray): Resampled data matrix.
    - resampled_times (numpy.ndarray): Resampled time vector.
    """
    resampled_times = np.arange(int(curr_t[0]), int(curr_t[-1]) + desired_fs, desired_fs)
    interp_func = interp1d(curr_t, mat.T, axis=0, kind=interp_kind, fill_value='extrapolate')
    new_mat = interp_func(resampled_times)
    
    return new_mat.T, resampled_times


def bpp(times, ff_ms, lf_ms):
    """
    Bin and count spikes within specified time intervals.

    Parameters:
    - times (numpy.ndarray): Spike times.
    - ff_ms (float): First frame time in milliseconds.
    - lf_ms (float): Last frame time in milliseconds.

    Returns:
    - spike_numbers (numpy.ndarray): Spike counts within each time bin.
    - bin_edges (numpy.ndarray): Bin edges.
    """
    first_frame_ms = int(ff_ms)
    last_frame_ms = int(lf_ms) + 2 #This works for ms bins, I guess I'll need to change for other stuff
    ms_bins = np.array(range(first_frame_ms, last_frame_ms))
    spike_numbers, bin_edges = np.histogram((times).astype(np.uint64), bins=ms_bins) 

    return spike_numbers, bin_edges


def get_electrode_pos(num_rows=2, num_columns=192, pitch=20):
    """
    Generate electrode positions based on the specified layout.

    Parameters:
    - num_rows (int): Number of rows in the layout.
    - num_columns (int): Number of columns in the layout.
    - pitch (int): Spacing between electrodes.

    Returns:
    - electrode_positions (numpy.ndarray): Array of electrode positions.
    """
    electrode_positions = np.zeros((num_columns, 1))
    
    for col in range(num_columns):
        electrode_index = col
        electrode_positions[electrode_index, 0] = col * pitch  # Assuming 20 micrometers spacing
    
    electrode_positions = np.tile(electrode_positions, num_rows).ravel()
    return electrode_positions


def cross_correlation(signal1, signal2, lag_max):
    """
    Compute the normalized cross-correlation between two signals and return corresponding lags.
    Parameters:
    signal1 (array_like): First input signal.
    signal2 (array_like): Second input signal, should have the same length as signal1.
    lag_max (int): Maximum lag for which to calculate the correlation.
    Returns:
    lags (array_like): Array of lags at which the correlation is computed.
    correlation (array_like): Normalized cross-correlation of the two signals for different lags.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Both signals must have the same length.")
    
    # Normalize the signals by subtracting mean and dividing by standard deviation
    signal1_norm = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)
    
    # Compute cross-correlation
    correlation_full = np.correlate(signal1_norm, signal2_norm, 'full')
    
    # Get the length and midpoint of the full correlation
    len_corr = len(correlation_full)
    mid_point = len_corr // 2
    
    # Select the relevant part of the correlation
    correlation = correlation_full[mid_point - lag_max : mid_point + lag_max + 1]
    
    # Normalize to ensure maximum correlation is 1
    correlation /= np.max(np.abs(correlation))
    
    # Generate lags
    lags = np.arange(-lag_max, lag_max + 1)
    
    return lags, correlation


def load_ephys_results(curr_experiment_path, exp_flag, load_behavior_analysis=False, curr_anatomy_path=False,atlas_10=None,dt1=None):
    """
    Load ephys results for a specific experiment.

    Parameters:
    - curr_ephys_path (str): Path to the root folder containing ephys data.
    - exp_flag (int): Experiment flag indicating the type of experiment.
    - load_behavior_analysis (bool): Flag to load behavior analysis results.

    Returns:
    - exp (dict): Dictionary containing experiment information and results.
    """
    
    #exp flags:
        #0: SN
        #1: DG
        #2: Gabors
        #3: DN
        #4: behavior
        #5: Habituation stim
        #6: 1d walk
      
    atlas_path='/home/sam/bucket/DBS/atlas/S_lessoniana_OL_v0.3/segmentation/atlas_seg10.nii.gz' 
    dt_path = '/home/sam/bucket/DBS/atlas/S_lessoniana_OL_v0.3/segmentation/d_ogl.nii.gz'  
   

    # Create an analysis folder for the current experiment
    analysis_path = make_analysis_folder(curr_experiment_path,'analysis_folder')

    # Load stimulus times
    if exp_flag != 4:
        stim_times = np.load(curr_experiment_path + '/STIM_WINDOWS.npy', allow_pickle=True)
    
    else:
        stim_times=[]
        
    # Initialize the experiment dictionary
    exp = {}
    
    # Load experiment-specific files
    if exp_flag == 0:
        sn_mat = np.load(glob.glob(curr_experiment_path + '/visual_stim/PROJECTOR_SN*')[0], allow_pickle=True)
      #  sn_mat = np.load(glob.glob(curr_experiment_path + '/analysis_folder/PROJECTOR_SN*')[0], allow_pickle=True)
      #  sn_mat = pd.read_csv(glob.glob(curr_experiment_path + '/visual_stim/SPARSE_NOISE*.csv')[0])
        exp['sn_mat'] = sn_mat
    
    elif exp_flag == 1: 
        dg_params = pd.read_csv(glob.glob(curr_experiment_path + '/visual_stim/DRIFTING_GRATINGS*.csv')[0])
        dg_params = dg_params[0:stim_times.shape[0]]  # Adjust to match the number of stim_times
        exp['dg_params'] = dg_params
        
    elif exp_flag == 2:
       # gabor_params = pd.read_csv(glob.glob(curr_experiment_path + '/visual_stim/GABOR_FIELDS*.csv')[0])
        #gabor_pos = np.load(curr_experiment_path + '/visual_stim/ACER_4DEG.npy', allow_pickle=True)
        #gabor_ori = np.load(curr_experiment_path + '/visual_stim/ACER_4DEG_ORIENTATIONS.npy', allow_pickle=True)
        
        gabor_pos = np.load(glob.glob(curr_experiment_path + '/visual_stim/TI_*DEG*.npy')[0], allow_pickle=True)
        gabor_ori  = np.load(glob.glob(curr_experiment_path + '/visual_stim/*ORIENT*')[0], allow_pickle=True)
        
     #   exp['gabor_params'] = gabor_params
        exp['gabor_pos'] = gabor_pos
        exp['gabor_ori'] = gabor_ori
     
    elif exp_flag == 3:
         dn_mat = np.load(glob.glob(curr_experiment_path + '/visual_stim/*PROJ*')[0], allow_pickle=True)
         exp['dn_mat'] = dn_mat
    
    elif exp_flag == 4:
         dummy=1
         
    elif exp_flag == 5:
         dn_mat = np.load(glob.glob(curr_experiment_path + '/visual_stim/*LIM*')[0], allow_pickle=True)
         exp['dn_mat'] = dn_mat
         
    elif exp_flag == 6:
         mat = np.load(glob.glob(curr_experiment_path + '/visual_stim/*1D*')[0], allow_pickle=True)
         exp['mat'] = mat
         
    elif exp_flag == 7:
       #  mat = np.load(glob.glob(curr_experiment_path + '/visual_stim/*OPTIC_FLOW*')[0], allow_pickle=True)
         exp['mat'] =  pd.read_csv(glob.glob(curr_experiment_path + '/visual_stim/OPTIC_FLOW*.csv')[0])
 
    elif exp_flag ==8:
        
         exp['isi'] =  np.load(glob.glob(curr_experiment_path + '/visual_stim/*ISI*')[0], allow_pickle=True)
         exp['pol_powers'] =  np.load(glob.glob(curr_experiment_path + '/visual_stim/*pol_powers*')[0], allow_pickle=True)
         exp['powers'] =  np.load(glob.glob(curr_experiment_path + '/visual_stim/powers.npy')[0], allow_pickle=True)
         exp['params'] = pd.read_pickle(glob.glob(curr_experiment_path + '/visual_stim/params.pkl')[0]) 
         
         
    elif exp_flag ==9: #LUM POL SN
        sn_mat_lum = np.load(glob.glob(curr_experiment_path + '/visual_stim/SN_LUM*')[0], allow_pickle=True)
        sn_mat_pol = np.load(glob.glob(curr_experiment_path + '/visual_stim/SN_POL*')[0], allow_pickle=True)
        cond = np.load(glob.glob(curr_experiment_path + '/visual_stim/COND_VECTOR*')[0], allow_pickle=True)
        exp['sn_mat_lum'] = sn_mat_lum
        exp['sn_mat_pol'] = sn_mat_pol
        exp['cond'] = cond
        
        
    elif exp_flag ==10: #texures
        image_list = np.load(glob.glob(curr_experiment_path + '/visual_stim/image_sequence*')[0], allow_pickle=True)
        #get image sorting
        import re

        # Regular expression to match filenames like 'img_001_v_25.png'
        pattern = re.compile(r"^img_(\d+)_v_(\d+)\.png$")

               # Initialize a dictionary with the three required keys
        images_dict = {
            'image_list': [],
            'image_id': [],
            'version': []
        }
        
       
        for filename in image_list:
            images_dict['image_list'].append(filename)
            
            match = pattern.match(filename)
            if match:
                # Extract as integers
                img_id = int(match.group(1))
                ver_id = int(match.group(2))
            else:
                # If it doesn't match the pattern, use None or 0 (depending on your needs)
                img_id = None
                ver_id = None
            
            images_dict['image_id'].append(img_id)
            images_dict['version'].append(ver_id)
        
        exp['image_seq'] = images_dict   
    
    elif exp_flag ==11: #lum pol shrimp
            exp['lum_vals'] = np.load(glob.glob(curr_experiment_path + '/visual_stim/LUM_VALUES*')[0], allow_pickle=True)
            exp['pol_vals'] = np.load(glob.glob(curr_experiment_path + '/visual_stim/POL_VALUES*')[0], allow_pickle=True)
            
    
    # Find paths for used probes
    used_probes_paths = sorted(glob.glob(curr_experiment_path + '/post_processed/' + '*probe*'))
    
    st_mat = []
    unit_chans = []
    # good_units = []
    # Load spike times and chan nums for each used probe
    for curr_probe_path in used_probes_paths:
        if len(os.listdir(curr_probe_path)) != 0:
            st_mat.append(np.load(curr_probe_path + '/spike_t_matrix.npy', allow_pickle=True))
            unit_chans.append(np.load(curr_probe_path + '/g_channels.npy', allow_pickle=True))
         #   good_units.append(np.load(curr_probe_path + '/qc_pass.npy', allow_pickle=True))
        else:
            unit_chans.append([])
            st_mat.append([])
         #   good_units.append([])
         #   
    # Load frame times
    frame_times = []
    
    # Load behavior analysis results if specified
    movement_energy = []
    # proc=[]
    

    #unit quality data
    data_folder=os.path.dirname(curr_experiment_path)

    if os.path.exists(data_folder + '/UNIT_METRICS/'): 

        probe_folders=sorted(glob.glob(data_folder + '/UNIT_METRICS/probe*'))
        

        for i, probe_folder in enumerate(probe_folders):
            
            probe_key = 'probe' + str(i)
            
            if 'unit_quality' not in exp:
                exp['unit_quality'] = {}
            if probe_key not in exp['unit_quality']:
                exp['unit_quality'][probe_key] = {}
                
            unit_amps = np.load(probe_folder + '/UNIT_METRICS/amps.npy', allow_pickle=True)
            unit_isiviol = np.load(probe_folder + '/UNIT_METRICS/isi_viols.npy', allow_pickle=True)
            unit_presence_ratios = np.load(probe_folder + '/UNIT_METRICS/presence_ratios_60s.npy', allow_pickle=True)
            real_chans = np.load(probe_folder + '/UNIT_METRICS/real_channels.npy', allow_pickle=True)
            unit_SNRs = np.load(probe_folder + '/UNIT_METRICS/SNRs.npy', allow_pickle=True)
         #   unit_waveforms = np.load(data_folder + '/UNIT_METRICS/waveforms.npy', allow_pickle=True)
            
         #   import pdb; pdb.set_trace()
          #  unit_chans[i]=real_chans  #overwrite unit_chans as this is a more accurate estimate
         
           # import pdb; pdb.set_trace()
            
            exp['unit_quality'][probe_key]['amps'] = unit_amps
            exp['unit_quality'][probe_key]['isi_viol'] = unit_isiviol
            exp['unit_quality'][probe_key]['presence_ratios'] = unit_presence_ratios
          #  exp['unit_quality'][probe_key]['unit_chans'] = real_chans
            exp['unit_quality'][probe_key]['SNRs'] = unit_SNRs
          #  exp['unit_quality'][probe_key]['waveforms'] = unit_waveforms
            
     
            exp['unit_quality'][probe_key]['good_units'] = check_unit_qual(exp['unit_quality'][probe_key])
    
    if load_behavior_analysis:
        # Load frame times
        frame_times = np.load(curr_experiment_path + '/CAM_TTLS.npy', allow_pickle=True)
        
        # proc = np.load(glob.glob(curr_experiment_path + '/video_files/*me')[0], allow_pickle=True).item()
       # vid_file=h5py.File(glob.glob(curr_experiment_path + '/analysis_folder/*.me')[0],'r')
       # movement_energy=vid_file['int'][:]
        
    if curr_anatomy_path:

        probe_info_files=sorted(glob.glob(curr_anatomy_path +'/analysis/probe_skeleton/probe_*transformed.csv'))
        probe_info=[]
        unit_anatomy=[] 
        elec_depths=[]  
        elec_positions=[]
        elec_brs=[] 
        

        for i,probe_info_file in enumerate(probe_info_files):

            if len(unit_chans[i])>0: #check if the probe has any units
                df=pd.read_csv(probe_info_file)
                # Filter out rows where 'channel' is less than 0
                df = df[df['channel'] >= 0]
                probe_info.append(df)
                unit_depths,elec_depth, elec_pos, elec_br=calculate_unit_positions(atlas_path,dt_path,df,unit_chans[i],atlas_10, dt1)
                elec_depths.append(elec_depth)
                elec_positions.append(elec_pos)
                unit_anatomy.append(unit_depths)
                elec_brs.append(elec_br)
            else:
                unit_anatomy.append([])
                elec_depths.append([])
                
        exp['probe_info'] = probe_info
        exp['unit_anatomy'] = unit_anatomy
        exp['elec_depths'] = elec_depths
        exp['elec_positions'] = elec_positions
        exp['elec_brs'] = elec_brs
    # Populate the experiment dictionary
  #  exp['curr_ephys_path'] = curr_ephys_path
    exp['curr_experiment_path'] = curr_experiment_path
    exp['analysis_path'] = analysis_path
    exp['stim_times'] = stim_times
    exp['st_mat'] = st_mat
    exp['unit_chans'] = unit_chans
    #exp['good_units'] = good_units
    exp['frame_times'] = frame_times
    #exp['proc'] = proc
    exp['movement_energy'] = movement_energy
    
    return exp

def check_unit_qual(unit_qual, ampThresh=0, isiThresh=5, presenceThresh=0, SNRthresh=5):
    """
    Filters unit indices based on quality thresholds.
    
    Parameters:
    unit_qual (dict): Dictionary containing unit quality metrics.
    ampThresh (float): Amplitude threshold.
    isiThresh (float): ISI violation threshold.
    presenceThresh (float): Presence ratio threshold.
    SNRthresh (float): Signal-to-noise ratio threshold.
    
    Returns:
    list: Indices of units that pass all thresholds.
    """
    good_units = []
    
    amps = unit_qual.get('amps', np.array([]))
    isi_viol = unit_qual.get('isi_viol', np.array([]))
    presence_ratios = unit_qual.get('presence_ratios', np.array([]))
    SNRs = unit_qual.get('SNRs', np.array([]))
    
    for idx in range(len(amps)):
        if (amps[idx] >= ampThresh and
            isi_viol[idx] <= isiThresh and
            presence_ratios[idx] >= presenceThresh and
            SNRs[idx] >= SNRthresh):
            good_units.append(idx)
    
    return good_units



def calculate_unit_positions(atlas_path,dt_path,df,unit_chans,atlas_10, dt1, voxel_size=10):
   
   
    if atlas_10 is None:
        atlas_10 = load_nifti(atlas_path)
    
    if dt1 is None:
       dt1 = load_nifti(dt_path)
        
      # dt1=np.load('/home/sam/bucket/DBS/atlas/S_lessoniana_OL_v0.4/tree_diameter.npy')
    
    elec_pos=[]
    for (i, row) in df.iterrows():
        x = int(row["X"] / voxel_size)
        x = np.clip(x, 0, atlas_10.shape[2])
        y = int(row["Y"] / voxel_size)
        y = np.clip(y, 0, atlas_10.shape[1])
        z = int(row["Z"] / voxel_size)
        z = np.clip(z, 0, atlas_10.shape[0])
        elec_pos.append([z,y,x])
    elec_pos=np.array(elec_pos)

   # medulla_mask=atlas_10==1
   # medulla_mask = binary_erosion(medulla_mask, ball(5)) #use newer atlas and expand the central brain a bit
  # dt1 = distance_transform_edt(atlas_10 != 5.) 
   
    elec_depth=[]
    probe_br=[]
    for e in range(0, elec_pos.shape[0]):
        curr_e=elec_pos[e]
        probe_br.append(atlas_10[curr_e[0],curr_e[1],curr_e[2]])
        elec_depth.append(dt1[curr_e[0],curr_e[1],curr_e[2]])
    probe_br=np.array(probe_br)
    elec_depth=np.array(elec_depth)
    


    #I get some brain regions as 0, this will show up as outside the OL but inside the brain in next atlas
    #1 Medulla, 2 peduncle lobe
    unit_br=[]
    unit_pos=[]
    unit_td=[]
   
   
    for un in range(0, len(unit_chans)): 
        try:
            curr_pos=elec_pos[unit_chans[un]]
        except:
             curr_pos=elec_pos[-1] #unit out of the brain, assign to top channel for now. Shouldnt happen

        unit_br.append(atlas_10[curr_pos[0],curr_pos[1],curr_pos[2]])
        unit_pos.append(curr_pos)
        unit_td.append(dt1[curr_pos[0],curr_pos[1],curr_pos[2]]) #true depth
    unit_br=np.array(unit_br)
    unit_td=np.array(unit_td)
   # unit_pos=np.array(unit_pos)
   

    
   
    unit_anatomy = pd.DataFrame()
    unit_anatomy['channel']=unit_chans
    unit_anatomy['brain_region']=unit_br
    unit_anatomy['pos']=unit_pos
    unit_anatomy['dist']=unit_td
 
    

    return(unit_anatomy,elec_depth, elec_pos, probe_br)
    
   
def make_analysis_folder(exp_path, folder_name):
    """
    Create an analysis folder within the specified experiment path.

    Parameters:
    - exp_path (str): Path to the experiment.

    Returns:
    - analysis_folder (str): Path to the created analysis folder.
    """
    analysis_folder = exp_path + '/' + folder_name
    if not os.path.exists(analysis_folder):
        # If not, create the folder
        os.makedirs(analysis_folder)
        print(f"Folder '{analysis_folder}' created.")
    else:
        print(f"Folder '{analysis_folder}' already exists.")
    return analysis_folder

def compute_STA(X, y, dims):
    """
    Compute the spike-triggered average.

    Parameters:
    - X (numpy.ndarray): Input data matrix.
    - y (numpy.ndarray): Binary spike train.
    - dims (tuple): Dimensions of the input data matrix.

    Returns:
    - w_STA (numpy.ndarray): Spike-triggered average.
    """
    if len(X.shape) == 1:
        X = X[:, None]

    if len(y.shape) == 1:
        y = y[:, None]

    n_spikes = np.sum(y)
    w_STA = X.T @ y / n_spikes if n_spikes != 0 else np.zeros((X.shape[1], 1))
    

    return w_STA.reshape(dims)


import numpy as np



def plot_STA(w_sta, dt):
    """
    Plot the spike-triggered average.

    Parameters:
    - w_sta (numpy.ndarray): Spike-triggered average.
    - dt (float): Time step.

    Returns:
    - fig: Matplotlib figure.
    """
    tt = np.arange(-w_sta.shape[0] * dt, dt, dt)
    w_sta=w_sta-np.mean(w_sta) #make it zero mean for nicer plotting
    
    vmax = np.max([np.abs(w_sta.min()), w_sta.max()])*.75
    fig, ax = plt.subplots(1, w_sta.shape[0], figsize=(25, 3))
    for i in range(w_sta.shape[0]):
        ax[i].imshow(gaussian_filter(w_sta[i],1), cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f't={tt[i]:.02f}')
        if i == 0:
            ax[i].set_ylabel('STA')
    fig.tight_layout()

    return fig
 

def smooth(y, box_pts):
    """
    Smooth a signal using a box filter.

    Parameters:
    - y (numpy.ndarray): Input signal.
    - box_pts (int): Size of the box filter.

    Returns:
    - y_smooth (numpy.ndarray): Smoothed signal.
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def interpolate_nans(time, positions):
    """
    Interpolate NaN values in a vector.

    Parameters:
    - time (numpy.ndarray): Time vector.
    - positions (numpy.ndarray): Vector with NaN values.

    Returns:
    - positions (numpy.ndarray): Vector with interpolated NaN values.
    """
    nan_mask = np.isnan(positions)
    not_nan_indices = np.arange(len(positions))[~nan_mask]
    nan_indices = np.arange(len(positions))[nan_mask]
    positions[nan_indices] = np.interp(nan_indices, not_nan_indices, positions[not_nan_indices])
    return positions

def get_event_trig_avg(sig, event_inds, backlag, forwardlag):
    """
    Calculate the event-triggered average.

    Parameters:
    - sig (numpy.ndarray): Input signal.
    - event_inds (numpy.ndarray): Indices of events.
    - backlag (int): Backward time lag.
    - forwardlag (int): Forward time lag.

    Returns:
    - ev_avg (numpy.ndarray): Event-triggered average.
    - ev_mat (numpy.ndarray): Event-triggered matrix.

    """
    event_inds = np.round(event_inds).astype(int) 
    if sig.ndim==1:
        sig=np.expand_dims(sig,0)
        
    min_nevents = 1  # minimum number of events where we will even compute a triggered avg

    orig_size = sig.shape

    lags = np.arange(-backlag, forwardlag + 1)

    # get rid of events that happen within the lag-range of the end points
    bad_ids = np.where(event_inds <= backlag)[0]
    if len(bad_ids) > 0:
        print(f'Dropping {len(bad_ids)} early events')
        event_inds = np.delete(event_inds, bad_ids)

    bad_ids = np.where(event_inds >= (orig_size[1] - forwardlag))[0]
    if len(bad_ids) > 0:
        print(f'Dropping {len(bad_ids)} late events')
        event_inds = np.delete(event_inds, bad_ids)

    n_events = len(event_inds)
    # check that we have at least the minimum number of events to work with
    if n_events < min_nevents:
        ev_avg = np.full((orig_size[0], len(lags)), np.nan)
        ev_mat = np.nan
        return ev_avg, ev_mat

    ev_avg = np.zeros((orig_size[0], len(lags)))
    ev_mat = np.zeros((n_events, orig_size[0], len(lags)))

    for i in range(n_events):
        cur_ids = np.arange(event_inds[i] - backlag, event_inds[i] + forwardlag + 1)
        temp_sig = sig[:, cur_ids]
        ev_avg += temp_sig
        ev_mat[i,:, :] = temp_sig

    ev_avg /= n_events

    return np.squeeze(ev_avg), np.squeeze(ev_mat)

def butter_highpass(data, cutoff_freq, fs, order=4):
    """
    Apply a high-pass Butterworth filter to the input data.

    Parameters:
    - data (numpy.ndarray): Input data.
    - cutoff_freq (float): Cutoff frequency of the high-pass filter.
    - fs (float): Sampling frequency.
    - order (int, optional): Order of the Butterworth filter. Defaults to 4.

    Returns:
    - filtered_data (numpy.ndarray): Filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    if data.ndim==2:
        filtered_data = filtfilt(b, a, data, axis=1)
    else:
        filtered_data = filtfilt(b, a, data)
    return filtered_data

def butter_bandpass(data, lowcut, highcut, fs, order=4):
    """
    Apply a band-pass Butterworth filter to the input data.
 
    Parameters:
    - data (numpy.ndarray): Input data.
    - lowcut (float): Lower cutoff frequency of the band-pass filter.
    - highcut (float): Upper cutoff frequency of the band-pass filter.
    - fs (float): Sampling frequency.
    - order (int, optional): Order of the Butterworth filter. Defaults to 4.

    Returns:
    - filtered_data (numpy.ndarray): Filtered data.
    """
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    if data.ndim==2:
        filtered_data = filtfilt(b, a, data, axis=1)
    else:
        filtered_data = filtfilt(b, a, data)
        
    return filtered_data

def butter_lowpass(data, highcut, fs, order=4):
    """
    Apply a band-pass Butterworth filter to the input data.

    Parameters:
    - data (numpy.ndarray): Input data.
    - lowcut (float): Lower cutoff frequency of the band-pass filter.
    - highcut (float): Upper cutoff frequency of the band-pass filter.
    - fs (float): Sampling frequency.
    - order (int, optional): Order of the Butterworth filter. Defaults to 4.

    Returns:
    - filtered_data (numpy.ndarray): Filtered data.
    """
    b, a = butter(order, highcut, btype='low', fs=fs)
    if data.ndim==2:
        filtered_data = filtfilt(b, a, data, axis=1)
    else:
        filtered_data = filtfilt(b, a, data)
        
    return filtered_data

def calculate_csd(lfp_data, electrode_spacing):
    """
    Calculate the current source density (CSD) from data.

    Parameters:
    - lfp_data (numpy.ndarray): LFP data matrix.
    - electrode_spacing (float): Spacing between electrodes.

    Returns:
    - csd (numpy.ndarray): Calculated CSD.
    """
    # Calculate second spatial derivative along the rows
    csd = np.diff(np.diff(lfp_data, axis=0), axis=0) / electrode_spacing**2

    # Pad the result to maintain the same size as the input LFP
    csd = np.concatenate((np.zeros((1, lfp_data.shape[1])), csd, np.zeros((1, lfp_data.shape[1]))), axis=0)

    return csd


from scipy.signal import welch, get_window

def calculate_power_spectral_density(
    signal,
    sampling_rate,
    nperseg=1024,
    noverlap=None,
    window='hann',
    detrend='constant',
    nfft=None
):
    """
    Calculate the power spectral density (PSD) using the Welch method with
    parameters optimized for better frequency resolution.

    Parameters
    ----------
    signal : numpy.ndarray
        1D array representing the LFP signal.
    sampling_rate : float
        Sampling rate of the signal in Hz.
    nperseg : int, optional
        Length of each segment. A larger segment yields finer frequency resolution.
        Defaults to 1024.
    noverlap : int or None, optional
        Number of points to overlap between segments. If None, it defaults to 50% of nperseg.
    window : str or tuple or array_like, optional
        Desired window to use to reduce spectral leakage (e.g., 'hann', 'hamming', etc.).
        Can also provide a user-defined window array or tuple for window function. 
        Defaults to 'hann'.
    detrend : str or function, optional
        Specifies how to detrend each segment. Defaults to 'constant'.
    nfft : int or None, optional
        Length of the FFT used; if None, the FFT length is `nperseg`. Larger values
        provide zero-padding and can help with interpolating the PSD at finer
        frequency intervals. Defaults to None.

    Returns
    -------
    frequencies : numpy.ndarray
        Array of sample frequencies.
    psd : numpy.ndarray
        Power spectral density of the signal.

    Notes
    -----
    - Increasing `nperseg` and/or `nfft` improves frequency resolution but
      reduces the number of averages (potentially increasing variance in the PSD).
    - The default 50% overlap is standard for Welch's method but can be adjusted
      (e.g., 75%) for different trade-offs in variance vs. resolution.
    """

    if noverlap is None:
        noverlap = nperseg // 2  # 50% overlap by default

    # If a string is passed for window, get the actual window array
    if isinstance(window, str):
        window_array = get_window(window, nperseg)
    else:
        window_array = window

    frequencies, psd = welch(
        x=signal,
        fs=sampling_rate,
        window=window_array,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        nfft=nfft,
        scaling='density',
        average='mean'
    )

    return frequencies, psd

from scipy.signal import butter, sosfiltfilt
def band_limited_power(data, fs, band, order=4):
    """
    data: 1D array (time series for one channel)
    fs: sampling frequency in Hz
    band: (low_freq, high_freq) tuple for band-pass
    order: filter order
    returns: band-limited power (sum of squares)
    """
    # Design a band-pass filter
    low, high = band
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype='bandpass', output='sos')
    
    # Apply the filter
    filtered = sosfiltfilt(sos, data)
    
    # Compute power (sum of squared values)
    return np.sum(filtered**2)


from scipy.signal import butter, filtfilt, hilbert

def rayleigh_test(angles):
    """
    Given a set of angles (in radians),
    returns (R, p-value) from the Rayleigh test.
    R = mean resultant length
    """
    N = len(angles)
    if N == 0:
        return np.nan, 1.0  # no spikes => no locking
    # Mean resultant length
    R = np.abs(np.mean(np.exp(1j * angles)))
    # Rayleigh statistic
    Z = N * (R ** 2)
    # p-value approximation for the Rayleigh test
    p = np.exp(-Z) * (1 + Z)
    return R, p


def plot_clusters_dendrogram(correlation_matrix, threshold=None, n_clusters=None):
    """
    Plot dendrogram of hierarchical clustering based on correlation matrix.

    Parameters:
    - correlation_matrix (numpy.ndarray): Correlation matrix.
    - threshold (float, optional): Threshold for clustering. Defaults to None.
    - n_clusters (int, optional): Number of clusters. Defaults to None.

    Returns:
    - cluster_labels (numpy.ndarray): Labels assigned to clusters.
    """
    # Convert the correlation matrix to a distance matrix
    distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

    # Perform hierarchical clustering with Ward's linkage
    linkage_matrix = linkage(distance_matrix, method='ward')

    # Extract clusters based on the specified threshold or number of clusters
    if threshold is not None:
        cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
    elif n_clusters is not None:
        cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    else:
        raise ValueError("Either 'threshold' or 'n_clusters' must be specified.")

    # Plot the dendrogram corresponding to the clustering result
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, orientation='left', labels=np.arange(1, correlation_matrix.shape[0] + 1),
               color_threshold=threshold, above_threshold_color='gray')

    plt.title('Clusters Dendrogram')
    plt.xlabel('Observations')
    plt.ylabel('Distance')
    plt.show()
    return cluster_labels

def plot_matrix_with_x_labels(matrix, x_values, x_range=None):
    """
    Plot a matrix and rename the x-axis values according to a vector of values.

    Parameters:
    - matrix (numpy.ndarray): 2D matrix to be plotted.
    - x_values (numpy.ndarray): Vector of values to rename the x-axis.
    - x_range (tuple, optional): Range of x values to display. Defaults to None.
    """

    if x_range is not None:
        x_min, x_max = x_range
        x_indices = np.where((x_values >= x_min) & (x_values <= x_max))[0]
        matrix = matrix[:, x_indices]
        x_values = x_values[x_indices]
    plt.figure()
    plt.imshow(matrix, aspect='auto', extent=(x_values.min(), x_values.max(), 0, matrix.shape[0]))
    plt.colorbar(label='Values')
    plt.show()




def peak_envelope(s, dmax=1):

    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    envelope_func = interp1d(lmax, s[lmax], kind='linear', fill_value="extrapolate", bounds_error=False)
    envelope = envelope_func(np.arange(len(s)))
    
    return envelope


def plot_spike_raster(spike_matrix):
    """
    Plots a spike raster plot from a matrix of spike times.

    Parameters:
    spike_matrix (array_like): A 2D array where rows represent different neurons or trials
                               and columns represent time. Each element should be 0 or 1,
                               where 1 indicates a spike.
    """
    n_neurons, n_time_points = spike_matrix.shape
    fig, ax = plt.subplots(figsize=(10, 6))

    for neuron_idx in range(n_neurons):
        spike_times = np.where(spike_matrix[neuron_idx, :] == 1)[0]
        ax.scatter(spike_times, np.ones_like(spike_times) * neuron_idx, marker='|', s=50,c='k')

    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron or Trial Index')
    ax.set_title('Spike Raster Plot')
    plt.show()

from scipy.interpolate import RegularGridInterpolator

def interpolate_3d_volume(x_orig, y_orig, z_orig, data, x_spacing=10, y_spacing=0.005, z_spacing=0.5):
    """
    Interpolates a 3D volume data from original unequal spacings to specified uniform spacings.

    Parameters:
    - x_orig: Original x coordinates (1D array)
    - y_orig: Original y coordinates (1D array)
    - z_orig: Original z coordinates (1D array)
    - data: Original 3D numpy array of data
    - x_spacing: Desired spacing for x dimension
    - y_spacing: Desired spacing for y dimension
    - z_spacing: Desired spacing for z dimension

    Returns:
    - data_new: Interpolated 3D numpy array with new spacings
    """
    
    # Adjust for the circular nature of x
    x_orig_circular = np.append(x_orig, 360)  # Close the loop for circular interpolation
    data_circular = np.append(data, data[:1, :, :], axis=0)  # Extend data at the boundary


    # Defining new grids with specified spacings
    x_new = np.arange(0, 361, x_spacing) #do it with orientations
    y_new = np.arange(min(y_orig), max(y_orig) + y_spacing, y_spacing)
    z_new = np.arange(min(z_orig), max(z_orig) + z_spacing, z_spacing)
    
    new_inds=(x_new,y_new,z_new)
    
    # Interpolator for circular data
    interpolator_circular = RegularGridInterpolator((x_orig_circular, y_orig, z_orig), data_circular)
    
    # New grid points
    points_new_circular = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    points_new_flat_circular = np.array([p.ravel() for p in points_new_circular]).T
    
    # Perform the interpolation
    data_new_flat_circular = interpolator_circular(points_new_flat_circular)
    data_new_circular = data_new_flat_circular.reshape(len(x_new), len(y_new), len(z_new))


    return data_new_circular, new_inds


def generalize_binning(params, spike_numbers):
    bin_edges = {}
    binned_data = {}
    dimensions = []
    
    # Creating bins for each parameter and digitizing the parameter data
    for key in params:
        unique_values = np.unique(params[key])
        bin_edges[key] = unique_values
        binned_data[key] = np.digitize(params[key], bin_edges[key], right=True)
        dimensions.append(len(unique_values))
    
    # Initializing the response matrices
    resp_mat = np.zeros(dimensions)
    resp_sd = np.zeros(dimensions)
    
    # Iterating over all combinations of bin indices
    for indices in np.ndindex(*dimensions):
    # Initializing a mask to select rows that match current bin indices for all parameters
        masks = []
        for param_idx, key in enumerate(params.keys()):
            # Creating a mask for each parameter based on current index
            mask = binned_data[key] == indices[param_idx]  
            masks.append(mask)

        # Combining masks: rows must satisfy all conditions (all masks must be True)
        combined_mask = np.logical_and.reduce(masks)

        if combined_mask.any():
            resp_mat[indices] = np.mean(spike_numbers[combined_mask])
            resp_sd[indices] = np.std(spike_numbers[combined_mask])
        else:
            resp_mat[indices] = np.nan  # or another placeholder to indicate no data for this combination
            resp_sd[indices] = np.nan
    
    return resp_mat, resp_sd
        

def gaussian_kernel(size, sigma):
    """Generates a Gaussian kernel."""
    position = np.arange(size) - size // 2
    kernel_raw = np.exp(-position**2 / (2 * sigma**2))
    kernel_normalized = kernel_raw / np.sum(kernel_raw)
    return kernel_normalized

def smoothen(binned_spike_train, sigma):
    
    kernel_size = int(np.ceil(sigma * 3)) * 2 + 1  # Ensure kernel size is odd
    gaussian_kernel_ = gaussian_kernel(kernel_size, sigma)
    smoothed_spike_rate = convolve(binned_spike_train, gaussian_kernel_, mode='same')
    return smoothed_spike_rate



def compute_real_and_null_etas(X, bpp_expanded, n_shuffles=1):
    """
    Compute real and null eta0 and eta1 values from input X and bpp_expanded arrays.

    Parameters:
        X (ndarray): A 2D array where rows are trials and columns are time points or features.
        bpp_expanded (ndarray): A 2D array of the same shape as X, containing stimulus or feature data.
        n_shuffles (int): Number of shuffles to perform for null distribution.

    Returns:
        eta0 (ndarray): Real STA for X > 0.
        eta1 (ndarray): Real STA for X < 0.
        null_eta0 (ndarray): Null STA (shuffled) for X > 0.
        null_eta1 (ndarray): Null STA (shuffled) for X < 0.
    """
    # -- Real STA --
    weighted_sum_0 = np.sum((X > 0) * bpp_expanded, axis=0)
    total_trials_0 = np.sum(X > 0, axis=0)
    eta0 = weighted_sum_0 / np.maximum(total_trials_0, 1e-12)

    weighted_sum_1 = np.sum((X < 0) * bpp_expanded, axis=0)
    total_trials_1 = np.sum(X < 0, axis=0)
    eta1 = weighted_sum_1 / np.maximum(total_trials_1, 1e-12)

    # -- Null STAs --
    null_etas0 = []
    null_etas1 = []
    for _ in range(n_shuffles):
        X_shuff = np.random.permutation(X)

        null_eta0 = np.sum((X_shuff > 0) * bpp_expanded, axis=0) / np.maximum(np.sum(X_shuff > 0, axis=0), 1e-12)
        null_eta1 = np.sum((X_shuff < 0) * bpp_expanded, axis=0) / np.maximum(np.sum(X_shuff < 0, axis=0), 1e-12)

        null_etas0.append(null_eta0)
        null_etas1.append(null_eta1)

    null_eta0 = np.mean(np.stack(null_etas0), axis=0) #this changes the std
    null_eta1 = np.mean(np.stack(null_etas1), axis=0)

    return eta0, eta1, null_eta0, null_eta1 



def calc_rf(sta,base_sta,peak_thresh=4,blob_thresh=3, gaus_f=1,med_f=1,upsample=1,force_polarity=None):
    

    sta = zoom(sta, zoom=upsample, order=1)
    base_sta = zoom(base_sta, zoom=upsample, order=1)
    
    
    
    if med_f>0:
        filt_sta=medfilt2d(sta, kernel_size=med_f)
        filt_sta=gaussian_filter(filt_sta, gaus_f)
        
        filt_base=medfilt2d(base_sta, kernel_size=med_f)
        filt_base=gaussian_filter(filt_base, gaus_f)
    
    else:
        filt_sta=gaussian_filter(sta, gaus_f)
        filt_base=gaussian_filter(base_sta, gaus_f)
    
    mean_sta = np.mean(filt_base)
    std_sta = np.std(filt_base)
    
    filt_sta=filt_sta-mean_sta
    
    max_deviation = np.max(filt_sta) 
    min_deviation = np.min(filt_sta) 
    
  #  import pdb;pdb.set_trace()
    
    if force_polarity == 'positive':
        peak_coords = np.unravel_index(np.argmax(filt_sta), filt_sta.shape)
        binary_rf = filt_sta > (blob_thresh * std_sta)
        cleared_peak_thresh = filt_sta[peak_coords] > (peak_thresh * std_sta)
        pos = 1
    
    elif force_polarity == 'negative':
        peak_coords = np.unravel_index(np.argmin(filt_sta), filt_sta.shape)
        binary_rf = filt_sta < (-1 * blob_thresh * std_sta)
        cleared_peak_thresh = filt_sta[peak_coords] < (-1 * peak_thresh * std_sta)
        pos = -1
    
    else:
        if np.abs(max_deviation) > np.abs(min_deviation):
            peak_coords = np.unravel_index(np.argmax(filt_sta), filt_sta.shape)
            binary_rf = filt_sta > (blob_thresh * std_sta)
            cleared_peak_thresh = filt_sta[peak_coords] > (peak_thresh * std_sta)
            pos = 1
        else:
            peak_coords = np.unravel_index(np.argmin(filt_sta), filt_sta.shape)
            binary_rf = filt_sta < (-1 * blob_thresh * std_sta)
            cleared_peak_thresh = filt_sta[peak_coords] < (-1 * peak_thresh * std_sta)
            pos = -1

        
    

    if cleared_peak_thresh:
              
        labeled_rf, num_features = label(binary_rf, return_num=True)
        regions = regionprops(labeled_rf)
        
        # Assuming the largest labeled area is the receptive field
        largest_area = max(regions, key=lambda x: x.area) if regions else None
            
        
        rf_blob = labeled_rf == largest_area.label if largest_area else np.zeros_like(binary_rf)
        rf_size = largest_area.area if largest_area else 0
        rf_loc = largest_area.centroid if largest_area else (0, 0)
        rf_diameter = largest_area.major_axis_length if largest_area else 0
        
       
        major_axis = largest_area.major_axis_length if largest_area else 0
        try:
            minor_axis = largest_area.minor_axis_length if largest_area else 0
        except:
            minor_axis=0
            
        rf_ellipse = np.pi * (major_axis / 2) * (minor_axis / 2)

        #this is peak deviation, with background subtracted
        rf_peakrate=sta[int(rf_loc[0]),int(rf_loc[1])]
        rf_minrate=np.min(filt_sta)
    else:
        rf_blob=np.zeros_like(filt_sta)
        rf_size=0
        rf_loc=(0,0)
        rf_diameter = 0
        rf_ellipse = 0
        rf_peakrate=mean_sta
        rf_minrate=np.min(filt_sta)
        pos=0
        
    return rf_blob, rf_size, rf_loc, filt_sta, rf_diameter, rf_peakrate,rf_minrate, pos



def calc_psth(stimulus_times, spike_times, time_window, bin_width=0.02, trial_samp=1000, plot=0):

    aligned_spikes = []
    for stimulus_onset_time in stimulus_times:
        window_start = stimulus_onset_time + time_window[0]
        window_end = stimulus_onset_time + time_window[1]
        spikes_in_window = spike_times[(spike_times >= window_start) & (spike_times <= window_end)]
        # Align spikes to stimulus onset
        aligned_spikes.append(spikes_in_window - stimulus_onset_time)
        
    # Subsample to make plotting possible
    if len(aligned_spikes) > trial_samp:
        aligned_spikes = random.sample(aligned_spikes, trial_samp)
    
    # Compile the PSTH
    # Flatten the list of aligned spikes and compute histogram
    all_aligned_spikes = np.concatenate(aligned_spikes)
    psth_bins = np.arange(time_window[0], time_window[1]+bin_width, bin_width)  # Define your bin width
    psth, bin_edges = np.histogram(all_aligned_spikes, bins=psth_bins)
    
    # Normalize the PSTH to spike rate
    num_trials = len(aligned_spikes)
    psth = psth / (num_trials * bin_width)
    
    if plot != 0:    
        # Plotting
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [5,1]})
        ax_raster, ax_psth = axes
        
        # Raster Plot
        for trial_idx, spikes in enumerate(aligned_spikes):
            ax_raster.scatter(spikes, np.ones_like(spikes) * trial_idx, marker='|', color='black', s=1)  # Smaller size
        ax_raster.set_ylabel('Trial')
        ax_raster.set_xlim(time_window)
        ax_raster.set_title('Raster Plot')
        ax_raster.invert_yaxis()  # Optionally invert y-axis so that the first trial is at the top
        
        # PSTH Plot
        bar_width = np.diff(bin_edges)[0] * 0.8  # Slightly narrower bar width
        ax_psth.bar((bin_edges[:-1] + bin_edges[1:]) / 2, psth, width=bar_width, color='gray', edgecolor='black', linewidth=0.5)  # Thin bar edges
        ax_psth.set_xlabel('Time relative to stimulus onset (s)')
        ax_psth.set_ylabel('Spike rate (spikes/s)')
        ax_psth.set_title('PSTH')
        ax_psth.set_xlim(time_window)
        
        # Save as JPG with DPI and anti-aliasing
        plt.tight_layout()
        plt.show()   
        
    return psth, bin_edges, aligned_spikes



def plot_cwt_spectrogram(signal, fs, start_sample, end_sample,plot=1):
    wavelet = "cmor1.5-1.0"
    nfreqs=100
    min_freq = 1
    max_freq = 100
    min_scale = 1/max_freq*fs
    max_scale = 1/min_freq*fs
    sampling_period = 1 / fs
    
    widths = np.geomspace(min_scale, max_scale, num=nfreqs)
    signal = signal[start_sample:end_sample]
    t = np.arange(start_sample, end_sample) / fs
    cwtmatr, freqs = pywt.cwt(signal, widths, wavelet, sampling_period=sampling_period)
    cwtmatr = np.abs(cwtmatr)
    
    if plot == 1:
        fig, axs = plt.subplots(1, 1)
        axs.pcolormesh(t, freqs, gaussian_filter(cwtmatr,1), cmap='bwr', shading='auto')  # Ensure 'shading' is set for better visualization
        axs.set_yscale("log")
        axs.set_xlabel("Time (s)")
        axs.set_ylabel("Frequency (Hz)")
        axs.set_title("Continuous Wavelet Transform (Scaleogram)")
    
    return cwtmatr,freqs
 

def run_dn(curr_ephys_path,dn_stimlist, time_bins=5, upsample=1, gf=1,vm=.6,plot_psth=0):

    for dn_stim in dn_stimlist:
        
        lum_folders = glob.glob(curr_ephys_path + '/' + '*' + dn_stim + '*')
        
        for curr_experiment_path in lum_folders:

            dn_exp = load_ephys_results(curr_experiment_path ,3, False)
            
            if dn_exp is not None:
                if not os.path.exists(dn_exp['analysis_path'] + '/rfs2'):
                    os.mkdir(dn_exp['analysis_path'] + '/rfs2')
                    
                orig_st = int(np.mean(np.diff(dn_exp['stim_times'][:, 0]))*1000)/1000
                dt = orig_st/upsample
        
                first_st=dn_exp['stim_times'][0, 0]
                stimulus_times=dn_exp['stim_times'][:, 0]-first_st
                
                dn_exp['dn_mat']=dn_exp['dn_mat'][0:len(stimulus_times)] #shouldn't be needed usually
            
                X, upsampled_times = resample_mat(
                    dn_exp['dn_mat'], stimulus_times, dt, interp_kind='nearest')
                dims = [time_bins, X.shape[1], X.shape[2]]
                
                for probe in range(0,len(dn_exp['st_mat'])):
                    
                    num_units = len(dn_exp['st_mat'][probe])
                    
                    for unit in range(0, num_units):
                    
                        curr_times = dn_exp['st_mat'][probe][unit]-first_st
                        
                        if len(curr_times)>0:
        
                            bpp, bin_edges = np.histogram(curr_times, bins=upsampled_times)
                            bpp = np.concatenate([bpp, [0]])
                        
                                 
                            bpp_expanded = bpp[:, np.newaxis, np.newaxis]
                            weighted_mean = np.mean((X>0) * bpp_expanded, axis=0)/dt
                            total_trials=np.sum(X>0,axis=0)
                            eta1 = weighted_mean / total_trials 
                      
                            
                            rf_blob1, rf_size, rf_loc, filt_eta1, rf_diameter,rf_e, pos  = calc_rf(eta1,eta1,peak_thresh=2,blob_thresh=2,gaus_f=0,med_f=0,upsample=5)
                            
                            
                            # Create a figure with 1 row and 2 columns
                            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                            axes = axes.ravel()
                          
                            # Plot 'filt_sta' in the first subplot
                            axes[0].imshow(filt_eta1, cmap=plt.cm.bwr)
                            axes[0].set_title('Filtered ETA')
                            axes[0].axis('off')  # Optionally hide axis
                            
                                         # Plot 'rf_blob' in the second subplot
                            axes[1].imshow(rf_blob1)
                            axes[1].set_title('RF Blob')
                            axes[1].axis('off')  # Optionally hide axis
                            
                                              
                            # Adjust the layout to avoid overlap
                            plt.tight_layout()
                            
                            # Show the figure
                            plt.savefig(dn_exp['analysis_path'] + '/rfs2/probe_' + str(probe) + '_unit_' + str(unit) + '.png')
                            plt.close()
                            
                            np.save(dn_exp['analysis_path'] + '/probe_' + str(probe) + '_unit_' +
                                        str(unit) + '_eta1.npy',eta1)
                          
                     
                            # psth, bin_edges, aligned_spikes= calc_psth(stimulus_times, curr_times, [0, 0.1],bin_width=0.002,trial_samp=1500,plot=1)
                            
                            
                            # plt.savefig(dn_exp['analysis_path'] + '/probe_' + str(probe) + '_unit_' + str(unit) +'_psth.png')
                            # plt.close() 
                            
                            # np.save(dn_exp['analysis_path'] + '/probe_' + str(probe) + '_unit_' +
                            #                 str(unit) + '_psth.npy',psth)
                    

def plot_smoothed_rf_size_vs_depth(depths_um, sizes_um, br,
                                   title="Receptive Field Size vs Depth",
                                   bandwidth=100.0, n_eval=100,
                                   dot_size=10, save_path='unit_rf_size.pdf'):
    
    x_eval      = np.linspace(np.nanmin(depths_um), np.nanmax(depths_um), n_eval)
    smooth_mean = np.empty_like(x_eval)
    smooth_sem  = np.empty_like(x_eval)

    for i, x0 in enumerate(x_eval):
        w = np.exp(-0.5 * ((depths_um - x0) / bandwidth) ** 2)
        if not np.any(w):
            smooth_mean[i] = np.nan
            smooth_sem[i]  = np.nan
            continue
        w /= w.sum()
        mu      = np.sum(w * sizes_um)
        var     = np.sum(w * (sizes_um - mu) ** 2)
        n_eff   = (w.sum() ** 2) / np.sum(w ** 2)
        smooth_mean[i] = mu
        smooth_sem[i]  = np.sqrt(var) / np.sqrt(n_eff)

    # Plotting
    plt.figure(figsize=(3,3))

    # Colored scatter using `br` values
    sc = plt.scatter(depths_um, sizes_um,
                     s=dot_size, c=br, cmap='viridis', alpha=0.8, edgecolors='k', linewidths=0.3)

    # plt.plot(x_eval, smooth_mean, color='crimson', lw=2, label='Smoothed mean')
    # plt.fill_between(x_eval,
    #                  smooth_mean - smooth_sem,
    #                  smooth_mean + smooth_sem,
    #                  color='crimson', alpha=0.3, label='± 1 SEM')


    plt.title(title)
    plt.xlabel('Depth (µm)')
    plt.ylabel('RF area (vis. degrees)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()




# ---------------------------------------------------------------------
# 1. Neuropixels-1.0 layout (phase-3a, 960 sites)
#    • 480 rows, 20 µm vertical pitch
#    • 4 staggered columns  (–18, –6, 6, 18 µm from centreline)
#      even rows: columns 0 & 2   (−18,   6)
#      odd  rows: columns 1 & 3   ( −6,  18)
# ---------------------------------------------------------------------
def np10_site_positions() -> np.ndarray:
    """Return (960, 2) array of [x, y] positions in µm for Neuropixels 1.0."""
    col_x = np.array([-18., -6., 6., 18.])    # lateral offsets (µm)
    n_rows = 480
    pos    = np.empty((960, 2), dtype=float)

    i = 0
    for row in range(n_rows):
        y = row * 20.0                        # 20 µm vertical pitch
        if row % 2 == 0:                      # even row  → outer columns
            xs = col_x[[0, 2]]               # -18, 6
        else:                                 # odd  row  → inner columns
            xs = col_x[[1, 3]]               # -6, 18
        for x in xs:
            pos[i] = (x, y)
            i += 1
    return pos                                # shape (960, 2)

# ---------------------------------------------------------------------
# 2. Convert a set of channels → spatial distance matrix (µm)
# ---------------------------------------------------------------------
def channel_distances(channels: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    channels : 1-D int array
        Channel indices (0-959).

    Returns
    -------
    dists : 2-D float array
        Square matrix of Euclidean distances in µm between the channels.
    """
    positions   = np10_site_positions()[channels]        # (n, 2)
    dists_vec   = pdist(positions, metric='euclidean')   # condensed vector
    return squareform(dists_vec)                         # (n, n) matrix




