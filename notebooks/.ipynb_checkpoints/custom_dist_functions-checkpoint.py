#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:39:59 2021

Collection of custom distance functions for UMAP

@author: marathomas
"""

import numpy as np 
import numba
from numba import jit

MIN_OVERLAP = 0.9

@numba.njit()
def unpack_specs(a,b):
    """
    Function that unpacks two specs that have been transformed into 
    a 1D array with preprocessing_functions.pad_transform_spec and 
    restores their original 2D shape

    Parameters
    ----------
    a,b : 1D numpy arrays (numeric)

    Returns
    -------
    spec_s, spec_l : 2D numpy arrays (numeric)
                     the restored specs 
    Example
    -------
    >>> 

    """

    a_shape0 = int(a[0])
    a_shape1 = int(a[1])
    b_shape0 = int(b[0])
    b_shape1 = int(b[1])

    spec_a= np.reshape(a[2:(a_shape0*a_shape1)+2], (a_shape0, a_shape1))
    spec_b= np.reshape(b[2:(b_shape0*b_shape1)+2], (b_shape0, b_shape1))
    
    len_a = a_shape1
    len_b = b_shape1
    
    # find bigger spec
    spec_s = spec_a
    spec_l = spec_b

    if len_a>len_b:
        spec_s = spec_b
        spec_l = spec_a
        
    return spec_s, spec_l

@numba.njit()
def spec_dist(a,b, size):
    """
    Function that calculates distance between two spectrograms
    Equivalent to manhattan distance of flattened vectors
    
    Parameters
    ----------
    a,b : 2D numpy arrays (numeric) of equal shape
          spectrograms S(X,Y) with X frequency bins and 
          Y timeframes
    size: Number of "pixels" in each spec (Integer)
          (a.shape[0]*b.shape[1])

    Returns
    -------
    dist : Manhattan distance between a and b (Float64) normalized to
           size
   
    Example
    -------
    >>> 

    """

    dist = (np.sum(np.abs(np.subtract(a, b)))) / size # manhattan
    
    #dist = (np.sum(np.subtract(a, b)*np.subtract(a, b))) / size # mean squared error
    #dist = np.sqrt((np.sum(np.subtract(a, b)*np.subtract(a, b))) / size)
    return dist
    

@numba.njit()
def calc_pairwise_pad(a, b):
    """
    Custom numba-compatible distance function for UMAP.
    Calculates distance between two pad-transformed spectrograms a,b 
    by zero-padding the shorter spectrogram to be equal to longer
    one, then calculate spec_dist.
    
    Parameters
    ----------
    a,b : 1D numpy arrays (numeric)
          pad_transformed spectrograms
          (with preprocessing_functions.pad_transform_spec)

    Returns
    -------
    dist : numeric (float64)
           distance between spectrograms a,b
    
    Example
    -------
    >>> 

    """
    
    spec_s, spec_l = unpack_specs(a,b)
    n_padding = int(spec_l.shape[1] - spec_s.shape[1])
    
    n_bins = spec_s.shape[0]
    
    # pad the smaller spec (if necessary)
    if n_padding!=0:
        pad = np.full((n_bins, n_padding), 0.0)
        spec_s_padded = np.concatenate((spec_s, pad), axis=1)
        spec_s_padded = spec_s_padded.astype(np.float64)
    else:
        spec_s_padded = spec_s.astype(np.float64)

    # compute distance

    spec_s_padded = np.reshape(spec_s_padded, (-1)).astype(np.float64)
    spec_l = np.reshape(spec_l, (-1)).astype(np.float64)
    size = spec_l.shape[0]
    
    dist = spec_dist(spec_s_padded, spec_l, size)
    
    return dist



@numba.njit()
def calc_overlap_only(a,b): 
    """
    Custom numba-compatible distance function for UMAP.
    Calculates distance between two pad-transformed spectrograms a,b 
    by aligning both from at timeframe 0 and calculating spec_dist only 
    between the overlapping segments of a and b
    
    Parameters
    ----------
    a,b : 1D numpy arrays (numeric)
          pad_transformed spectrograms
          (with preprocessing_functions.pad_transform_spec)

    Returns
    -------
    dist : numeric (float64)
           distance between spectrograms a,b
    
    Example
    -------
    >>> 

    """
    spec_s, spec_l = unpack_specs(a,b)
    
    #only use overlap section from longer spec
    spec_l = spec_l[:spec_s.shape[0],:spec_s.shape[1]]
    
    spec_s = spec_s.astype(np.float64)
    spec_l = spec_l.astype(np.float64)
    
    size = spec_s.shape[1]*spec_s.shape[0]   
    dist = spec_dist(spec_s, spec_l, size)
    
    return dist

@numba.njit()
def calc_timeshift(a,b):
    """
    Custom numba-compatible distance function for UMAP.
    Calculates distance between two pad-transformed spectrograms a,b 
    by shifting the shorter spectrogram along the longer
    one and finding the minimum distance overlap (according to
    spec_dist). Non-overlapping sections are ignored when
    calculating the distance. Uses global variable OVERLAP to constrain
    shifting to have OVERLAP*100 % of overlap between specs.
    
    Parameters
    ----------
    a,b : 1D numpy arrays (numeric)
          pad_transformed spectrograms
          (with preprocessing_functions.pad_transform_spec)

    Returns
    -------
    dist : numeric (float64)
           distance between spectrograms a,b
    
    Example
    -------
    >>> 

    """
    
    spec_s, spec_l = unpack_specs(a,b)  
    len_l = spec_l.shape[1]
    len_s = spec_s.shape[1]


    # define start position
    min_overlap_frames = int(MIN_OVERLAP * len_s)
    start_timeline = min_overlap_frames-len_s
    max_timeline = len_l - min_overlap_frames
    
    n_of_calculations = (max_timeline+1-start_timeline)+(max_timeline+1-start_timeline)

    distances = np.full((n_of_calculations),3.)

    count=0
    
    for timeline_p in range(start_timeline, max_timeline+1):
        # mismatch on left side
        if timeline_p < 0:
            start_col_l = 0
            len_overlap = len_s - abs(timeline_p)

            end_col_l = start_col_l + len_overlap

            end_col_s = len_s # until the end
            start_col_s = end_col_s - len_overlap

        # mismatch on right side
        elif timeline_p > (len_l-len_s):
            start_col_l = timeline_p
            len_overlap = len_l - timeline_p
            end_col_l = len_l

            start_col_s = 0
            end_col_s = start_col_s + len_overlap

        # no mismatch on either side
        else:
            start_col_l = timeline_p
            len_overlap = len_s
            end_col_l = start_col_l + len_overlap

            start_col_s = 0
            end_col_s = len_s # until the end
            

        s_s = spec_s[:,start_col_s:end_col_s].astype(np.float64)
        s_l = spec_l[:,start_col_l:end_col_l].astype(np.float64)
        
        size = s_s.shape[0]*s_s.shape[1]
        distances[count] = spec_dist(s_s, s_l, size)

        count = count + 1
    
    min_dist = np.min(distances)
                                                     
    return min_dist



@numba.njit()
def calc_timeshift_pad(a,b):
    """
    Custom numba-compatible distance function for UMAP.
    Calculates distance between two pad-transformed spectrograms a,b 
    by shifting the shorter spectrogram along the longer
    one and finding the minimum distance overlap (according to
    spec_dist). Non-overlapping sections of the shorter spec are 
    zero-padded to match the longer spec when calculating the distance. 
    Uses global variable OVERLAP to constrain shifting to have 
    OVERLAP*100 % of overlap between specs.
    
    Parameters
    ----------
    a,b : 1D numpy arrays (numeric)
          pad_transformed spectrograms
          (with preprocessing_functions.pad_transform_spec)

    Returns
    -------
    dist : numeric (float64)
           distance between spectrograms a,b
    
    Example
    -------
    >>> 

    """
    
    spec_s, spec_l = unpack_specs(a,b)
    
    len_s = spec_s.shape[1]
    len_l = spec_l.shape[1]

    nfreq = spec_s.shape[0] 

    # define start position
    min_overlap_frames = int(MIN_OVERLAP * len_s)
    start_timeline = min_overlap_frames-len_s
    max_timeline = len_l - min_overlap_frames
    
    n_of_calculations = int((((max_timeline+1-start_timeline)+(max_timeline+1-start_timeline))/2) +1)

    distances = np.full((n_of_calculations),999.)

    count=0
    
    for timeline_p in range(start_timeline, max_timeline+1,2):
        #print("timeline: ", timeline_p)
        # mismatch on left side
        if timeline_p < 0:

            len_overlap = len_s - abs(timeline_p)
            
            pad_s = np.full((nfreq, (len_l-len_overlap)),0.)
            pad_l = np.full((nfreq, (len_s-len_overlap)),0.)

            s_config = np.append(spec_s, pad_s, axis=1).astype(np.float64)
            l_config = np.append(pad_l, spec_l, axis=1).astype(np.float64)

        # mismatch on right side
        elif timeline_p > (len_l-len_s):
            
            len_overlap = len_l - timeline_p

            pad_s = np.full((nfreq, (len_l-len_overlap)),0.)
            pad_l = np.full((nfreq, (len_s-len_overlap)),0.)

            s_config = np.append(pad_s, spec_s, axis=1).astype(np.float64)
            l_config = np.append(spec_l, pad_l, axis=1).astype(np.float64)

        # no mismatch on either side
        else:
            len_overlap = len_s
            start_col_l = timeline_p
            end_col_l = start_col_l + len_overlap

            pad_s_left = np.full((nfreq, start_col_l),0.)
            pad_s_right = np.full((nfreq, (len_l - end_col_l)),0.)

            l_config = spec_l.astype(np.float64)
            s_config = np.append(pad_s_left, spec_s, axis=1).astype(np.float64)
            s_config = np.append(s_config, pad_s_right, axis=1).astype(np.float64)
        
        size = s_config.shape[0]*s_config.shape[1]
        distances[count] = spec_dist(s_config, l_config, size)
        count = count + 1


    min_dist = np.min(distances)
    return min_dist

