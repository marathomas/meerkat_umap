#!/usr/bin/env python
# coding: utf-8

# In[4]:


# -*- coding: utf-8 -*-
"""
Created on Created on Tue May  4 12:20:04 2021

Collection of preprocessing functions for analysis of spectrograms

@author: marathomas
"""

import numpy as np 
import numba
from numba import jit

MEL_BINS_REMOVED_LOWER = 5
MEL_BINS_REMOVED_UPPER = 5


@jit(nopython=True)
def calc_zscore(s):
    """
    Function that z-score transforms each value of a 2D array 
    (not along any axis). numba-compatible.

    Parameters
    ----------
    spec : 2D numpy array (numeric)

    Returns
    -------
    spec : 2D numpy array (numeric)
           the z-transformed array

    Example
    -------
    >>> 

    """
    spec = s.copy()
    mn = np.mean(spec)
    std = np.std(spec)
    for i in range(spec.shape[0]):
        for j in range(spec.shape[1]):
            spec[i,j] = (spec[i,j]-mn)/std
    return spec


@jit(nopython=True)
def preprocess_spec_numba_fl(spec, n_lower, n_upper):
    """
    Function that preprocesses a spectrogram
    i) removing mel bins
    ii) calculating zscore
    iiii) setting floor
    
    numba compatible.

    Parameters
    ----------
    spec : 2D numpy array (numeric)
           a spectrogram S(X,Y) with X frequency bins and 
           Y timeframes

    Returns
    -------
    s : 2D numpy array (numeric)
        the preprocessed spectrogram S

    Example
    -------
    >>> 

    """ 
    s = np.copy(spec)
    n_mels = s.shape[0]
    s = s[n_lower:(n_mels-n_upper),:]
    s = calc_zscore(s)
    s = np.where(s < 0, 0, s)
    
    return s

@jit(nopython=True)
def preprocess_spec_numba(spec,n_lower, n_upper):
    """
    Function that preprocesses a spectrogram
    i) removing mel bins
    ii) calculating zscore
    iii) setting ceiling
    iiii) setting floor
    
    numba compatible.

    Parameters
    ----------
    spec : 2D numpy array (numeric)
           a spectrogram S(X,Y) with X frequency bins 
           and Y timeframes

    Returns
    -------
    s : 2D numpy array (numeric)
        the preprocessed spectrogram S
          
    Example
    -------
    >>> 

    """ 
    s = spec.copy()
    n_mels = s.shape[0]
    s = s[n_lower:(n_mels-n_upper),:]
    s = calc_zscore(s)
    s = np.where(s > 3, 3, s)
    s = np.where(s < 0, 0, s)
    
    return s


def pad_spectro(spec,maxlen):
    """
    Function that Pads a spectrogram with shape (X,Y) with 
    zeros, so that the result is in shape (X,maxlen)

    Parameters
    ----------
    spec : 2D numpy array (numeric)
           a spectrogram S(X,Y) with X frequency bins and Y timeframes
    maxlen: maximal length (integer)

    Returns
    -------
    padded_spec : 2D numpy array (numeric)
                  a zero-padded spectrogram S(X,maxlen) with X frequency bins 
                  and maxlen timeframes

    Example
    -------
    >>> 

    """
    padding = maxlen - spec.shape[1]
    z = np.zeros((spec.shape[0],padding))
    padded_spec=np.append(spec, z, axis=1)
    return padded_spec


def create_padded_data(specs_list):
    """
    Function that creates a 2D array from a list of spectrograms
    by zero-padding and flattening each spectrogram. All specs
    are zero-padded to the length of the longest spectrogram in the
    list. Each row in the resulting data represents one spectrogram.

    Parameters
    ----------
    specs : list of 2D numeric numpy arrays
            list of spectrograms S(X,Y) with X frequency bins and 
            Y timeframes

    Returns
    -------
    data : 2D numpy array (numeric)
           the array of padded and flattened spectrograms
           A(X,Y) with X flattened spectrograms

    Example
    -------
    >>> 

    """       
    maxlen= np.max([spec.shape[1] for spec in specs_list])
    flattened_specs = [pad_spectro(spec, maxlen).flatten() for spec in specs_list]
    data = np.asarray(flattened_specs)
    return data




def pad_transform_spectro(spec,maxlen):
    """
    Function that encodes a 2D spectrogram in a 1D array, so that it 
    can later be restored again.
    Flattens and pads a spectrogram with default value 999
    to a given length. Size of the original spectrogram is encoded
    in the first two cells of the resulting array

    Parameters
    ----------
    spec : 2D numpy array (numeric)
           a spectrogram S(X,Y) with X frequency bins and Y timeframes
    maxlen: Integer 
            n of timeframes to which spec should be padded

    Returns
    -------
    trans_spec : 1D numpy array (numeric)
                 the padded and flattened spectrogram
               
    Example
    -------
    >>> 

    """       
    flat_spec = spec.flatten()
    trans_spec = np.concatenate((np.asarray([spec.shape[0], spec.shape[1]]), flat_spec, np.asarray([999]*(maxlen-flat_spec.shape[0]-2))))
    trans_spec = np.float64(trans_spec)
    
    return trans_spec






