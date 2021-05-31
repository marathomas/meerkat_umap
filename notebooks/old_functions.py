#!/usr/bin/env python
# coding: utf-8

# In[4]:


# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:39:59 2021

Collection of custom evaluation functions for embedding

@author: marathomas
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt


def make_nn_stats_dict(calltypes, labels, nb_indices):
    """
    Function that evaluates the labels of the k nearest neighbors of 
    all datapoints in a dataset.

    Parameters
    ----------
    calltypes : 1D numpy array (string) or list of strings
                set of class labels
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset
    nb_indices: 2D numpy array (numeric integer)
                Array I(X,k) containing the indices of the k nearest
                nearest neighbors for each datapoint X of a
                dataset

    Returns
    -------
    nn_stats_dict : dictionary[<class label string>] = 2D numpy array (numeric)
                    dictionary that contains one array for each type of label.
                    Given a label L, nn_stats_dict[L] contains an array A(X,Y), 
                    where Y is the number of class labels in the dataset and each
                    row X represents a datapoint of label L in the dataset.
                    A[i,j] is the number of nearest neighbors of datapoint i that
                    are of label calltypes[j].
                    
    Example
    -------
    >>> 

    """
    nn_stats_dict = {}
    
    for calltype in calltypes:
        # which datapoints in the dataset are of this specific calltype?
        # -> get their indices
        call_indices = np.asarray(np.where(labels==calltype))[0]
        
        # initialize array that can save the class labels of the k nearest
        # neighbors of all these datapoints
        calltype_counts = np.zeros((call_indices.shape[0],len(calltypes)))
        
        # for each datapoint
        for i,ind in enumerate(call_indices):
            # what are the indices of its k nearest neighbors
            nearest_neighbors = nb_indices[ind]
            # for eacht of these neighbors
            for neighbor in nearest_neighbors:
                # what is their label
                neighbor_label = labels[neighbor]
                # put a +1 in the array
                calltype_counts[i,np.where(np.asarray(calltypes)==neighbor_label)[0][0]] += 1 
        
        # save the resulting array in dictionary 
        # (1 array per calltype)
        nn_stats_dict[calltype] = calltype_counts 
  
    return nn_stats_dict

def get_knn(k,embedding):
    """
    Function that finds k nearest neighbors (based on 
    euclidean distance) for each datapoint in a multidimensional 
    dataset 

    Parameters
    ----------
    k : integer
        number of nearest neighbors
    embedding: 2D numpy array (numeric)
               a dataset E(X,Y) with X datapoints and Y dimensions

    Returns
    -------
    indices: 2D numpy array (numeric)
             Array I(X,k) containing the indices of the k nearest
             nearest neighbors for each datapoint X of the input
             dataset
             
    distances: 2D numpy array (numeric)
               Array D(X,k) containing the euclidean distance to each
               of the k nearest neighbors for each datapoint X of the 
               input dataset. D[i,j] is the euclidean distance of datapoint
               embedding[i,:] to its jth neighbor.
                    
    Example
    -------
    >>> 

    """

    # Find k nearest neighbors
    nbrs = NearestNeighbors(metric='euclidean',n_neighbors=k+1, algorithm='brute').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    # need to remove the first neighbor, because that is the datapoint itself
    indices = indices[:,1:]  
    distances = distances[:,1:]
    
    return indices, distances


def make_statstabs(nn_stats_dict, calltypes, labels,k):
    """
    Function that generates two summary tables containing
    the frequency of different class labels among the k nearest 
    neighbors of datapoints belonging to a class.

    Parameters
    ----------
    nn_stats_dict : dictionary[<class label string>] = 2D numpy array (numeric)
                    dictionary that contains one array for each type of label.
                    Given a label L, nn_stats_dict[L] contains an array A(X,Y), 
                    where Y is the number of class labels in the dataset and each
                    row X represents a datapoint of label L in the dataset.
                    A[i,j] is the number of nearest neighbors of datapoint i that
                    are of label calltypes[j].
                    (is returned from evaulation_functions.make_nn_statsdict)
    calltypes : 1D numpy array (string) or list of strings
                set of class labels
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset
    k: Integer
       number of nearest neighbors

    Returns
    -------
    stats_tab: 2D pandas dataframe (numeric)
               Summary table T(X,Y) with X,Y = number of classes.
               T[i,j] is the average percentage of datapoints with class label j
               in the neighborhood of datapoints with class label i
             
    stats_tab_norm: 2D pandas dataframe (numeric)
                   Summary table N(X,Y) with X,Y = number of classes.
                   N[i,j] is the log2-transformed ratio of the percentage of datapoints 
                   with class label j in the neighborhood of datapoints with class label i
                   to the percentage that would be expected by random chance and random
                   distribution. (N[i,j] = log2(T[i,j]/random_expect))
              
    Example
    -------
    >>> 

    """
    
    # Get the class frequencies in the dataset
    overall = np.zeros((len(calltypes)))  
    for i,calltype in enumerate(calltypes):
        overall[i] = sum(labels==calltype) 
    overall = (overall/np.sum(overall))*100
    
    # Initialize empty array for stats_tab and stats_tab_norm
    stats_tab = np.zeros((len(calltypes),len(calltypes)))
    stats_tab_norm = np.zeros((len(calltypes),len(calltypes)))

    # For each calltype
    for i, calltype in enumerate(calltypes):
        # Get the table with all neighbor label counts per datapoint
        stats = nn_stats_dict[calltype]
        # Average across all datapoints and transform to percentage
        stats_tab[i,:] = (np.mean(stats,axis=0)/k)*100
        # Divide by overall percentage of this class in dataset 
        # for the normalized statstab version
        stats_tab_norm[i,:] = ((np.mean(stats,axis=0)/k)*100)/overall
    
    # Turn into dataframe
    stats_tab = pd.DataFrame(stats_tab)
    stats_tab_norm = pd.DataFrame(stats_tab_norm)
    
    # Add row with overall frequencies to statstab
    stats_tab.loc[len(stats_tab)] = overall
    
    # Name columns and rows
    stats_tab.columns = calltypes
    stats_tab.index = calltypes+['overall']

    stats_tab_norm.columns = calltypes
    stats_tab_norm.index = calltypes
    
    # Replace zeros with small value as otherwise log2 transform cannot be applied
    x=stats_tab_norm.replace(0, 0.0001)
    
    # log2-tranform the ratios that are currently in statstabnorm
    stats_tab_norm = np.log2(x)

    return stats_tab, stats_tab_norm
