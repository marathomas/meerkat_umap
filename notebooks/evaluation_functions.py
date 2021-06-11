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


class nn:
    """
    A class to represent nearest neighbor statistics for a
    given latent space representation of a labelled dataset

    Attributes
    ----------
    embedding : 2D numpy array (numeric)
                a dataset E(X,Y) with X datapoints and Y dimensions
                
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset
    k : integer
        number of nearest neighbors to consider
    
    statstab: 2D pandas dataframe (numeric)
               Summary table T(X,Y) with X,Y = number of classes.
               T[i,j] is the average percentage of datapoints with class label j
               in the neighborhood of datapoints with class label i
             
    statstabnorm: 2D pandas dataframe (numeric)
                   Summary table N(X,Y) with X,Y = number of classes.
                   N[i,j] is the log2-transformed ratio of the percentage of datapoints 
                   with class label j in the neighborhood of datapoints with class label i
                   to the percentage that would be expected by random chance and random
                   distribution. (N[i,j] = log2(T[i,j]/random_expect))      

    Methods
    -------
    
    def knn_cc():
        returns k nearest neighbor fractional consistency for each class
        (1D numpy array). What percentage of datapoints (of this class)
        have fully consistent k neighbors (all k are also of the same class)
    
    get_statstab():
        returns statstab
    
    get_statstabnorm():
        returns statstabnorm
    
    get_S():    
        returns S score of embedding
        S(class X) is the average percentage of same-class neighbors
        among the k nearest neighbors of all datapoints of
        class X. S of an embedding is the average of S(class X) over all
        classes X (unweighted, e.g. does not consider class frequencies).
    
    get_Snorm():
        returns Snorm score of embedding
        Snorm(class X) is the log2 transformed, normalized percentage of 
        same-class neighbors among the k nearest neighbors of all datapoints of
        class X. Snorm of an embedding is the average of Snorm(class X) over all
        classes X.
    
    get_ownclass_S():
        returns array of S(class X) score for each class X in the dataset
        (alphanumerically sorted by class name)
        S(class X) is the average percentage of same-class neighbors
        among the k nearest neighbors of all datapoints of
        class X.
    
    get_ownclass_Snorm():
        returns array of Snorm(class X) score for each class X in the dataset
        (alphanumerically sorted by class name)
        Snorm(class X) is the log2 transformed, normalized percentage of 
        same-class neighbors among the k nearest neighbors of all datapoints of
        class X. 
    
    plot_heat_S(vmin, vmax, center, cmap, cbar, outname)
        plots heatmap of S scores
    
    plot_heat_S(vmin, vmax, center, cmap, cbar, outname)
        plots heatmap of Snorm scores
        
    plot_heat_S(center, cmap, cbar, outname)
        plots heatmap of fold likelihood (statstabnorm scores to the power of 2)
    
    """
    def __init__(self, embedding, labels, k):
        
        self.embedding = embedding
        self.labels = labels
        self.k = k
        
        label_types = sorted(list(set(labels)))        
        
        indices, distances = get_knn(k,embedding)
        nn_stats_dict = make_nn_stats_dict(label_types, labels, indices)
        stats_tab, stats_tab_norm = make_statstabs(nn_stats_dict, label_types, labels, k)
        
        self.nn_stats_dict = nn_stats_dict
        self.statstab = stats_tab
        self.statstabnorm = stats_tab_norm
    
    def knn_cc(self):
        label_types = sorted(list(set(self.labels)))        
        consistent = []
        for i,labeltype in enumerate(label_types):
            statstab = self.nn_stats_dict[labeltype] 
            x = statstab[:,i]
            cc = (np.count_nonzero(x == self.k) / statstab.shape[0])*100
            consistent.append(cc)
        return np.asarray(consistent)
          
    def get_statstab(self):
        return self.statstab
    
    def get_statstabnorm(self):
        return self.statstabnorm
    
    def get_S(self):    
        return np.mean(np.diagonal(self.statstab))
    
    def get_Snorm(self):
        return np.mean(np.diagonal(self.statstabnorm))
    
    def get_ownclass_S(self):
        return np.diagonal(self.statstab)
    
    def get_ownclass_Snorm(self):
        return np.diagonal(self.statstabnorm)
    
    def plot_heat_S(self,vmin=0, vmax=100, center=50, cmap='YlOrRd', cbar=None, outname=None):
        plt.figure(figsize=(6,6))
        ax=sns.heatmap(self.statstab, annot=True, vmin=vmin, vmax=vmax, center=center, cmap=cmap, cbar=cbar)
        plt.xlabel("neighbor label")
        plt.ylabel("datapoint label")
        plt.title("Nearest Neighbor Frequency S")
        if outname:
            plt.savefig(outname)

    def plot_heat_Snorm(self,vmin=-13, vmax=13, center=1, cmap='YlOrRd', cbar=None, outname=None):
        plt.figure(figsize=(6,6))
        ax=sns.heatmap(self.statstabnorm, annot=True, vmin=vmin, vmax=vmax, center=center, cmap=cmap, cbar=cbar)
        plt.xlabel("neighbor label")
        plt.ylabel("datapoint label")
        plt.title("Normalized Nearest Neighbor Frequency Snorm")
        if outname:
            plt.savefig(outname)
    
    def plot_heat_fold(self, center=1, cmap='YlOrRd', cbar=None, outname=None):
        plt.figure(figsize=(6,6))
        ax=sns.heatmap(np.power(2,self.statstabnorm), annot=True, center=center, cmap=cmap, cbar=cbar)
        plt.xlabel("neighbor label")
        plt.ylabel("datapoint label")
        plt.title("Nearest Neighbor fold likelihood")
        if outname:
            plt.savefig(outname) 
    
    
class sil:
    """
    A class to represent Silhouette score statistics for a
    given latent space representation of a labelled dataset

    Attributes
    ----------
    embedding : 2D numpy array (numeric)
                a dataset E(X,Y) with X datapoints and Y dimensions
                
    labels: 1D numpy array (string) or list of strings
            vector/list of class labels in dataset

    
    labeltypes: list of strings
                 alphanumerically sorted set of class labels
                 
    avrg_SIL: Numeric (float)
              The average Silhouette score of the dataset
             
    sample_SIL: 1D numpy array (numeric)
                The Silhouette scores for each datapoint in the dataset
    
    Methods
    -------
    
    get_avrg_score():
        returns the average Silhouette score of the dataset
    
    get_score_per_class():
        returns the average Silhouette score per class for each
        class in the dataset as 1D numpy array
        (alphanumerically sorted classes)
        
    get_sample_scores():
        returns the Silhouette scores for each datapoint in the dataset
        (1D numpy array, numeric)
             
    
    """
    def __init__(self, embedding, labels):
        
        self.embedding = embedding
        self.labels = labels
        self.labeltypes = sorted(list(set(labels)))
        
        self.avrg_SIL = silhouette_score(embedding, labels)
        self.sample_SIL = silhouette_samples(embedding, labels)
    
    def get_avrg_score(self):
        return self.avrg_SIL
    
    def get_score_per_class(self):
        scores = np.zeros((len(self.labeltypes),))
        for i, label in enumerate(self.labeltypes):
            ith_cluster_silhouette_values = self.sample_SIL[self.labels == label]
            scores[i] = np.mean(ith_cluster_silhouette_values)
            #scores_tab = pd.DataFrame([scores],columns=self.labeltypes)
        return scores
    
    def get_sample_scores(self):
        return self.sample_SIL

    

import sklearn
from sklearn.metrics.pairwise import euclidean_distances  

def next_sameclass_nb(embedding, labels):
    indices = []
    distmat = euclidean_distances(embedding, embedding)
    k = embedding.shape[0]-1
    
    nbs_to_sameclass = []

    for i in range(distmat.shape[0]):
        neighbors = []
        distances = distmat[i,:]
        ranks = np.array(distances).argsort().argsort()
        for j in range(1,embedding.shape[0]):
            ind = np.where(ranks==j)[0]
            nb_label = labels[ind[0]]
            neighbors.append(nb_label)
        
        neighbors = np.asarray(neighbors)
            
        # How many neighbors until I encounter a same-class neighbor?
        own_type = labels[i]
        distances = distmat[i,:]
        ranks = np.array(distances).argsort().argsort()
        neighbors = []
        for j in range(1,embedding.shape[0]):
            ind = np.where(ranks==j)[0]
            nb_label = labels[ind[0]]
            neighbors.append(nb_label)
        
        neighbors = np.asarray(neighbors)
            
        # How long to same-class label?
        own_type = labels[i]
        first_occurrence = np.where(neighbors==labels[i])[0][0]
    
        nbs_to_sameclass.append(first_occurrence)
    
    return(np.asarray(nbs_to_sameclass))



