#!/usr/bin/env python
# coding: utf-8

# Evaluate an embedding

import os
import pandas as pd
import sys
import numpy as np
from pandas.core.common import flatten
import pickle
from pathlib import Path
import datetime
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import librosa.display
import random
from scipy.spatial.distance import pdist, squareform
import umap

from plot_functions import umap_2Dplot, mara_3Dplot, plotly_viz
from preprocessing_functions import pad_spectro, calc_zscore, create_padded_data
from evaluation_functions import nn,sil, plot_within_without


# Setting project, input and output folders.
wd = os.getcwd()
DATA = os.path.join(os.path.sep, str(Path(wd).parents[0]), "data", "processed")
FIGURES = os.path.join(os.path.sep, str(Path(wd).parents[0]), "reports", "figures")

LOAD_EXISTING = True
PLOTTING = False

for embedding_id in ['full', 'reduced', 'balanced', 'reducedrandom']:
    
    if embedding_id=="reducedrandom":
        spec_df = pd.read_pickle(os.path.join(os.path.sep, DATA, "df_focal_reduced.pkl"))
    else:
        spec_df = pd.read_pickle(os.path.join(os.path.sep, DATA, "df_focal_"+embedding_id+".pkl"))
    
    labels = spec_df.call_lable.values
    
    # Prepare data
    specs = spec_df.spectrograms.copy()
    specs = [calc_zscore(x) for x in specs] 
    data = create_padded_data(specs)
    
    # UMAP
    #embedding_filename = os.path.join(os.path.sep, DATA,'basic_UMAP_'+embedding_id+'_default_params.csv')
    
    embeddings = {}
    
    for n_dims in [2,3]:
        dim_type = str(int(n_dims))+'D'
        embedding_filename = os.path.join(os.path.sep, DATA, 'basic_UMAP_'+dim_type+'_'+embedding_id+'_default_params.csv')
        print(embedding_filename)
        
        if (LOAD_EXISTING and os.path.isfile(embedding_filename)):
            embeddings[dim_type] = np.loadtxt(embedding_filename, delimiter=";")
            print("File already exists")
        else:
            if embedding_id=='reducedrandom':
                distmat = squareform(pdist(data, metric='euclidean'))

                flattened_dists = distmat[np.triu_indices(n=distmat.shape[0], k=1)]
                random.seed(100)
                np.random.shuffle(flattened_dists)
                random_distmat = np.zeros(distmat.shape)
                random_distmat[np.triu_indices(n=distmat.shape[0], k=1)] = flattened_dists
                for i in range(random_distmat.shape[0]):
                    for j in range(i,random_distmat.shape[1]):
                        random_distmat[j,i] = random_distmat[i,j]  

                reducer = umap.UMAP(n_components=n_dims, min_dist=0, metric='precomputed', random_state=2204)
                embeddings[dim_type] = reducer.fit_transform(random_distmat)

            else:
                reducer = umap.UMAP(n_components=3, min_dist = 0, random_state=2204)
                embeddings[dim_type] = reducer.fit_transform(data)

            np.savetxt(embedding_filename, embeddings[dim_type], delimiter=";")
    
    
    embedding = embeddings['3D']
    embedding_2D = embeddings['2D']
    # Plotting
    pal="Set2"

    ## 2D Plots
    if PLOTTING:
        umap_2Dplot(embedding_2D[:,0], 
                    embedding_2D[:,1], 
                    labels, 
                    pal, 
                    os.path.join(os.path.sep, FIGURES, 'UMAP_2D_plot_'+embedding_id+'_nolegend.jpg'), 
                    showlegend=False)
        plt.close()


        ## 3D Plot
        mara_3Dplot(embedding[:,0],
                    embedding[:,1],
                    embedding[:,2],
                    labels,
                    pal,
                    os.path.join(os.path.sep, FIGURES, 'UMAP_3D_plot_'+embedding_id+'_nolegend.jpg'),
                    showlegend=False)
        plt.close()


    # Embedding evaluation

    # Evaluate the embedding based on calltype labels of nearest neighbors.

    nn_stats = nn(embedding, np.asarray(labels), k=5)
    print("Log final metric (unweighted):",nn_stats.get_S())
    print("Abs final metric (unweighted):",nn_stats.get_Snorm())
    

    if PLOTTING:
        nn_stats.plot_heat_S(outname=os.path.join(os.path.sep, FIGURES, 'heatS_UMAP_'+embedding_id+'.png'))
        nn_stats.plot_heat_Snorm(outname=os.path.join(os.path.sep, FIGURES, 'heatSnorm_UMAP_'+embedding_id+'.png'))
        nn_stats.plot_heat_fold(outname=os.path.join(os.path.sep, FIGURES, 'heatfold_UMAP_'+embedding_id+'.png'))


        ## Within vs. outside distances
        plot_within_without(embedding=embedding, labels=labels, outname="distanceswithinwithout_"+embedding_id+"_.png")
        plt.close()


    ## Silhouette Plot
    sil_stats = sil(embedding, labels)    
    print("SIL: ", sil_stats.get_avrg_score())
    
    if PLOTTING:
        sil_stats.plot_sil(outname=os.path.join(os.path.sep, FIGURES, 'silplot_UMAP_'+embedding_id+'.png'))
        plt.close()

        ## Graph from embedding evaluation
        outname = os.path.join(os.path.sep,FIGURES,'simgraph_'+embedding_id+'.png')
        nn_stats.draw_simgraph(outname)
        plt.close()