import os
import ot
import ot.plot

from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

from scipy.signal import find_peaks

###################################################################################################################
###################################################################################################################

# Denoising the data

###################################################################################################################
###################################################################################################################

def knearest_denoising(data, ws):
    return [np.average(data[(t - ws):(t + ws)]) for t in range(len(data))]

###################################################################################################################
###################################################################################################################

# Compute two sample test stats

###################################################################################################################
###################################################################################################################

"""
parameters:
    data array:            data
    bin size:              bs
    sliding window length: s
returns:
    Wasserstein distances between each bin
"""
def qW2(data, n, s):
    cpd_stat = np.zeros(len(data)// s, )
    count = 0
    for i in range(0, len(data)-s, s):
        if i<n or i>= len(data)- n:
            cpd_stat[count] = 0
        else:
            x, y = data[i-n:i], data[i: i+n]
            cpd_stat[count] = ot.emd2_1d(x, y)
        count += 1
    return cpd_stat  

"""
#computing Kolmogorov Smirnoff scores
def get_KS_score(samples, n, s):
    ks_score = np.zeros(len(samples)// s, )
    count = 0
    for i in range(0, len(samples)-s, s):
        if i<n or i>= len(samples)- n:
            ks_score[count] = 0
        else:
            x, y = data[i-n:i], data[i: i+n]
            stat, pval = stats.ks_2samp(x,y)
            ks_score[count] = stat
        count += 1
    return ks_score

#computing empirical characteristic function scores
def get_ECF_score(samples, n, s):
    ECF_score = np.zeros(len(samples)// s, )
    count = 0
    for i in range(0, len(samples)-s, s):
        if i<n or i>= len(samples)- n:
            ECF_score[count] = 0
        else:
            x, y = data[i-n:i], data[i: i+n]
            stat, pval = stats.epps_singleton_2samp(x,y)
            ECF_score[count] = stat
        count += 1
    return ECF_score
"""

###################################################################################################################
###################################################################################################################

# Clustering Raw data

###################################################################################################################
###################################################################################################################
"""
parameters:
    data array:         data
    bin size:           bs
    number of clusters: k
returns:
    cluster labels for each bin
"""
def raw_cluster_Kmeans(data, bs, k):
    num_bins = int(len(data) / bs)
    bins = np.array_split(data, num_bins)
    X = np.array(bins)
    
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    
    return labels

def raw_cluster_spectral(data, bs, k):
    num_bins = int(len(data) / bs)
    bins = np.array_split(data, num_bins)
    X = np.array(bins)
    
    spec = SpectralClustering(k)
    labels = spec.fit_predict(X)
    
    return labels

###################################################################################################################
###################################################################################################################

# Clustering Empirical PDFs

###################################################################################################################
###################################################################################################################
"""
parameters:
    data array:         data
    bin size:           bs
    number of clusters: k
returns:
    cluster labels for each bin
"""
def ECDF_cluster_Kmeans(data, bs, k):
    num_bins = int(len(data) / bs)
    bins = np.array_split(data, num_bins)
    
    Points = []
    for b in bins:
        pts = np.sort(b)
        Points.append(pts)
    
    X = np.array(Points)
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    
    return labels

def ECDF_cluster_spectral(data, bs, k):
    num_bins = int(len(data) / bs)
    bins = np.array_split(data, num_bins)
    
    Points = []
    for b in bins:
        pts = np.sort(b)
        Points.append(pts)
    
    X = np.array(Points)
    spec = SpectralClustering(k)
    labels = spec.fit_predict(X)
    
    return labels

###################################################################################################################
###################################################################################################################

# Compute Change Points

###################################################################################################################
###################################################################################################################
"""
parameters:
    cluster labels
returns:
    change points
"""
def get_change_points_clusters(cluster_label):
    cp_detected = []
    
    for i in range(1,len(cluster_label)):
        if (cluster_label[i-1] != cluster_label[i]):
            cp_detected.append(i)
    
    return cp_detected  

"""
parameters:
    computed Wasserstein distances: wass_dist
    quantile:                       q
    width of peak filter:           w
returns:
    change points
"""
def get_change_points_Wass(wass_dist, q, w):
    #Need to optimize over width as well otherwise noise will affect detection
    quantile = np.quantile(wass_dist, q)
    peaks, _ = find_peaks(wass_dist, height=quantile, width=w)
    
    return peaks

###################################################################################################################
###################################################################################################################

# Computing F1 Scores

###################################################################################################################
###################################################################################################################
def test_func_detection_stat(labels, true_cp, bs):
    TP=0
    for i in labels:
        for j in true_cp:
            if abs(i*bs - j) < 100:
                TP+=1
                break
    FP = len(labels) - TP
    FN = len(true_cp) - TP
    F1 = (2*TP)/(2*TP+FN+FP)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return F1, Precision, Recall

"""
parameters:
    detected change points: detected
    true change points:     true_cp
    bin size:               bs
    threshold for error
returns:
    F1 Score, Precision, Recall

"""
def detection_statistic_Wass(data, true_cp, bs, q, s, threshold=100):
    
    detected_cps = qW2(data, bs, s)
    quantile = np.quantile(detected_cps, q)
    peaks, _ = find_peaks(detected_cps, height=quantile, width=0)
    TP = 0
    
    #Threshold set at 100pts away from true change point
    for i in peaks:
        for j in true_cp:
            if abs(i*bs - j) < 100:
                TP+=1
                break
    FP = len(peaks) - TP
    FN = len(true_cp) - TP
    F1 = (2*TP)/(2*TP+FN+FP)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return F1, Precision, Recall

def detection_statistic_clustering(detected, true_cp, bs, threshold=100):
    TP = 0
    
    #Threshold set at 100pts away from true change point
    for i in detected:
        for j in true_cp:
            if abs(i*bs - j) < 100:
                TP+=1
                break
    FP = len(detected) - TP
    FN = len(true_cp) - TP
    F1 = (2*TP)/(2*TP+FN+FP)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return F1, Precision, Recall

###################################################################################################################
###################################################################################################################

# Visualizations

###################################################################################################################
###################################################################################################################

def make_graph_clusters(data, labels, bs):
    plt.figure(figsize=(20, 5))
    plt.plot(np.arange(100000), data, 'r', alpha=0.7)

    for i in range(len(labels)):
        if labels[i] == 0:
            plt.plot(np.arange(bs * i,bs * (i+1)), data[bs * i:bs * (i+1)], 'b')
        if labels[i] == 1:
            plt.plot(np.arange(bs * i,bs * (i+1)), data[bs * i:bs * (i+1)], 'orange')
        if labels[i] == 2:
            plt.plot(np.arange(bs * i,bs * (i+1)), data[bs * i:bs * (i+1)], 'g')

def make_graph_given_cp_labels(data,labels):
    fig, ax = plt.subplots()
    fig.set_size_inches(24, 5)
    ax.vlines(labels, -2, 2, color='purple', linestyle='dashed')
    ax.plot(np.arange(len(data)), data, 'r', alpha=0.7)
    plt.title("Langevin Trajectory with labelled change points")
    
            
def make_graph_CPS(data, bs, q, s):
    fig, ax = plt.subplots()
    fig.set_size_inches(24, 5)
    
    detected_cps = qW2(data, bs, s)
    quantile = np.quantile(detected_cps, q)
    peaks, _ = find_peaks(detected_cps, height=quantile, width=0)
    
    ax.vlines(peaks*bs, -2, 2, color='purple', linestyle='dashed')
    
    ax.plot(np.arange(len(data)), data, 'r', alpha=0.7)
    plt.title("Langevin Trajectory with labelled change points")

def plt_empirical_cdf(dat, start, stop, bs):
    bins = np.array_split(dat, int(len(dat) / bs))
    #bins = splt_data[start:stop]
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(bins)-1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    
    ax1.plot(np.arange(100000), dat, 'r')
    ax1.vlines(start*bs, -2, 2, linestyle='dashed')
    ax1.vlines(stop*bs, -2, 2, linestyle='dashed')
    
    for i in range(start, stop):
        ax2.plot(np.sort(bins[i]), np.arange(100), c=cmap(norm(i-start)))
        ax2.plot(min(bins[i]), 0, 'x') 
        ax2.annotate((i)*bs, (min(bins[i]), 0))

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Time')
    
    # Fix the tick locations etc.. for the colorbar
    cbar.set_ticks([0,len(bins)-1])  
    cbar.set_ticklabels([start*bs, stop*bs])

    plt.xlabel('Position')
    plt.title('Empirical CDFs')
    
    str1_lst = ['Fig (', start, '-', stop, ').png']
    str1 = ''.join(map(str,str1_lst))
    plt.savefig(str1)
    plt.show()

def plt_cluster_cdf(data, labels, start, stop, bs):  
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    
    ax1.plot(np.arange(100000), data, 'r', alpha=0.7)

    for i in range(len(labels)):
        if labels[i] == 0:
            ax1.plot(np.arange(bs * i,bs * (i+1)), data[bs * i:bs * (i+1)], 'b')
        if labels[i] == 1:
            ax1.plot(np.arange(bs * i,bs * (i+1)), data[bs * i:bs * (i+1)], 'orange')
        if labels[i] == 2:
            ax1.plot(np.arange(bs * i,bs * (i+1)), data[bs * i:bs * (i+1)], 'g')
    
    ax1.vlines(start*bs, -2, 2, linestyle='dashed')
    ax1.vlines(stop*bs, -2, 2, linestyle='dashed')
    
    for j in range(start,stop+1):
        if labels[j] == 0:
            ax2.plot(np.sort(data[bs * j:bs * (j+1)]), np.arange(bs), 'b')
        if labels[j] == 1:
            ax2.plot(np.sort(data[bs * j:bs * (j+1)]), np.arange(bs), 'orange')
        if labels[j] == 2:
            ax2.plot(np.sort(data[bs * j:bs * (j+1)]), np.arange(bs), 'g')
    
    
    
    
    
    