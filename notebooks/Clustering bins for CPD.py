#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import ot
import ot.plot
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

data = np.loadtxt("data/Langevin_1D.txt")

SAMPLES_PER_BIN = 100
SAMPLE_COUNT = len(data)
BIN_COUNT = int(SAMPLE_COUNT / SAMPLES_PER_BIN)
bins = np.array_split(data, BIN_COUNT)

Points = []
for b in bins:
    pts = np.sort(b)
    Points.append(pts)
    
df = pd.DataFrame(Points) 
np_arr = np.array(Points)

X = np_arr
# Choose the number of clusters (k)
k = 3

# Create and fit the KMeans clusterer
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(X)

plt.figure(figsize=(20, 5))
plt.plot(np.arange(100000), data, 'r', alpha=0.7)

for i in range(len(labels)):
    if labels[i] == 0:
        plt.plot(np.arange(SAMPLES_PER_BIN * i,SAMPLES_PER_BIN * (i+1)), data[SAMPLES_PER_BIN * i:SAMPLES_PER_BIN * (i+1)], 'b')
    if labels[i] == 1:
        plt.plot(np.arange(SAMPLES_PER_BIN * i,SAMPLES_PER_BIN * (i+1)), data[SAMPLES_PER_BIN * i:SAMPLES_PER_BIN * (i+1)], 'orange')
    if labels[i] == 2:
        plt.plot(np.arange(SAMPLES_PER_BIN * i,SAMPLES_PER_BIN * (i+1)), data[SAMPLES_PER_BIN * i:SAMPLES_PER_BIN * (i+1)], 'g')


# In[ ]:




