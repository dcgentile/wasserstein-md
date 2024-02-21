#!/usr/bin/env python3
#
# the plan:
# read in the data
# doing a moving window scan to get a moving mean value of the data over time
# then do Wasserstein CPD on that information
# eventually, we should try to define terminals of a geodesic in W2 space using this information


import os
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

IMG_FOLDER = '../img/'

raw_data = np.loadtxt("../data/Langevin_1D.txt")
data = [np.array([v]) for v in raw_data]

SAMPLES_PER_BIN = 100
SAMPLE_COUNT = len(data)

means = [np.mean(data[i:i + SAMPLES_PER_BIN]) for i in range(SAMPLE_COUNT - SAMPLES_PER_BIN)]
kmeans = KMeans(n_clusters=2, n_init='auto').fit(data)

fig, ax = plt.subplots()
ax.plot(np.arange(len(means)), means)
plt.savefig(os.path.join(IMG_FOLDER, 'means.png'))

def dfScatter(df, xcol='time', ycol='val', catcol='cluster'):
    fig, ax = plt.subplots()
    categories = np.unique(df[catcol])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))

    df["Color"] = df[catcol].apply(lambda x: colordict[x])
    ax.scatter(df[xcol], df[ycol], c=df.Color)
    return fig

if 1:
    df = pd.DataFrame(data, columns=['val'])
    df['time'] = np.arange(SAMPLE_COUNT)
    df['cluster'] = kmeans.labels_
    fig = dfScatter(df)
    fig.savefig(os.path.join(IMG_FOLDER, 'fig1.png'))
