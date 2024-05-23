#!/usr/bin/env python3
"""
author: david gentile
date: 05/15/2024
utiliy functions for high dimensional state analysis in molecular dynamics
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import ot
from sklearn.cluster import KMeans

"""
given a pandas dataframe with rows corresponding to
time-ordered observations of a molecular configuration,
compute the 2-Wasserstein distance between neighboring
distributions of size w and return the (time-ordered)
list of distances
"""


def compute_wasserstein_distances(data: pd.DataFrame, w: int, reg=0.01) -> np.ndarray:
    n = len(data)
    distances = np.zeros((n))
    for t in tqdm(range(w, n - w)):
        a = data[t - w : t].to_numpy()
        b = data[t + 1 : t + w].to_numpy()
        distances[t] = ot.bregman.empirical_sinkhorn2(a, b, reg)
    return distances


"""
given a list of time-ordered 2-Wasserstein distances,
and a cutoff quantile q
return a list of candidate change points
"""


def indentify_change_points(differences: np.ndarray, q=0.85) -> np.ndarray:
    change_pts = []
    cutoff = np.quantile(differences, q)
    for index, dist in enumerate(differences):
        if dist > cutoff:
            change_pts.append(index)
    return np.array(change_pts)


"""
given a list of candidate change points and the
distances data, filter out consecutive candidates
"""


def filter_change_points(
    differences: np.ndarray,
    change_points: np.ndarray,
) -> np.ndarray:
    # find the boundaries of the regions containing consecutive CP candidates
    boundary_pts = []
    for i in range(len(change_points) - 1):
        if change_points[i + 1] - change_points[i] > 1:
            boundary_pts.append(change_points[i])
            boundary_pts.append(change_points[i + 1])

    # filter for and output the final change points
    filtered_points = []
    for i in range(len(boundary_pts) - 1):
        lo = boundary_pts[i]
        hi = boundary_pts[i + 1]
        if hi - lo > 1:
            subset = differences[lo:hi]
            if type(subset) is not np.ndarray:
                subset = np.array(subset)
            filtered_points.append(subset.argmax() + lo)
    return np.array(filtered_points)


"""
given a dataframe containing time ordered observations
of a molecular configuration, and a series of change points,
create a matrix of high dimensional vectors formed by
sampling from the slices of the data induced by the change points
"""


def create_sample_distributions(
    data: pd.DataFrame, change_points: np.ndarray, start: int, end: int
) -> np.ndarray:
    ECDF = []
    for i in range(len(change_points)):
        if i!=0 and i < len(change_points)-1:
            prev_cp = change_points[i]
            curr_cp = change_points[i+1]
        elif i==0:
            prev_cp = start
            curr_cp = change_points[i+1]
        else:
            prev_cp = change_points[i]
            curr_cp = end
        # Construct empirical CDF over datapoints between two change points
        orig_CDF = np.sort(data[prev_cp:curr_cp])
        # Interpolate to obtain an expanded empirical CDF of 500 points
        ecdf = np.interp(np.linspace(0, len(orig_CDF) - 1, 500), np.arange(len(orig_CDF)), orig_CDF)
        ECDF.append(ecdf)
    return np.array(ECDF)


"""
given an array of vectors representing the slices of the original dataset
perform a clustering method to label the sample distributions and return
a dictionary containing the distributions and their labels
"""


def cluster_sample_distributions(samples: np.ndarray) -> np.ndarray:
    Kmeans = KMeans(max_iter=1000)
    Kmeans.fit(samples)
    return Kmeans.labels_


"""
given a dictionary containing the cluster label information for
the dataset, return a markov model correspondig to the transitions
between labels
"""


def create_markov_model_from_labels(
    label_dict: dict,
) -> np.ndarray:
    return np.array([0])
