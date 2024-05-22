#!/usr/bin/env python3
"""
author: david gentile
date: 05/15/2024
utiliy functions for high dimensional state analysis in molecular dynamics
"""
import pandas as pd
import numpy as np

"""
given a pandas dataframe with rows corresponding to
time-ordered observations of a molecular configuration,
compute the 2-Wasserstein distance between neighboring
distributions of size w and return the (time-ordered)
list of distances
"""


def compute_wasserstein_distances(
    data: pd.DataFrame, w: int
) -> np.ndarray[int, np.dtype[np.int128]]:
    return np.array([0])


"""
given a list of time-ordered 2-Wasserstein distances,
return a list of candidate change points
"""


def filter_change_points(
    change_points: np.ndarray,
) -> np.ndarray[int, np.dtype[np.int128]]:
    return np.array([0])


"""
given a dataframe containing time ordered observations
of a molecular configuration, and a series of change points,
create a matrix of high dimensional vectors formed by
sampling from the slices of the data induced by the change points
"""


def create_sample_distributions(
    data: pd.DataFrame, change_points: np.ndarray
) -> np.ndarray[int, np.dtype[np.float128]]:
    return np.array([0.0])


"""
given an array of vectors representing the slices of the original dataset
perform a clustering method to label the sample distributions and return
a dictionary containing the distributions and their labels
"""


def cluster_sample_distributions(samples: np.ndarray) -> dict:
    return {}


"""
given a dictionary containing the cluster label information for
the dataset, return a markov model correspondig to the transitions
between labels
"""


def create_markov_model_from_labels(
    label_dict: dict,
) -> np.ndarray[int, np.dtype[np.int128]]:
    return np.array([0])
