#!/usr/bin/env python3
#

"""
some utility functions for the ChangePointDetector class
these functions should not be exposed via that class,
but are necessary for it to perform its prescribed functionality
"""


from ot import emd2_1d
from scipy.stats import kstest
import numpy as np


"""
given a timeseries data, a windowzize, and a quantile cutoff
use the metric wasserstein derivative to predict change points in the time series
at each position t, let mu_0 be the windowsize points to the left of t
and mu_1 be the windowsize points to the right of t
compute the wasserstein distance between mu_0 and mu_1
return the the indices of the top quantile of distances
"""


def metric_wasserstein_change_points(data, windowsize, quantile):
    # we can't say there's a change point until we've collected
    # enough data to set a baseline -- we use the windowsize to
    # set this threshold of "enough data"
    change_points = []
    differences = [
        emd2_1d(data[(t - windowsize) : t], data[t : (t + windowsize)])
        for t in range(windowsize, len(data) - windowsize + 1)
    ]
    cutoff = np.quantile(differences, quantile)
    for t in range(len(differences)):
        if differences[t] > cutoff:
            change_points.append(t + windowsize)
    return change_points


"""
given a timeseries data, a windowzize, and a quantile cutoff
use the Kolmogorov-Smirnov test to predict change points in the time series
at each position t, let mu_0 be the windowsize points to the left of t
and mu_1 be the windowsize points to the right of t
compute the KS Statistic between the emprical CDFs of mu_0 and mu_1
return all indices for which the statistic exceeds the cutoff
"""


def kolmogorov_smirnov_change_points(data, windowsize, cutoff):
    # we can't say there's a change point until we've collected
    # enough data to set a baseline -- we use the windowsize to
    # set this threshold of "enough data"
    change_points = []
    kolmogorov_statistics = [
        kstest(data[(t - windowsize) : t], data[t : (t + windowsize)])
        for t in range(windowsize, len(data) - windowsize)
    ]
    for t in range(len(kolmogorov_statistics)):
        if kolmogorov_statistics[t] > cutoff:
            change_points.append(t + windowsize)
    return change_points
