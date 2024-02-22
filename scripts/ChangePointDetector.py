#!/usr/bin/env python3
#

import numpy as np
from utils import metric_wasserstein_change_points, kolmogorov_smirnov_change_points


class ChangePointDetector:
    """
    initialize an empty ChangePointDetectorObject
    """

    def __init__(self):
        self._data = None
        self._ground_truth = None
        self._predictions = None
        self._f1 = None
        self._recall = None
        self._precision = None
        return

    """
    set the data property of the ChangePointDetector
    expects a string filename to represent the path to a
    CSV txt file
    resets the ground_truth and the predictions to None
    """

    def set_data(self, filename):
        try:
            self._data = np.loadtxt(filename)
            self._ground_truth = None
            self._predictions = None
        except:
            print("could not load data, check file name and type")
        return

    """
    return a copy of the data as it is currently set
    """

    def get_data(self):
        if self._data is None:
            raise Exception("data has not been loade")
        return self._data

    """
    set the ground_truth property of the ChangePointDetector
    expects a string filename to represent the path to a
    CSV txt file
    """

    def set_truth(self, filename):
        try:
            self._ground_truth = np.loadtxt(filename)
        except:
            print("could not set ground truth, check file name and type")
        return

    """
    sets the predictions property of the ChangePointDetector
    via the specified method and hyperparameters
    expects method to be a string, windowsize to be an integer, and cutoff to be
    some floating point number between 0 and 1
    """

    def get_f1_stats(self):
        return self._recall, self._precision, self._f1

    def compute_change_points(self, method="wasserstein", windowsize=250, cutoff=0.85):
        if method == "wasserstein":
            self._predictions = metric_wasserstein_change_points(
                self._data, windowsize=windowsize, quantile=cutoff
            )
        elif method == "kolmogorov":
            self._predictions = kolmogorov_smirnov_change_points(
                self._data, windowsize=windowsize, cutoff=cutoff
            )
        else:
            raise Exception("passed method is not valid")
        return

    """
    if data has already been provided, denoise it by taking local averages
    over a prescribed windowsize
    """

    def denoise_data(self, windowsize):
        if self._data is None:
            raise Exception("no data has been set")
        leading_batch = self._data[0:windowsize]
        terminal_batch = self._data[(len(self._data) - windowsize) : len(self._data)]
        if np.isnan(np.mean(terminal_batch)):
            print(terminal_batch)
            raise Exception("bad terminal_batch")
        front_padding = [np.mean(leading_batch)] * windowsize
        back_padding = [np.mean(terminal_batch)] * windowsize
        denoised_data = []
        for t in range(windowsize, len(self._data) - windowsize):
            window = self._data[(t - windowsize) : (t + windowsize)]
            denoised_data.append(np.mean(window))
        print(len(front_padding))
        print(len(back_padding))
        print(len(denoised_data))
        self._data = np.array(front_padding + denoised_data + back_padding)
        return

    """
    sets the f1, recall, and precision properties of the ChangePointDetector
    expects a specificed integer error tolerance e, set to 50 by default
    """

    def compute_f1_stats(self, e=50):
        if self._data is None:
            raise Exception("no dataset has been loaded")
        if self._ground_truth is None:
            raise Exception("no set of ground truth labels has been loaded")
        if self._predictions is None:
            raise Exception("predictions have not been made")
        tp, fp, fn = 0, 0, 0
        change_points = [t in self._predictions for t in range(len(self._data))]
        truth = [t in self._ground_truth for t in range(len(self._data))]
        for position, candidate in enumerate(change_points):
            if position - e < 0:
                truth_subset = truth[0 : (position + e)]
            elif position + e > len(self._data):
                truth_subset = truth[(position - e) :]
            else:
                truth_subset = truth[(position - e) : (position + e)]

            if candidate and (True in truth_subset):
                tp += 1
            elif candidate and (not True in truth_subset):
                fp += 1
            elif (not candidate) and (True in truth_subset):
                fn += 1

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 / ((1 / recall) + (1 / precision))
        self._f1 = f1
        self._recall = recall
        self._precision = precision

        return
