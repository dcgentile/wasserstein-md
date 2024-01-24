#!/usr/bin/env python3

import os
import ot
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class W2ChangePoints:
    def __init__(self,
                 data_file: str,
                 truth_file: str):
        self.data = [np.array([v]) for v in np.loadtxt(data_file)]
        self.truth = [np.array([v]) for v in np.loadtxt(truth_file)]

    def compute_wass_diffs(self, samples_per_bin):
        bin_count = int(len(data) / samples_per_bin)
        bins = np.array_split(self.data, bin_count)
        return [ot.emd2_1d(bins[i], bins[i + 1]) for i in range(BIN_COUNT-1)]

    def compute_change_points(self,
                              q:int,
                              alpha: int):
         change_points = []
         dw = self.compute_wass_diffs(alpha)
         quantile = np.quantile(dw, q)
         for index, sample in enumerate(dw):
             if sample - quantile > 0:
                 change_points.append(index)
         return change_points

    def f1_score(self,
                 t: int,
                 q: int,
                 alpha: int):
        # compute the F1 score with tolerance t, quantile q, samples/bin alpha
        change_points = self.compute_change_points(q, alpha)

        pass
