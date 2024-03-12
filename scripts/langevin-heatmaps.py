#!/usr/bin/env python3
from ChangePointDetector import ChangePointDetector
import numpy as np
from tqdm import tqdm

detector = ChangePointDetector()
detector.set_data("../data/Langevin_1D.txt")
detector.set_truth("../data/Langevin_1D_change_points.txt")


def generate_matrices(
    method, f1_filename, recall_filename, i_lo, i_hi, i_step, j_lo, j_hi, j_step
):
    row_count = int((i_hi - i_lo) / i_step)
    col_count = int((j_hi - j_lo) / j_step)
    recall_matrix = np.empty((row_count, col_count))
    f1_matrix = np.empty((row_count, col_count))

    for i, q in tqdm(enumerate(range(i_lo, i_hi, i_step))):
        # save the data each time we make it through a quantile setting
        np.save(f1_filename, f1_matrix)
        np.save(recall_filename, recall_matrix)
        for j, w in enumerate(range(j_lo, j_hi, j_step)):
            detector.compute_change_points(
                method=method, windowsize=w, cutoff=(0.01 * q)
            )
            detector.compute_f1_stats()
            recall_matrix[i, j] = detector.get_f1_stats()[0]
            f1_matrix[i, j] = detector.get_f1_stats()[2]

    np.save(f1_filename, f1_matrix)
    np.save(recall_filename, recall_matrix)


wass_f1 = "wass_f1.npy"
wass_recall = "wass_recall.npy"
ks_f1 = "ks_f1.npy"
ks_recall = "ks_recall.npy"
smooth_wass_f1 = "smooth_wass_f1.npy"
smooth_wass_recall = "smooth_wass_recall.npy"
smooth_ks_f1 = "smooth_ks_f1.npy"
smooth_ks_recall = "smooth_ks_recall.npy"

generate_matrices("wasserstein", wass_f1, wass_recall, 78, 96, 9, 150, 750, 150)
