#/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import W2ChangePoints
from scipy.stats import iqr

DATA_PATH = os.path.join("..", "data", "Langevin_1D.txt")
GROUND_PATH = os.path.join("..", "data", "ChangePts.txt")
IMG_FOLDER = os.path.join("..", "img")

def cdf_grapher(samples, filename, cdf=False):
    plt.clf()
    fig, ax = plt.subplots()
    for sample in samples:
        if cdf:
            sample = np.sort(sample)
        #ax.plot(np.sort(sample), np.linspace(0, 1, len(sample), endpoint=False))
        ax.plot(np.linspace(0, 1, len(sample), endpoint=False), np.sort(sample))
    plt.savefig(filename)

def fourier_cdf_comp(samples):
    sorted = [np.sort(sample) for sample in samples]
    transforms = [np.fft.fft(cdf) for cdf in sorted]
    ratios = [ratio_of_transforms(transforms[i], transforms[i + 1]) for i in range(len(transforms) - 1)]
    norms = [np.linalg.norm(ratio) for ratio in ratios]
    return norms


def ratio_of_transforms(ft1, ft2):
    return([x / y for (x,y) in zip(ft1, ft2)])

def bin_op(bin1, bin2):
    ft1 = np.fft.fft(bin1)
    ft2 = np.fft.fft(bin2)
    logft1 = np.array([np.log(x) for x in ft1])
    logft2 = np.array([np.log(x) for x in ft2])
    return logft1 - logft2


def foo():
    samples = DATA
    first_bins = BINS
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    norms = fourier_cdf_comp(first_bins)
    norms.append(norms[-1])
    ax.plot(np.arange(len(samples)), samples)
    ax.plot(np.arange(len(norms)), norms)
    cutoff = 1
    for index, diff in enumerate(norms):
        if abs(diff) > cutoff:
            ax.vlines(index, -2, 2, color='red', linestyle='dashed')
    print(np.mean(norms), np.median(norms), iqr(norms))
    plt.savefig(os.path.join(IMG_FOLDER, "test2.png"))


def make_bins(path: str, alpha: int):
    raw_data = np.loadtxt(path)
    return np.array_split(raw_data, int(len(raw_data) / alpha))

def plot_wass_diff_change_point_cdfs():
    q = 95
    alpha = 100
    cpd = W2ChangePoints.ChangePointDetector(DATA_PATH, GROUND_PATH)
    bins = make_bins(DATA_PATH, alpha)
    change_points = cpd.compute_change_points(quantile=q, sample_ratio=alpha)
    fig, ax = plt.subplots()
    for index, point in enumerate(change_points):
        if point:
            sample = np.sort(bins[index])
            ax.plot(sample, np.linspace(0, 1, len(sample), endpoint=False))
    plt.savefig(os.path.join(IMG_FOLDER, "change_pt_cdfs.png"))




if __name__ == "__main__":
    main()
