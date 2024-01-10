#!/usr/bin/env python3

import os
import ot
import ot.plot
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

IMG_FOLDER = './img'

SAMPLES_PER_BIN = 100
data = np.loadtxt("Langevin_1D.txt")
data = data
SAMPLE_COUNT = len(data)
BIN_COUNT = int(SAMPLE_COUNT / SAMPLES_PER_BIN)
bins = np.array_split(data, BIN_COUNT)

dw = [ot.emd2_1d(bins[i], bins[i + 1]) for i in range(BIN_COUNT-1)]


def make_graph(samples, q, filename):
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 5)
    fig.set_size_inches(20, 5)
    quantile = np.quantile(samples, q)
    for index, sample in enumerate(samples):
        if sample - quantile > 0:
            ax.vlines(SAMPLES_PER_BIN * index, -2, 2, color='purple', linestyle='dashed')

    ax.plot(np.arange(SAMPLE_COUNT), data, 'r', alpha=0.7)
    plt.savefig(filename)


for i in tqdm(range(50, 100, 5)):
    path = os.path.join(IMG_FOLDER, f'quantile_{i}.png')
    make_graph(dw, 0.01 * i, path)

for i in tqdm(range(95, 100)):
    path = os.path.join(IMG_FOLDER, f'quantile_{i}.png')
    make_graph(dw, 0.01 * i, path)
