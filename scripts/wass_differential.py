#!/usr/bin/env python3

import os
import ot
import ot.plot
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

IMG_FOLDER = './img/'

SAMPLES_PER_BIN = 100
raw_data = np.loadtxt("Langevin_1D.txt")
data = [np.array([v]) for v in raw_data]
SAMPLE_COUNT = len(data)
BIN_COUNT = int(SAMPLE_COUNT / SAMPLES_PER_BIN)
bins = np.array_split(data, BIN_COUNT)
dw = [ot.emd2_1d(bins[i], bins[i + 1]) for i in range(BIN_COUNT-1)]


def make_graph(samples, q, filename):
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    fig.set_size_inches(15, 5)
    quantile = np.quantile(samples, q)
    for index, sample in enumerate(samples):
        if sample - quantile > 0:
            ax.vlines(SAMPLES_PER_BIN * index, -2, 2, color='blue', linestyle='dashed')
    ax.plot(np.arange(SAMPLE_COUNT), data, 'r')
    plt.title("Change Point Detection for a 1D Langevin Trajectory\n" + fr"$\alpha = {SAMPLES_PER_BIN}, q = {q:.2f}$")
    plt.savefig(filename)


def print_graphs(lower_bound, upper_bound, step_size):
   for i in tqdm(range(lower_bound, upper_bound, step_size)):
       path = os.path.join(IMG_FOLDER, f'quantile_{i}.png')
       make_graph(dw, 0.01 * i, path)

print_graphs(95, 96, 1)

plt.clf()
fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
fig.set_size_inches(15, 5)
ax.plot(np.arange(SAMPLE_COUNT), data, 'r')
plt.title("Sample 1D Langevin Trajectory")
plt.savefig("sample_traj.png")
