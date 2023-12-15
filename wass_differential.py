#!/usr/bin/env python3

import ot
import ot.plot
import numpy as np
import matplotlib.pyplot as plt

BIN_COUNT = 1000
data = np.loadtxt("Langevin_1D.txt")
DATA_COUNT = len(data)
SAMPLES_PER_BIN = DATA_COUNT / BIN_COUNT
bins = np.array_split(data, BIN_COUNT)

dw = [ot.emd2_1d(bins[i], bins[i + 1]) for i in range(len(bins)-1)]
mean_dw = np.mean(dw)
std_dw = np.std(dw)
filtered_dw = [x if np.abs(x - mean_dw) > 2*std_dw else 0 for x in dw]

fig, ax = plt.subplots()
plt.figure(1, figsize=(100,5))
ax.plot(np.arange(DATA_COUNT), data, 'r', alpha=0.7)
# ax.scatter(SAMPLES_PER_BIN * np.arange(BIN_COUNT - 1), filtered_dw, c='b')
for index, sample in enumerate(dw):
    if np.abs(sample - mean_dw) > 2 * std_dw:
        ax.vlines(SAMPLES_PER_BIN * index, -2, 2, linestyle='dashed')

plt.savefig('dw.png')

fig.clear()
