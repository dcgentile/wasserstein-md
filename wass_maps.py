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
map_data = [np.linalg.norm(ot.lp.emd_1d(bins[i], bins[i + 1])) for i in range(BIN_COUNT - 1)]# pyright: ignore[reportGeneralTypeIssues]

fig, ax = plt.subplots()
ax.scatter(SAMPLES_PER_BIN * np.arange(BIN_COUNT - 1), map_data)
fig.savefig('dets.png')
