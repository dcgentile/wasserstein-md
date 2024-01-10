#!/usr/bin/env python3

import ot
import ot.plot
import numpy as np
import matplotlib.pyplot as plt

SAMPLES_PER_BIN = 100
data = np.loadtxt("Langevin_1D.txt")[0:10000]
SAMPLE_COUNT = len(data)
BIN_COUNT = int(SAMPLE_COUNT / SAMPLES_PER_BIN)
I = np.identity(int(SAMPLES_PER_BIN))
bins = np.array_split(data, BIN_COUNT)
map_norms = [
    np.linalg.norm(np.diag(bins[i]) - ot.lp.emd_1d(bins[i], bins[i + 1])) for i in range(BIN_COUNT - 1) # pyright: ignore[reportGeneralTypeIssues]
]
fig, ax = plt.subplots()
ax.scatter(SAMPLES_PER_BIN * np.arange(BIN_COUNT - 1), map_norms)
ax.plot(np.arange(SAMPLE_COUNT), data, 'r', alpha=0.7)

fig.savefig('norms.png')
