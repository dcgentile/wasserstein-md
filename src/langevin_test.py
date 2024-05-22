#!/usr/bin/env python3
import util
import pandas as pd
import numpy as np
from tqdm import tqdm
import ot
import matplotlib.pyplot as plt

lange_data = "../data/Langevin_1D.txt"
df = pd.read_csv(lange_data)
w = 300
n = len(df)
distances = np.zeros((n))
for t in tqdm(range(w, n - w)):
    a = df[t - w : t].to_numpy()
    b = df[t + 1 : t + w].to_numpy()
    distances[t] = ot.emd2_1d(a, b)
candidates = util.indentify_change_points(distances)
change_points = util.filter_change_points(distances, candidates)
print(change_points.shape)
fig, ax = plt.subplots()
for t in change_points:
    ax.vlines(t, -1.5, 1.5, colors="red", linestyle="dashdot")
plt.plot(df)
plt.show()
