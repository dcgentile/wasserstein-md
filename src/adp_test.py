#!/usr/bin/env python3

import utils
import pandas as pd

adp_file = "../data/A_2D_1ps.dat"
names = ["phi", "psi"]
df = pd.read_csv(adp_file, names=names, sep=" ")
w = 25
n = len(df)
distances = utils.compute_wasserstein_distances(df, w)
candidates = utils.identify_change_points(distances)
change_points = utils.filter_change_points(distances, candidates)
samples = utils.create_sample_distributions(df, change_points, w, n - w)
labels = utils.cluster_sample_distributions(samples)
print(len(labels))
