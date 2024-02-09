#!/usr/bin/env python3


import os
import ot
import numpy as np
import matplotlib.pyplot as plt


class ChangePointDetector:
    def __init__(self, datafile: str, ground_truth: str):
        data = np.loadtxt(datafile)
        truth = np.loadtxt(ground_truth)
        self.data = [np.array([v]) for v in data]
        self.truth = [i in truth for i in range(len(data))]

    def compute_change_points(
        self, quantile: int, sample_ratio: int, method="wass_diff"
    ):
        if method == "wass_diff":
            change_points = self.wass_diff_cpd(quantile, sample_ratio)
        elif method == "cdf_diff":
            change_points = self.cdf_diff_cpd(quantile, sample_ratio)
        elif method == "fourier":
            change_points = self.fourier_cpd(quantile, sample_ratio)
        else:
            change_points = []
        return change_points

    def ratio_of_transforms(self, ft1, ft2):
        return [x / y for (x, y) in zip(ft1, ft2)]

    def fourier_cpd(self, quantile: int, sample_ratio: int):
        bin_count = int(len(self.data) / sample_ratio)
        bins = np.array_split(self.data, bin_count)
        sorted = [np.sort(sample) for sample in bins]
        transforms = [np.fft.fft(cdf) for cdf in sorted]
        ratios = [
            self.ratio_of_transforms(transforms[i], transforms[i + 1])
            for i in range(len(transforms) - 1)
        ]
        norms = [np.linalg.norm(ratio) for ratio in ratios]
        cutoff = np.quantile(norms, 0.01 * quantile)
        return [norm > cutoff for norm in norms]

    def cdf_diff_cpd(self, quantile: int, sample_ratio: int):
        bin_count = int(len(self.data) / sample_ratio)
        bins = np.array_split(self.data, bin_count)
        sorted = [np.sort(sample) for sample in bins]
        diffs = [sorted[i] - sorted[i + 1] for i in range(len(sorted) - 1)]
        norms = [np.linalg.norm(diff) for diff in diffs]
        cutoff = np.quantile(norms, 0.01 * quantile)
        return [sample > cutoff for sample in norms]

    def wass_diff_cpd(self, quantile: int, sample_ratio: int):
        bin_count = int(len(self.data) / sample_ratio)
        bins = np.array_split(self.data, bin_count)
        dw = [ot.emd2_1d(bins[i], bins[i + 1]) for i in range(bin_count - 1)]
        cutoff = np.quantile(dw, 0.01 * quantile)
        return [sample > cutoff for sample in dw]

    def compute_score_components(
        self, tolerance: int, quantile: int, sample_ratio: int, method="wass_diff"
    ):
        tp = 0
        fp = 0
        fn = 0

        change_points = self.compute_change_points(
            quantile, sample_ratio, method=method
        )

        for index, candidate in enumerate(change_points):
            position = sample_ratio * index
            if position - tolerance < 0:
                truth_subset = self.truth[0 : (position + tolerance)]
            elif position + tolerance > len(self.data):
                truth_subset = self.truth[(position - tolerance) :]
            else:
                truth_subset = self.truth[
                    (position - tolerance) : (position + tolerance)
                ]

            if candidate and (True in truth_subset):
                tp += 1
            elif candidate and (not True in truth_subset):
                fp += 1
            elif (not candidate) and (True in truth_subset):
                fn += 1

        return tp, fp, fn

    def f1_score(
        self, tolerance: int, quantile: int, sample_ratio: int, method="wass_diff"
    ):
        tp, fp, fn = self.compute_score_components(
            tolerance, quantile, sample_ratio, method=method
        )
        return (2 * tp) / ((2 * tp) + fp + fn)

    def precision_score(self, tolerance: int, quantile: int, sample_ratio: int):
        tp, fp, fn = self.compute_score_components(tolerance, quantile, sample_ratio)
        return tp / (tp + fp)

    def recall_score(self, tolerance: int, quantile: int, sample_ratio: int):
        tp, fp, fn = self.compute_score_components(tolerance, quantile, sample_ratio)
        return tp / (tp + fn)

    def full_report(
        self, tolerance: int, quantile: int, sample_ratio: int, method="wass_diff"
    ):

        tp, fp, fn = self.compute_score_components(
            tolerance, quantile, sample_ratio, method=method
        )
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 / ((1 / recall) + (1 / precision))

        # print(f"{tp}, {fp}, {fn}")
        return recall, precision, f1

    def make_graph(
        self, tolerance: int, quantile: int, sample_ratio: int, method="wass_diff"
    ):

        filename = os.path.join(
            "..", "img", f"gt-comparison-{method}-q-{quantile}-alpha-{sample_ratio}.png"
        )
        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 5)
        change_points = self.compute_change_points(
            quantile, sample_ratio, method=method
        )
        f1, precision, recall = self.full_report(
            tolerance, quantile, sample_ratio, method=method
        )
        for index, sample in enumerate(change_points):
            if sample:
                ax.vlines(sample_ratio * index, -2, 2, color="blue", linestyle="dashed")
        for index, sample in enumerate(self.truth):
            if sample:
                ax.vlines(index, -2, 2, color="green", linestyle="dashed")
        ax.plot(np.arange(len(self.data)), self.data, "r")
        plt.title(
            "Change Point Detection for a 1D Langevin Trajectory\n"
            + rf"{method}, $\alpha = {sample_ratio}, q = {quantile}$"
            + f"\nf1 score:{f1}, prec: {precision}, rec: {recall}"
        )
        plt.savefig(filename)
