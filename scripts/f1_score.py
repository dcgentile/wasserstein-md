#!/usr/bin/env python3

from W2ChangePoints import ChangePointDetector
import os
import numpy as np
from tqdm import tqdm
import itertools

def main():
    data_path = os.path.join('..', 'data', 'Langevin_1D.txt')
    truth_path = os.path.join('..', 'data', 'ChangePts.txt')
    w2cp = ChangePointDetector(data_path, truth_path)

    w2cp.make_graph(95, 100)

    q_low = 85
    q_high = 100
    alpha_low = 25
    alpha_high = 500
    alpha_step = 25
    tolerance_low = 1
    tolerance_high = 1000
    tolerance_step = 25

    for tolerance in tqdm(range(tolerance_low, tolerance_high, tolerance_step)):
        score_matrix = np.empty((q_high - q_low, int(alpha_high / alpha_step)))

        for q, alpha in tqdm(itertools.product(range(q_low, q_high), range(alpha_low, alpha_high, alpha_step))):
            i = q_low - q
            j = int((alpha_high - alpha) / alpha_step)
            score_matrix[i][j] = w2cp.f1_score(tolerance, q, alpha)

        np.savetxt(os.path.join('..', 'data', f'f1-scores-tolerance-{tolerance}.txt'), score_matrix)


if __name__ == "__main__":
    main()
