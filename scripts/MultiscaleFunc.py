import Functions

import os
import ot
import ot.plot

from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from scipy.signal import find_peaks

###################################################################################################################
###################################################################################################################

# Approach 1

###################################################################################################################
###################################################################################################################

def multiscale_1(binsizes, data):
    wass_d = []
    cp = []
    detected = []
    for i in binsizes:
        wass_d=Functions.qW2(data, i, i)
        cp=Functions.get_change_points_Wass(wass_d, 0.85, 0)
        detected.append(cp*i)
        wass_d=[]
        cp=[]

    stacked = np.concatenate(detected)
    sorted_arr = np.sort(stacked)
    #new_sorted are the detected change points
    new_sorted = np.unique(sorted_arr)
    
    #detected is a matrix of original detected cps from each binsize
    return detected, new_sorted

###################################################################################################################
###################################################################################################################

# Approach 1

###################################################################################################################
###################################################################################################################

def multiscale_2():
    
    return 0


###################################################################################################################
###################################################################################################################

# Graph separate bins

###################################################################################################################
###################################################################################################################

def grapher(binsizes, colors, bins_labels, data):
    fig, ax = plt.subplots()
    fig.set_size_inches(24, 5)
    for i in range(len(colors)):
        ax.vlines(bins_labels[i], -2, 2, color=colors[i], linestyle='dashed', label=str(binsizes[i]))
    
    ax.plot(np.arange(len(data)), data, 'r', alpha=0.7)
    ax.legend()
    plt.title("Langevin Trajectory with labelled change points")







