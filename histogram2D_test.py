import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import dask.array as da
import fast_histogram
import tables
import scipy.stats as stats
import os
import glob


def hist2D(x,y, bins):
    return np.histogram2d(x,y, bins, [[0,1],[0,1]])

def fast_hist2D(x,y, bins):
    return fast_histogram.histogram2d(x,y, bins, range=[[0,1],[0,1]])

def dask_hist2D(x,y, bins):
    return da.histogram2d(x, y, bins, range=[[0,1],[0,1]])

def scipy_hist2D(x,y, values,  bins):
    return stats.binned_statistic_2d(x, y, values, 'count', bins, range=[[0,1],[0,1]])

# --------------- fast-histogram -----------------

data_sizes = []
times = []

data_path = "/home/jackson/research/code/binning_benchmarks/pytables"
output_path = "/home/jackson/research/code/binning_benchmarks/benchmarks_output"
output_file = "histogram_benchmark2.png"

pytable_names = glob.glob(os.path.join(data_path, "*.pytable"))
pytable_names = [pytable_names[1]]

for name in tqdm(pytable_names):
    with tables.open_file(name, mode='r') as hdf5_file:
        x = hdf5_file.root.embed_x[:]
        y = hdf5_file.root.embed_y[:]
        pred = hdf5_file.root.pred[:]
    
    data_sizes.append(len(x))

    # time to plot
    start = time.time()


    # FAST-HISTOGRAM
    # fast_hist2D(x, y, bins=100)

    # SCIPY HISTOGRAM
    scipy_hist2D(x,y, pred, bins=100)

    end = time.time()
    times.append(end - start)

plt.plot(data_sizes, times)
plt.xlabel("Number of points")
plt.ylabel("Time to run (s)")   
plt.savefig(os.path.join(output_path, output_file))