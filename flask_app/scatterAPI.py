from flask import render_template, Blueprint, request, current_app, send_file, send_from_directory
from flask_sock import Sock
import json
import os
import time
from binning import histogrammap_scipy_bins, generate_xybins, generate_partitions, get_aggregator, get_info
import tables
import numpy as np
from multiprocessing import Pool, cpu_count
import functools
import concurrent

scatterAPI = Blueprint('scatterAPI', __name__, template_folder='templates')

sock = Sock(scatterAPI)

@sock.route('/hello')
def socket(ws):
    while True:
        ws.send(json.dumps({'message': 'Hello!'}))
        time.sleep(1)
        # ws.recv()


# call the render function with xy bounds: x0, x1, y0, y1
@sock.route('/render')
def render(ws):
    filename = "/home/jackson/code/research/binned_statistic_flask_app/flask_app/data_store/1000000001_points.pytable"

    # get the xy bounds from the request. TODO ADD THESE PARAMETERS TO THE REQUEST
    # xmin = request.args.get('x1', type=float)
    # xmax = request.args.get('x2', type=float)
    # ymin = request.args.get('y1', type=float)
    # ymax = request.args.get('y2', type=float)

    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    N_steps_per_partition = 50

    # loop through each chunk of data and send it to the client

    # ----------------- GENERATE INFORMATION FOR MULTIPROCESSING ----------------- #
    with tables.open_file(filename=filename, mode='r') as hdf5_file:
        col_x = hdf5_file.root.embed_x
        col_y = hdf5_file.root.embed_y
        nitems = len(col_x) - 1 # exclude the fake point at the end
        step_size = col_x.chunkshape[0]
        nbins = (100, 100) 
        xbins, ybins = generate_xybins([xmin, xmax], [ymin, ymax], nbins[0], nbins[1])
        partitions = generate_partitions(nitems, step_size, N=N_steps_per_partition, method='n_steps')
        # breakpoint()
        cumulative_histogram = -1 * np.ones((nbins[0], nbins[1]), dtype=np.int64)

    print(f'number of partitions: {len(partitions)-1}')

    # ----------------- BEGIN PARTITIONED MULTIPROCESSING ----------------- #
    aggregator = get_aggregator()
    p=Pool(cpu_count())
    start = time.time()
    for i in range(len(partitions)-1):  # -1 needed because we are using i+1
        results=p.map(functools.partial(histogrammap_scipy_bins, 
                                        fname=filename,xbins=xbins, 
                                        ybins=ybins,
                                        step_size=step_size,
                                        randompercent=None,
                                        colval='idx'),
                    range(partitions[i], partitions[i+1], step_size))
        print(i)
        
        for result in results:
            cumulative_histogram = aggregator(cumulative_histogram, result)

        # get the filtered color matrix.
        ids = cumulative_histogram.flatten().tolist()
        
        # replace all nan values with -1
        # this operation multiplies the time by 2
        ids = [-1 if np.isnan(x) else int(x) for x in ids]

        colors = get_info(ids, filename)

        ws.send(json.dumps({'indices': ids,
                            'colors': colors,
                            'iteration:': i}))
        


    # ----------------- ALTERNATE MULTIPROCESSING METHOD ----------------- #

    # -------------------------------------------------------------------- #
    
    end = time.time()
    print(end-start)
