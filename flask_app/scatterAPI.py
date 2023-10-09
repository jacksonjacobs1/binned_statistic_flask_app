from flask import render_template, Blueprint, request, current_app, send_file, send_from_directory
from flask_sock import Sock
import json
import os
import time
from binning import histogrammap_scipy_bins, generate_xybins, generate_partitions
import tables
import numpy as np
from multiprocessing import Pool, cpu_count
import functools


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
    filename = "./data_store/testData5h_1000000000.pytable"

    # get the xy bounds from the request. TODO ADD THESE PARAMETERS TO THE REQUEST
    # xmin = request.args.get('x1', type=float)
    # xmax = request.args.get('x2', type=float)
    # ymin = request.args.get('y1', type=float)
    # ymax = request.args.get('y2', type=float)

    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    N_steps_per_partition = 100

    # loop through each chunk of data and send it to the client

    # ----------------- GENERATE INFORMATION FOR MULTIPROCESSING ----------------- #
    with tables.open_file(filename=filename, mode='r') as hdf5_file:
        col_x = hdf5_file.root.embed_x
        col_y = hdf5_file.root.embed_y
        nitems = len(col_x)
        step_size = col_x.chunkshape[0]
        nbins = (100, 100) 
        xbins, ybins = generate_xybins([xmin, xmax], [ymin, ymax], nbins[0], nbins[1])
        partitions = generate_partitions(nitems, step_size, N=N_steps_per_partition, method='n_steps')
        # breakpoint()
        cumulative_histogram = np.zeros((nbins[0], nbins[1]), dtype=np.int64)



    # ----------------- BEGIN PARTITIONED MULTIPROCESSING ----------------- #
    p=Pool(cpu_count())
    start = time.time()
    for i in range(len(partitions)-1):  # -1 needed because we are using i+1
        results=p.map(functools.partial(histogrammap_scipy_bins, 
                                        fname=filename,xbins=xbins, 
                                        ybins=ybins,
                                        step_size=step_size,
                                        randompercent=None,
                                        colval='pred'),
                    range(partitions[i], partitions[i+1], step_size))
        # breakpoint()
        cumulative_histogram += np.sum(results, axis=0, dtype=np.int64)
        
        ws.send(json.dumps({'message': f'{int(cumulative_histogram.sum())} / {nitems}'}))
        # ws.send(json.dumps({'message': 'step'}))

    # ----------------- ALTERNATE MULTIPROCESSING METHOD ----------------- #
    # results=p.map_async(functools.partial(histogrammap_scipy_bins, 
    #                             fname=filename,xbins=xbins, 
    #                             ybins=ybins,
    #                             step_size=step_size,
    #                             randompercent=None,
    #                             colval='pred'),
    #         range(0, nitems, step_size))

    # while 

    # -------------------------------------------------------------------- #
    
    end = time.time()
    ws.send(json.dumps({'message': f'pool time {end-start} (s)'}))
