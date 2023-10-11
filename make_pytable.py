# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import tables
from tqdm.autonotebook import tqdm

data = int(1e9)
with tables.open_file(f"{data + 1}_points.pytable", mode='w') as hdf5_file:

    filters = tables.Filters(complevel=9,complib='blosc')

    hdf5_file.create_earray(hdf5_file.root, "embed_x", tables.FloatAtom(), shape=[0], #chunkshape=[1],
                            filters=filters,expectedrows=(data+1))
    hdf5_file.create_earray(hdf5_file.root, "embed_y", tables.FloatAtom(), shape=[0], #chunkshape=[1],
                            filters=filters,expectedrows=(data+1))
    hdf5_file.create_earray(hdf5_file.root, "pred", tables.IntAtom(), shape=[0], #chunkshape=[1],
                            filters=filters,expectedrows=(data+1))
    hdf5_file.create_earray(hdf5_file.root, "idx", tables.IntAtom(), shape=[0], #chunkshape=[1],
                        filters=filters,expectedrows=(data+1))

    step=10000000

    for i in tqdm(range(0, data, step)): 
        rand=np.random.rand(min(data,step))
        hdf5_file.root.embed_x.append(rand)
        
        rand=np.random.rand(min(data,step))
        hdf5_file.root.embed_y.append(rand)

        pred=np.random.choice([1,2,3],size=min(data,step),p=[.5,.25,.25])
        hdf5_file.root.pred.append(pred)

        idx=np.arange(i, i+min(data,step))
        hdf5_file.root.idx.append(idx)


    # add a single point to the end to signify an empty bin. When a bin is empty, the id array will be -1. When the id array is -1, the client will know to color the bin gray.
    hdf5_file.root.embed_x.append([-1.0]) 
    hdf5_file.root.embed_y.append([-1.0])
    hdf5_file.root.pred.append([-1])
    hdf5_file.root.idx.append([data+1])
    

    y=hdf5_file.root.embed_y

    y.size_in_memory

    y.size_on_disk

    # +
    # # %%timeit
    # a=y[:]
    # -


