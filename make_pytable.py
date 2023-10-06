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

data = int(1e5)
with tables.open_file(f"testData5h_{data}.pytable", mode='w') as hdf5_file:

    filters = tables.Filters(complevel=9,complib='blosc')

    hdf5_file.create_earray(hdf5_file.root, "embed_x", tables.FloatAtom(), shape=[0], #chunkshape=[1],
                            filters=filters,expectedrows=data)
    hdf5_file.create_earray(hdf5_file.root, "embed_y", tables.FloatAtom(), shape=[0], #chunkshape=[1],
                            filters=filters,expectedrows=data )
    hdf5_file.create_earray(hdf5_file.root, "pred", tables.IntAtom(), shape=[0], #chunkshape=[1],
                            filters=filters,expectedrows=data )

    step=10000000

    for i in tqdm(range(0, data, step)): 
        rand=np.random.rand(min(data,step))
        hdf5_file.root.embed_x.append(rand)
        
        rand=np.random.rand(min(data,step))
        hdf5_file.root.embed_y.append(rand)

        pred=np.random.choice([1,2,3],size=min(data,step),p=[.5,.25,.25])
        hdf5_file.root.pred.append(pred)

    y=hdf5_file.root.embed_y

    y.size_in_memory

    y.size_on_disk

    # +
    # # %%timeit
    # a=y[:]
    # -


