#!/usr/bin/env python
# coding: utf-8
import numpy as np
import tables
from tqdm.autonotebook import tqdm
from multiprocessing import Pool
import functools
from scipy.stats import binned_statistic_2d
from collections import defaultdict
import time

def getminmax(val):
    minval = 100000
    maxval = -100000
    for i in range(0, val.nrows, val.chunkshape[0]):
        chunk = val[i:i + val.chunkshape[0]]
        tmax = chunk.max()
        maxval = tmax if tmax > maxval else maxval

        tmin = chunk.min()
        minval = tmin if tmin < minval else minval

    return minval,maxval

def generate_xybins(xrange, yrange, xbins, ybins):
    xbins=np.linspace(xrange[0], xrange[1], xbins+1)
    ybins=np.linspace(yrange[0], yrange[1], ybins+1)
    return xbins, ybins


def histogrammap_scipy_bins(i, fname=None, xbins=None, ybins=None,step_size=None,randompercent=None, colval=None):
    with tables.open_file(fname, mode='r') as hdf5_file:
        chunkx = hdf5_file.root.embed_x[i:i + step_size]
        chunky = hdf5_file.root.embed_y[i:i + step_size]
        if colval:
            chunkval = hdf5_file.root[colval][i:i + step_size] 

    if randompercent: #--- using a subset via sampling will result in faster speed, but less accuracy
        idx=np.random.randint(0,len(chunkx),int(len(chunkx)*randompercent)) #-- note,faster to load the entire chucnk and then subset, versus subsetting directly
        chunkx=chunkx[idx]
        chunky=chunky[idx]
        if colval:
            chunkval=chunkval[idx]

    counts, xedges, yedges, binnumber=binned_statistic_2d(chunkx, chunky,None,statistic='count',bins=[xbins,ybins])

    
    #--- identify bins which only have a single point, these are not "superpoints" but regular points and should be presented seperately
    xx,yy=(counts==1).nonzero()
    if len(xx)>0:
        #print(f'lencoiunt:{len(xx)}')
        xx+=1 #compensate for the "too small" bin
        yy+=1

        pointbins=np.ravel_multi_index([xx,yy],[len(xbins)+1,len(ybins)+1])
        singlepoints=np.where(np.isin(binnumber,pointbins))[0]+i #--- these points should be presented as "points", note they are offset by "i"
    else:
        singlepoints=None
    
    res=None
    if colval: #if we want statistics of a particular column, e.g., pred or GT, or whatever, this will do it
        nclass=3 #-- should be from config file
        nbins = (len(xbins)+1)*(len(ybins)+1)
        res=np.zeros((nclass,nbins),dtype=np.int64)

        for n in range(nclass):
            idx=chunkval==(n+1)
            c=np.bincount(binnumber[idx],minlength=nbins)
            res[n,:]=c #output is a nclass x nbin matrix

    return (counts,res,singlepoints)

# +
if __name__ == '__main__':
    
    fname="/home/jackson/research/code/binning_benchmarks/pytables/testData5h_1000000000.pytable"
    hdf5_file = tables.open_file(fname, mode='r')
    

    col_x=hdf5_file.root.embed_x
    col_y=hdf5_file.root.embed_y

    nitems = len(col_x)
    print(f'number of items: {nitems}')
    step_size=col_x.chunkshape[0]*10
    nbins=100

    xmin, xmax = getminmax(col_x)
    ymin, ymax = getminmax(col_y)
    print(f'xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}')

    xbins, ybins = generate_xybins([xmin, xmax], [ymin, ymax], nbins, nbins)
    print(f'xbins: {xbins}')
    print(f'ybins: {ybins}')

    hdf5_file.close()


    print("pool start")
    start = time.time()
    p=Pool(32)
    print("pool job exec")
    results=p.map(functools.partial(histogrammap_scipy_bins, fname=fname,xbins=xbins, ybins=ybins,
                                    step_size=step_size,randompercent=None,colval='pred'),range(0, nitems, step_size))
    counts= np.sum([r[0] for r in results],axis=0)
    binpreds= np.sum([r[1] for r in results],axis=0)
    singlepoints = [r[2] for r in results]
    
    end = time.time()

    print(f"pool time {end-start} (s)")
    
    print(counts)
    print(np.sum(binpreds))
    #print(singlepoints)
