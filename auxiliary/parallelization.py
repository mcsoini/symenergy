#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the auxiliary functions.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import multiprocessing
import numpy
import pandas as pd
import itertools
try:
    from pathos.multiprocessing import ProcessingPool as Pool
except Exception as e: print(e)

def parallelize_df(df, func, nthreads, use_pathos=False, **kwargs):
    nthreads = min(nthreads, len(df))
    nchunks = min(nthreads * 2, len(df))

    df_split = numpy.array_split(df, nchunks)
    if use_pathos:
        pool = Pool(nthreads)
        results = pool.map(func, df_split, **kwargs)
    else:
        pool = multiprocessing.Pool(nthreads)
        results = pool.map(func, df_split)
    pool.close()
    pool.join()
    pool.clear()
    pool.restart()
    print('parallelize_df: concatenating', end=' ... ')
    if isinstance(results[0], (list, tuple)):
        result = list(itertools.chain.from_iterable(results))
    else:
        result = pd.concat(results)
    print('done.')
    return result


