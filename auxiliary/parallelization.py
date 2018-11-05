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

from pathos.multiprocessing import ProcessingPool as Pool

def parallelize_df(df, func, nthreads, use_pathos=True, **kwargs):
    nthreads = min(nthreads, len(df))
    nchunks = min(nthreads * 4, len(df))

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


